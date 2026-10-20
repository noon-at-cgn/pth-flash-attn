# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the correctness grid harness (tests/gpu/grid_cases.py).

The GPU grid runs ~250 cases through the scalar reference kernel and
gates each with ``|out - ref| <= bound`` plus canary checks. A harness bug
there would either fail correct kernels (wasting a GPU run) or, worse, pass
broken ones. So the whole harness runs here first, on CPU, with the numpy
emulator of the kernel's arithmetic standing in for the kernel:

* the grid is the one the test matrix asks for (every axis value is
  present, the case count is pinned so it cannot silently shrink);
* a correct kernel passes (emulator: 0 violations, canaries intact);
* broken kernels fail: off-by-one mutants at the causal edge, tile edge,
  page edge and split edge, a wrong KV head, writes to padding rows and
  writes outside a strided view, and a window one key too wide
  and a non-causal case run as causal. These are the negative controls:
  without them a green grid would mean nothing.

The default tier takes about a minute; ``-m full`` runs every grid case
through the emulator (process pool).
"""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from pth_testing.reference import DenseKV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
grid = importlib.import_module("grid_cases")

# Pinned grid size. A change must be deliberate: edit
# the grid, this number and the runbook.
FULL_CASES = 328
QUICK_CASES = 66
# Worker processes for the full-tier emulator run: the CI container's --cpus.
CPU_JOBS = 4


# ---------------------------------------------------------------- grid shape
def test_grid_size_is_pinned():
    """The full grid has exactly FULL_CASES cases and the quick tier is a
    subset of QUICK_CASES of them.

    Why: a refactor that drops a family would otherwise shrink the grid's
    evidence without anyone noticing; the GPU run instructions quote these counts.
    """
    full = grid.cases("full")
    quick = grid.cases("quick")
    assert len(full) == FULL_CASES
    assert len(quick) == QUICK_CASES
    assert {c.name for c in quick} <= {c.name for c in full}
    assert len({c.name for c in full}) == len(full)
    # Distinct data per case, except the production cases of one descriptor,
    # which share their data on purpose (see test_production_cases_share_data).
    assert len({c.seed for c in full}) == len({c.data_name or c.name for c in full})


def test_grid_covers_every_plan_axis_value():
    """Every value of every launch axis (as assigned for the grid) is exercised.

    Why: the grid is a covering design, not a product; this test is what
    makes "covers B {1,2,3,7}, q {...}, L {...}, ..." a checked statement.
    """
    full = grid.cases("full")
    seen_q = {q for c in full for q in c.q_lens}
    seen_L = {L for c in full for L in c.seq_lens}
    assert {len(c.q_lens) for c in full} >= {1, 2, 3, 7}
    assert seen_q >= {1, 2, 5, 10, 11, 16, 37, 512, 2048}
    assert seen_L >= {1, 15, 16, 63, 64, 65, 815, 816, 817, 4096, 65536, 262144}
    assert {c.G for c in full} >= {1, 4, 6, 8, 16}
    assert {c.D for c in full} == {64, 128, 256}
    assert {c.block_size for c in full} >= {16, 64, 528, 816}
    assert {c.layout for c in full} == {"NHD", "HND"}
    assert {c.num_splits for c in full} >= {1, 2, 34, grid.S_MAX}
    assert grid.S_MAX == 204
    assert {c.q_dtype for c in full} == {"bf16", "fp16"}
    assert {c.out_dtype for c in full} == {"f32", "q"}
    modes = {
        m
        for c in full
        for m in ([c.scale_mode] if isinstance(c.scale_mode, str) else c.scale_mode)
    }
    assert modes >= {"normal", "floor", "extreme", "mixed"}
    assert any(c.pad_page_bytes for c in full)
    assert any(c.storage_offset_bytes % 16 for c in full)
    assert any(c.high_block_ids for c in full)
    assert any(c.strided_io for c in full)
    assert any(c.pad_rows for c in full), "padding rows (all-masked) missing"
    assert any(0 in c.q_lens for c in full), "empty request missing"
    assert any(c.ws_token_splits for c in full)
    # The window axes.
    win = [c for c in full if c.windowed]
    assert {c.window for c in win} >= {64, 65, 2047, 2048, 4096}
    assert {c.causal for c in win} == {True, False}
    assert {L for c in win for L in c.seq_lens} >= {
        63,
        64,
        65,
        2046,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        815,
        816,
        817,
        65536,
    }
    assert {q for c in win for q in c.q_lens} >= {1, 8, 16}
    assert {c.G for c in win} >= {4, 6}
    assert {c.D for c in win} == {128}
    assert {c.Hkv for c in win if c.window == 2048 and not c.causal} >= {4, 8}
    assert {c.num_splits for c in win} >= {1, 2, 3, 34}
    # L = 262144 only with B = 1 (memory and run time).
    assert all(len(c.q_lens) == 1 for c in full if 262144 in c.seq_lens)
    fams = Counter(c.family for c in full)
    assert set(fams) == {
        "paging",
        "rows",
        "batch",
        "splits",
        "long",
        "scales",
        "dtypes",
        "fuzz",
        "production",
        "window",
    }


def test_sanitizer_cases_are_small_grid_cases():
    """The compute-sanitizer subset names real cases with L <= 5000.

    Why: sanitizers slow kernels 10-100x; a long case there would stall the
    GPU run, a typo would silently run nothing.
    """
    by_name = {c.name: c for c in grid.cases("full")}
    for n in grid.SANITIZER_CASES:
        assert n in by_name
        # 5000 = the fuzz family's longest draw; nothing from
        # the 64k/256k families.
        assert max(by_name[n].seq_lens) <= 5000


def test_sentinel_positions_hit_every_boundary():
    """Sentinels sit on first/last token, tile edge, page edge, split edges.

    Why: at long L one wrong token is invisible to the gate unless it carries
    weight; these are exactly the tokens that boundary bugs
    touch. L = 817, block 816, S = 2: 13 tiles, split 1 starts at tile 6.
    """
    assert grid.sentinel_positions(817, 816, (2,)) == [0, 63, 64, 383, 384, 815, 816]
    assert grid.sentinel_positions(1, 16, (1,)) == [0]
    assert grid.sentinel_positions(17, 16, (1,)) == [0, 15, 16]
    # Several split counts: the union of their edges (13 tiles: S=2 -> 6, S=3
    # -> 4 and 8).
    assert grid.sentinel_positions(817, 816, (2, 3)) == [
        0,
        63,
        64,
        255,
        256,
        383,
        384,
        511,
        512,
        815,
        816,
    ]


def test_production_rows_use_the_planner():
    """Production cases carry the planner's split count and fit W_ts = 1088.

    Why: (f) split-count independence must include the S the planner really
    launches, inside the workspace really reserved.
    """
    assert grid.production_ws_token_splits() == 1088
    prod = [c for c in grid.cases("full") if c.family == "production"]
    for name, q, _ in grid.production_descriptors():
        s_plan = grid.production_splits(q)
        mine = {c.num_splits for c in prod if c.name.startswith(f"prod-{name}-")}
        assert mine == {s_plan, 1, 2, 34}
        assert all(c.num_splits * c.num_tokens <= 1088 for c in prod)


def test_production_cases_share_data():
    """All S of one production descriptor see identical q and cache bytes.

    Why: split-count independence compares outputs of different S with each
    other; with different data per case the comparison would be meaningless
    (Seeding by case name instead made the comparison fail by 100-800x although
    every case passed its own gate.)
    """
    cs = [c for c in grid.cases("full") if c.name.startswith("prod-dflash-b2-")]
    assert len(cs) >= 3
    first = grid.build_inputs(cs[0])
    for c in cs[1:]:
        other = grid.build_inputs(c)
        assert torch.equal(first.q, other.q)
        assert torch.equal(first.cache.storage, other.cache.storage)
        assert torch.equal(first.cache.block_table, other.cache.block_table)


# ---------------------------------------------------------------- buffers
def test_canary_marks_exactly_the_written_rows():
    """The out/workspace canary check accepts writes to request rows only.

    Why: the C ABI promises rows >= qsl[num_reqs] and bytes outside the views
    are never written; the check must see a single stray byte, and must not
    flag legitimate writes (positive and negative control).
    """
    c = grid.case_by_name("fuzz-00")  # strided out
    assert c.strided_io
    b = grid.make_buffers(c)
    n = sum(c.q_lens)
    b.out[:n] = 0
    assert grid._outside_rows_untouched(b.out_buf, b.out, 0, slice(0, n))
    b.out_buf.view(torch.uint8).view(-1)[-1] = 0  # last byte of the stride padding
    assert not grid._outside_rows_untouched(b.out_buf, b.out, 0, slice(0, n))


def test_rebase_view_keeps_strides_and_offset():
    """Moving a buffer keeps its view's strides and offset."""
    c = grid.case_by_name("fuzz-00")
    b = grid.make_buffers(c)
    clone = b.out_buf.clone()
    v = grid.rebase_view(b.out, b.out_buf, clone)
    assert v.stride() == b.out.stride() and v.shape == b.out.shape
    clone[0, 0, 0] = 7
    assert float(v[0, 0, 0]) == 7.0


def test_expected_cache_key_changes_with_the_case(tmp_path, monkeypatch):
    """Cached fp64 references are reused only for the identical case and code.

    Why: the bound is precomputed before the GPU run; a stale entry used for
    a different case would turn the gate into a comparison with the wrong
    answer.
    """
    monkeypatch.setattr(grid, "EXPECTED_DIR", tmp_path)
    c = grid.case_by_name("page-L1-bs16-NHD-D64-gqa1")
    other = grid.dataclasses.replace(c, num_splits=2)
    assert grid.expected_path(c) != grid.expected_path(other)
    assert grid.expected_path(c, "code-a") != grid.expected_path(c, "code-b")
    exp1 = grid.load_expected(c)
    assert grid.expected_path(c).exists()
    exp2 = grid.load_expected(c)
    np.testing.assert_array_equal(exp1.ref, exp2.ref)
    np.testing.assert_array_equal(exp1.bound, exp2.bound)


# ---------------------------------------------------------------- emulator
SMOKE = (
    "page-L65-bs16-NHD-D128-gqa1",
    "page-L817-bs816-NHD-D128-gqa1",
    "batch-b7-mixed-pad5-bs16",
    "split-L817-S34",
    "fuzz-00",
    "dtype-qfp16-outq-decode",
    "win-W64-L65-nc-q16-gqa4-D128-S3",
    "win-drafter-b3-empty-pad3",
)


@pytest.mark.parametrize("name", SMOKE)
def test_emulator_passes_grid_case(name):
    """The emulated kernel passes the gate on representative grid cases.

    Why: the control for every mutant below; it also shows that the gate
    (bound for the reference tiling) is satisfiable by the coarser tensor-core
    arithmetic, so a grid failure is the kernel's, not the grid's.
    """
    r = grid.run_case(grid.case_by_name(name), grid.emulator_kernel(), use_cache=False)
    assert r.passed, r.message()
    assert 0.0 < r.tightness <= 1.0


def _mutant(name, mutate):
    return grid.run_case(
        grid.case_by_name(name), grid.emulator_kernel(mutate), use_cache=False
    )


def _drop(x: DenseKV, idx) -> DenseKV:
    keep = torch.ones(x.seq_len, dtype=torch.bool)
    keep[idx] = False
    return DenseKV(x.k_int[keep], x.k_scale[keep], x.v_int[keep], x.v_scale[keep])


def test_mutant_drops_the_diagonal_key():
    """`<` instead of `<=` at the causal limit (kernel sees L-1 tokens) fails.

    The sentinel at L-1 = 816 carries most of the weight, so missing it
    changes the output by O(1).
    """
    r = _mutant(
        "page-L817-bs816-NHD-D128-gqa1",
        lambda inp: [_drop(x, slice(x.seq_len - 1, None)) for x in inp.kv],
    )
    assert not r.passed and r.violations > 0, r.message()
    assert "page-L817-bs816-NHD-D128-gqa1" in r.message()


def test_mutant_skips_the_last_token_of_a_tile():
    """A tile loop that stops at 63 instead of 64 tokens (token 63 lost) fails."""
    r = _mutant(
        "page-L65-bs16-NHD-D128-gqa1", lambda inp: [_drop(x, 63) for x in inp.kv]
    )
    assert not r.passed and r.violations > 0, r.message()


def test_mutant_reads_the_wrong_slot_at_a_page_edge():
    """Token BS read from slot BS-1 (off-by-one in t % BS / t // BS) fails."""

    def mutate(inp):
        out = []
        for x in inp.kv:
            k, ks, v, vs = (a.clone() for a in (x.k_int, x.k_scale, x.v_int, x.v_scale))
            for a in (k, ks, v, vs):
                a[16] = a[15]
            out.append(DenseKV(k, ks, v, vs))
        return out

    r = _mutant("page-L65-bs16-NHD-D128-gqa1", mutate)
    assert not r.passed and r.violations > 0, r.message()


def test_mutant_loses_the_first_token_of_every_split():
    """Splits starting one token late (tile range off by one) fail."""
    c = grid.case_by_name("split-L817-S34")
    tiles = -(-817 // 64)
    starts = sorted(
        {s * tiles // c.num_splits * 64 for s in range(1, c.num_splits)} - {0}
    )
    assert starts, "the case must have split edges inside the sequence"
    r = _mutant(c.name, lambda inp: [_drop(x, starts) for x in inp.kv])
    assert not r.passed and r.violations > 0, r.message()


def test_mutant_uses_the_wrong_kv_head():
    """Query head h reading KV head (h // G + 1) % Hkv fails."""

    def mutate(inp):
        return [
            DenseKV(*(a.roll(1, 1) for a in (x.k_int, x.k_scale, x.v_int, x.v_scale)))
            for x in inp.kv
        ]

    r = _mutant("page-L65-bs16-NHD-D128-gqa1", mutate)
    assert not r.passed and r.violations > 0, r.message()


# The smallest non-causal window case (q = 16 over L = 65, W = 64,
# S = 3): sentinels at both sides of the first and last rows' window edges.
WINDOW_MUTANT_CASE = "win-W64-L65-nc-q16-gqa4-D128-S3"


def test_mutant_window_one_key_too_wide():
    """A window of W + 1 keys (lo = p - W instead of p - W + 1) fails.

    The last row (p = 64) then sees key 0, a sentinel just outside its window.
    """
    r = grid.run_case(
        grid.case_by_name(WINDOW_MUTANT_CASE),
        grid.emulator_kernel(mutate_window=lambda c: (c.window + 1, c.causal)),
        use_cache=False,
    )
    assert not r.passed and r.violations > 0, r.message()


def test_mutant_noncausal_run_as_causal():
    """A non-causal case computed causally (the block's later keys masked) fails.

    The first row (p = 49) loses keys 50..64, the sentinel at 64 among them.
    """
    r = grid.run_case(
        grid.case_by_name(WINDOW_MUTANT_CASE),
        grid.emulator_kernel(mutate_window=lambda c: (c.window, True)),
        use_cache=False,
    )
    assert not r.passed and r.violations > 0, r.message()


def test_mutant_writes_padding_rows():
    """A kernel that writes rows >= qsl[num_reqs] (graph padding) fails on
    the canary even though every checked value is right."""
    c = grid.case_by_name("batch-b7-mixed-pad5-bs16")
    base = grid.emulator_kernel()

    def kernel(inp, bufs):
        run = base(inp, bufs)
        bufs.out[int(inp.qsl[-1]) :] = 0
        return run

    r = grid.run_case(c, kernel, use_cache=False)
    assert r.violations == 0 and not r.canary_out_ok and not r.passed


def test_mutant_writes_outside_a_strided_view():
    """A kernel that assumes contiguous out (writes into the stride padding)
    fails on the canary."""
    c = grid.case_by_name("fuzz-00")
    base = grid.emulator_kernel()

    def kernel(inp, bufs):
        run = base(inp, bufs)
        bufs.out_buf[0, 0, c.D] = 0  # first padding element after head 0's row
        return run

    r = grid.run_case(c, kernel, use_cache=False)
    assert not r.canary_out_ok and not r.passed


def test_nan_output_is_a_violation_with_ratio_inf():
    """A NaN in a valid row counts as a violation and reports ratio inf.

    Why: NaN compares False with everything; a gate written as
    ``(err > bound).any()`` would pass it.
    """
    c = grid.case_by_name("page-L1-bs16-NHD-D64-gqa1")
    base = grid.emulator_kernel()

    def kernel(inp, bufs):
        run = base(inp, bufs)
        bufs.out[0, 0, 0] = float("nan")
        return run

    r = grid.run_case(c, kernel, use_cache=False)
    assert r.violations == 1 and r.nonfinite == 1 and r.tightness == float("inf")
    assert r.worst["index"] == [0, 0, 0]


@pytest.mark.full
@pytest.mark.slow
@pytest.mark.timeout(0)
def test_emulator_passes_the_full_grid():
    """Every grid case passes with the emulator in place of the kernel.

    Why: run before a GPU run, this proves the whole grid (all ~250 cases,
    up to L = 262144 and q = 2048) is satisfiable by a correct design, so a
    grid failure is a kernel bug. Measured with the CLI at 16 workers: 41 min
    (2026-10-11); at --cpus 4 expect several times that, so before a GPU run
    prefer `python tests/gpu/grid_cases.py emulate --tier full --jobs 16`. The
    only test marked both full and slow: CI deselects it (`-m "not (full and
    slow)"` in tools/run_ci_local.sh), it cannot finish within the suite's
    600 s per-test timeout, hence no timeout here for a run by hand.
    """
    names = [c.name for c in sorted(grid.cases("full"), key=grid._cost, reverse=True)]
    recs = grid.pool_map(grid._emulate_worker, names, CPU_JOBS)
    bad = [r for r in recs if not r["passed"]]
    assert len(recs) == FULL_CASES
    assert not bad, [(r["case"], r["violations"], r["tightness"]) for r in bad]
