# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the kernel benchmark (bench/kernel_bench.py) and the
statistics it adds to bench/stats.py (eta_ci, choose_clock_domain).

The benchmark's numbers decide the decode gates (decode eta_mem >= 0.90 of
the copy ceiling, verify eta_mma >= 0.60). Everything except the GPU timing
itself is checked here on CPU:

* the work each launch is credited with (KV bytes read once, useful and
  executed flops) against brute-force counts: a wrong formula moves eta
  directly;
* the grid contains every row the gates need, and the launch geometry is
  the planner's;
* the CUPTI span bookkeeping refuses a block whose kernel count is wrong;
* eta_ci covers the truth when processes differ and is wider than the
  bootstrap alone (the cluster-aware interval), and the measured
  clock-domain choice picks the memory clock only for clock-insensitive
  arms;
* the whole CPU pipeline: kernel_bench run on the host reference library ->
  raw files -> bench_report.py records that validate against
  results.schema.json -> kernel_gate.py, which must refuse the host timer.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
kb = importlib.import_module("kernel_bench")
stats = importlib.import_module("stats")
bench_report = importlib.import_module("bench_report")
kernel_gate = importlib.import_module("kernel_gate")
from pth_flash_attn import planner  # noqa: E402

NUM_SMS = 68


# ---------------------------------------------------------------- work
def test_kv_bytes_are_one_read_of_every_cell():
    """(2D + 8) bytes per token per KV head: the 520-B cell at D=256 that
    sol_copy streams, so eta_mem compares equal amounts of data."""
    assert kb.kv_bytes(1, 1, 256) == 520
    assert (
        kb.kv_bytes(131072, 2, 256) == 136314880
    )  # sol_copy L=128k record of the ceilings runs


@pytest.mark.parametrize(("q", "L"), [(1, 1), (1, 1000), (4, 4), (10, 817), (33, 33)])
def test_useful_flops_count_visible_keys(q, L):
    """Closed form = brute force over bottom-right causal visibility.

    Why: eta_mma_useful divides by this; an off-by-one in the causal count
    is a few per cent at q = L (prefill)."""
    Hq, D = 3, 64
    brute = sum(4 * D * Hq * (L - q + i + 1) for i in range(q))
    assert kb.useful_flops(q, L, Hq, D) == brute


def test_kernel_source_root_follows_the_built_commit(tmp_path):
    """The Python that fills the kernel's C structs comes from the commit the
    library was built from (build/k/src), not from a working tree that may be
    edited meanwhile; stand-in kernels and CPU runs use the working tree.
    Why: an uncommitted edit of tests/gpu/attn_lib.py (a symbol the benchmarked
    commit lacks) would kill a rerun in the middle of a benchmark run."""
    k = tmp_path / "build" / "k"
    (k / "src" / "python" / "pth_flash_attn").mkdir(parents=True)
    lib = str(k / "libpth_attn.so")
    assert kb.kernel_source_root(["x", "run", "--kernel-lib", lib]) == k / "src"
    assert kb.kernel_source_root(["x", f"--kernel-lib={lib}"]) == k / "src"
    assert kb.kernel_source_root(["x", "run"]) == kb.ROOT
    other = tmp_path / "other" / "libpth_attn.so"  # no src next to it
    assert kb.kernel_source_root(["x", "--kernel-lib", str(other)]) == kb.ROOT


def test_executed_flops_pad_rows_and_tiles():
    """Executed flops = whole 64-token tiles x the MMA rows issued per row tile.

    q = 10, L = 100, bq = 2, 16 issued rows (tile16, wm = 1): five row tiles
    ending at keys 92, 94, ..., 100 -> 2 tiles each; 4 D x 16 x 64 x 10 tiles
    x Hkv. And executed >= useful always (padding never removes work)."""
    D, Hkv = 256, 2
    assert (
        kb.executed_flops(10, 100, Hkv, D, bq=2, rows=16) == 4 * D * 16 * 64 * 10 * Hkv
    )
    for q, L, bq, rows in ((1, 65536, 1, 8), (10, 4096, 10, 64), (512, 512, 10, 64)):
        assert kb.executed_flops(q, L, Hkv, D, bq, rows) >= kb.useful_flops(
            q, L, 6 * Hkv, D
        )


@pytest.mark.parametrize(
    ("kernel", "rows", "wm", "issued"),
    [
        # decode kernel: whole 8-row n-tiles (S^T = K Q^T has the rows on N)
        (kb.KERNEL_DECODE, 6, 1, 8),  # q = 1, G = 6: one n-tile
        (kb.KERNEL_DECODE, 12, 1, 16),  # the planner's unbalanced bq = 2 at q = 1
        (kb.KERNEL_DECODE, 24, 2, 24),  # q = 4: three n-tiles, no padding
        (kb.KERNEL_DECODE, 30, 2, 32),  # bq = 5: padded to four n-tiles
        # 16-row-tile kernel: 16 wm rows whatever the fill
        (kb.KERNEL_TILE16, 12, 1, 16),
        (kb.KERNEL_TILE16, 24, 2, 32),
        (kb.KERNEL_TILE16, 60, 4, 64),
    ],
)
def test_issued_rows_per_kernel_variant(kernel, rows, wm, issued):
    """The 16-row-tile kernel multiplies 16 wm rows per KV tile, the swap-AB
    decode kernel 8 ceil(rows / 8). Why: the old model charged 16 rows to a
    kernel that issues 8 at q = 1 (and 24 at q = 4), so eta_mma of the decode
    rows would have been off by 2x and 1.33x."""
    assert kb.issued_rows(kernel, rows, wm) == issued


def test_executed_flops_of_the_decode_rows():
    """Closed forms of the quick-tier decode rows at L = 64k (1024 tiles),
    D = 256, Hkv = 2, G = 6: q = 1 issues 8 rows, q = 4 issues 24 (one row
    tile of the balanced bq = 4), q = 10 runs the 16-row-tile kernel (60 rows
    in WM = 4, 64 issued)."""
    tiles, D, Hkv = 1024, 256, 2
    want = {1: 8, 4: 24, 10: 64}
    for q, rows in want.items():
        s = next(x for x in kb.grid("decode") if x.name == f"dec-q{q}-b1-L64k")
        (p,) = kb.plan_shape(s, NUM_SMS)
        assert p.flops_executed == 4 * D * rows * 64 * tiles * Hkv, q
    d1 = kb.plan_shape(
        next(x for x in kb.grid("decode") if x.name == "dec-q1-b1-L64k"), NUM_SMS
    )
    assert d1[0].kernel == kb.KERNEL_DECODE


def _fake_lib(wn_for_rows, n_tile=64):
    """A library double with the pth_attn_tiling contract (n_tile, sub, wn)."""

    class Lib:
        def tiling(self, head_size, bq, group):
            return n_tile, 64, wn_for_rows(bq * group)

    return Lib()


def test_library_variant_follows_the_loaded_library():
    """The variant comes from pth_attn_tiling, not from the dispatch rule: a
    library without the decode kernel reports
    wn = 4 / wm at <= 32 rows and is modelled as the 16-row-tile kernel; wn = 1
    at 60 rows (WM = 4) is the 16-row-tile kernel too, never the decode one."""
    wm_of = lambda rows: 1 if rows <= 16 else 2 if rows <= 32 else 4  # noqa: E731
    with_decode = kb.library_variant(
        _fake_lib(lambda rows: 1 if rows <= 32 else 4 // wm_of(rows))
    )
    assert with_decode(256, 1, 6) == kb.KERNEL_DECODE
    assert with_decode(256, 5, 6) == kb.KERNEL_DECODE
    assert with_decode(256, 10, 6) == kb.KERNEL_TILE16
    no_decode = kb.library_variant(_fake_lib(lambda rows: 4 // wm_of(rows)))
    assert no_decode(256, 1, 6) == kb.KERNEL_TILE16
    assert no_decode(256, 10, 6) == kb.KERNEL_TILE16  # wn = 1, 60 rows
    assert with_decode(128, 1, 6) == kb.KERNEL_TILE16  # decode kernel is D = 256
    s = next(x for x in kb.grid("decode") if x.name == "dec-q1-b1-L64k")
    (old,) = kb.plan_shape(s, NUM_SMS, no_decode)
    (new,) = kb.plan_shape(s, NUM_SMS, with_decode)
    assert old.flops_executed == 2 * new.flops_executed  # 16 rows vs 8


def test_library_variant_refuses_another_load_tile():
    """executed_flops counts 64-token tiles; a library with another tile is an
    error, not a silently wrong eta_mma."""
    variant = kb.library_variant(_fake_lib(lambda rows: 1, n_tile=32))
    with pytest.raises(RuntimeError, match="tile 32"):
        variant(256, 1, 6)


def test_plan_is_the_planner_s():
    """Each slice gets the planner's wm, S and the balanced bq = min(planner
    bq, q) for the device's SM count, with a workspace never below the
    production reservation. The planner's bq is 2 at q = 1 (12
    rows, two n-tiles of the decode kernel); the harness passes 1."""
    s = next(x for x in kb.grid("decode") if x.name == "dec-q1-b1-L64k")
    (p,) = kb.plan_shape(s, NUM_SMS)
    ref = planner.plan_launch(
        num_tokens=1,
        num_reqs=1,
        max_query_len=1,
        G=6,
        Hkv=2,
        D=256,
        num_sms=NUM_SMS,
        is_prefill=False,
        ws_token_splits=1088,
    )
    assert (ref.wm, ref.bq, ref.num_splits) == (1, 2, 34)
    assert (p.wm, p.bq, p.num_splits) == (1, 1, 34)
    assert p.bytes_kv == kb.kv_bytes(65536, 2, 256)
    s4 = next(x for x in kb.grid("decode") if x.name == "dec-q4-b1-L64k")
    (p4,) = kb.plan_shape(s4, NUM_SMS)
    assert (p4.wm, p4.bq) == (2, 4)  # planner bq 5: 4 tokens, 24 rows, 3 n-tiles
    s10 = next(x for x in kb.grid("decode") if x.name == "dec-q10-b1-L64k")
    (p10,) = kb.plan_shape(s10, NUM_SMS)
    assert (p10.wm, p10.bq) == (4, 10)  # unchanged: bq <= q
    mix = next(x for x in kb.grid("mixed"))
    pre, dec = kb.plan_shape(mix, NUM_SMS)
    assert pre.is_prefill and not dec.is_prefill and dec.reqs == (1,)
    assert pre.bq == 10 and dec.bq == 4  # the prefill chunk keeps the planner's


def test_grid_holds_every_gate_row():
    """The quick tier is exactly the decode gate rows (KD: q in {1,4},
    VERIFY: q = 10; B in {1,2}; L >= 64k) plus prefill 8192 and the mixed
    batch; the full grid has 192 decode, 24 prefill and 1 mixed shape.

    Why: a gate can only pass on rows that exist; kernel_gate.py requires
    every one of them."""
    full = kb.grid("all")
    assert [s.family for s in full].count("decode") == 192
    assert [s.family for s in full].count("prefill") == 24
    assert len({s.name for s in full}) == len(full)
    quick_dec = {s.name for s in kb.select("decode", "quick")}
    want = {
        f"dec-q{q}-b{b}-L{L // 1024}k"
        for q in (1, 4, 10)
        for b in (1, 2)
        for L in (65536, 131072, 204800, 262144)
    }
    assert quick_dec == want
    # The prefill chunk's longest prefix is full-tier only (~800 s per
    # process): the PREFILL gate still expects it, so a quick run reports
    # that row as missing instead of passing on three of four prefixes.
    quick_pre = {s.name for s in kb.select("prefill", "quick")}
    assert quick_pre == {"pre-q8192-P0k", "pre-q8192-P32k", "pre-q8192-P128k"}
    prefill_rule = next(r for r in kernel_gate.RULES if r.id == "PREFILL")
    assert "pre-q8192-P200k" in kernel_gate.expected_rows(prefill_rule)
    for rule in kernel_gate.RULES:
        assert kernel_gate.expected_rows(rule), rule.id


def test_rotations_give_four_l2_between_reuses():
    """Cold L2: copies x working set >= 4 x L2, one copy when a
    launch alone exceeds that."""
    l2 = 5 * 2**20
    assert kb.rotations(1 * 2**20, l2) == 20
    assert kb.rotations(4 * l2, l2) == 1
    for ws in (12345, 2**20, 3 * 2**20):
        n = kb.rotations(ws, l2)
        assert n * ws >= 4 * l2 > (n - 1) * ws


def test_spans_group_kernels_per_launch():
    """A launch of attention + combine is timed from the first kernel's
    start to the last kernel's end; a missing record is an error, never a
    shorter block (negative control)."""
    recs = [(0, 10), (12, 15), (100, 130), (131, 140)]
    assert kb.spans(recs, 2, 2) == [15.0, 40.0]
    with pytest.raises(RuntimeError, match="expected 4 kernel records"):
        kb.spans(recs[:3], 2, 2)


def test_ref_arm_only_on_short_launches():
    """The scalar reference (~10 GB/s) joins only shapes whose launch stays
    short; it is a sanity row, never a ceiling."""
    names = {s.name: s.has_ref for s in kb.grid("all")}
    assert names["dec-q1-b1-L64k"] and names["dec-q1-b1-L1k"]
    assert not names["dec-q1-b2-L64k"] and not names["pre-q8192-P0k"]


# ---------------------------------------------------------------- stats
def _process_sample(rng, P, n, true, offs_sd, sd):
    offs = rng.normal(0.0, offs_sd, P)
    x = np.concatenate(
        [true * np.exp(offs[p] + rng.normal(0, sd, n)) for p in range(P)]
    )
    return x, np.repeat(np.arange(P), n)


def test_eta_ci_is_the_ratio_and_covers_with_process_offsets():
    """eta = work / (median T x median F); with between-process offsets
    (~0.3-0.6 % on the ceilings runs) the cluster-aware interval covers the
    true eta in >= 90 % of 40 synthetic runs and is wider than the bootstrap
    alone, which covers far less often (the reason for the cluster-aware
    interval)."""
    rng = np.random.default_rng(1)
    work, T, F = 1e9, 1.5e-3, 7e11  # true eta = 1e9 / (1.5e-3 * 7e11) = 0.952
    true = work / (T * F)
    cover_c = cover_b = 0
    for seed in range(40):
        t, ts = _process_sample(rng, 6, 30, T, 0.005, 0.002)
        f, fs = _process_sample(rng, 6, 30, F, 0.005, 0.002)
        c = stats.eta_ci(t, ts, work, f, fs, n_resamples=2000, seed=seed)
        b = stats.eta_ci(
            t, ts, work, f, fs, n_resamples=2000, seed=seed, cluster_robust=False
        )
        assert c.lo <= b.lo and c.hi >= b.hi and c.method.endswith("+cluster-t")
        cover_c += c.lo <= true <= c.hi
        cover_b += b.lo <= true <= b.hi
    assert cover_c >= 36, cover_c
    assert cover_b < cover_c
    assert math.isclose(
        stats.eta_ci(
            [2.0, 2.0], [0, 0], 4.0, [1.0, 1.0], [0, 0], n_resamples=10
        ).estimate,
        2.0,
    )


def test_eta_ci_single_process_side_has_no_between_term():
    """With one process on both sides no between-process sd exists: plain
    bootstrap, and the method string says so."""
    c = stats.eta_ci(
        [1.0, 1.1, 0.9], [0, 0, 0], 1.0, [1.0, 1.0, 1.1], [0, 0, 0], n_resamples=200
    )
    assert "cluster" not in c.method


def _blocks_and_samples(clock_of, time_of, n=60):
    blocks, samples = [], []
    for i in range(n):
        t0 = i * 1_000_000_000
        clk = clock_of(i)
        blocks.append(
            stats.Block(
                i % 3, i // 3, i, "kernel", t0, t0 + 10_000_000, (time_of(clk),) * 5
            )
        )
        samples.append(
            {
                "t_mono_ns": t0 + 5_000_000,
                "pci_bus_id": "0000:04:00.0",
                "sm_clock_mhz": clk,
                "mem_clock_mhz": 9251,
            }
        )
    return blocks, samples


def test_clock_domain_is_measured_per_experiment():
    """Memory clock only when the time does not follow the SM clock:
    copy-like arm (flat time, clock 1155-1905 MHz) -> "mem"; compute-like
    arm (time ~ 1/clock) -> "both" (negative control); a clock that never
    moved by the 3 % tolerance -> "both", slope None (nothing to drop)."""

    def plateau(i):
        return 1155.0 + 750.0 * (i % 2)

    b, s = _blocks_and_samples(plateau, lambda c: 1e6)
    dom, slope = stats.choose_clock_domain(b, s, "0000:04:00.0")
    assert dom == "mem" and abs(slope) < 1e-9
    b, s = _blocks_and_samples(plateau, lambda c: 1e9 / c)
    dom, slope = stats.choose_clock_domain(b, s, "0000:04:00.0")
    assert dom == "both" and slope == pytest.approx(-1.0)
    b, s = _blocks_and_samples(lambda i: 1800.0 + (i % 2), lambda c: 1e9 / c)
    assert stats.choose_clock_domain(b, s, "0000:04:00.0") == ("both", None)


# ---------------------------------------------------------------- pipeline
def _nvml_log(path: Path, raws: list[dict]) -> None:
    t = [
        b[k]
        for r in raws
        for b in r["blocks"]
        for k in ("t_start_mono_ns", "t_end_mono_ns")
    ]
    lines = [
        json.dumps(
            {
                "kind": "header",
                "log_format": 1,
                "interval_ms": 100,
                "t_mono_ns": min(t) - 10**9,
                "t_unix_ns": 0,
                "host": "x",
                "gpus": [],
            }
        )
    ]
    for i, tm in enumerate(range(min(t) - 10**8, max(t) + 2 * 10**8, 5 * 10**7)):
        lines.append(
            json.dumps(
                {
                    "kind": "sample",
                    "tick": i,
                    "t_mono_ns": tm,
                    "t_unix_ns": 0,
                    "gpu_index": 0,
                    "pci_bus_id": "cpu",
                    "sm_clock_mhz": 1800,
                    "mem_clock_mhz": 9251,
                    "temp_c": 50,
                    "power_w": 100,
                    "throttle_reasons": 0,
                    "util_gpu_pct": 0,
                    "compute_procs": [{"pid": 1, "used_mem_mib": 1}],
                    "errors": {},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.slow
def test_cpu_pipeline_end_to_end(tmp_path):
    """kernel_bench on the host reference library -> raw files ->
    bench_report.py report -> schema-valid records with the kernel's roles and
    work -> kernel_gate.py refuses them (host timer: never gate evidence).

    Why: the GPU run must not be the first time these pieces meet."""
    raw_dir = tmp_path / "raw"
    for p in (0, 1):
        assert (
            kb.main(
                [
                    "run",
                    "--family",
                    "selftest",
                    "--only",
                    "self-dec-b2",
                    "--process",
                    str(p),
                    "--out-dir",
                    str(raw_dir),
                    "--device",
                    "cpu",
                    "--kernel-kind",
                    "ref",
                    "--num-sms",
                    str(NUM_SMS),
                    "--rounds",
                    "2",
                    "--replays",
                    "2",
                    "--warmup-s",
                    "0",
                ]
            )
            == 0
        )
    raws = sorted(raw_dir.glob("*.json"))
    assert len(raws) == 2
    loaded = [json.loads(p.read_text()) for p in raws]
    assert loaded[0]["timer"] == "host" and loaded[0]["design"]["n_arms"] == 3
    _nvml_log(tmp_path / "nvml.jsonl", loaded)
    (tmp_path / "env.json").write_text(
        json.dumps(
            {
                "git_sha": "0" * 40,
                "git_dirty": False,
                "image_digest": "sha256:" + "0" * 64,
                "nvcc_version": "13.0.88",
                "driver_version": "x",
                "power_limit_w": None,
                "captures": [],
            }
        )
    )
    out = tmp_path / "res"
    assert (
        bench_report._main(
            [
                "report",
                "--raw",
                *map(str, raws),
                "--nvml",
                str(tmp_path / "nvml.jsonl"),
                "--env",
                str(tmp_path / "env.json"),
                "--session",
                "2026-10-20-kernel-220w-gpu0",
                "--experiment",
                "self-dec-b2",
                "--out-dir",
                str(out),
                "--aa",
                "kernel",
                "kernel#aa",
                "--min-accepted",
                "1",
                "--cluster-robust",
            ]
        )
        == 0
    )
    schema = json.loads((ROOT / "bench" / "results.schema.json").read_text())
    recs = {}
    for f in sorted(out.glob("*.json")):
        r = json.loads(f.read_text())
        jsonschema.validate(r, schema)
        recs[r["arm"]["name"]] = r
    assert {n: r["arm"]["role"] for n, r in recs.items()} == {
        "kernel": "subject",
        "kernel#aa": "aa_control",
        "ref": "comparison",
    }
    assert recs["ref"]["arm"]["label"] == "comparison, not a target"
    assert recs["kernel"]["arm"]["params"]["bytes_per_launch"] == (
        kb.kv_bytes(130, 2, 64) + kb.kv_bytes(41, 2, 64)
    )
    assert kernel_gate.main([str(out)]) == 2  # host timer refused


# ---------------------------------------------------------------- op_vs_triton
# bench/op_vs_triton.py needs a GPU and vLLM to run; its cache layout, block
# tables and argument handling are checked here on the CPU.
ovt = importlib.import_module("op_vs_triton")


def test_op_vs_triton_defaults_and_masks():
    """Page sizes of the two cache shapes; the Triton mask a layer maps to."""
    assert ovt.default_block_size(256) == 816 and ovt.default_block_size(128) == 864
    with pytest.raises(SystemExit):
        ovt.default_block_size(64)  # no default page size: --block-size is required
    assert ovt.triton_mask(0, True) == (True, (-1, -1))
    assert ovt.triton_mask(0, False) == (False, (-1, -1))
    assert ovt.triton_mask(2048, False) == (False, (2047, 0))


def test_op_vs_triton_window_keeps_only_the_blocks_it_reads():
    """Without a window every block is real; with window W the query rows sit in
    the last q positions, so blocks wholly before L - W - q are null (block 0)."""
    assert ovt.real_block_range(131080, 1, 0, 816) == (0, 161)
    assert ovt.real_block_range(2056, 8, 2048, 864) == (0, 3)
    first, count = ovt.real_block_range(131080, 8, 2048, 864)
    assert (first, count) == ((131080 - 2048 - 8) // 864, 3)
    assert first + count == -(-131080 // 864)


def test_op_vs_triton_block_tables_are_disjoint_per_launch():
    """Every (rotation, request) gets its own run of real blocks from id 1, so
    cold launches share no cache line; the null prefix stays 0; the block count
    covers the null block plus every real run."""
    tables, nb = ovt.build_block_tables(
        131080, 8, 2048, 864, batch=2, rotations=3, device="cpu"
    )
    first, count = ovt.real_block_range(131080, 8, 2048, 864)
    assert nb == 1 + 3 * 2 * count and len(tables) == 3
    seen = set()
    for bt in tables:
        assert bt.shape == (2, first + count) and bt.dtype == torch.int32
        assert int(bt[:, :first].abs().sum()) == 0
        for row in bt:
            ids = set(row[first:].tolist())
            assert len(ids) == count and not ids & seen and 0 not in ids
            seen |= ids
    assert seen == set(range(1, nb))
    full, nb_full = ovt.build_block_tables(2056, 8, 0, 864, 1, 2, "cpu")
    assert int(full[0].min()) == 1 and nb_full == 1 + 2 * 3


def test_op_vs_triton_cache_layout_matches_the_scale_views():
    """The scale views alias the head padding of the packed int8 cache: a float
    written through ``k_scale`` / ``v_scale`` appears at bytes D..D+3 and
    PHS+D..PHS+D+3 of the (slot, head) cell, key / value are the padded views,
    and the cell is ``2 * (D + 4)`` bytes (520 at D = 256)."""
    D, nb, block, hkv = 8, 3, 4, 2
    raw, key, value, ks, vs = ovt.build_cache(nb, block, hkv, D, torch.device("cpu"))
    phs = D + 4
    assert raw.shape == (nb, block, hkv, 2 * phs) and raw.dtype == torch.int8
    assert key.shape == value.shape == (nb, block, hkv, phs)
    assert ks.shape == vs.shape == (nb, block, hkv) and ks.dtype == torch.float32
    ks[2, 1, 1] = 0.25
    vs[2, 3, 0] = -0.5
    assert raw[2, 1, 1, D : D + 4].contiguous().view(torch.float32).item() == 0.25
    assert (
        raw[2, 3, 0, phs + D : phs + D + 4].contiguous().view(torch.float32).item()
        == -0.5
    )
    # block 0 is the null block: zeros, scale 1.0
    assert int(raw[0, :, :, :D].abs().sum()) == 0 and bool((ks[0] == 1.0).all())
    assert ovt.cache_bytes(10, 816, 2, 256) == 10 * 816 * 2 * 520


def test_op_vs_triton_arguments_and_help(capsys):
    """--help explains inputs, outputs and the GPU requirement without a GPU or
    vLLM; options that need the windowed kernel are refused at other head sizes."""
    with pytest.raises(SystemExit) as e:
        ovt.parse_args(["--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    assert "Needs: one NVIDIA GPU" in text and "--json" in text and "vLLM" in text
    a = ovt.parse_args(["--json", "x.json"])
    assert (a.head_size, a.block_size, a.query_len, a.seq_len) == (256, 816, 1, 131072)
    a = ovt.parse_args(
        ["--json", "x", "--head-dim", "128", "--window", "2048", "--non-causal"]
    )
    assert a.block_size == 864 and a.window == 2048 and a.non_causal
    for bad in (["--window", "2048"], ["--non-causal"], ["--num-q-heads", "13"]):
        with pytest.raises(SystemExit) as e:
            ovt.parse_args(["--json", "x", *bad])
        assert e.value.code == 2


def test_op_vs_triton_quantiles():
    q = ovt.quantiles(np.arange(1.0, 101.0))
    assert q["median_us"] == 50.5 and q["min_us"] == 1 and q["max_us"] == 100
    assert q["iqr_rel"] == pytest.approx((75.25 - 25.75) / 50.5)
