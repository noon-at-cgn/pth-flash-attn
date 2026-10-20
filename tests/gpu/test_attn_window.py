# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The windowed kernels on the GPU: fuzz, poisoned pages, oracle.

The grid's ``window`` family (test_attn_grid.py) covers the window
axes with sentinels. This module adds what a grid of named cases cannot:

* 2,000 seeded fuzz launches (random window, causal flag, batch, head
  geometry, block size, layout, dtypes, splits, padding rows), each gated
  like a grid case (bound of the kernel's own tiling, canaries);
* poisoned null pages: every page wholly below a request's window points to
  block 0, vLLM's null block (evicted window pages map there), whose bytes
  are all 0xFF (NaN scales). The windowed kernel must not read it (loader
  zero fill below the range begin, block_table not read there). Negative
  control: the same cache without a window reads it and the output is not
  finite;
* the D = 64 and D = 256 refusals (the windowed kernels exist at D = 128);
* vLLM's Triton ``unified_attention``, the kernel that serves the DFlash2
  drafter in vLLM 0.31, on the same int8 cache with ``causal=False,
  window_size=(W - 1, 0)`` as a second oracle, at the drafter's shapes
  (D = 128, 16/4 and 32/8 heads, block 864, a null-page prefix) and
  L in {8, 2047, 2048, 2049, 2055} x q in {1, 8, 16}. Its arithmetic is not
  ours (P * v_scale rounded to bf16), so it is compared to the
  fp64 reference with a semantic tolerance (TRITON_TOL of the largest |v|)
  on caches with sentinel keys on both sides of every row's window edge,
  where a semantic disagreement is an O(1) error. Negative controls: Triton
  with a window one key wider, and run causally, exceed that tolerance on
  the cases where they change the visible set. The oracle must be vLLM 0.31
  (the version this project pins; its ``triton_unified_attention.py``
  defines the windowed semantics): another version fails the test; only a
  missing vLLM skips it.
"""

from __future__ import annotations

import dataclasses
import random

import attn_grid
import grid_cases as grid
import numpy as np
import pytest
import torch
from attn_lib import attn_case
from grid_cuda import cache_to, ref_kernel
from pth_testing import ref_capi
from pth_testing.reference import make_int8_cache, make_queries, visible_bounds

pytestmark = pytest.mark.attn

FUZZ_LAUNCHES = 2000
FUZZ_CHUNKS = 20
FUZZ_MAX_L = 700  # keeps the fp64 bound of one launch well under a second
# Triton vs fp64: 2^-6 of max |v|. Triton's own rounding is ~2^-9 relative
# (bf16 P * v_scale); a misplaced sentinel key moves the output by
# O(max |v|).
TRITON_TOL = 2.0**-6
ORACLE_VLLM = "0.31"  # the vLLM version this project pins
DRAFTER_W = 2048
DRAFTER = dict(G=4, Hkv=4, D=128, block_size=grid.DRAFTER_BLOCK)
# The drafter's heads per rank: TP=2 16/4, TP=1 32/8.
DRAFTER_HEADS = ((16, 4), (32, 8))
ORACLE_L = (8, 2047, 2048, 2049, 2055)
ORACLE_Q = (1, 8, 16)
# Second request of every oracle case: long enough that its first pages lie
# wholly below its window (null-page prefix: 5000 - 16 - 2047 >= 2 pages).
NULL_PREFIX_L = 5000
NULL_BLOCK = 0  # vLLM's null block id


def _fuzz_case(i: int) -> grid.GridCase:
    rng = random.Random(1_000_003 * (i + 1))
    B = rng.choice((1, 2, 3))
    q = tuple(rng.randint(1, 16) for _ in range(B))
    window = rng.choice((0, 1, 2, 63, 64, 65, rng.randint(1, FUZZ_MAX_L + 100)))
    causal = rng.random() < 0.5 and window > 0  # every launch is windowed
    return grid._check_case(
        grid.GridCase(
            f"wfuzz-{i:04d}",
            "wfuzz",
            q,
            tuple(rng.randint(ql, FUZZ_MAX_L) for ql in q),
            G=rng.choice((1, 2, 4, 6, 8)),
            Hkv=rng.choice((1, 2, 4)),
            D=grid.W_D_VALUES[0],
            block_size=rng.choice((16, 32, 64, 816, grid.DRAFTER_BLOCK)),
            layout=rng.choice(("NHD", "HND")),
            q_dtype=rng.choice(("bf16", "fp16")),
            out_dtype=rng.choice(("f32", "q")),
            num_splits=rng.choice((1, 2, 3, 5, 34)),
            pad_rows=rng.choice((0, 0, 2)),
            sentinel=rng.random() < 0.5,
            window=window,
            causal=causal,
        )
    )


@pytest.mark.parametrize("chunk", range(FUZZ_CHUNKS))
def test_window_fuzz_within_bound(device, record, chunk):
    """FUZZ_LAUNCHES / FUZZ_CHUNKS seeded windowed launches per chunk pass."""
    lib = attn_grid.lib()
    kernel = ref_kernel(lib, device)
    n = FUZZ_LAUNCHES // FUZZ_CHUNKS
    failed = []
    for i in range(chunk * n, (chunk + 1) * n):
        r = grid.run_case(attn_case(_fuzz_case(i), lib), kernel, use_cache=False)
        record(test="attn_window_fuzz", **r.record())
        if not r.passed:
            failed.append(r.message())
    assert not failed, f"{len(failed)}/{n} launches failed:\n" + "\n".join(failed[:5])


def _null_pages_below_windows(inp: grid.CaseInputs) -> tuple[grid.CaseInputs, int]:
    """Point every page wholly below a request's first window position at
    block 0 (vLLM's null block) and fill block 0 with 0xFF (NaN scales).

    A real page that sat in block 0 is moved to an unused block first, so
    the visible data is unchanged. Returns the new inputs and the number of
    null pages."""
    c, cache = inp.case, inp.cache
    bt = cache.block_table.clone()
    used = set(bt.flatten().tolist())
    spare = next(b for b in range(cache.k_cache.shape[0]) if b not in used)
    cache.kv_packed[spare].copy_(cache.kv_packed[NULL_BLOCK])
    bt[bt == NULL_BLOCK] = spare
    cache.kv_packed[NULL_BLOCK].fill_(-1)
    nulls = 0
    for r, (ql, L) in enumerate(zip(c.q_lens, c.seq_lens)):
        if ql == 0:
            continue
        lo, _ = visible_bounds([L - ql], L, c.window, c.causal)
        pages = lo[0] // c.block_size
        bt[r, :pages] = NULL_BLOCK
        nulls += pages
    cache = dataclasses.replace(cache, block_table=bt)
    return dataclasses.replace(inp, cache=cache), nulls


@pytest.mark.parametrize("S", [1, 3, 34])
def test_null_pages_below_the_window_are_never_read(device, record, S):
    """NaN null pages below the window leave the output within the bound;
    the same cache read without a window (causal, window 0) is outside its own
    bound against the clean fp64 reference (control: the poison is read)."""
    lib = attn_grid.lib()
    case = attn_case(
        grid.GridCase(
            f"wnull-S{S}",
            "wnull",
            (8, 8),
            (3000, 5000),
            num_splits=S,
            sentinel=True,
            window=DRAFTER_W,
            causal=False,
            **{**DRAFTER, "block_size": 16},
        ),
        lib,
    )
    inp = grid.build_inputs(case)
    exp = grid.load_expected(case, inp, use_cache=False)  # from the clean cache
    bad, nulls = _null_pages_below_windows(inp)
    assert nulls > 0, "the case must have whole pages below its window"
    run = ref_kernel(lib, device)(bad, grid.make_buffers(case))
    r = grid.check(case, bad, run, exp)
    record(test="attn_window_null_pages", null_pages=nulls, **r.record())
    assert r.passed, r.message()
    # Control: the causal no-window kernel reads every page, the null ones
    # included. Its output is checked against its own clean fp64 reference
    # and bound (bad.kv is the dense clean cache; only the paged copy is
    # poisoned). Finiteness alone cannot be asserted: at S > 1 a split whose
    # tiles all lie in null pages has only NaN scores, writes lse = -inf, and
    # the combine skips it as an empty split, so the output stays
    # finite but wrong (GPU run 2026-10-17: S = 3 and 34 finite, S = 1 all
    # NaN). Either way the result is outside the bound.
    full = attn_case(dataclasses.replace(case, window=0, causal=True), lib)
    bad_full = dataclasses.replace(bad, case=full)
    exp_full = grid.load_expected(full, bad_full, use_cache=False)
    run = ref_kernel(lib, device)(bad_full, grid.make_buffers(full))
    ctl = grid.check(full, bad_full, run, exp_full)
    record(test="attn_window_null_pages_control", null_pages=nulls, **ctl.record())
    assert ctl.violations > 0, f"control: poison not read: {ctl.message()}"


@pytest.mark.parametrize("D", [64, 256])
def test_window_outside_head_size_128_is_refused(device, D):
    """No windowed kernel at D = 64 or 256: EINVAL with the reason."""
    lib = attn_grid.lib()
    c = cache_to(make_int8_cache([100], Hkv=1, D=D, block_size=16, seed=1), device)
    q, qsl = make_queries([4], 6, D, seed=2, dtype=torch.bfloat16)
    q, qsl = q.to(device), qsl.to(device)
    out = torch.zeros_like(q)
    for window, causal in ((2048, False), (64, True), (0, False)):
        a = ref_capi.attn_args(
            c, q, qsl, out, softmax_scale=D**-0.5, bq=4, num_splits=1,
            window=window, causal=causal,
        )  # fmt: skip
        assert lib.attention(a, host=False) == ref_capi.EINVAL
        assert "head size D = 128" in lib.last_error()


def _vllm_031():
    """vLLM's Triton kernel and the quant mode enum, from vLLM 0.31 only."""
    vllm = pytest.importorskip("vllm")  # absent: nothing to compare with
    assert vllm.__version__.startswith(ORACLE_VLLM), (
        f"the Triton oracle must be vLLM {ORACLE_VLLM} (the pinned version), "
        f"this environment has {vllm.__version__}"
    )
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    return vllm.__version__, unified_attention, KVQuantMode


def _oracle_params():
    for hq, hkv in DRAFTER_HEADS:
        for L in ORACLE_L:
            for q in ORACLE_Q:
                if q <= L:
                    yield pytest.param(hq, hkv, L, q, id=f"H{hq}-{hkv}-L{L}-q{q}")


@pytest.mark.parametrize(("hq", "hkv", "L", "q"), list(_oracle_params()))
def test_triton_unified_attention_oracle(device, record, hq, hkv, L, q):
    """vLLM 0.31's Triton kernel and ours agree with the fp64 reference."""
    version, unified_attention, KVQuantMode = _vllm_031()
    lib = attn_grid.lib()
    case = attn_case(
        grid.GridCase(
            f"woracle-H{hq}-{hkv}-L{L}-q{q}",
            "woracle",
            (q, q),
            (L, NULL_PREFIX_L),
            G=hq // hkv,
            Hkv=hkv,
            D=128,
            block_size=grid.DRAFTER_BLOCK,
            out_dtype="q",
            sentinel=True,
            window=DRAFTER_W,
            causal=False,
        ),
        lib,
    )
    clean = grid.build_inputs(case)
    exp = grid.load_expected(case, clean, use_cache=False)
    inp, nulls = _null_pages_below_windows(clean)
    assert nulls > 0, "the second request must have a null-page prefix"
    run = ref_kernel(lib, device)(inp, grid.make_buffers(case))
    ours = grid.check(case, inp, run, exp)
    assert ours.passed, ours.message()
    dev = cache_to(inp.cache, device)
    qd, qsl = inp.q.to(device), inp.qsl.to(device)
    out = torch.empty_like(qd)

    def triton(window: int, causal: bool) -> np.ndarray:
        out.zero_()
        unified_attention(
            q=qd, k=dev.key_cache, v=dev.value_cache, out=out, cu_seqlens_q=qsl,
            max_seqlen_q=max(case.q_lens), seqused_k=dev.seq_lens,
            max_seqlen_k=max(case.seq_lens), softmax_scale=inp.softmax_scale,
            causal=causal, window_size=(window - 1, 0), block_table=dev.block_table,
            softcap=0, q_descale=None, k_descale=None, v_descale=None,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            k_scale_cache=dev.k_scale, v_scale_cache=dev.v_scale,
        )  # fmt: skip
        torch.cuda.synchronize()
        return out.double().cpu().numpy()

    vmax = max(
        float((x.v_int.double() * x.v_scale.double()[..., None]).abs().max())
        for x in clean.kv
    )
    tol = TRITON_TOL * vmax
    valid = exp.valid
    tri = triton(DRAFTER_W, False)
    err_tri = float(np.abs(tri - exp.ref)[valid].max())
    mine = run.bufs.out.double().numpy()
    gap = float((np.abs(mine - tri) - exp.bound - tol)[valid].max())
    # Controls: a wider window changes the second request's rows (L = 5000);
    # causal changes every row with a later key, i.e. every q > 1 case.
    wider = float(np.abs(triton(DRAFTER_W + 1, False) - exp.ref)[valid].max())
    as_causal = float(np.abs(triton(DRAFTER_W, True) - exp.ref)[valid].max())
    record(
        test="attn_window_triton_oracle",
        vllm=version,
        case=case.name,
        null_pages=nulls,
        tol=tol,
        err_triton=err_tri,
        ours_minus_triton_over=gap,
        control_wider=wider,
        control_causal=as_causal,
        tightness=ours.tightness,
    )
    assert err_tri <= tol, f"Triton vs fp64 {err_tri:.3g} > {tol:.3g}"
    assert gap <= 0.0, f"ours vs Triton exceeds bound + tol by {gap:.3g}"
    assert wider > tol, f"control: a wider window must differ ({wider:.3g})"
    if q > 1:
        assert as_causal > tol, f"control: causal must differ ({as_causal:.3g})"
