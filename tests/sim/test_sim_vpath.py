# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the V path: byte pairing and d-permutation.

V is token-major int8 in shared memory, but the PV MMA's B operand needs,
in each 32-bit register, two consecutive *tokens* at one d. Lane (g, q)
loads 4 bytes (d = 32a+4g .. +3) of each of tokens 2q, 2q+1, 2q+8, 2q+9 with
LDS.32; `__byte_perm(w0, w1, 0x5140 / 0x7362)` interleaves two token rows so
each 16-bit slot holds (token t, token t+1) at one d; cvt4 converts. That
makes B-fragment column g of n-tile 4a+b physical d = 32a + 4g + b, and the
C fragment of each thread then covers 8 contiguous d.
"""

import numpy as np
import pytest
from ptxsim.dequant import V_SEL_D01, V_SEL_D23
from ptxsim.fp import round_to_f16
from ptxsim.isa import scatter_c
from ptxsim.layouts import N_TOKENS, v_ntile_col_to_d
from ptxsim.paths import (
    build_stage,
    epilogue_columns,
    o_matrix,
    p_a_fragment_from_c,
    pv_warp,
)

RNG_SEED = 20261012


def _p_frags_from_matrix(p):
    """P [16, 64] -> 4 PV A fragments, built through the S accumulator layout.

    p is first scattered into QK n-tile accumulator fragments (as if it were
    S after softmax), then packed with the FA2 register reuse.
    """
    c = [scatter_c(p[:, 8 * nt : 8 * nt + 8]) for nt in range(N_TOKENS // 8)]
    return [
        p_a_fragment_from_c(c[2 * kt], c[2 * kt + 1]) for kt in range(N_TOKENS // 16)
    ]


def _inputs(D, seed):
    rng = np.random.default_rng(seed)
    v = rng.integers(-128, 128, size=(N_TOKENS, D)).astype(np.int8)
    # Per-token V scales spanning several binades, like real caches.
    vs = (
        rng.uniform(0.5, 1.0, N_TOKENS) * 2.0 ** rng.integers(-12, -4, N_TOKENS)
    ).astype(np.float32)
    p = rng.uniform(0, 1, size=(16, N_TOKENS))
    return v, vs, p


def test_v_columns_contiguous():
    """Each thread's 8 output values for a row are 8 contiguous d values
    d = 32a + 8q .. 32a + 8q + 7, in register order.

    Why it matters: contiguous columns let the epilogue write one 16-byte
    store per row per group. It also proves the d-permutation is
    a bijection: over all lanes and groups every d is written exactly once.
    """
    for D in (64, 128, 256):
        seen = np.zeros(D, dtype=np.int64)
        for lane in range(32):
            g, q = lane >> 2, lane & 3
            for a in range(D // 32):
                cols = epilogue_columns(lane, a)
                assert cols == list(range(32 * a + 8 * q, 32 * a + 8 * q + 8))
                if g == 0:  # rows g and g+8 use the same columns
                    seen[cols] += 1
        assert np.all(seen == 1)


def test_v_d_permutation_is_bijection():
    """n-tile j, column n -> d = 32(j//4) + 4n + j%4 hits every d once."""
    for D in (64, 128, 256):
        j, n = np.meshgrid(np.arange(D // 8), np.arange(8), indexing="ij")
        d = v_ntile_col_to_d(j, n).ravel()
        assert sorted(d.tolist()) == list(range(D))


@pytest.mark.parametrize("D", [64, 128, 256])
def test_pv_equals_plain_product(D):
    """O from the simulated PV path equals P'' @ V_int exactly, and equals
    sum_t p_t vs_t v_t within the P'' rounding budget.

    Why it matters: this is the end-to-end proof for the V path. P'' =
    fp16(p * vs * 2^-c) (the V scale folded into P) enters as the
    A fragment built from S accumulators; V goes through the swizzled stage,
    LDS.32, __byte_perm, cvt4 and the MMA; the result is placed with the
    d-permutation. (1) The layout check is exact: same rounded P'' in a plain
    product. (2) The numerics check: one fp16 rounding per P'' entry costs at
    most 2^-11 relative, or 2^-25 absolute (times 2^c) where P'' falls below
    the fp16 normal range 2^-14; the MMA sum is
    exact in this model.
    """
    v, vs, p = _inputs(D, RNG_SEED + D)
    c_tile = int(np.ceil(np.log2(vs.max())))
    pv = p * vs[None, :].astype(np.float64)
    p2 = round_to_f16(pv * 2.0**-c_tile)
    stage = build_stage(np.zeros_like(v), v, vs, vs)
    o = o_matrix(pv_warp(_p_frags_from_matrix(p2), stage, D))
    v64 = v.astype(np.float64)
    assert np.array_equal(o, p2 @ v64)
    exact = pv @ v64
    budget = 2.0**-11 * (np.abs(pv) @ np.abs(v64)) + 2.0 ** (c_tile - 25) * (
        np.ones_like(pv) @ np.abs(v64)
    )
    assert np.all(np.abs(o * 2.0**c_tile - exact) <= budget)


def test_v_without_d_permutation_fails():
    """Negative control: reading the n-tiles as d = 8j + n (no permutation)
    gives a wrong O.

    Why it matters: shows the d-permutation is load-bearing, not cosmetic.
    """
    D = 128
    v, vs, p = _inputs(D, RNG_SEED + 1)
    p2 = round_to_f16(p)
    stage = build_stage(np.zeros_like(v), v, vs, vs)
    o = o_matrix(pv_warp(_p_frags_from_matrix(p2), stage, D), d_permuted=False)
    assert np.mean(o != p2 @ v.astype(np.float64)) > 0.9


@pytest.mark.parametrize(
    "sel01,sel23",
    [(0x5410, 0x7632), (V_SEL_D23, V_SEL_D01), (0x1504, 0x3726)],
    ids=["halfword-pairing", "swapped", "token-order-reversed"],
)
def test_wrong_byte_perm_selectors_fail(sel01, sel23):
    """Negative control: plausible but wrong selectors give a wrong O.

    Why it matters: 0x5410/0x7632 is the common halfword interleave (pairs d,
    not tokens); swapping the two selectors mixes up d+0/1 and d+2/3;
    0x1504/0x3726 puts token t+1 before t (k order reversed in the pair).
    Each must be detected by the end-to-end product.
    """
    D = 64
    v, vs, p = _inputs(D, RNG_SEED + 2)
    p2 = round_to_f16(p)
    stage = build_stage(np.zeros_like(v), v, vs, vs)
    o = o_matrix(pv_warp(_p_frags_from_matrix(p2), stage, D, sel01=sel01, sel23=sel23))
    assert not np.array_equal(o, p2 @ v.astype(np.float64))


def test_s_accumulators_are_the_pv_a_fragment():
    """The C layout of two adjacent QK n-tiles is the A layout of one PV k16
    step, so P needs no shuffle between the two MMAs (FA2 register reuse).

    Why it matters: the kernel relies on this; if it were false the kernel
    would need warp shuffles or a shared-memory round trip for P.
    """
    from ptxsim.isa import gather_a

    rng = np.random.default_rng(RNG_SEED + 3)
    p = rng.integers(-1000, 1000, size=(16, 16)).astype(np.float64)
    frag = p_a_fragment_from_c(scatter_c(p[:, :8]), scatter_c(p[:, 8:]))
    assert np.array_equal(gather_a(frag), p)
