# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""One warp's QK and PV data path, executed instruction by instruction.

These functions run the kernel's inner loop for one warp on a byte-exact
model of shared memory: the swizzled stage, ldmatrix/LDS.32 with the real
lane addresses, `cvt4`, `__byte_perm` and `mma.m16n8k16`. The tests compare
the result with a plain matrix product. Because every step is an
instruction model, a wrong lane address, swizzle, selector, pi or fragment
map shows up as a wrong product.
"""

from __future__ import annotations

import numpy as np

from .dequant import V_SEL_D01, V_SEL_D23, cvt4_s8_to_f16x2
from .fp import (
    f16_bits_to_value,
    f16_value_to_bits,
    pack_half2,
    round_to_f16,
    unpack_half2,
)
from .isa import (
    B_MAP,
    C_MAP,
    byte_perm,
    gather_c,
    ldmatrix,
    lds32,
    mma_m16n8k16,
)
from .layouts import (
    N_TOKENS,
    k_offset,
    permute_q_chunks,
    q_offset,
    stage_offsets,
    v_ntile_col_to_d,
    v_offset,
)
from .patterns import V_TOKEN_OFFSETS

LANES = np.arange(32)
GROUP = LANES >> 2  # groupID g
TIG = LANES & 3  # threadID_in_group q


# ---------------------------------------------------------------------------
# Building shared-memory images
# ---------------------------------------------------------------------------


def build_stage(
    k_int8: np.ndarray,
    v_int8: np.ndarray,
    ks: np.ndarray,
    vs: np.ndarray,
    swizzled: bool = True,
) -> np.ndarray:
    """Byte image of one stage, written directly through the swizzle maps.

    (The loader test builds the same image through simulated cp.async and
    checks the two agree.)
    """
    n, D = k_int8.shape
    assert n == N_TOKENS
    offs = stage_offsets(D)
    smem = np.zeros(offs["bytes"], dtype=np.uint8)
    t, b = np.meshgrid(np.arange(n), np.arange(D), indexing="ij")
    smem[offs["K"] + k_offset(t, b, D, swizzled)] = k_int8.view(np.uint8)
    smem[offs["V"] + v_offset(t, b, D, swizzled)] = v_int8.view(np.uint8)
    smem[offs["ks"] : offs["ks"] + 4 * n] = ks.astype(np.float32).view(np.uint8)
    smem[offs["vs"] : offs["vs"] + 4 * n] = vs.astype(np.float32).view(np.uint8)
    return smem


def build_q_smem(
    q_f16: np.ndarray, permute: bool = True, swizzled: bool = True
) -> np.ndarray:
    """Q tile as the prologue stores it: fp16, pi-permuted, swizzled.

    q_f16: [rows, D] float values that are exactly representable in fp16.
    """
    rows, D = q_f16.shape
    qp = permute_q_chunks(q_f16) if permute else q_f16
    bits = f16_value_to_bits(qp)
    smem = np.zeros(rows * 2 * D, dtype=np.uint8)
    r, k = np.meshgrid(np.arange(rows), np.arange(D), indexing="ij")
    off = q_offset(r, k, D, swizzled)
    smem[off] = (bits & 0xFF).astype(np.uint8)
    smem[off + 1] = (bits >> 8).astype(np.uint8)
    return smem


# ---------------------------------------------------------------------------
# Lane -> shared-memory coordinates of the fragment loads
# (single source for the fragment paths below and for golden.py, which the
# CUDA lane-address helpers in smem_layout.cuh are checked against)
# ---------------------------------------------------------------------------


def q_ldsm_rows_k(row0: int, s: int):
    """(row, k) each lane addresses in ldmatrix.x4 of Q for k16 step s.

    Lane l supplies row row0 + l % 16 at 8-element chunk 2s + l // 16, i.e.
    element index k = 8 * (2s + l // 16) (post-pi order).
    """
    return row0 + LANES % 16, 8 * (2 * s + LANES // 16)


def k_ldsm_tokens(tok0: int):
    """Token each lane addresses in ldmatrix.x4 of K (4 n-tiles of 8)."""
    return tok0 + 8 * (LANES // 8) + LANES % 8


def v_lds32_token_byte(t16: int, a: int, i: int):
    """(token, byte) of LDS.32 number i (0..3) of lane (g, q): token
    t16 + 2q + {0, 1, 8, 9}[i], bytes 32a + 4g .. +3."""
    return t16 + 2 * TIG + V_TOKEN_OFFSETS[i], 32 * a + 4 * GROUP


# ---------------------------------------------------------------------------
# QK
# ---------------------------------------------------------------------------


def q_a_fragment(q_smem: np.ndarray, D: int, s: int, swizzled: bool = True):
    """A fragment (a0..a7) for k16 step s via ldmatrix.x4 of Q smem.

    Lane l supplies row (l % 16) at 8-element chunk 2s + l // 16: matrices
    0..3 are (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15),
    (rows 8-15, k 8-15), which is exactly the register order (a0a1),
    (a2a3), (a4a5), (a6a7) of PTX ISA Figure 82.
    """
    rows, k = q_ldsm_rows_k(0, s)
    addr = q_offset(rows, k, D, swizzled)
    return ldmatrix(q_smem, addr, 4)


def k_b_fragments(
    stage: np.ndarray, D: int, tok0: int, s: int, swizzled: bool = True, k_fn=None
):
    """B fragments of 4 consecutive n-tiles (32 tokens) for k16 step s.

    One ldmatrix.x4 (b16 view on int8 data): lanes 8m..8m+7 address token
    tok0 + 8m + r, chunk s, so register m of lane (g, q) holds K bytes
    d = 16s + 4q .. 16s + 4q + 3 of token tok0 + 8m + g. cvt4 turns that into
    (b0, b1) = bytes (4q, 4q+1) and (b2, b3) = bytes (4q+2, 4q+3), the B
    fragment at MMA k = (2q, 2q+1, 2q+8, 2q+9), n = g. Returns [4][32, 2].
    """
    fn = k_fn or (lambda t, b: k_offset(t, b, D, swizzled))
    base = stage_offsets(D)["K"]
    addr = base + fn(k_ldsm_tokens(tok0), 16 * s)
    regs = ldmatrix(stage, addr, 4)
    out = []
    for mi in range(4):
        lo, hi = cvt4_s8_to_f16x2(regs[:, mi])
        out.append(np.stack([lo, hi], axis=1))
    return out


def qk_warp(
    q_smem: np.ndarray,
    stage: np.ndarray,
    D: int,
    tok0: int = 0,
    n_tiles: int = 4,
    swizzled: bool = True,
    k_fn=None,
):
    """S = Q_tile . K[tok0 : tok0 + 8*n_tiles]^T for 16 rows, as C fragments.

    Returns a list of n_tiles accumulator fragments [32, 4] (QK n-tile i
    covers tokens tok0 + 8i .. +7 in natural order).
    """
    assert n_tiles % 4 == 0
    acc = [np.zeros((32, 4)) for _ in range(n_tiles)]
    for s in range(D // 16):
        a = q_a_fragment(q_smem, D, s, swizzled)
        for grp in range(n_tiles // 4):
            bs = k_b_fragments(stage, D, tok0 + 32 * grp, s, swizzled, k_fn)
            for i in range(4):
                nt = 4 * grp + i
                acc[nt] = mma_m16n8k16(a, bs[i], acc[nt])
    return acc


def s_matrix(c_frags) -> np.ndarray:
    """Concatenate n-tile accumulators into the [16, 8*n] S matrix."""
    return np.concatenate([gather_c(c) for c in c_frags], axis=1)


def derive_pi(D: int = 256) -> tuple[int, ...]:
    """Re-derive pi from the instruction models alone (no documented input).

    Fill every K row with its own in-chunk d offset (K[t][d] = d mod 16), run
    the real K fragment path (ldmatrix b16 + cvt4) for chunk 0, and read
    which d offset each B element received. PTX Figure 84 says element b_i of
    lane l sits at MMA k = B_MAP[l, i, 0]; so pi[k] = that received offset.
    Asserts that every lane and n-tile agrees (pi must be a single map).
    """
    _, d = np.meshgrid(np.arange(N_TOKENS), np.arange(D), indexing="ij")
    k_int8 = (d % 16).astype(np.int8)
    zeros = np.zeros((N_TOKENS, D), dtype=np.int8)
    ones = np.ones(N_TOKENS, dtype=np.float32)
    stage = build_stage(k_int8, zeros, ones, ones)
    pi = [-1] * 16
    for frag in k_b_fragments(stage, D, 0, 0):
        lo, hi = unpack_half2(frag)  # [32, 2]: low/high halves of reg 0, 1
        # b0, b1 are the low/high halves of register 0; b2, b3 of register 1.
        vals = [lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]]
        for i in range(4):
            got_all = f16_bits_to_value(vals[i])
            for lane in range(32):
                k = int(B_MAP[lane, i, 0])
                got = int(got_all[lane])
                assert pi[k] in (-1, got), "inconsistent pi across lanes"
                pi[k] = got
    assert -1 not in pi
    return tuple(pi)


# ---------------------------------------------------------------------------
# P (A operand of PV) from the S accumulators
# ---------------------------------------------------------------------------


def p_a_fragment_from_c(c_even: np.ndarray, c_odd: np.ndarray) -> np.ndarray:
    """FA2 register reuse: two QK accumulator n-tiles -> one PV A fragment.

    For tokens 16kt..16kt+15, n-tile 2kt holds (row g, tokens 2q, 2q+1) in
    c0,c1 and (row g+8, same tokens) in c2,c3; n-tile 2kt+1 the same for
    tokens 8 + 2q, 8 + 2q + 1. The A fragment wants a0,a1 = (g, 2q..),
    a2,a3 = (g+8, 2q..), a4,a5 = (g, 2q+8..), a6,a7 = (g+8, 2q+8..). So
    the registers are pack(c0,c1), pack(c2,c3) of the even tile, then the
    same of the odd tile, each rounded to fp16 (cvt.rn).
    """

    def h(x):
        return f16_value_to_bits(round_to_f16(x))

    regs = [
        pack_half2(h(c_even[:, 0]), h(c_even[:, 1])),
        pack_half2(h(c_even[:, 2]), h(c_even[:, 3])),
        pack_half2(h(c_odd[:, 0]), h(c_odd[:, 1])),
        pack_half2(h(c_odd[:, 2]), h(c_odd[:, 3])),
    ]
    return np.stack(regs, axis=1)


# ---------------------------------------------------------------------------
# PV
# ---------------------------------------------------------------------------


def v_b_fragments(
    stage: np.ndarray,
    D: int,
    t16: int,
    a: int,
    swizzled: bool = True,
    sel01: int = V_SEL_D01,
    sel23: int = V_SEL_D23,
    v_fn=None,
):
    """B fragments of PV n-tiles 4a..4a+3 for the 16 tokens t16..t16+15.

    Lane (g, q) issues four LDS.32 of V[t][32a+4g .. +3] for
    t = t16 + 2q + {0, 1, 8, 9}; `__byte_perm` pairs tokens (t, t+1) per d
    and cvt4 converts. Result [4][32, 2]: entry b is the B fragment of n-tile
    4a + b, whose column g is physical d = 32a + 4g + b.
    """
    fn = v_fn or (lambda t, b: v_offset(t, b, D, swizzled))
    base = stage_offsets(D)["V"]
    w = []
    for i in range(len(V_TOKEN_OFFSETS)):
        t, b = v_lds32_token_byte(t16, a, i)
        w.append(lds32(stage, base + fn(t, b)))
    return v_words_to_b_fragments(w, sel01, sel23)


def v_words_to_b_fragments(w, sel01: int = V_SEL_D01, sel23: int = V_SEL_D23):
    """The register part of the V path: 4 LDS.32 words -> 4 B fragments.

    w[0..3] = V bytes d..d+3 of tokens 2q, 2q+1, 2q+8, 2q+9. Returns
    [4][..., 2]: entry b = (reg0, reg1) of n-tile 4a + b, reg0 the token pair
    (2q, 2q+1) and reg1 the pair (2q+8, 2q+9), both at d + b.
    """
    # Register 0 of the B fragment: k = 2q, 2q+1 (tokens w[0], w[1]).
    r0_d0, r0_d1 = cvt4_s8_to_f16x2(byte_perm(w[0], w[1], sel01))
    r0_d2, r0_d3 = cvt4_s8_to_f16x2(byte_perm(w[0], w[1], sel23))
    # Register 1: k = 2q+8, 2q+9 (tokens w[2], w[3]).
    r1_d0, r1_d1 = cvt4_s8_to_f16x2(byte_perm(w[2], w[3], sel01))
    r1_d2, r1_d3 = cvt4_s8_to_f16x2(byte_perm(w[2], w[3], sel23))
    reg0 = (r0_d0, r0_d1, r0_d2, r0_d3)
    reg1 = (r1_d0, r1_d1, r1_d2, r1_d3)
    return [np.stack([reg0[b], reg1[b]], axis=-1) for b in range(4)]


def pv_warp(
    p_frags,
    stage: np.ndarray,
    D: int,
    tok0: int = 0,
    swizzled: bool = True,
    sel01: int = V_SEL_D01,
    sel23: int = V_SEL_D23,
    v_fn=None,
):
    """O accumulators (D/8 n-tiles of [32, 4]) for P . V over len(p_frags)*16
    tokens starting at tok0. p_frags[kt] is the A fragment for tokens
    tok0 + 16kt .. +15."""
    acc = [np.zeros((32, 4)) for _ in range(D // 8)]
    for kt, a_regs in enumerate(p_frags):
        for a in range(D // 32):
            bs = v_b_fragments(
                stage, D, tok0 + 16 * kt, a, swizzled, sel01, sel23, v_fn
            )
            for b in range(4):
                j = 4 * a + b
                acc[j] = mma_m16n8k16(a_regs, bs[b], acc[j])
    return acc


def o_matrix(o_frags, d_permuted: bool = True) -> np.ndarray:
    """Place PV n-tile accumulators at their physical d columns.

    With the d-permutation, n-tile j column n is d = 32(j//4) + 4n + j%4.
    `d_permuted=False` assumes the naive map d = 8j + n (negative control).
    """
    D = 8 * len(o_frags)
    out = np.full((16, D), np.nan)
    n = np.arange(8)
    for j, c in enumerate(o_frags):
        cols = v_ntile_col_to_d(j, n) if d_permuted else 8 * j + n
        out[:, cols] = gather_c(c)
    return out


def epilogue_columns(lane: int, a: int) -> list[int]:
    """Physical d of the 8 values lane writes for its row g in group a.

    Register order: c0 of n-tiles 4a..4a+3, then c1 of n-tiles 4a..4a+3
    (c2/c3 give the same columns for row g+8). c0 sits at fragment column
    2q, c1 at 2q+1 (PTX Figure 86).
    """
    cols = []
    for ci in (0, 1):
        n = int(C_MAP[lane, ci, 1])
        for b in range(4):
            cols.append(int(v_ntile_col_to_d(4 * a + b, n)))
    return cols
