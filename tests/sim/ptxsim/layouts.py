# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Shared-memory stage layout, XOR swizzles, the K permutation pi, the V
d-permutation and the cp.async loader's lane -> destination mapping.

This is the simulator twin of the CUDA `smem_layout.cuh` (stage layout and
swizzles) and of the loader part of `cache_geometry.cuh` under csrc/pth_attn.
Those headers are checked against these functions through `golden.py`.

Stage layout per pipeline stage, N = 64 tokens:

    offset 0            K   [N][D] int8, 16-B chunks XOR-swizzled
    offset N*D          V   [N][D] int8, 16-B chunks XOR-swizzled
    offset 2*N*D        ks  [N] f32 (unswizzled)
    offset 2*N*D + 4N   vs  [N] f32 (unswizzled)
    total N*(2D+8) bytes  (33,280 at D=256)

Swizzle convention used here. Shared memory is 32 banks of 4 B,
so a 128-B "line" covers every bank once and a 16-B "chunk" covers 4 banks
(a bank group). A swizzle is a permutation of the 8 chunks inside each
128-B line, keyed by the token index t:

    K:  chunk' = chunk ^ (t & 7)
    V:  chunk' = chunk ^ (2 * ((t >> 1) & 3))

where "chunk" is the chunk index within its 128-B line. For D >= 128 (a
token row is a whole number of lines) this is exactly the row-chunk form
`c ^ (t&7)` / `c ^ 2((t>>1)&3)` with c the chunk index in the row, because
the XOR key is < 8 and only changes the low 3 bits of c. For D = 64 a token
row is half a line; the row-chunk reading would move chunks outside
the 64-B row (see `k_offset_row_chunks`), the line-chunk reading does not.
"""

from __future__ import annotations

import numpy as np

N_TOKENS = 64  # load tile N
LINE_BYTES = 128  # 32 banks x 4 B: one pass over every bank
CHUNK_BYTES = 16  # ldmatrix row = 16 B = 4 banks
CHUNKS_PER_LINE = LINE_BYTES // CHUNK_BYTES  # 8
K_SWIZZLE_MASK = 7  # t & 7: 8 rows of one ldmatrix 8x8 matrix
V_SWIZZLE_MASK = 3  # (t >> 1) & 3: 4 token pairs read by one LDS.32
SCALE_BYTES = 4  # f32 per-token scale
CELL_PAD_PITCH = 16  # whole-cell pitch 2D + 16: cell 2D + 8 rounded to 16 B, odd x16

# K permutation pi. pi[k] = d offset
# inside a 16-element chunk that logical MMA index k reads.
PI_PERMUTATION = (0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15)


# ---------------------------------------------------------------------------
# Stage geometry
# ---------------------------------------------------------------------------


def stage_offsets(D: int, n: int = N_TOKENS) -> dict[str, int]:
    """Byte offsets of the four regions of one stage and its total size."""
    k = 0
    v = k + n * D
    ks = v + n * D
    vs = ks + SCALE_BYTES * n
    return {"K": k, "V": v, "ks": ks, "vs": vs, "bytes": vs + SCALE_BYTES * n}


def _line_swizzle(t, byte, D, key):
    t = np.asarray(t, dtype=np.int64)
    byte = np.asarray(byte, dtype=np.int64)
    o = t * D + byte  # dense row-major offset (int64: no overflow)
    line = o // LINE_BYTES
    chunk = (o // CHUNK_BYTES) % CHUNKS_PER_LINE
    within = o % CHUNK_BYTES
    return line * LINE_BYTES + (chunk ^ key) * CHUNK_BYTES + within


def k_offset(t, byte, D: int, swizzled: bool = True):
    """Physical byte offset (inside the K region) of K[t][byte]."""
    key = (np.asarray(t) & K_SWIZZLE_MASK) if swizzled else 0
    return _line_swizzle(t, byte, D, key)


def v_offset(t, byte, D: int, swizzled: bool = True):
    """Physical byte offset (inside the V region) of V[t][byte]."""
    key = 2 * ((np.asarray(t) >> 1) & V_SWIZZLE_MASK) if swizzled else 0
    return _line_swizzle(t, byte, D, key)


def k_offset_row_chunks(t, byte, D: int):
    """The K swizzle read literally on chunks of the token row.

    Used only to show that this reading is not a permutation for D = 64
    (a 64-B row has 4 chunks; t & 7 can move a chunk to index 4..7, i.e. into
    the next token's row). Identical to `k_offset` for D >= 128.
    """
    t = np.asarray(t, dtype=np.int64)
    byte = np.asarray(byte, dtype=np.int64)
    c = byte // CHUNK_BYTES
    return t * D + (c ^ (t & K_SWIZZLE_MASK)) * CHUNK_BYTES + byte % CHUNK_BYTES


def v_offset_row_chunks(t, byte, D: int):
    """The V swizzle read literally on chunks of the token row."""
    t = np.asarray(t, dtype=np.int64)
    byte = np.asarray(byte, dtype=np.int64)
    c = byte // CHUNK_BYTES
    key = 2 * ((t >> 1) & V_SWIZZLE_MASK)
    return t * D + (c ^ key) * CHUNK_BYTES + byte % CHUNK_BYTES


def q_offset(row, k, D: int, swizzled: bool = True):
    """Physical byte offset of Q_smem[row][k] (fp16, pitch 2D bytes).

    k is the position *after* the pi permutation (the MMA's k order). The
    swizzle is `chunk ^ (row & 7)` on the 16-B chunks of the row; the row
    pitch 2D >= 128 B for every supported D, so rows are whole lines.
    """
    row = np.asarray(row, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    byte = 2 * k
    c = byte // CHUNK_BYTES
    key = (row & K_SWIZZLE_MASK) if swizzled else 0
    return row * 2 * D + (c ^ key) * CHUNK_BYTES + byte % CHUNK_BYTES


# ---------------------------------------------------------------------------
# K permutation pi
# ---------------------------------------------------------------------------


def pi_inverse(pi) -> tuple[int, ...]:
    inv = [0] * len(pi)
    for k, d in enumerate(pi):
        inv[d] = k
    return tuple(inv)


def kappa_closed_form(d: int) -> int:
    """Closed form of the map kappa: position of d.

    kappa(d) = 2*floor(d/4) + (d mod 4 if d mod 4 < 2 else 6 + d mod 4).
    """
    r = d % 4
    return 2 * (d // 4) + (r if r < 2 else 6 + r)


def permute_q_chunks(q: np.ndarray, pi=PI_PERMUTATION) -> np.ndarray:
    """Q_perm[..., 16s + k] = Q[..., 16s + pi[k]] for every 16-wide chunk s.

    This is what the prologue writes into Q smem. Then the MMA's k index k
    multiplies Q at d = 16s + pi[k], which is exactly the K byte the K
    fragment delivers for that k (see `derive_pi`).
    """
    q = np.asarray(q)
    D = q.shape[-1]
    idx = np.concatenate([16 * s + np.asarray(pi) for s in range(D // 16)])
    return q[..., idx]


# ---------------------------------------------------------------------------
# V d-permutation
# ---------------------------------------------------------------------------


def v_ntile_col_to_d(j, n):
    """Physical d of PV n-tile j, fragment column n.

    n-tile j = 4a + b covers d = 32a + 4n + b for n = 0..7: the 8 columns of
    one n-tile are 4 apart, so the 4 n-tiles 4a..4a+3 together cover the
    32-wide group a, and the 4 bytes one LDS.32 brings in (d = 32a+4g..+3)
    feed n-tiles 4a+0..4a+3 at column g.
    """
    j = np.asarray(j)
    n = np.asarray(n)
    return 32 * (j // 4) + 4 * n + (j % 4)


# ---------------------------------------------------------------------------
# Loader: cp.async lane -> (token, byte) -> destination
# ---------------------------------------------------------------------------


def _chunk_instructions(cpr: int, width: int, warps: int, n: int):
    """Shared chunk enumeration of every loader stream.

    Warp w copies its n/warps tokens as one run of `cpr` chunks of `width`
    bytes per token; instruction `it` of the warp gives lane ln chunk
    i = 32*it + ln, i.e. token t0 + i // cpr, byte (i % cpr) * width. Lanes
    whose chunk is past the end are predicated off. This is the order of
    csrc/pth_attn/loader.cuh (`stream_chunk`).
    """
    tokens_per_warp = n // warps
    total = tokens_per_warp * cpr
    instrs = []
    for w in range(warps):
        t0 = w * tokens_per_warp
        for it in range(-(-total // 32)):
            ins = []
            for ln in range(32):
                i = 32 * it + ln
                if i < total:
                    ins.append((w, ln, t0 + i // cpr, (i % cpr) * width))
            instrs.append(ins)
    return instrs


def loader_instructions(row_bytes: int, width: int, warps: int = 4, n=N_TOKENS):
    """The cp.async instruction stream that loads one K or V stream of a tile.

    Each warp loads 16 of the 64 tokens; one warp-instruction
    moves 32 * width contiguous bytes. Lanes cover a token row in order;
    when a row is shorter than 32 * width bytes, one instruction covers
    several consecutive tokens. Returns a list of instructions, each a list
    of (warp, lane, token, byte) for the active lanes.
    """
    assert row_bytes % width == 0 and row_bytes > SCALE_BYTES
    return _chunk_instructions(row_bytes // width, width, warps, n)


def scale_pair_instructions(warps: int = 4, n=N_TOKENS):
    """The one cp.async (4 B) per warp that loads both scale streams.

    Lanes 0..15 of warp w copy k_scale of the warp's own tokens 16w + l;
    lanes 16..31 copy v_scale of the tokens of warp w ^ 1 (16(w^1) + l - 16).
    Taking the partner warp's tokens for vs puts the two halves of the
    instruction in different bank halves (vs region = ks region + 256 B, a
    multiple of 128, so same-token pairing would be 2-way).
    Returns a list of instructions of (warp, lane, stream, token, byte) with
    stream "ks" or "vs".
    """
    tpw = n // warps
    instrs = []
    for w in range(warps):
        ins = [(w, ln, "ks", tpw * w + ln, 0) for ln in range(16)]
        ins += [(w, ln, "vs", tpw * (w ^ 1) + ln - 16, 0) for ln in range(16, 32)]
        instrs.append(ins)
    return instrs


def cell_pitch(D: int) -> int:
    """Whole-cell stage row pitch 2D + 16 (the whole-cell alternative: an odd
    multiple of 16 B, so consecutive cells start in different bank groups)."""
    return 2 * D + CELL_PAD_PITCH


def cell_bytes(D: int) -> int:
    """vLLM int8_per_token_head cell [K D | ks f32 | V D | vs f32]."""
    return 2 * D + 2 * SCALE_BYTES


def cell_loader_instructions(D: int, width: int, warps: int = 4, n=N_TOKENS):
    """cp.async stream of the whole-cell challenger: (2D+8)/width chunks per
    token, same chunk enumeration as the separate streams; byte b of token t
    lands at t * cell_pitch(D) + b (no swizzle; the odd pitch does its job)."""
    assert cell_bytes(D) % width == 0
    return _chunk_instructions(cell_bytes(D) // width, width, warps, n)


def stream_width(*byte_quantities: int) -> int:
    """Largest w in {16, 8, 4} dividing every given byte quantity.

    For each stream the host passes the largest of {16, 8, 4}
    that divides the base pointer, every byte stride and the row bytes. A
    stream with none of these (not even 4-B aligned) is unsupported.
    """
    for w in (16, 8, 4):
        if all(q % w == 0 for q in byte_quantities):
            return w
    raise ValueError("stream is not 4-byte aligned")
