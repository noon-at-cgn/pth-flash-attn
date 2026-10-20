# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Per-lane models of the PTX instructions used on the fragment path.

Every function models one instruction for a whole warp (32 lanes) and cites
the PTX ISA section it was written from. Source: "PTX ISA 9.4"
(https://docs.nvidia.com/cuda/parallel-thread-execution/), fetched 2026-10-10.
Section numbers drift between ISA versions; the anchors in the URLs below are
stable.

Registers are numpy uint32 arrays. A warp-wide register is an array of shape
(32,); an instruction with several destination registers returns shape
(32, n). Shared memory is a flat numpy uint8 array addressed in bytes and read
little-endian, which is what NVIDIA GPUs are.

Modelled exactly: data movement (which byte goes where), bitwise logic and the
binary16 rounding of `sub.f16x2`. Modelled as exact real arithmetic: the MMA
accumulation. The tensor core's own fp32 accumulation rounding is a numerics
question owned by `bound.py`/`emulate.py`, not a layout question; the
fragment tests choose inputs whose products and sums are exact in fp32, so
any mismatch they report is a layout bug and never a rounding effect.
"""

from __future__ import annotations

import numpy as np

from .fp import (
    f16_bits_to_value,
    f16_value_to_bits,
    pack_half2,
    round_to_f16,
    unpack_half2,
)

WARP = 32
U32 = np.uint32
U64 = np.uint64

PTX = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"
CITE_PRMT = f"{PTX}#data-movement-and-conversion-instructions-prmt (9.7.10.7)"
CITE_LOP3 = f"{PTX}#logic-and-shift-instructions-lop3 (9.7.9.6)"
CITE_HSUB = f"{PTX}#half-precision-floating-point-instructions-sub (9.7.4.2)"
CITE_LDMATRIX = f"{PTX}#warp-level-matrix-instructions-ldmatrix (9.7.16.5.15)"
CITE_MMA16816 = f"{PTX}#warp-level-matrix-fragment-mma-16816-float (9.7.16.5.8)"
CITE_CP_ASYNC = (
    f"{PTX}#data-movement-and-conversion-instructions-cp-async (9.7.10.28.3.1)"  # noqa: E501
)
CITE_BYTE_PERM = (
    "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/"
    "group__CUDA__MATH__INTRINSIC__INT.html (__byte_perm)"
)


# ---------------------------------------------------------------------------
# Byte and bit manipulation
# ---------------------------------------------------------------------------


def prmt(a: np.ndarray, b: np.ndarray, c: int | np.ndarray) -> np.ndarray:
    """`prmt.b32 d, a, b, c` in the generic (no-mode) form.

    PTX ISA 9.7.10.7: the 8 source bytes are {b, a} = {b7..b4, b3..b0}
    (a supplies bytes 0-3, b supplies bytes 4-7). For each destination byte
    i, the nibble ctl = c[4i+3 : 4i] selects it: the 3 low bits pick the
    source byte; if the nibble's msb is set, the selected byte's sign bit is
    replicated over all 8 bits instead of copying the byte.
    """
    tmp64 = (np.asarray(b, dtype=U64) << U64(32)) | np.asarray(a, dtype=U64)
    c = np.asarray(c, dtype=U64)
    out = np.zeros(np.broadcast(tmp64, c).shape, dtype=U64)
    for i in range(4):
        ctl = (c >> U64(4 * i)) & U64(0xF)
        byte = (tmp64 >> (U64(8) * (ctl & U64(7)))) & U64(0xFF)
        sign_fill = np.where(byte & U64(0x80), U64(0xFF), U64(0))
        byte = np.where(ctl & U64(8), sign_fill, byte)
        out |= byte << U64(8 * i)
    return out.astype(U32)


def byte_perm(x: np.ndarray, y: np.ndarray, s: int) -> np.ndarray:
    """CUDA `__byte_perm(x, y, s)`.

    CUDA Math API: tmp64 = (y << 32) | x and selector n = (s >> 4n) & 0x7,
    i.e. the same byte numbering as `prmt` but only 3 selector bits (no
    sign-replicate mode). It therefore equals `prmt(x, y, s & 0x7777)`.
    """
    return prmt(x, y, s & 0x7777)


def lop3(a: np.ndarray, b: np.ndarray, c: np.ndarray, imm_lut: int) -> np.ndarray:
    """`lop3.b32 d, a, b, c, immLut`: arbitrary 3-input bitwise function.

    PTX ISA 9.7.9.6 defines immLut = F(0xF0, 0xCC, 0xAA). Reading that
    backwards, for input bits (a_j, b_j, c_j) the output bit is bit number
    4*a_j + 2*b_j + c_j of immLut (0xF0 is 1 exactly where that index has
    bit 2 set, 0xCC where it has bit 1 set, 0xAA where it has bit 0 set).
    """
    a = np.asarray(a, dtype=U32)
    b = np.asarray(b, dtype=U32)
    c = np.asarray(c, dtype=U32)
    out = np.zeros(np.broadcast(a, b, c).shape, dtype=U32)
    for j in range(8):
        if not (imm_lut >> j) & 1:
            continue
        ma = a if j & 4 else ~a
        mb = b if j & 2 else ~b
        mc = c if j & 1 else ~c
        out |= ma & mb & mc
    return out


def lut_of(fn) -> int:
    """The immLut for a Python bitwise function, per PTX ISA 9.7.9.6."""
    return fn(0xF0, 0xCC, 0xAA) & 0xFF


def hsub2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`sub.rn.f16x2 d, a, b` (default rounding is .rn), no .ftz.

    PTX ISA 9.7.4.2: d[i] = fA[i] - fB[i] for the two halves, rounded to
    nearest even. The exact difference of two binary16 values is exactly
    representable in float64, so `round_to_f16(fa - fb)` is the IEEE result.
    """
    a_lo, a_hi = unpack_half2(a)
    b_lo, b_hi = unpack_half2(b)
    d = []
    for x, y in ((a_lo, b_lo), (a_hi, b_hi)):
        exact = f16_bits_to_value(x) - f16_bits_to_value(y)
        d.append(f16_value_to_bits(round_to_f16(exact)))
    return pack_half2(d[0], d[1])


# ---------------------------------------------------------------------------
# ldmatrix
# ---------------------------------------------------------------------------


def _load_u16(smem: np.ndarray, addr: np.ndarray) -> np.ndarray:
    lo = smem[addr].astype(U32)
    hi = smem[addr + 1].astype(U32)
    return lo | (hi << U32(8))


def ldmatrix_lane_map(trans: bool) -> np.ndarray:
    """(row, col) of the two b16 elements lane l receives from one 8x8 matrix.

    Returns int array [32, 2, 2]: [lane, element (low, high), (row, col)].

    PTX ISA 9.7.16.5.15, Figure 107 ("ldmatrix fragment layout for one 8x8
    matrix with 16-bit elements"): row r is loaded by the four threads
    4r..4r+3, thread 4r+j receiving columns 2j and 2j+1, the lower column in
    the low half. With `.trans` the matrix "is loaded in column-major format",
    i.e. the same lane receives elements (2j, r) and (2j+1, r).
    """
    m = np.zeros((WARP, 2, 2), dtype=np.int64)
    for lane in range(WARP):
        r, j = lane >> 2, lane & 3
        for e in range(2):
            m[lane, e] = (2 * j + e, r) if trans else (r, 2 * j + e)
    return m


def ldmatrix(
    smem: np.ndarray, row_addr: np.ndarray, num: int, trans: bool = False
) -> np.ndarray:
    """`ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16`.

    row_addr[lane] is the byte address lane supplies. Per PTX ISA 9.7.16.5.15
    only threads 0-7 (x1), 0-15 (x2) or 0-31 (x4) contribute addresses:
    threads 8m..8m+7 give rows 0..7 of matrix m, each the start of 16
    contiguous bytes (8 b16 elements) that must be 16-B aligned. Matrix m
    lands in destination register m of every lane, laid out per
    `ldmatrix_lane_map`. Returns uint32 [32, num].
    """
    assert num in (1, 2, 4)
    row_addr = np.asarray(row_addr, dtype=np.int64)
    used = row_addr[: 8 * num]
    assert np.all(used % 16 == 0), "ldmatrix row addresses must be 16-B aligned"
    lane_map = ldmatrix_lane_map(trans)
    out = np.zeros((WARP, num), dtype=U32)
    for m in range(num):
        rows = row_addr[8 * m : 8 * m + 8]
        r = lane_map[:, :, 0]
        c = lane_map[:, :, 1]
        elems = _load_u16(smem, rows[r] + 2 * c)  # [32, 2]
        out[:, m] = elems[:, 0] | (elems[:, 1] << U32(16))
    return out


# ---------------------------------------------------------------------------
# mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
# ---------------------------------------------------------------------------


def _group(lane: int) -> tuple[int, int]:
    # PTX ISA 9.7.16.5.8: groupID = laneid >> 2, threadID_in_group = laneid % 4
    return lane >> 2, lane % 4


def mma_a_map() -> np.ndarray:
    """(row, col) of a0..a7 for every lane; int [32, 8, 2].

    PTX ISA 9.7.16.5.8, Figure 82: row = groupID for a_i with i in {0,1,4,5},
    groupID + 8 otherwise; col = 2*threadID_in_group + (i & 1), plus 8 for
    i >= 4. Registers hold the pairs (a0,a1), (a2,a3), (a4,a5), (a6,a7).
    """
    m = np.zeros((WARP, 8, 2), dtype=np.int64)
    for lane in range(WARP):
        g, t = _group(lane)
        for i in range(8):
            row = g if (i < 2 or 4 <= i < 6) else g + 8
            col = 2 * t + (i & 1) + (8 if i >= 4 else 0)
            m[lane, i] = (row, col)
    return m


def mma_b_map() -> np.ndarray:
    """(row=k, col=n) of b0..b3 for every lane; int [32, 4, 2].

    PTX ISA 9.7.16.5.8, Figure 84: row = 2*threadID_in_group + (i & 1), plus
    8 for i >= 2; col = groupID. Registers hold (b0,b1) and (b2,b3).
    """
    m = np.zeros((WARP, 4, 2), dtype=np.int64)
    for lane in range(WARP):
        g, t = _group(lane)
        for i in range(4):
            m[lane, i] = (2 * t + (i & 1) + (8 if i >= 2 else 0), g)
    return m


def mma_c_map() -> np.ndarray:
    """(row, col) of c0..c3 (f32 accumulators) for every lane; int [32, 4, 2].

    PTX ISA 9.7.16.5.8, Figure 86: row = groupID for i < 2, groupID + 8
    otherwise; col = 2*threadID_in_group + (i & 1).
    """
    m = np.zeros((WARP, 4, 2), dtype=np.int64)
    for lane in range(WARP):
        g, t = _group(lane)
        for i in range(4):
            m[lane, i] = (g if i < 2 else g + 8, 2 * t + (i & 1))
    return m


A_MAP = mma_a_map()
B_MAP = mma_b_map()
C_MAP = mma_c_map()


def _regs_to_halves(regs: np.ndarray) -> np.ndarray:
    """[32, R] f16x2 registers -> [32, 2R] float64 element values (lo, hi)."""
    regs = np.asarray(regs, dtype=U32)
    lo, hi = unpack_half2(regs)
    vals = np.empty((regs.shape[0], 2 * regs.shape[1]), dtype=np.float64)
    vals[:, 0::2] = f16_bits_to_value(lo)
    vals[:, 1::2] = f16_bits_to_value(hi)
    return vals


def gather_a(a_regs: np.ndarray) -> np.ndarray:
    """Rebuild the 16x16 A matrix a warp holds in fragments a0..a7."""
    vals = _regs_to_halves(a_regs)  # [32, 8]
    a = np.full((16, 16), np.nan)
    a[A_MAP[..., 0], A_MAP[..., 1]] = vals
    assert not np.isnan(a).any()
    return a


def gather_b(b_regs: np.ndarray) -> np.ndarray:
    """Rebuild the 16x8 (k x n) B matrix from fragments b0..b3."""
    vals = _regs_to_halves(b_regs)  # [32, 4]
    b = np.full((16, 8), np.nan)
    b[B_MAP[..., 0], B_MAP[..., 1]] = vals
    assert not np.isnan(b).any()
    return b


def gather_c(c_vals: np.ndarray) -> np.ndarray:
    """Rebuild the 16x8 accumulator matrix from c0..c3 ([32, 4] values)."""
    c = np.full((16, 8), np.nan)
    c[C_MAP[..., 0], C_MAP[..., 1]] = np.asarray(c_vals, dtype=np.float64)
    assert not np.isnan(c).any()
    return c


def scatter_c(c_mat: np.ndarray) -> np.ndarray:
    """Distribute a 16x8 matrix into accumulator fragments [32, 4]."""
    return np.asarray(c_mat, dtype=np.float64)[C_MAP[..., 0], C_MAP[..., 1]]


def mma_m16n8k16(
    a_regs: np.ndarray, b_regs: np.ndarray, c_vals: np.ndarray
) -> np.ndarray:
    """`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for one warp.

    a_regs: [32, 4] f16x2, b_regs: [32, 2] f16x2, c_vals: [32, 4] values.
    Returns D = A*B + C as [32, 4] float64 accumulators in the C layout.
    The sum is computed exactly (float64 over exact fp16 products); see the
    module docstring for why accumulation rounding is out of scope here.
    """
    a = gather_a(a_regs)
    b = gather_b(b_regs)
    c = gather_c(c_vals)
    return scatter_c(a @ b + c)


# ---------------------------------------------------------------------------
# Shared-memory load/store helpers used by the data-path simulators
# ---------------------------------------------------------------------------


def lds32(smem: np.ndarray, addr: np.ndarray) -> np.ndarray:
    """`ld.shared.b32` per lane (4-B aligned), little-endian."""
    addr = np.asarray(addr, dtype=np.int64)
    assert np.all(addr % 4 == 0), "LDS.32 needs 4-B alignment"
    v = np.zeros(addr.shape, dtype=U32)
    for k in range(4):
        v |= smem[addr + k].astype(U32) << U32(8 * k)
    return v


def cp_async(
    smem: np.ndarray,
    gmem: np.ndarray,
    dst: np.ndarray,
    src: np.ndarray,
    width: int,
    written: np.ndarray | None = None,
) -> None:
    """`cp.async.{ca,cg}.shared.global [dst], [src], width` for active lanes.

    PTX ISA 9.7.10.28.3.1: cp-size is 4, 8 or 16 bytes (`.cg` only allows
    16, so widths 4 and 8 must use `.ca`); dst and src must be aligned to the
    size ("Addresses as Operands"). Completion ordering (commit/wait groups)
    is not modelled: the simulator only checks where bytes land. If `written`
    (a counter per smem byte) is given, it is incremented so callers can
    prove every destination byte is written exactly once.
    """
    assert width in (4, 8, 16)
    dst = np.asarray(dst, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    assert np.all(dst % width == 0) and np.all(src % width == 0), "misaligned"
    for k in range(width):
        smem[dst + k] = gmem[src + k]
        if written is not None:
            np.add.at(written, dst + k, 1)
