# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Bit-exact models of the 16-bit float formats the kernel touches.

Why this module exists: every exactness claim in the fragment path (the magic
int8->fp16 conversion, the Q prescale) is a claim about IEEE-754 binary16
rounding. numpy's float16 cast is one implementation of that rounding; this
module carries a second, independent one (`round_to_f16`) written directly
from the definition, so the simulator does not have to trust numpy blindly.
`tests/sim/test_sim_prescale.py` cross-checks the two on every bf16 and every
fp16 bit pattern.

Formats (IEEE 754-2008 binary16 and Google bfloat16):
- fp16: 1 sign, 5 exponent (bias 15), 10 fraction bits. Normal range
  [2^-14, 65504]; subnormals are multiples of 2^-24.
- bf16: the top 16 bits of an IEEE binary32 (8 exponent bits, 7 fraction bits),
  so bf16 -> f32 is a 16-bit left shift and always exact.

All values are carried as float64 inside the simulator. float64 represents
every fp16, bf16 and fp32 value exactly, and the difference of two fp16 values
exactly (at most 40 significant bits), so float64 is a sound carrier for the
exact real-number result that an IEEE operation then rounds.
"""

from __future__ import annotations

import numpy as np

# binary16 parameters, from the format definition (IEEE 754-2008 table 3.5).
F16_MANT_BITS = 10  # stored fraction bits; significand has 11 bits
F16_EMIN = -14  # exponent of the smallest normal number 2^-14
F16_MAX = 65504.0  # (2 - 2^-10) * 2^15, the largest finite fp16


def round_to_f16(x: np.ndarray) -> np.ndarray:
    """Round float64 values to the nearest binary16 value, ties to even.

    This is the `.rn` rounding of `cvt.rn.f16.f32` and `sub.rn.f16`
    (PTX ISA 9.4, section 6.5.2 "Rounding Modifiers" and 9.7.4.2): the exact
    real input is rounded to the binary16 grid. Overflow follows IEEE: any
    magnitude that rounds above 65504 becomes infinity (that happens from 65520
    upward, the midpoint between 65504 and the would-be 65536).

    Returns float64 values that are exactly representable in binary16
    (or +-inf). NaN inputs are not supported (the kernel never produces them
    on this path).
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    # frexp gives ax = m * 2^E with m in [0.5, 1), so floor(log2 ax) = E - 1.
    # frexp is exact (it only splits the exponent field).
    _, e = np.frexp(ax)
    exp = e - 1
    # The grid spacing ("quantum") of binary16 around ax: 2^(exp-10) for
    # normals, and the fixed subnormal spacing 2^-24 below 2^-14.
    q_exp = np.maximum(exp, F16_EMIN) - F16_MANT_BITS
    # ldexp by a power of two is exact in float64 for the ranges involved
    # (|x| < 2^130 and quanta >= 2^-24).
    steps = np.ldexp(ax, -q_exp)
    # np.rint rounds half to even under the default IEEE rounding mode.
    rounded = np.ldexp(np.rint(steps), q_exp)
    rounded = np.where(rounded > F16_MAX, np.inf, rounded)
    return np.copysign(rounded, x)


def f16_value_to_bits(v: np.ndarray) -> np.ndarray:
    """Encode float64 values that are already binary16-representable.

    The numpy cast is exact here because no rounding decision is needed;
    callers must round first (`round_to_f16`). Checked with an assertion so a
    caller that forgets cannot silently pick up numpy's rounding.
    """
    v = np.asarray(v, dtype=np.float64)
    h = v.astype(np.float16)
    same = (h.astype(np.float64) == v) | (np.isnan(v))
    assert bool(np.all(same)), "value not representable in binary16"
    return h.view(np.uint16)


def f16_bits_to_value(bits: np.ndarray) -> np.ndarray:
    """Decode binary16 bit patterns (uint16) to float64 values (exact)."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)


def bf16_bits_to_value(bits: np.ndarray) -> np.ndarray:
    """Decode bf16 bit patterns: bf16 is the high half of a binary32."""
    b = np.asarray(bits, dtype=np.uint32) << np.uint32(16)
    # NaN payloads raise a harmless "invalid value" warning when widened.
    with np.errstate(invalid="ignore"):
        return b.view(np.float32).astype(np.float64)


def all_bf16_bits(finite_only: bool = True) -> np.ndarray:
    """Every bf16 bit pattern (65,536), optionally without inf/NaN."""
    bits = np.arange(1 << 16, dtype=np.uint32)
    if finite_only:
        bits = bits[np.isfinite(bf16_bits_to_value(bits))]
    return bits


def all_f16_bits(finite_only: bool = True) -> np.ndarray:
    """Every fp16 bit pattern (65,536), optionally without inf/NaN."""
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    if finite_only:
        bits = bits[np.isfinite(f16_bits_to_value(bits))]
    return bits


def pack_half2(lo_bits: np.ndarray, hi_bits: np.ndarray) -> np.ndarray:
    """Pack two binary16 bit patterns into one 32-bit `.f16x2` register.

    PTX convention (9.7.4.2 semantics, `fA[0] = a[0:15]`): element 0 lives in
    bits 0-15 ("low"), element 1 in bits 16-31 ("high").

    Trap worth knowing: `cvt.rn.f16x2.f32 d, a, b` puts `a` in the *upper*
    half (PTX ISA 9.4, 9.7.10.24), so packing (lo, hi) with cvt means
    `cvt.rn.f16x2.f32 d, hi, lo`.
    """
    lo = np.asarray(lo_bits, dtype=np.uint32)
    hi = np.asarray(hi_bits, dtype=np.uint32)
    return (lo & np.uint32(0xFFFF)) | (hi << np.uint32(16))


def unpack_half2(reg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 32-bit `.f16x2` register into (low, high) binary16 bits."""
    r = np.asarray(reg, dtype=np.uint32)
    lo = (r & np.uint32(0xFFFF)).astype(np.uint16)
    hi = (r >> np.uint32(16)).astype(np.uint16)
    return lo, hi
