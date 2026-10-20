# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Per-row power-of-two Q prescale into fp16.

Kernel path being modelled, per query row:
    e      = floor(log2(absmax)) - 14         (0 for an all-zero row)
    x_f32  = float(x)                         (bf16 -> f32 is a shift: exact)
    y_f32  = x_f32 * 2^-e                     (FMUL.rn by a power of two)
    q_f16  = cvt.rn.f16.f32(y_f32)            (one rounding, RNE)
and later the score is multiplied by 2^e (folded into the per-row factor).

Why it is (almost) exact: multiplying by a power of two only changes the
exponent, so y is exact in f32 as long as it stays a normal f32. The only
rounding is the final cvt, and it is exact when y fits the fp16 grid: the
significand fits (bf16 has 8 significant bits, fp16 has 11) and y is not in
the part of the subnormal range whose spacing 2^-24 is coarser than y's own
last bit.
"""

from __future__ import annotations

import numpy as np

from .fp import round_to_f16

WINDOW_LOG2_LO = 14  # absmax * 2^-e lands in [2^14, 2^15)
F16_SUBNORMAL_QUANTUM_LOG2 = -24  # fp16 subnormal spacing 2^-24
BF16_SIG_BITS = 8  # significant bits of bf16 (7 stored + hidden)
F16_SIG_BITS = 11  # significant bits of fp16 (10 stored + hidden)
F32_MAX_EXP = 127  # 2^127 is the largest power of two in f32
F32_MIN_NORMAL_EXP = -126  # 2^-126 is the smallest normal f32
# Lower clamp for e:
# keeps 2^-e <= 2^112 and 2^e >= 2^-112, i.e. both factors normal f32 with
# 14 binades of headroom for the softmax_scale * log2(e) multiplier.
E_MIN_CLAMP = F32_MIN_NORMAL_EXP + WINDOW_LOG2_LO  # -112


def floor_log2(x: np.ndarray) -> np.ndarray:
    """floor(log2 |x|) for nonzero finite x, exact (frexp splits exponents)."""
    _, e = np.frexp(np.abs(np.asarray(x, dtype=np.float64)))
    return (e - 1).astype(np.int64)


def prescale_exponent(absmax: np.ndarray, clamp: bool = False) -> np.ndarray:
    """e such that absmax * 2^-e is in [2^14, 2^15); 0 for absmax == 0.

    With clamp=True, e is raised to at least E_MIN_CLAMP (see module
    constants); rows that small then sit below the window, which costs
    nothing in exactness (the per-element bound 2^(e-25) still holds).
    """
    absmax = np.asarray(absmax, dtype=np.float64)
    safe = np.where(absmax > 0, absmax, 1.0)
    e = floor_log2(safe) - WINDOW_LOG2_LO
    e = np.where(absmax > 0, e, 0)
    if clamp:
        e = np.maximum(e, E_MIN_CLAMP)
    return e


def pow2_f32(k: np.ndarray) -> np.ndarray:
    """2^k as an f32 the way the kernel would materialise it (exponent bits).

    Returns inf where 2^k exceeds the f32 range and 0 below the subnormal
    range, which is what building the bit pattern (or ldexpf) produces.
    """
    k = np.asarray(k, dtype=np.int64)
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(np.float32(1.0), k).astype(np.float32)


def prescale_to_f16(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Model of the kernel path: f32 multiply by 2^-e, then cvt.rn.f16.f32.

    x: float64 values that are exact bf16 or fp16 values. Returns float64
    values on the fp16 grid (inf on overflow).
    """
    x32 = np.asarray(x, dtype=np.float32)
    scale = pow2_f32(-np.asarray(e))
    with np.errstate(over="ignore", invalid="ignore"):
        y = (x32 * scale).astype(np.float64)  # IEEE f32 RNE multiply
    return round_to_f16(y)


def guaranteed_exact_floor_log2(sig_bits: int) -> int:
    """Smallest floor(log2|y|) at which every value with `sig_bits`
    significant bits is exactly representable in fp16.

    y = M * 2^(E - sig_bits + 1) with an integer M of sig_bits bits; it lies
    on the fp16 grid iff its last bit 2^(E - sig_bits + 1) >= 2^-24 (the
    finest fp16 spacing). So E >= -24 + sig_bits - 1:
    bf16 (8 bits) -> E >= -17;  fp16 (11 bits) -> E >= -14.
    """
    return F16_SUBNORMAL_QUANTUM_LOG2 + sig_bits - 1
