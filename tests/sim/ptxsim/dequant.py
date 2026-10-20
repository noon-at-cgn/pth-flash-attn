# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The int8 -> fp16 "magic number" conversion and the V byte pairing.

This is the simulator twin of the CUDA `dequant.cuh` (`cvt4`, V pairing).
Each step is one PTX instruction modelled in `isa.py`, so the proof in
`test_sim_magic.py` is a proof about the instruction sequence, not about a
Python re-statement of the idea.

The idea (Kim et al., arXiv 2211.10017, section on fast int8->fp16):
the binary16 number with bit pattern 0x64XX is 1024 + XX for every byte XX,
because exponent field 0b11001 = 25 means 2^(25-15) = 1024 and at that
binade the 10 fraction bits count in steps of exactly 1. So writing a byte
into the low 8 fraction bits of 0x6400 *is* an integer-to-float conversion
for unsigned bytes; subtracting 1024 removes the offset.

Signed bytes: a two's-complement int8 x read as unsigned is x mod 256, which
is not monotone in x. XOR with 0x80 maps x to x + 128 (in 0..255, monotone),
so 0x64XX then encodes 1024 + 128 + x and one subtraction of 1152 (bit
pattern 0x6480) yields x. Both 1024+128+x (in 1024..1279) and x (in
-128..127) are exactly representable in binary16, so the subtraction is exact
(IEEE subtraction is correctly rounded; an exactly representable result is
returned unrounded).
"""

from __future__ import annotations

import numpy as np

from .isa import byte_perm, hsub2, lop3, lut_of, prmt

U32 = np.uint32

# Constants of the conversion.
MAGIC_SIGN_FLIP = 0x80808080  # XOR flips the sign bit of each of 4 bytes
MAGIC_EXP_BYTES = 0x64646464  # source of the 0x64 high byte (fp16 1024 binade)
MAGIC_BIAS_HALF2 = 0x64806480  # 1152.0 in both halves = 1024 + 128
MAGIC_BIAS_UNSIGNED_HALF2 = 0x64006400  # 1024.0 in both halves (uint8 data)
SEL_BYTES_01 = 0x4140  # prmt: (u.b0, 0x64, u.b1, 0x64) -> half2 of bytes 0,1
SEL_BYTES_23 = 0x4342  # prmt: (u.b2, 0x64, u.b3, 0x64) -> half2 of bytes 2,3
LUT_XOR_AB = lut_of(lambda a, b, c: a ^ b)  # 0x3C: lop3 immLut for a ^ b

# V byte pairing selectors (__byte_perm): pair token rows w0 and w1 per d.
V_SEL_D01 = 0x5140  # (w0.b0, w1.b0, w0.b1, w1.b1): d+0 and d+1 token pairs
V_SEL_D23 = 0x7362  # (w0.b2, w1.b2, w0.b3, w1.b3): d+2 and d+3 token pairs

# bf16 analogue used only as a negative control: 0x43XX is 128 + XX only for
# XX <= 128, because bf16 has 7 fraction bits (the 8th byte bit spills into
# the exponent field: 0x43XX = 256 + 2(XX - 128) above that).
BF16_MAGIC_HIGH_BYTE = 0x43


def cvt4_s8_to_f16x2(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4 packed int8 values to two f16x2 registers.

    Instruction sequence (1 LOP3 + 2 PRMT + 2 HSUB2 for 4 values):
        u  = lop3(w, 0x80808080, 0, 0x3C)        // u = w ^ 0x80808080
        lo = prmt(u, 0x64646464, 0x4140)         // {0x64,u.b1,0x64,u.b0}
        hi = prmt(u, 0x64646464, 0x4342)         // {0x64,u.b3,0x64,u.b2}
        lo = sub.f16x2(lo, 0x64806480)           // (b0, b1) exactly
        hi = sub.f16x2(hi, 0x64806480)           // (b2, b3) exactly
    Returns (lo, hi): lo holds bytes 0 and 1 (low half = byte 0), hi holds
    bytes 2 and 3. This byte order is what makes the K permutation pi
    necessary (see `paths.derive_pi`).
    """
    w = np.asarray(w, dtype=U32)
    u = lop3(w, U32(MAGIC_SIGN_FLIP), U32(0), LUT_XOR_AB)
    lo = prmt(u, U32(MAGIC_EXP_BYTES), SEL_BYTES_01)
    hi = prmt(u, U32(MAGIC_EXP_BYTES), SEL_BYTES_23)
    bias = U32(MAGIC_BIAS_HALF2)
    return hsub2(lo, bias), hsub2(hi, bias)


def cvt4_u8_to_f16x2(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned variant: no sign flip, subtract 1024 instead of 1152."""
    w = np.asarray(w, dtype=U32)
    lo = prmt(w, U32(MAGIC_EXP_BYTES), SEL_BYTES_01)
    hi = prmt(w, U32(MAGIC_EXP_BYTES), SEL_BYTES_23)
    bias = U32(MAGIC_BIAS_UNSIGNED_HALF2)
    return hsub2(lo, bias), hsub2(hi, bias)


def cvt4_s8_without_sign_flip(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Negative control: the signed bias without the XOR. Must be wrong."""
    w = np.asarray(w, dtype=U32)
    lo = prmt(w, U32(MAGIC_EXP_BYTES), SEL_BYTES_01)
    hi = prmt(w, U32(MAGIC_EXP_BYTES), SEL_BYTES_23)
    bias = U32(MAGIC_BIAS_HALF2)
    return hsub2(lo, bias), hsub2(hi, bias)


def v_pair_tokens(
    w_even: np.ndarray, w_odd: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair two token rows of V bytes so each 16-bit slot holds one d.

    w_even = V[t][d..d+3], w_odd = V[t+1][d..d+3] (one LDS.32 each). The
    MMA B operand wants, per register, two consecutive k (= tokens) at one
    n (= d). `__byte_perm` with 0x5140 builds (t,d) (t+1,d) (t,d+1) (t+1,d+1)
    and 0x7362 the same for d+2, d+3. After `cvt4` each half2 holds the token
    pair for one d. Returns the two pre-conversion words.
    """
    return byte_perm(w_even, w_odd, V_SEL_D01), byte_perm(w_even, w_odd, V_SEL_D23)
