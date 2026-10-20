# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the magic-number int8 -> fp16 conversion.

The conversion is 1 LOP3 + 2 PRMT + 2 HSUB2 per 4 values, each modelled
bit-exactly in ptxsim.isa from the PTX ISA. The claim under test is that
this sequence returns *exactly* the int8 value for every possible byte, in
every byte position, for both halves of the output.
"""

import numpy as np
from ptxsim.dequant import (
    BF16_MAGIC_HIGH_BYTE,
    LUT_XOR_AB,
    MAGIC_BIAS_HALF2,
    SEL_BYTES_01,
    SEL_BYTES_23,
    cvt4_s8_to_f16x2,
    cvt4_s8_without_sign_flip,
    cvt4_u8_to_f16x2,
)
from ptxsim.fp import (
    bf16_bits_to_value,
    f16_bits_to_value,
    f16_value_to_bits,
    pack_half2,
    unpack_half2,
)
from ptxsim.isa import byte_perm, hsub2, lop3, lut_of, prmt

ALL_INT8 = np.arange(-128, 128, dtype=np.int64)


def _pack_bytes(b0, b1, b2, b3):
    u = [
        np.asarray(b, dtype=np.int64).astype(np.uint8).astype(np.uint32)
        for b in (b0, b1, b2, b3)
    ]
    return u[0] | (u[1] << 8) | (u[2] << 16) | (u[3] << 24)


def _decode(lo, hi):
    """(lo, hi) f16x2 registers -> values of bytes 0, 1, 2, 3."""
    l0, l1 = unpack_half2(lo)
    h0, h1 = unpack_half2(hi)
    return [f16_bits_to_value(x) for x in (l0, l1, h0, h1)]


def test_magic_exact_all_256():
    """cvt4 returns exactly x for all 256 int8 values (incl. -128) in every
    byte position.

    Why it matters: K and V reach the tensor cores only through this
    conversion; a single wrong value would bias every score or output that
    touches it, and the error bound assumes the conversion is exact
    (QK products are exact). The four byte positions
    are rotated so every value is seen in position 0, 1, 2 and 3.
    """
    for rot in range(4):
        cols = [np.roll(ALL_INT8, 64 * k + rot) for k in range(4)]
        lo, hi = cvt4_s8_to_f16x2(_pack_bytes(*cols))
        got = _decode(lo, hi)
        for k in range(4):
            assert np.array_equal(got[k], cols[k].astype(np.float64))
    # -128 explicitly, all four positions at once: u = 0x00000000 after the
    # XOR, so every half is 0x6400 = 1024 and 1024 - 1152 = -128.
    lo, hi = cvt4_s8_to_f16x2(_pack_bytes(-128, -128, -128, -128))
    assert [v.item() for v in _decode(lo, hi)] == [-128.0] * 4


def test_magic_exact_all_byte_pairs():
    """Each f16x2 half pair is exact for all 65,536 (b0, b1) combinations.

    Why it matters: the two halves of a register are produced together by one
    PRMT and one HSUB2, so a cross-talk bug between neighbouring bytes would
    only show up for particular pairs. This covers every pair exhaustively
    (and the (b2, b3) half with the pair reversed).
    """
    x, y = np.meshgrid(ALL_INT8, ALL_INT8, indexing="ij")
    x, y = x.ravel(), y.ravel()
    lo, hi = cvt4_s8_to_f16x2(_pack_bytes(x, y, y, x))
    got = _decode(lo, hi)
    for k, want in enumerate((x, y, y, x)):
        assert np.array_equal(got[k], want.astype(np.float64))


def test_magic_result_bits_are_canonical():
    """The outputs are the canonical fp16 encodings (0 is +0, never -0).

    Why it matters: an MMA treats -0 and +0 identically, but golden-vector
    comparisons in the header tests compare bits; this pins the expected bits.
    """
    lo, _ = cvt4_s8_to_f16x2(_pack_bytes(ALL_INT8, ALL_INT8, 0, 0))
    l0, _ = unpack_half2(lo)
    assert np.array_equal(l0, f16_value_to_bits(ALL_INT8.astype(np.float64)))


def test_magic_constants_mean_what_the_docs_say():
    """0x6480 is 1152.0, 0x64XX is 1024 + XX, and the selectors pick bytes.

    Why it matters: these are the literal constants that will appear in
    dequant.cuh; this test is their derivation.
    """
    lo, hi = unpack_half2(np.uint32(MAGIC_BIAS_HALF2))
    assert f16_bits_to_value(lo) == 1152.0 == f16_bits_to_value(hi)
    xx = np.arange(256, dtype=np.uint16)
    assert np.array_equal(f16_bits_to_value(0x6400 | xx), 1024.0 + xx)
    # prmt(a=0xDDCCBBAA, b=0x64646464, 0x4140) = {0x64, 0xBB, 0x64, 0xAA}.
    a = np.uint32(0xDDCCBBAA)
    assert int(prmt(a, np.uint32(0x64646464), SEL_BYTES_01)) == 0x64BB64AA
    assert int(prmt(a, np.uint32(0x64646464), SEL_BYTES_23)) == 0x64DD64CC
    assert LUT_XOR_AB == 0x3C


def test_unsigned_variant_exact_and_signed_needs_the_xor():
    """uint8 needs no XOR (bias 1024); int8 without the XOR is always wrong.

    Why it matters: documents *why* the LOP3 is in the sequence. Two's
    complement bytes read as unsigned are x mod 256; only after flipping the
    sign bit is the byte x + 128, monotone in x, so one subtraction works.
    The negative control must fail for every value (no x with byte(x) =
    x + 128 exists).
    """
    u = np.arange(256, dtype=np.int64)
    lo, hi = cvt4_u8_to_f16x2(_pack_bytes(u, u, u, u))
    for got in _decode(lo, hi):
        assert np.array_equal(got, u.astype(np.float64))
    lo, hi = cvt4_s8_without_sign_flip(
        _pack_bytes(ALL_INT8, ALL_INT8, ALL_INT8, ALL_INT8)
    )
    for got in _decode(lo, hi):
        assert np.all(got != ALL_INT8)


def test_bf16_magic_is_not_exact():
    """Negative control: the same trick in bf16 (0x43XX) is wrong for 127 of
    the 256 bytes.

    Why it matters: this is the arithmetic reason the kernel uses fp16 MMA.
    bf16 has 7 fraction bits, so 0x43XX = 128 + XX only for
    XX < 128. For XX = 128 + m the top bit spills into the exponent field:
    0x43XX = 256 + 2m, which equals 128 + XX only for m = 0. So bytes
    129..255 come out wrong.
    """
    xx = np.arange(256, dtype=np.uint32)
    vals = bf16_bits_to_value((BF16_MAGIC_HIGH_BYTE << 8) | xx)
    ok = vals == 128.0 + xx
    assert ok[:129].all() and not ok[129:].any()
    assert np.array_equal(vals[128:], 256.0 + 2.0 * np.arange(128))


def test_fp32_magic_then_bf16_truncation_is_exact():
    """The bf16 alternative path (fp32 magic, keep the upper half) is exact.

    Why it matters: evidence for the bf16 MMA alternative:
    it is *exact* too, but needs a 32-bit magic per value (2 PRMT per value
    instead of per pair) -- the cost difference is the reason, not accuracy.
    f32 0x4B0000XX = 2^23 + XX; subtracting 2^23 + 128 gives x exactly; x
    has <= 8 significant bits, so its low 16 f32 bits are zero and taking the
    high half is an exact bf16.
    """
    xx = (ALL_INT8 + 128).astype(np.uint32)
    f = (np.uint32(0x4B000000) | xx).view(np.float32)
    x32 = f - np.float32(2**23 + 128)
    assert np.array_equal(x32.astype(np.float64), ALL_INT8.astype(np.float64))
    bits = x32.view(np.uint32)
    assert np.all((bits & 0xFFFF) == 0)
    assert np.array_equal(bf16_bits_to_value(bits >> 16), ALL_INT8.astype(np.float64))


def test_instruction_models_against_isa_examples():
    """prmt, __byte_perm, lop3 and sub.f16x2 models match the ISA's own
    examples and IEEE ties-to-even.

    Why it matters: the magic proof is only as good as these models. Checks:
    lop3 immLut examples from PTX ISA 9.7.9.6 (0x80, 0xFE, 0x40, 0x1A);
    lop3 against Python bit ops for every immLut; prmt sign-replicate mode;
    __byte_perm ignoring the selector msb; sub.f16x2 rounding a tie to even
    and operating on both halves independently.
    """
    assert lut_of(lambda a, b, c: a & b & c) == 0x80
    assert lut_of(lambda a, b, c: a | b | c) == 0xFE
    assert lut_of(lambda a, b, c: a & b & ~c) == 0x40
    assert lut_of(lambda a, b, c: (a & b | c) ^ a) == 0x1A
    rng = np.random.default_rng(7)
    a, b, c = (
        rng.integers(0, 2**32, size=64, dtype=np.uint64).astype(np.uint32)
        for _ in range(3)
    )
    for lut in range(256):
        want = np.zeros(64, dtype=np.uint32)
        for j in range(32):
            idx = (((a >> j) & 1) << 2) | (((b >> j) & 1) << 1) | ((c >> j) & 1)
            want |= ((np.uint32(lut) >> idx.astype(np.uint32)) & 1) << np.uint32(j)
        assert np.array_equal(lop3(a, b, c, lut), want)
    # prmt: nibble msb set -> replicate the sign of the selected byte.
    # a = 0x00008001 (byte1 = 0x80, byte0 = 0x01); selector nibbles, low
    # first: 9 (byte1, replicate sign -> 0xFF), 8 (byte0, sign 0 -> 0x00),
    # 0 (byte0 copy -> 0x01), 0 (byte0 copy -> 0x01).
    assert int(prmt(np.uint32(0x00008001), np.uint32(0), 0x0089)) == 0x010100FF
    # Selector 0x4448: byte 0 <- sign of a.b0; bytes 1-3 <- b.b0 (= 0).
    assert int(prmt(np.uint32(0x00000080), np.uint32(0), 0x4448)) == 0xFF
    assert int(prmt(np.uint32(0x0000007F), np.uint32(0), 0x4448)) == 0x00
    # __byte_perm uses only 3 selector bits: 8 is byte 0 copied, no sign mode.
    assert int(byte_perm(np.uint32(0x00000080), np.uint32(0), 0x4448)) == 0x80
    # sub.f16x2: (1 + 2^-10) - (-2^-11) = 1 + 3*2^-11: a tie, rounds to even
    # 1 + 2^-9; low half 2048 - 1 = 2047 exactly.
    one_p = f16_value_to_bits(np.array([2048.0, 1 + 2**-10]))
    sub = f16_value_to_bits(np.array([1.0, -(2**-11)]))
    d = hsub2(pack_half2(one_p[0], one_p[1]), pack_half2(sub[0], sub[1]))
    lo, hi = unpack_half2(d)
    assert f16_bits_to_value(lo) == 2047.0
    assert f16_bits_to_value(hi) == 1 + 2**-9
