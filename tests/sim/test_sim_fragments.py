# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the ldmatrix and mma.m16n8k16 fragment models.

Everything else in tests/sim is built on these maps, so they are checked two
ways: against the PTX ISA formulas typed in independently here, and
end-to-end, by loading matrices with the ldmatrix model and multiplying them
with the MMA model. Two independently transcribed tables that multiply
correctly cannot both be wrong in a compensating way for all inputs.
"""

import numpy as np
import pytest
from ptxsim.fp import f16_value_to_bits, round_to_f16
from ptxsim.isa import (
    A_MAP,
    B_MAP,
    C_MAP,
    gather_c,
    ldmatrix,
    ldmatrix_lane_map,
    mma_m16n8k16,
)

RNG_SEED = 20261010


def _f16_ints(rng, shape, lo=-64, hi=64):
    """Integer-valued fp16 matrices: every product and sum below is exact."""
    return rng.integers(lo, hi, size=shape).astype(np.float64)


def _smem_rowmajor_f16(mat):
    """fp16 matrix -> little-endian bytes, row-major, 16-B aligned rows."""
    bits = f16_value_to_bits(mat).astype(np.uint16)
    return bits.view(np.uint8).ravel().copy()


def test_ldmatrix_lane_map_matches_isa_figure_107():
    """Lane l gets row l//4, columns 2(l%4) and 2(l%4)+1 (low half first).

    Why it matters: this one table decides which K bytes and which Q
    elements every lane holds; pi, the V path and the swizzle analysis all
    inherit it. Transcribed from PTX ISA 9.4 Figure 107 (T0 gets (0,0),(0,1);
    T1 gets (0,2),(0,3); ... T31 gets (7,6),(7,7)).
    """
    m = ldmatrix_lane_map(trans=False)
    for lane in range(32):
        assert m[lane, 0].tolist() == [lane // 4, 2 * (lane % 4)]
        assert m[lane, 1].tolist() == [lane // 4, 2 * (lane % 4) + 1]
    t = ldmatrix_lane_map(trans=True)
    assert np.array_equal(t[..., 0], m[..., 1]) and np.array_equal(t[..., 1], m[..., 0])
    # Every element of the 8x8 matrix is delivered exactly once.
    cells = {tuple(x) for x in m.reshape(-1, 2).tolist()}
    assert len(cells) == 64


def test_mma_maps_match_isa_formulas_and_are_bijections():
    """A/B/C fragment maps equal PTX ISA 9.7.16.5.8 and cover each element once.

    Why it matters: a fragment map that dropped or duplicated an element
    would make the MMA silently compute a different product.
    """
    for lane in range(32):
        g, t = lane >> 2, lane % 4
        for i in range(8):
            row = g if i in (0, 1, 4, 5) else g + 8
            col = t * 2 + (i & 1) + (8 if i >= 4 else 0)
            assert A_MAP[lane, i].tolist() == [row, col]
        for i in range(4):
            assert B_MAP[lane, i].tolist() == [t * 2 + (i & 1) + 8 * (i >= 2), g]
            assert C_MAP[lane, i].tolist() == [g + 8 * (i >= 2), t * 2 + (i & 1)]
    assert len({tuple(x) for x in A_MAP.reshape(-1, 2).tolist()}) == 16 * 16
    assert len({tuple(x) for x in B_MAP.reshape(-1, 2).tolist()}) == 16 * 8
    assert len({tuple(x) for x in C_MAP.reshape(-1, 2).tolist()}) == 16 * 8


@pytest.mark.parametrize("b_style", ["k_major_x2", "n_major_x2_trans"])
def test_mma_16x8x16_through_ldmatrix_equals_matmul(b_style):
    """ldmatrix + mma.m16n8k16 reproduces a plain 16x8x16 product.

    Why it matters: this is the validation of the simulator itself. A comes
    from ldmatrix.x4 with the standard address pattern (lane l -> row l%16,
    k-half l//16). B is loaded in both ways the kernel family uses:
    `k_major_x2` stores B^T as [n][k] (like the K tile, token-major) and uses
    ldmatrix.x2; `n_major_x2_trans` stores B as [k][n] and uses .trans.
    Integer-valued inputs keep the product exact, so equality is exact.
    """
    rng = np.random.default_rng(RNG_SEED)
    a = _f16_ints(rng, (16, 16))
    b = _f16_ints(rng, (16, 8))
    c = _f16_ints(rng, (16, 8))
    lanes = np.arange(32)

    smem_a = _smem_rowmajor_f16(a)  # row pitch 32 B
    a_addr = (lanes % 16) * 32 + (lanes // 16) * 16
    a_regs = ldmatrix(smem_a, a_addr, 4)

    if b_style == "k_major_x2":
        smem_b = _smem_rowmajor_f16(b.T.copy())  # [n][k], pitch 32 B
        b_addr = (lanes % 8) * 32 + ((lanes // 8) % 2) * 16
        b_regs = ldmatrix(smem_b, b_addr, 2)
    else:
        smem_b = _smem_rowmajor_f16(b)  # [k][n], pitch 16 B
        b_addr = (lanes % 16) * 16
        b_regs = ldmatrix(smem_b, b_addr, 2, trans=True)

    d = mma_m16n8k16(a_regs, b_regs, c[C_MAP[..., 0], C_MAP[..., 1]])
    assert np.array_equal(gather_c(d), a @ b + c)


def test_ldmatrix_x1_x2_ignore_upper_lane_addresses():
    """.x1 reads addresses from lanes 0-7 only, .x2 from lanes 0-15 only.

    Why it matters: the kernel may leave garbage addresses in unused lanes;
    the ISA table says they are ignored, and the model must agree.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    smem = rng.integers(0, 256, size=4096, dtype=np.uint8)
    lanes = np.arange(32)
    good = (lanes % 16) * 64
    garbage = good.copy()
    garbage[8:] = 3 * 16  # different rows for lanes >= 8
    assert np.array_equal(ldmatrix(smem, good, 1), ldmatrix(smem, garbage, 1))
    garbage2 = good.copy()
    garbage2[16:] = 5 * 16
    assert np.array_equal(ldmatrix(smem, good, 2), ldmatrix(smem, garbage2, 2))
    # x4 of the same addresses returns x1's register as register 0.
    assert np.array_equal(ldmatrix(smem, good, 4)[:, :1], ldmatrix(smem, good, 1))


def test_mma_detects_a_swapped_fragment_register():
    """Negative control: swapping a2a3 with a4a5 must change the product.

    Why it matters: proves the end-to-end check above is sensitive to
    register order, so passing it means something.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    a = _f16_ints(rng, (16, 16))
    b = _f16_ints(rng, (16, 8))
    lanes = np.arange(32)
    a_regs = ldmatrix(_smem_rowmajor_f16(a), (lanes % 16) * 32 + (lanes // 16) * 16, 4)
    smem_b = _smem_rowmajor_f16(b)
    b_regs = ldmatrix(smem_b, (lanes % 16) * 16, 2, trans=True)
    zero = np.zeros((32, 4))
    good = gather_c(mma_m16n8k16(a_regs, b_regs, zero))
    bad_regs = a_regs[:, [0, 2, 1, 3]]
    bad = gather_c(mma_m16n8k16(bad_regs, b_regs, zero))
    assert np.array_equal(good, a @ b)
    assert not np.array_equal(bad, a @ b)


def test_round_to_f16_is_used_for_fragment_packing():
    """Packing non-representable values rounds them (cvt.rn), not truncates.

    Why it matters: P'' enters the PV MMA through this packing; the rounding
    must be round-to-nearest-even, the u = 2^-11 the error bound assumes.
    """
    x = np.array([1 + 2**-11, 1 + 3 * 2**-11, 2049.0, 2051.0])
    assert round_to_f16(x).tolist() == [1.0, 1 + 2 * 2**-10, 2048.0, 2052.0]
