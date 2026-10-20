# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the kernel-arithmetic emulator (``pth_testing.emulate``).

The emulator is only evidence for the bound if its building blocks model the
hardware: the tensor-core accumulation, the power-of-two bookkeeping and the Q
prescale. Each block is pinned to a published or provable fact here.
"""

from __future__ import annotations

import numpy as np
import pytest
from pth_testing.bound import TC_ABS_PER_TERM, gamma_tc, pow2_ceil_exp
from pth_testing.emulate import (
    TensorCoreModel,
    f16,
    f32,
    prescale_q,
    tc_dot,
    tc_step,
)

_TC = TensorCoreModel()


def _dot(c, x, ex, tc=_TC):
    return float(tc_step(np.array([c]), np.array([x]), np.array([ex]), tc)[0])


def test_tc_model_published_vectors():
    """The tensor-core model reproduces published A100 test vectors.

    1. Khattak & Mikaitis (arXiv 2512.07004 §4.1.2): c = 0, p1 = 1.5 x 1.5,
       p2 = 2^-23, p3 = p4 = 2^-24 gives 2.25 + 2^-22: products are aligned to
       the largest *exponent sum* (0 here, not 1) with 24 fraction bits.
    2. Fasi et al. 2021 (rounding-mode test): 2 + (3/4) 2^-22 gives 2, i.e. the
       result is truncated (round-to-nearest would give 2 + 2^-22).
    Negative control: aligning to the value's own exponent (1) loses the two
    2^-24 terms and gives 2.25.
    """
    ex = [0, -23, -24, -24]  # exponent sums of the four products
    got = _dot(0.0, [2.25, 2.0**-23, 2.0**-24, 2.0**-24], ex)
    assert got == 2.25 + 2.0**-22
    wrong = _dot(0.0, [2.25, 2.0**-23, 2.0**-24, 2.0**-24], [1, -23, -24, -24])
    assert wrong == 2.25, "negative control: value-exponent alignment differs"
    rz = _dot(
        2.0, [0.75 * 2.0**-22, 0.0, 0.0, 0.0], [-23, -(10**6), -(10**6), -(10**6)]
    )
    assert rz == 2.0
    assert (
        float(np.float32(2.0 + 0.75 * 2.0**-22)) == 2.0 + 2.0**-22
    )  # RN would round up


@pytest.mark.parametrize("align", ["rz", "rd"])
@pytest.mark.parametrize("n", [16, 64, 256])
def test_tc_accumulation_within_gamma_tc(align, n):
    """|tc_dot - exact| <= gamma_tc(n) sum|a b| + n 2^-31, random and cancelling.

    Why: ``gamma_tc`` is the coefficient the bound uses for QK (n = D) and PV
    (n = tokens per warp); the emulator's accumulation must respect it, with
    both alignment conventions (truncation toward zero and toward -inf). Inputs
    include fp16 subnormals (tiny Q after prescale) and exact cancellation.
    """
    rng = np.random.default_rng(n)
    tc = TensorCoreModel(align=align)
    worst = 0.0
    for trial in range(200):
        a = f16(rng.standard_normal((1, n)) * 2.0 ** rng.integers(-20, 15))
        b = rng.integers(-127, 128, size=(1, n)).astype(np.float64)
        if trial % 4 == 0:  # cancellation: second half negates the first
            a[0, n // 2 :] = a[0, : n // 2]
            b[0, n // 2 :] = -b[0, : n // 2]
        got = tc_dot(np.zeros(1), a, b, tc)[0]
        exact = float((a * b).sum())
        lim = gamma_tc(n) * float(np.abs(a * b).sum()) + n * TC_ABS_PER_TERM
        assert abs(got - exact) <= lim
        worst = max(worst, abs(got - exact) / lim if lim else 0.0)
    assert worst > 1e-3  # the model does round: not a vacuous comparison


def test_pow2_rescale_exact():
    """Scaling fp32 values by 2^k and back is exact; P'' = p vs 2^-c lies in (0, 1].

    Why: the V-scale normalisation keeps O in units of 2^c and rescales by
    ``alpha 2^(c_old - c_new)``. That is only free of error
    because multiplying by a power of two changes the exponent alone (inside
    the normal range). ``c = ceil(log2 max vs)`` from the exponent bits makes
    ``vs 2^-c`` in (0.5, 1] for the largest scale, including exact powers of
    two. Negative control: the same round trip with a non-power-of-two
    factor (1.1) is not exact.
    """
    rng = np.random.default_rng(0)
    x = f32(rng.standard_normal(10000) * 2.0 ** rng.integers(-40, 40, 10000))
    for k in (-20, -3, 1, 7, 20):
        y = f32(np.ldexp(x, k))
        assert np.array_equal(f32(np.ldexp(y, -k)), x)
    z = f32(f32(x * 1.1) / 1.1)
    assert not np.array_equal(z, x), "negative control"
    vs = np.concatenate(
        [f32(10.0 ** rng.uniform(-6, 2, 1000)), [2.0**-20, 0.5, 1.0, 4.0]]
    )
    c = pow2_ceil_exp(vs)
    r = np.ldexp(vs, -c)
    assert np.all((r > 0.5) & (r <= 1.0))
    assert np.all(c == np.ceil(np.log2(vs)))


def test_q_prescale_window_and_exactness():
    """Q prescale puts max|q| 2^-e in [2^14, 2^15), clamps e >= -112, and is exact.

    Why: the kernel feeds fp16 tensor cores for bf16 models.
    A bf16 value has 8 significant bits, so after the power-of-two prescale it
    is exactly representable in fp16 unless it falls into fp16's subnormal
    range; the bound charges at most 2^(e-25) for those. An all-zero row gets
    e = 0. Rows with max|q| < 2^-98 hit the clamp e = -112 (which keeps
    2^-e and 2^e normal in fp32) and sit below the window; the 2^(e-25)
    per-element bound still holds for them. Negative control: without the
    prescale, bf16 queries outside fp16's range are not exact in fp16.
    """
    rng = np.random.default_rng(1)
    q = rng.standard_normal((64, 256)) * 2.0 ** rng.integers(-30, 30, (64, 1))
    q = q.astype(np.float32)
    q = (
        (q.view(np.uint32) & 0xFFFF0000).view(np.float32).astype(np.float64)
    )  # bf16 values
    q[3] = 0.0
    # power-of-two scaling keeps the 8-bit significands: max|q| ~ 2^-105 < 2^-98
    q[5] = np.ldexp(q[5], -(int(np.floor(np.log2(np.abs(q[5]).max()))) + 105))
    qt, e = prescale_q(q)
    amax = np.abs(q).max(-1)
    nz = amax > 0
    win = nz & (amax >= 2.0**-98)
    scaled = np.ldexp(amax[win], -e[win])
    assert np.all((scaled >= 2.0**14) & (scaled < 2.0**15))
    assert e[5] == -112 and np.ldexp(amax[5], -e[5]) < 2.0**14
    assert e[3] == 0 and np.all(qt[3] == 0)
    back = np.ldexp(qt, e[:, None])
    big = np.abs(q) >= np.ldexp(1.0, e[:, None] - 14)
    assert np.array_equal(back[big], q[big])
    assert np.all(np.abs(back - q) <= np.ldexp(1.0, e[:, None] - 25))
    assert not np.array_equal(f16(q[nz][:4]), q[nz][:4]), "negative control"
