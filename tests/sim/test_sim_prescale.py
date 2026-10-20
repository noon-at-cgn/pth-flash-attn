# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the per-row power-of-two Q prescale into fp16.

Claims under test (scaled value y = x * 2^-e, original value x):
1. The fp16 rounding model is IEEE round-to-nearest-even (two independent
   implementations agree on every bf16 and fp16 input).
2. For every bf16 x and every e the kernel can produce, cvt(y) is exact when
   |y| >= 2^-17 and in error by at most 2^-25 (i.e. 2^(e-25) in x units)
   otherwise; 2^-17 is tight. For fp16 inputs the threshold is 2^-14.
3. absmax * 2^-e lands in [2^14, 2^15), so nothing overflows.
"""

import numpy as np
import pytest
from ptxsim.fp import (
    all_bf16_bits,
    all_f16_bits,
    bf16_bits_to_value,
    f16_bits_to_value,
    round_to_f16,
)
from ptxsim.prescale import (
    BF16_SIG_BITS,
    E_MIN_CLAMP,
    F16_SIG_BITS,
    WINDOW_LOG2_LO,
    floor_log2,
    guaranteed_exact_floor_log2,
    pow2_f32,
    prescale_exponent,
    prescale_to_f16,
)

BF16_ALL = bf16_bits_to_value(all_bf16_bits())
F16_ALL = f16_bits_to_value(all_f16_bits())
# e range to sweep per x: from the smallest e that keeps y below 2^15 for
# this x as the row max, to well past total underflow (y < 2^-26 rounds to 0).
SWEEP_BINADES = 45


def test_rne_model_matches_numpy_on_every_input():
    """Own RNE (round_to_f16) == numpy's float16 cast on all finite bf16 and
    fp16 values, on the bf16 values scaled into every fp16 binade, and on 10^6
    random f32 values.

    Why it matters: every exactness claim below rests on the rounding model;
    two independent implementations agreeing everywhere they are compared is
    the evidence that it is right.
    """
    rng = np.random.default_rng(1)
    r32 = (rng.standard_normal(10**6) * 2.0 ** rng.integers(-30, 17, 10**6)).astype(
        np.float32
    )
    samples = [BF16_ALL, F16_ALL, r32.astype(np.float64)]
    for k in range(-40, 20, 3):
        samples.append(BF16_ALL * 2.0**k)
    for x in samples:
        x = x[np.abs(x) < 2.0**126]
        with np.errstate(over="ignore"):
            ref = x.astype(np.float32).astype(np.float16).astype(np.float64)
        assert np.array_equal(round_to_f16(x), ref, equal_nan=False)


def test_bf16_to_fp16_boundaries():
    """Hand-checked bf16 -> fp16 conversions at every boundary.

    Why it matters: these are the edges the prescale window and the error
    bound are built on; each line is one fact about binary16.
    """
    cases = [
        (65280.0, 65280.0),  # largest bf16 below 2^16: fits fp16 exactly
        (65536.0, np.inf),  # 2^16 overflows (fp16 max is 65504)
        (2.0**15 * (2 - 2**-7), 2.0**15 * (2 - 2**-7)),  # top of the window's binade
        (2.0**-14, 2.0**-14),  # smallest normal fp16
        (2.0**-17 * (1 + 2**-7), 2.0**-17 * (1 + 2**-7)),  # 8 sig bits at 2^-17: exact
        (2.0**-18 * (1 + 2**-7), 2.0**-18),  # 2^-25 tie -> even (down)
        (
            2.0**-18 * (1 + 3 * 2**-7),
            2.0**-18 * (1 + 2**-5),
        ),  # 65.5 quanta -> 66 (even)
        (2.0**-24, 2.0**-24),  # smallest subnormal
        (2.0**-25, 0.0),  # exactly half the smallest subnormal: tie -> 0 (even)
        (2.0**-25 * (1 + 2**-7), 2.0**-24),  # just above half: rounds up
        (-(2.0**-25), -0.0),  # sign kept
    ]
    for x, want in cases:
        got = float(round_to_f16(np.array([x]))[0])
        assert got == want and np.signbit(got) == np.signbit(want), (x, got, want)


def _sweep(values, e_max):
    """For every x and every e the kernel can give a row containing x.

    e starts at x's own e (x is the row max) and grows (larger row maxima)
    up to e_max, the e of the largest finite input; it is clamped below at
    E_MIN_CLAMP.
    """
    x = values[(values != 0) & np.isfinite(values)]
    e_lo = floor_log2(x) - WINDOW_LOG2_LO
    for de in range(SWEEP_BINADES):
        e = np.clip(e_lo + de, E_MIN_CLAMP, e_max)
        y_f16 = prescale_to_f16(x, e)
        y_exact = x * 2.0 ** (-e.astype(np.float64))
        yield y_exact, np.abs(y_f16 - y_exact)


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_prescale_exact_above_threshold_exhaustive(dtype):
    """For all finite bf16 (or fp16) x and every e that a row containing x can
    get: exact when |y| >= 2^T, error <= 2^-25 (scaled units) always, never
    overflow. T = -17 for bf16, -14 for fp16 (`guaranteed_exact_floor_log2`).

    Why it matters: this is the precise statement of the Q prescale
    underflow bound. In x units the error is <= 2^(e-25) per element.
    Relative to the row max (absmax * 2^-e >= 2^14), every element with
    |x| >= 2^(T-14) * absmax is exact: 2^-31 for bf16, 2^-28 for fp16.
    """
    values = BF16_ALL if dtype == "bf16" else F16_ALL
    sig = BF16_SIG_BITS if dtype == "bf16" else F16_SIG_BITS
    e_max = int(floor_log2(np.abs(values[np.isfinite(values)]).max())) - WINDOW_LOG2_LO
    t_exact = guaranteed_exact_floor_log2(sig)
    for y, err in _sweep(values, e_max):
        assert np.all(np.isfinite(y)) and np.all(np.abs(y) < 2.0**15)
        assert np.all(err[np.abs(y) >= 2.0**t_exact] == 0)
        assert np.all(err <= 2.0**-25)


@pytest.mark.parametrize("dtype,t", [("bf16", -17), ("fp16", -14)])
def test_exactness_threshold_is_tight(dtype, t):
    """Just below 2^T some values are inexact, so T cannot be lowered.

    Why it matters: documents that the thresholds are the true boundary, not
    a loose bound (bf16: y = 2^-18(1+2^-7) needs a 2^-25 bit; fp16:
    y = 2^-15(1+2^-10) needs 2^-25 too).
    """
    sig = BF16_SIG_BITS if dtype == "bf16" else F16_SIG_BITS
    y = 2.0 ** (t - 1) * (1 + 2.0 ** (1 - sig))
    assert round_to_f16(np.array([y]))[0] != y
    assert guaranteed_exact_floor_log2(sig) == t


def test_prescale_window_all_bf16_absmax():
    """For every finite nonzero bf16 absmax, absmax * 2^-e is in
    [2^14, 2^15) and converts exactly; an all-zero row gets e = 0.

    Why it matters: the window is what guarantees no fp16 overflow (fp16 max
    65504 >= 2^15) and puts the row max 14 binades above the normal range.
    """
    a = np.abs(BF16_ALL[BF16_ALL != 0])
    e = prescale_exponent(a)
    y = a * 2.0 ** (-e.astype(np.float64))
    assert np.all((y >= 2.0**14) & (y < 2.0**15))
    assert np.array_equal(round_to_f16(y), y)
    assert int(prescale_exponent(np.array([0.0]))[0]) == 0


def test_relative_threshold_claims():
    """A relative threshold of "exact for elements >= 2^-29 * absmax" is true for bf16
    inputs (their guaranteed threshold is 2^-31), false for fp16 inputs
    (their guaranteed threshold is 2^-28).

    Why it matters: it limits what can be claimed. Same formula, both dtypes: the
    bound 2^(e-25) per element holds for both (previous tests), but the
    "exact above 2^-29" sentence is bf16-only. Counterexample for fp16:
    absmax = 2^15 (e = 1), x = 2^-14 * (1 + 2^-10) = 2^-29 * absmax * (1 +
    2^-10) is >= 2^-29 * absmax, yet y = x/2 = 2^-15 (1 + 2^-10) needs a
    2^-25 bit and rounds.
    """
    absmax = 2.0**15
    x = 2.0**-14 * (1 + 2**-10)  # an fp16 normal number
    assert x >= 2.0**-29 * absmax
    e = prescale_exponent(np.array([absmax]))
    assert int(e[0]) == 1
    y = prescale_to_f16(np.array([x]), e)
    assert y[0] != x / 2  # inexact
    assert abs(y[0] - x / 2) <= 2.0**-25


def test_pow2_scale_overflows_without_clamp():
    """For row maxima below 2^-113, e < -127 and the
    f32 factor 2^-e is +inf. Clamping e >= E_MIN_CLAMP (-112) keeps both
    2^-e and 2^e normal f32 and keeps the per-element error bound.

    Why it matters: a bf16 row with absmax 2^-120 is legal input; without
    the clamp the prescale multiply produces inf (then NaN scores).
    """
    absmax = np.array([2.0**-120])
    e = prescale_exponent(absmax)
    assert int(e[0]) == -134 and np.isinf(pow2_f32(-e)[0])
    ec = prescale_exponent(absmax, clamp=True)
    assert int(ec[0]) == E_MIN_CLAMP
    assert np.isfinite(pow2_f32(-ec)[0]) and pow2_f32(ec)[0] >= np.float32(2.0**-126)
    # Every bf16 below 2^-98 (the clamped regime) still meets the bound.
    xs = BF16_ALL[(BF16_ALL != 0) & (np.abs(BF16_ALL) < 2.0**-98)]
    y = prescale_to_f16(xs, np.full(xs.shape, E_MIN_CLAMP))
    err_x_units = np.abs(y * 2.0**E_MIN_CLAMP - xs)
    assert np.all(err_x_units <= 2.0 ** (E_MIN_CLAMP - 25))


def test_window_one_binade_higher_is_also_overflow_free():
    """Evidence for an alternative window: a window
    [2^15, 2^16) also never overflows (largest bf16 below 2^16 is 65280,
    largest fp16 is 65504) and would make fp16 inputs never scale down.

    Why it matters: documents that [2^14, 2^15) is a choice with one binade
    of margin, not a necessity; the alternative gains one bit of underflow
    headroom.
    """
    for vals in (BF16_ALL, F16_ALL):
        a = np.abs(vals[(vals != 0) & np.isfinite(vals)])
        # Same lower clamp as the chosen window (below it rows sit under the
        # window and the f32 factor 2^-e would overflow otherwise).
        e = np.maximum(floor_log2(a) - (WINDOW_LOG2_LO + 1), E_MIN_CLAMP)
        y = prescale_to_f16(a, e)
        assert np.all(np.isfinite(y))
        y_exact = a * 2.0 ** (-e.astype(np.float64))
        assert np.all(y_exact < 2.0**16)
        # The row max itself is exact wherever it is in the window.
        in_window = y_exact >= 2.0**15
        assert np.array_equal(y[in_window], y_exact[in_window])
