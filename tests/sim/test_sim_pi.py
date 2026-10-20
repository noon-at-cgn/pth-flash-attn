# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the K-index permutation pi.

`ldmatrix.b16` on int8 K hands lane (g, q) the four bytes d = 4q .. 4q+3 of
token g in the current 16-byte chunk. After cvt4 those become B elements
b0..b3, which the MMA uses at k = 2q, 2q+1, 2q+8, 2q+9. So MMA index k reads
K at d = pi[k]. If Q is stored with the same permutation, the k-sum is the
d-sum in a different order, which is the same number.
"""

import numpy as np
import pytest
from ptxsim.layouts import (
    N_TOKENS,
    PI_PERMUTATION,
    kappa_closed_form,
    permute_q_chunks,
    pi_inverse,
)
from ptxsim.paths import build_q_smem, build_stage, derive_pi, qk_warp, s_matrix

RNG_SEED = 20261011


def _qk_inputs(D, seed):
    rng = np.random.default_rng(seed)
    # Q: integers times 2^-4, |q| <= 2^7 -> exact in fp16; K: full int8 range.
    q = rng.integers(-2048, 2048, size=(16, D)).astype(np.float64) / 16
    k = rng.integers(-128, 128, size=(N_TOKENS, D)).astype(np.int8)
    ks = np.ones(N_TOKENS, dtype=np.float32)
    v = np.zeros((N_TOKENS, D), dtype=np.int8)
    return q, k, v, ks


def test_pi_rederived_from_layouts_matches_reference_pi():
    """pi derived from the ldmatrix/cvt4/B-fragment models equals the
    documented pi.

    Why it matters: the documented pi was derived by hand. Here it is
    derived mechanically from the transcribed ISA tables, so a transcription
    slip in the documentation or in the simulator would show up as a mismatch.
    """
    assert (
        derive_pi()
        == PI_PERMUTATION
        == (0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15)
    )


def test_pi_bijection_preserves_dot():
    """pi is a permutation of 0..15, and permuting q and k by the same pi
    leaves the dot product unchanged (exactly, for integer data).

    Why it matters: this is the whole justification for storing Q permuted:
    sum_k q[pi k] k[pi k] = sum_d q[d] k[d] because pi only reorders the
    terms. With integer inputs the equality is exact, not approximate.
    """
    assert sorted(PI_PERMUTATION) == list(range(16))
    rng = np.random.default_rng(RNG_SEED)
    for D in (64, 128, 256):
        q = rng.integers(-1000, 1000, size=(32, D))
        k = rng.integers(-128, 128, size=(32, D))
        assert np.array_equal(permute_q_chunks(q) @ permute_q_chunks(k).T, q @ k.T)


def test_kappa_is_pi_inverse():
    """The closed-form kappa map (d -> position) is exactly pi^-1.

    Why it matters: the closed form and the permutation table were derived
    independently; this pins that they agree, so one map can be cited.
    """
    assert tuple(kappa_closed_form(d) for d in range(16)) == pi_inverse(PI_PERMUTATION)


@pytest.mark.parametrize("D", [64, 128, 256])
def test_qk_through_int8_ldmatrix_and_pi_equals_matmul(D):
    """One warp's S = Q K^T via the real instruction path equals Q @ K^T.

    Why it matters: this runs the inner QK loop end to end on a byte-exact
    shared-memory model: pi-permuted, swizzled Q smem -> ldmatrix.x4 (A);
    swizzled int8 K stage -> ldmatrix.x4 b16 -> cvt4 (B); mma.m16n8k16. Any
    error in pi, the swizzles, the lane addresses, the magic conversion or a
    fragment map changes S. All 64 tokens and all D/16 k-steps are covered.
    """
    q, k, v, ks = _qk_inputs(D, RNG_SEED + D)
    stage = build_stage(k, v, ks, ks)
    q_smem = build_q_smem(q, permute=True)
    s = s_matrix(qk_warp(q_smem, stage, D, tok0=0, n_tiles=N_TOKENS // 8))
    assert np.array_equal(s, q @ k.astype(np.float64).T)


@pytest.mark.parametrize("D", [64, 256])
def test_unpermuted_q_fails(D):
    """Negative control: the same path with Q stored *unpermuted* is wrong.

    Why it matters: proves the permutation is necessary (the int8 ldmatrix
    trick really does deliver K in pi order) and that the end-to-end test
    above is sensitive to it. Expect most entries of S to differ.
    """
    q, k, v, ks = _qk_inputs(D, RNG_SEED + 7 * D)
    stage = build_stage(k, v, ks, ks)
    q_smem = build_q_smem(q, permute=False)
    s = s_matrix(qk_warp(q_smem, stage, D, tok0=0, n_tiles=N_TOKENS // 8))
    want = q @ k.astype(np.float64).T
    assert not np.array_equal(s, want)
    assert np.mean(s != want) > 0.9
