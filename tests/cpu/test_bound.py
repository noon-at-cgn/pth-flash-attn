# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the data-dependent error bound (``pth_testing.bound``).

The bound is the correctness gate for every kernel run:
``|kernel - fp64 reference| <= bound`` element by element. Two properties
make it a useful gate, and both are tested here against the CPU emulator of
the kernel's arithmetic:

* **sound**: the emulated kernel never exceeds it (0 violations), and
* **sharp**: it is not vacuous (tightness ``max(err/bound)`` >= 1e-4), and
  real bugs (an off-by-one token, a wrong scale, a wrong head) exceed it by a
  wide margin. The mutation tests below are the negative controls of the
  soundness tests: they must fail the gate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pth_testing.bound import (
    DEEP_LOG2,
    LOG2E,
    Tiling,
    bound,
    gamma,
    gamma_tc,
    tightness,
    token_plan,
)
from pth_testing.emulate import Case, default_matrix, emulate, make_case, run_case
from pth_testing.reference import DenseKV, quantize_kv, ref_attention_dense

TIGHTNESS_FLOOR = 1e-4  # below this the bound is too loose to catch bugs

_SMALL = {c.name: c for c in default_matrix("small")}


@pytest.mark.parametrize("name", sorted(_SMALL))
def test_bound_holds_on_emulator(name):
    """Every emulated output element is within the bound, and the bound is sharp.

    Why: this establishes that the bound is a valid gate for the kernel design
    independently of any GPU code. The matrix covers
    decode (WN=4), verify (WN=2) and prefill (WN=1) tilings, splits S in
    {1, 2, 3}, D in {64, 128, 256}, G in {1, 4, 6}, V scales from the 1e-6
    floor to 1e-1, peaked and flat logits, tiny queries (Q prescale
    underflow), fp16/bf16 queries and fp32/bf16/fp16 outputs.
    """
    r = run_case(_SMALL[name])
    assert r.violations == 0, (
        f"{name}: {r.violations} violations, tightness {r.tightness:.3g}"
    )
    assert r.tightness >= TIGHTNESS_FLOOR, (
        f"{name}: bound vacuous, tightness {r.tightness:.3g}"
    )
    assert r.tightness <= 1.0
    assert r.eps_max < 0.5  # the derivation's first-order regime


def test_bound_reference_matches_ref_attention():
    """The exact output that ``bound`` computes internally equals ``ref_attention``.

    Why: the bound and the gate must talk about the same "right answer"; the
    bound recomputes it in log2 units with split-wise sums, so this pins it to
    the reference (including S > 1 and WN > 1 regrouping).
    """
    c = Case("x", L=300, q_len=5, D=64, Hkv=2, G=3, tiling=Tiling(64, 16, 4, 3, 2))
    q, kv, qsl, scale = make_case(c)
    b = bound(q, kv, qsl, scale, tiling=c.tiling, out_dtype=torch.float32)
    ref = ref_attention_dense(q, kv, qsl, scale)
    np.testing.assert_allclose(b.ref, ref.out.numpy(), rtol=1e-12, atol=1e-15)


def test_bound_cannot_be_halved():
    """Half the bound is violated by the unmutated emulator (bf16 output case).

    Why: a gate that is far above the errors a correct kernel makes would let
    small bugs through. Where output rounding dominates (bf16 output) the
    bound is within 2x of an error the correct design actually makes, so it
    cannot be halved; ``test_bound_holds_on_emulator`` shows the full bound
    holds on the same case. (With fp32 output the bound is 2.6-17x above the
    worst observed error, mostly from the truncation-based score and P''
    terms.)
    """
    c = _SMALL["bf16-out"]
    q, kv, qsl, scale = make_case(c)
    emu = emulate(q, kv, qsl, scale, tiling=c.tiling, out_dtype=c.out_dtype)
    b = bound(q, kv, qsl, scale, tiling=c.tiling, out_dtype=c.out_dtype)
    err = np.abs(emu.out.double().numpy() - b.ref)
    assert (err <= b.bound)[b.valid].all()
    assert (err > b.bound / 2)[b.valid].any()


def test_gamma_tc_dominates_closed_form():
    """The tensor-core coefficient is <= the closed form D 2^-23 and grows linearly.

    Why: the closed form for the QK accumulation error is ``D 2^-23 sum|q k|``.
    The derivation gives ``n (2^-24 + 2^-23/4)(1 + n 2^-23)``
    for >= 4 products per fused step and 24 alignment fraction bits, which is
    at most ``0.75 n 2^-23 (1 + n 2^-23)``. ``gamma`` is the standard
    ``n u / (1 - n u)``.
    """
    for n in (16, 64, 256, 4096, 2**20):
        assert gamma_tc(n) <= n * 2.0**-23
        assert gamma_tc(n) >= 0.75 * n * 2.0**-23
    assert gamma(1) == pytest.approx(2.0**-24, rel=1e-6)


def test_token_plan_partitions_tiles():
    """Splits own whole, contiguous, balanced tile ranges; warps cover each tile.

    Why: the bound's step counts (n_acc, n_l, rescale events) come from this
    assignment, and it must match the planner's rule (split s owns
    tiles [floor(sT/S), floor((s+1)T/S))). Any two splits differ by at most one
    tile.
    """
    t = Tiling(n_tile=64, sub_tile=16, wn=4, num_splits=5, bq=1)
    tp = token_plan(64 * 13 + 7, t)
    counts = tp.n_tiles_split
    assert counts.sum() == 14 and counts.max() - counts.min() <= 1
    assert np.all(np.diff(tp.split) >= 0)  # contiguous, in order
    for tile in range(14):
        sl = slice(tile * 64, min(tile * 64 + 64, tp.range_end))
        assert len(set(tp.split[sl].tolist())) == 1
    assert set(tp.warp[:64].tolist()) == {0, 1, 2, 3}
    assert np.all(tp.warp[:16] == 0) and np.all(tp.warp[48:64] == 3)
    with pytest.raises(ValueError):
        Tiling(n_tile=64, sub_tile=24, wn=1)


# --------------------------------------------------------------------------
# Mutations: deliberate kernel bugs must fail the gate
# --------------------------------------------------------------------------
def _gate(q, kv_kernel, kv_true, qsl, scale, tiling, out_dtype=torch.bfloat16):
    """(fraction of valid elements violating, max err/bound) for a mutated kernel."""
    emu = emulate(q, kv_kernel, qsl, scale, tiling=tiling, out_dtype=out_dtype)
    b = bound(q, kv_true, qsl, scale, tiling=tiling, out_dtype=out_dtype)
    err = np.abs(emu.out.double().numpy() - b.ref)
    v = b.valid
    return float((err > b.bound)[v].mean()), tightness(err, b.bound, v)


_MUT_TILING = Tiling(n_tile=64, sub_tile=32, wn=2, num_splits=2, bq=4)


def _mut_case(L=64, q_len=4):
    c = Case("mut", L=L, q_len=q_len, D=128, Hkv=2, G=6, peaks=0, tiling=_MUT_TILING)
    return make_case(c)


def test_bound_catches_off_by_one_token():
    """A kernel that drops the diagonal key (``<`` instead of ``<=``) fails the gate.

    Mutation: the emulated kernel sees seq_len - 1 tokens, so with bottom-right
    alignment every query row loses exactly its own (newest) key. The gate is
    evaluated against the bound for the correct problem; even with bf16
    output (the loosest bound) and flat attention over 64 keys the error must
    exceed the bound on most elements. Detection power is weight-dependent: a
    missing token is caught when ``pbar_t |v_t - o| > bound`` (about
    ``2^-11 sum pbar |v|`` + output rounding), so flat attention over very long
    contexts needs the per-row check of a peaked case too (second assert).
    """
    q, kv, qsl, scale = _mut_case()
    x = kv[0]
    short = [DenseKV(x.k_int[:-1], x.k_scale[:-1], x.v_int[:-1], x.v_scale[:-1])]
    frac, worst = _gate(q, short, kv, qsl, scale, _MUT_TILING)
    assert frac > 0.5 and worst > 10.0, (frac, worst)
    # Control: the unmutated kernel passes the same gate.
    frac0, worst0 = _gate(q, kv, kv, qsl, scale, _MUT_TILING)
    assert frac0 == 0.0 and worst0 <= 1.0


def test_bound_catches_shifted_token():
    """A kernel that reads token t+1's K/V instead of token t's fails the gate."""
    q, kv, qsl, scale = _mut_case()
    x = kv[0]
    rolled = [
        DenseKV(
            *(torch.roll(a, -1, 0) for a in (x.k_int, x.k_scale, x.v_int, x.v_scale))
        )
    ]
    frac, worst = _gate(q, rolled, kv, qsl, scale, _MUT_TILING)
    assert frac > 0.5 and worst > 10.0, (frac, worst)


def test_bound_catches_wrong_scale():
    """A kernel using the other head's V scale or a neighbour's K scale fails.

    Why: scale streams are separate loads with their own strides;
    a wrong head or token index in the scale stream is a plausible bug that
    leaves the int data correct. Both variants must exceed the bound.
    """
    q, kv, qsl, scale = _mut_case()
    x = kv[0]
    vs_swapped = [DenseKV(x.k_int, x.k_scale, x.v_int, x.v_scale.flip(1))]
    frac, worst = _gate(q, vs_swapped, kv, qsl, scale, _MUT_TILING)
    assert frac > 0.5 and worst > 10.0, (frac, worst)
    ks_shift = [DenseKV(x.k_int, torch.roll(x.k_scale, 1, 0), x.v_int, x.v_scale)]
    frac, worst = _gate(q, ks_shift, kv, qsl, scale, _MUT_TILING)
    assert frac > 0.5 and worst > 10.0, (frac, worst)


def test_bound_catches_wrong_head():
    """A kernel that maps query head h to the wrong KV head fails the gate."""
    q, kv, qsl, scale = _mut_case()
    x = kv[0]
    swapped = [DenseKV(*(a.flip(1) for a in (x.k_int, x.k_scale, x.v_int, x.v_scale)))]
    frac, worst = _gate(q, swapped, kv, qsl, scale, _MUT_TILING)
    assert frac > 0.5 and worst > 10.0, (frac, worst)


# Sentinel K = 4 sqrt(D) q/|q|: its logit is ~4 sqrt(D) (45 at D=128), far above
# the N(0, 1) logits of the other keys, so it carries almost all the weight.
SENTINEL_GAIN = 4.0


def test_off_by_one_at_long_L_needs_a_sentinel():
    """At L = 4096 a dropped token barely shows unless it carries weight.

    Why (detection power): the gate catches a wrong token only
    where ``pbar_t |v_t - o|`` exceeds the bound, roughly ``2^-11 sum pbar |v|``.
    With flat attention over 4096 keys one token has weight ~1/4096, so a
    kernel that drops the newest key violates the gate on only a few elements
    and by a small factor (first assert, the measured limitation: about 7% of
    elements, at most 2.5x here; at 16k keys none). Planting a *sentinel* (the
    newest key aligned with the query, so it dominates the softmax) makes the
    same bug fail almost every element by orders of magnitude (second
    assert). Correctness grids therefore include sentinel tokens at the
    boundaries (first/last token, tile, page and split edges), not only random
    data.
    """
    t = Tiling(n_tile=64, sub_tile=16, wn=4, num_splits=1, bq=1)
    L, D, G = 4096, 128, 2
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, 2 * G, D, generator=g)
    k = torch.randn(L, 2, D, generator=g)
    v = torch.randn(L, 2, D, generator=g)
    qsl, scale = [0, 1], D**-0.5

    def gate_drop_last(kk):
        kv = [quantize_kv(kk.to(torch.bfloat16), v.to(torch.bfloat16))]
        x = kv[0]
        short = [DenseKV(x.k_int[:-1], x.k_scale[:-1], x.v_int[:-1], x.v_scale[:-1])]
        return _gate(q.bfloat16(), short, kv, qsl, scale, t, torch.float32)

    frac, worst = gate_drop_last(k)
    assert frac < 0.2 and worst < 10.0, (frac, worst)  # weak detection: the limit
    ks = k.clone()
    for h in range(2):
        ks[-1, h] = q[0, h * G] / q[0, h * G].norm() * SENTINEL_GAIN * D**0.5
    frac, worst = gate_drop_last(ks)
    assert frac > 0.5 and worst > 100.0, (frac, worst)


def test_bound_finite_when_a_whole_split_is_deep():
    """A split whose every token is far below the row max keeps a finite bound.

    Why: with S > 1 and a peaked row (one token ~200 log2 above the rest) most
    splits hold only tokens deeper than BOUND_DEEP_LOG2 below the row max.
    Inside such a split the kernel works relative to the split's own maximum,
    so nothing flushes there; only the combine weight 2^(lse_s - max lse) can
    flush. Earlier versions of the bound charged
    every such token a 100% weight error inside the split, so the split's denominator
    sum pbar (1 - eps) became 0 and the bound NaN on the whole row (seen on
    synthetic data at S = 34). The case must really
    contain such splits (first assert), the bound must be finite and it must
    hold on the emulated kernel and stay sharp.
    """
    c = _SMALL["deep-splits-S16"]
    q, kv, qsl, scale = make_case(c)
    x = kv[0]
    G, D = c.G, c.D
    qd = q.double().reshape(c.Hkv, G, D) * scale * LOG2E
    s2 = torch.einsum("hgd,lhd->hgl", qd, x.k_int.double() * x.k_scale[..., None])
    tp = token_plan(c.L, c.tiling)
    split = torch.as_tensor(tp.split)
    depth = s2.amax(-1, keepdim=True) - s2
    split_depth = torch.stack(
        [depth[..., split == s].amin(-1) for s in range(c.tiling.num_splits)]
    )
    assert (split_depth > DEEP_LOG2).any(), split_depth.min()
    b = bound(q, kv, qsl, scale, tiling=c.tiling, out_dtype=c.out_dtype)
    assert np.isfinite(b.bound).all()
    r = run_case(c)
    assert r.violations == 0 and TIGHTNESS_FLOOR <= r.tightness <= 1.0, r


# The longest case (decode, 128k) takes about 6 minutes alone and several times
# that on a loaded host, which is more than the suite-wide 600 s limit of
# pyproject.toml. The check itself is unchanged; only its time limit is raised.
@pytest.mark.timeout(3600)
@pytest.mark.full
@pytest.mark.parametrize(
    "case", default_matrix("full")[len(_SMALL) :], ids=lambda c: c.name
)
def test_bound_holds_on_emulator_long(case):
    """Long contexts (16k-128k, S up to 34): still 0 violations and a sharp bound."""
    r = run_case(case)
    assert r.violations == 0 and TIGHTNESS_FLOOR <= r.tightness <= 1.0, r
