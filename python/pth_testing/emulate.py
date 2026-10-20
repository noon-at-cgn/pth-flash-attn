# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Bit-level CPU emulator of the kernel's arithmetic, in kernel order.

This emulator executes the kernel's numeric design (the
kernels in ``csrc/pth_attn/attn_kernel.cuh`` and ``attn_decode.cuh`` follow it)
operation by operation, with the same roundings, so that without a GPU we can
check two things about the error bound of :mod:`pth_testing.bound`:

(a) it is never violated, and
(b) it is not vacuous (``max(err / bound)`` is not tiny).

What it models, per (CTA row tile, KV head, split, warp):

* Q prescale: ``e = floor(log2 max|q|) - 14``, ``q~ = fp16_rn(q 2^-e)``.
* QK on tensor cores: products exact; fused steps of ``n_fma`` products plus
  the accumulator, aligned to the largest *exponent sum* keeping 24 fraction
  bits (the rest truncated), result truncated to fp32 (Fasi et al. 2021;
  Khattak & Mikaitis 2025: A100 and L40S fp16->fp32 has N_FMA = 8 and one extra
  alignment bit).
* Scores ``s = fp32(fp32(acc * ks) * c_row)`` with ``c_row = fp32(fp32(scale)
  * fp32(log2 e)) * 2^e``; online softmax in the exp2 domain with ``ex2.approx``
  (correctly rounded result moved by up to 2 ulp, flushed to zero below
  2^-126); ``l`` sums the unscaled fp32 ``p``.
* P'' = ``fp16_rn(fp32(p * vs) * 2^-c)`` with ``c = ceil(log2 max vs)`` over the
  sub-tile; O is kept in units of ``2^c`` and rescaled by ``alpha * 2^(c_old -
  c_new)`` (the power of two is exact); PV on the tensor-core model.
* The in-CTA merge of ``wn`` warps, the normalisation with an approximate
  reciprocal, log2 partials ``lse = m + lg2(l)`` and the split combine.

It is a *model* of hardware behaviour; the bound's derivation does not depend
on it, and the two modules share no arithmetic code except the tiling
vocabulary.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from pth_testing.bound import (
    FP16_EMIN,
    LOG2E,
    Q_E_MIN,
    Q_TOP,
    Tiling,
    bound,
    floor_log2,
    pow2_ceil_exp,
    tightness,
)
from pth_testing.reference import DenseKV, quantize_kv, visible_bounds

_NEG_BIG = -(10**6)  # stand-in exponent for exact zeros (never the maximum)
FP32_MANT_BITS = 23  # fp32 fraction bits (24 significant bits)
FTZ_EXP = -126  # ex2.approx.ftz flushes results below 2^-126
QK_CHUNK = 256  # tokens per QK chunk (memory: rows x 256 x D float64)


@dataclass(frozen=True)
class TensorCoreModel:
    """fp16 x fp16 -> fp32 MMA accumulation model.

    Attributes:
        n_fma: Products per fused step (8 on A100 and L40S, Khattak & Mikaitis).
        frac_bits: Fraction bits kept when aligning to the largest exponent
            (24 on Turing/Ampere/Ada; 23 on Volta).
        align: ``"rz"`` truncates aligned magnitudes toward zero
            (sign-magnitude), ``"rd"`` floors them (two's complement); the
            published tests do not distinguish the two, so both are run.
    """

    n_fma: int = 8
    frac_bits: int = 24
    align: str = "rz"


@dataclass(frozen=True)
class ApproxModel:
    """MUFU approximations inside their PTX error envelope.

    ``mode="random"`` adds a seeded random-sign error of exactly the documented
    worst case (ex2: 2 ulp, rcp: 1 ulp, lg2: 2^-22 absolute in (0.5, 2), else
    relative) to the correctly rounded result; ``"high"``/``"low"`` always add
    or subtract it; ``"exact"`` uses correctly rounded results.
    """

    mode: str = "random"
    ex2_ulps: float = 2.0
    rcp_ulps: float = 1.0
    lg2_err: float = 2.0**-22


def f32(x: np.ndarray | float) -> np.ndarray:
    """Round to fp32 (round to nearest even) and return as float64."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def f16(x: np.ndarray) -> np.ndarray:
    """Round to fp16 (round to nearest even, subnormals kept), as float64."""
    with np.errstate(over="ignore"):
        return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def _ulp32(x: np.ndarray) -> np.ndarray:
    ax = np.where(x != 0, np.abs(x), 1.0)
    return np.ldexp(1.0, floor_log2(ax) - FP32_MANT_BITS)


class _Mufu:
    """ex2 / lg2 / rcp with the :class:`ApproxModel` envelope."""

    def __init__(self, model: ApproxModel, seed: int) -> None:
        self.m = model
        self.rng = np.random.default_rng(seed)

    def _sign(self, shape: tuple[int, ...]) -> np.ndarray:
        if self.m.mode == "random":
            return self.rng.choice((-1.0, 1.0), size=shape)
        if self.m.mode == "high":
            return np.ones(shape)
        if self.m.mode == "low":
            return -np.ones(shape)
        return np.zeros(shape)

    def ex2(self, a: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            y = f32(np.exp2(a))
        y = f32(y + self._sign(y.shape) * self.m.ex2_ulps * _ulp32(y) * (y > 0))
        return np.where(y < 2.0**FTZ_EXP, 0.0, y)

    def rcp(self, x: np.ndarray) -> np.ndarray:
        y = f32(1.0 / x)
        return f32(y + self._sign(y.shape) * self.m.rcp_ulps * _ulp32(y))

    def lg2(self, x: np.ndarray) -> np.ndarray:
        y = np.log2(x)
        near = (x > 0.5) & (x < 2.0)
        err = self.m.lg2_err * np.where(near, 1.0, np.abs(y))
        return f32(y + self._sign(y.shape) * err)


def fp16_exponent(a: np.ndarray) -> np.ndarray:
    """Exponent the tensor core aligns with for an fp16 value.

    Normal values: ``floor(log2 |a|)``; subnormals: -14 (their significand is
    below one); zero: a large negative stand-in so it never sets the maximum.
    """
    aa = np.abs(a)
    normal = aa >= 2.0**FP16_EMIN
    e = np.where(normal, floor_log2(np.where(normal, aa, 1.0)), FP16_EMIN)
    return np.where(aa > 0, e, _NEG_BIG)


def tc_step(
    c: np.ndarray, x: np.ndarray, ex: np.ndarray, tc: TensorCoreModel
) -> np.ndarray:
    """One fused tensor-core step: ``c + sum(x)`` over the last axis of ``x``.

    ``x`` holds exact products and ``ex`` their exponent sums. Every operand is
    aligned to ``E = max(exponent(c), max ex)`` keeping ``frac_bits`` fraction
    bits (bits below ``2^(E - frac_bits)`` are dropped), the aligned operands
    are added exactly, and the sum is truncated to 24 significant bits.
    """
    ac = np.abs(c)
    ec = np.where(ac > 0, floor_log2(np.where(ac > 0, ac, 1.0)), _NEG_BIG)
    big_e = np.maximum(ec, ex.max(-1))
    live = big_e > _NEG_BIG // 2
    unit = np.ldexp(1.0, np.where(live, big_e, 0) - tc.frac_bits)
    rnd = np.trunc if tc.align == "rz" else np.floor
    xs = rnd(x / unit[..., None]) * unit[..., None]
    cs = rnd(c / unit) * unit
    # Exact in float64: each term is a multiple of unit below 2^(frac_bits+3)
    # units, and at most 17 terms are added.
    tot = cs + xs.sum(-1)
    at = np.abs(tot)
    ur = np.ldexp(1.0, floor_log2(np.where(at > 0, at, 1.0)) - FP32_MANT_BITS)
    res = np.trunc(tot / ur) * ur
    return np.where(live & (tot != 0), res, np.where(live, 0.0, c))


def tc_dot(
    c: np.ndarray, a: np.ndarray, b: np.ndarray, tc: TensorCoreModel
) -> np.ndarray:
    """Accumulate ``c + sum_k a[..., k] b[..., k]`` in ``n_fma``-wide steps.

    ``a`` and ``b`` broadcast against each other; both hold fp16-representable
    values (int8 values are exact in fp16). ``k`` runs in MMA order (16 per
    instruction, ``n_fma`` per fused step).
    """
    ea, eb = fp16_exponent(a), fp16_exponent(b)
    K = max(a.shape[-1], b.shape[-1])
    for k0 in range(0, K, tc.n_fma):
        k1 = min(k0 + tc.n_fma, K)
        x = a[..., k0:k1] * b[..., k0:k1]
        ex = np.where(x != 0, ea[..., k0:k1] + eb[..., k0:k1], _NEG_BIG)
        c = tc_step(c, x, np.broadcast_to(ex, x.shape), tc)
    return c


def prescale_q(qrows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kernel Q prescale: ``(q~ as fp16 values, e)`` per row."""
    amax = np.abs(qrows).max(-1)
    nz = amax > 0
    e = np.where(
        nz, np.maximum(floor_log2(np.where(nz, amax, 1.0)) - Q_TOP, Q_E_MIN), 0
    )
    return f16(np.ldexp(qrows, -e[:, None])), e


@dataclass
class EmuResult:
    """Emulator output: ``out`` in the output dtype ``[T, Hq, D]``; ``valid`` rows.

    Rows that belong to no request are left at 0 (the kernel does not write
    them).
    """

    out: torch.Tensor
    valid: np.ndarray


@dataclass
class _Ctx:
    """Per (CTA row tile, KV head) inputs of the emulation."""

    s: np.ndarray  # [R, n] fp32 scores (log2 units), -inf where masked
    cexp: np.ndarray  # [n] ceil(log2 vs)
    vs: np.ndarray  # [n]
    vint: np.ndarray  # [n, D]
    tiling: Tiling
    tc: TensorCoreModel
    mufu: _Mufu
    range_begin: int = 0  # the CTA's first KV position


def emulate(
    q: torch.Tensor,
    kv: Sequence[DenseKV],
    query_start_loc: torch.Tensor | Sequence[int],
    softmax_scale: float,
    *,
    tiling: Tiling,
    out_dtype: torch.dtype,
    tc: TensorCoreModel | None = None,
    approx: ApproxModel | None = None,
    seed: int = 0,
    window: int = 0,
    causal: bool = True,
) -> EmuResult:
    """Run the kernel's arithmetic on CPU (see the module docstring).

    Args:
        q: ``[T, Hq, D]`` queries (bf16/fp16/fp32 values used exactly).
        kv: One :class:`DenseKV` per request.
        query_start_loc: ``[R + 1]`` cumulative query counts.
        softmax_scale: Softmax scale (rounded to fp32 as the kernel does).
        tiling: Tile N, sub-tile, warps per tile, splits, row tile.
        out_dtype: Output dtype.
        tc: Tensor-core model (default :class:`TensorCoreModel`).
        approx: MUFU model (default :class:`ApproxModel`).
        seed: Seeds the MUFU error signs.
        window: Sliding window in tokens, 0 = none.
        causal: Bottom-right causal (True) or non-causal (False).
    """
    tc = tc or TensorCoreModel()
    mufu = _Mufu(approx or ApproxModel(), seed)
    qsl = [int(v) for v in query_start_loc]
    qn = q.detach().double().numpy()
    T, Hq, D = qn.shape
    out = np.zeros((T, Hq, D))
    valid = np.zeros(T, dtype=bool)
    c_base = f32(f32(softmax_scale) * f32(LOG2E))
    for r, x in enumerate(kv):
        q0, q1 = qsl[r], qsl[r + 1]
        ql = q1 - q0
        if ql == 0:
            continue
        valid[q0:q1] = True
        L = x.seq_len
        Hkv = x.k_int.shape[1]
        G = Hq // Hkv
        for j0 in range(0, ql, tiling.bq):
            j1 = min(j0 + tiling.bq, ql)
            # The CTA's KV range: lo of the first token
            # to hi of the last.
            lo, hi = visible_bounds(
                [L - ql + j for j in range(j0, j1)], L, window, causal
            )
            range_begin, range_end = lo[0], hi[-1] + 1
            for h in range(Hkv):
                heads = slice(h * G, (h + 1) * G)
                qrows = qn[q0 + j0 : q0 + j1, heads].reshape(-1, D)
                vis_lo = np.repeat(np.asarray(lo, dtype=np.int64), G)
                vis_hi = np.repeat(np.asarray(hi, dtype=np.int64), G)
                vs = x.v_scale[:range_end, h].double().numpy()
                ctx = _Ctx(
                    _scores(
                        qrows,
                        vis_lo,
                        vis_hi,
                        x.k_int[:range_end, h].double().numpy(),
                        x.k_scale[:range_end, h].double().numpy(),
                        c_base,
                        tc,
                    ),
                    pow2_ceil_exp(vs),
                    vs,
                    x.v_int[:range_end, h].double().numpy(),
                    tiling,
                    tc,
                    mufu,
                    range_begin,
                )
                o = _cta(ctx)
                out[q0 + j0 : q0 + j1, heads] = o.reshape(j1 - j0, G, D)
    res = torch.from_numpy(out.astype(np.float32)).to(out_dtype)
    return EmuResult(res, valid)


def _scores(
    qrows: np.ndarray,
    vis_lo: np.ndarray,
    vis_hi: np.ndarray,
    kint: np.ndarray,
    ks: np.ndarray,
    c_base: np.ndarray,
    tc: TensorCoreModel,
) -> np.ndarray:
    """fp32 scores in log2 units ``[R, n]``; -inf outside each row's [lo, hi]."""
    R, _ = qrows.shape
    n = kint.shape[0]
    qt, e = prescale_q(qrows)
    c_row = f32(np.ldexp(c_base, e))  # power-of-two scaling: exact
    s = np.full((R, n), -np.inf)
    for t0 in range(0, n, QK_CHUNK):
        t1 = min(t0 + QK_CHUNK, n)
        acc = tc_dot(np.zeros((R, t1 - t0)), qt[:, None, :], kint[None, t0:t1], tc)
        st = f32(f32(acc * ks[None, t0:t1]) * c_row[:, None])
        pos = np.arange(t0, t1)[None, :]
        allowed = (pos >= vis_lo[:, None]) & (pos <= vis_hi[:, None])
        s[:, t0:t1] = np.where(allowed, st, -np.inf)
    return s


@dataclass
class _State:
    """Online-softmax state of one warp: max m, sum l, O in units of 2^c."""

    m: np.ndarray
    l_sum: np.ndarray
    acc_o: np.ndarray
    c: int | None


def _warp(ctx: _Ctx, split_tiles: range, w: int) -> _State:
    """One warp's pass over its sub-tiles of the split's tiles."""
    R, n = ctx.s.shape
    D = ctx.vint.shape[1]
    t = ctx.tiling
    per_warp = t.n_tile // t.wn
    st = _State(np.full(R, -np.inf), np.zeros(R), np.zeros((R, D)), None)
    for tile in split_tiles:
        for k in range(per_warp // t.sub_tile):
            lo = tile * t.n_tile + w * per_warp + k * t.sub_tile
            idx = np.arange(lo, lo + t.sub_tile)
            inr = (idx >= ctx.range_begin) & (idx < n)  # zero-filled outside
            idv = idx[inr]
            s_sub = np.full((R, t.sub_tile), -np.inf)
            s_sub[:, inr] = ctx.s[:, idv]
            m_new = np.maximum(st.m, s_sub.max(-1))
            # Guard: an all-masked row so far uses offset 0,
            # and alpha = 0 when there was nothing before.
            m_use = np.where(np.isfinite(m_new), m_new, 0.0)
            alpha = np.where(np.isfinite(st.m), ctx.mufu.ex2(f32(st.m - m_use)), 0.0)
            with np.errstate(invalid="ignore"):
                p = ctx.mufu.ex2(f32(s_sub - m_use[:, None]))
            rowsum = p.astype(np.float32).sum(-1, dtype=np.float32).astype(np.float64)
            st.l_sum = f32(f32(st.l_sum * alpha) + rowsum)
            # V-scale exponent of this sub-tile (tokens inside the CTA range).
            if idv.size:
                c_new = int(ctx.cexp[idv].max())
            else:
                c_new = st.c if st.c is not None else 0
            # O in units of 2^c: alpha_O = alpha 2^(c_old - c_new), exact pow2.
            alpha_o = np.zeros(R) if st.c is None else np.ldexp(alpha, st.c - c_new)
            st.acc_o = f32(st.acc_o * alpha_o[:, None])
            vs_sub = np.zeros(t.sub_tile)
            vs_sub[inr] = ctx.vs[idv]
            pdd = f16(np.ldexp(f32(p * vs_sub[None, :]), -c_new))
            v_sub = np.zeros((t.sub_tile, D))
            v_sub[inr] = ctx.vint[idv]
            # PV: the MMA's k dimension is the token: O[r, d] += P''[r, j] v[j, d]
            st.acc_o = tc_dot(st.acc_o, pdd[:, None, :], v_sub.T[None, :, :], ctx.tc)
            st.m, st.c = m_new, c_new
    return st


def _merge_warps(ctx: _Ctx, states: list[_State]) -> tuple[_State, int]:
    """In-CTA merge of the ``wn`` warp states (decode merge).

    Returns the merged state with O in units of ``2^C`` and ``C``.
    """
    if len(states) == 1:
        st = states[0]
        return st, (st.c if st.c is not None else 0)
    R, D = states[0].acc_o.shape
    big_m = np.stack([s.m for s in states]).max(0)
    m_use = np.where(np.isfinite(big_m), big_m, 0.0)
    cs = [s.c for s in states if s.c is not None]
    big_c = max(cs) if cs else 0
    acc_o = np.zeros((R, D))
    l_sum = np.zeros(R)
    for s in states:
        f = np.where(np.isfinite(s.m), ctx.mufu.ex2(f32(s.m - m_use)), 0.0)
        f_o = np.ldexp(f, (s.c if s.c is not None else big_c) - big_c)  # exact
        acc_o = f32(acc_o + f32(f_o[:, None] * s.acc_o))
        l_sum = f32(l_sum + f32(f * s.l_sum))
    return _State(big_m, l_sum, acc_o, big_c), big_c


def _cta(ctx: _Ctx) -> np.ndarray:
    """All splits of one (row tile, KV head), then the combine; fp32 output values."""
    t = ctx.tiling
    R, n = ctx.s.shape
    n_tiles = -(-n // t.n_tile)
    first = ctx.range_begin // t.n_tile  # the CTA's tiles start here
    T = n_tiles - first
    S = t.num_splits
    parts_o, parts_lse = [], []
    for sp in range(S):
        tiles = range(first + (sp * T) // S, first + ((sp + 1) * T) // S)
        st, big_c = _merge_warps(ctx, [_warp(ctx, tiles, w) for w in range(t.wn)])
        live = st.l_sum > 0
        rcp = ctx.mufu.rcp(np.where(live, st.l_sum, 1.0))
        o = f32(
            np.where(live[:, None], np.ldexp(f32(st.acc_o * rcp[:, None]), big_c), 0.0)
        )
        lg = ctx.mufu.lg2(np.where(live, st.l_sum, 1.0))
        parts_o.append(o)
        parts_lse.append(np.where(live, f32(st.m + lg), -np.inf))
    if S == 1:
        return parts_o[0]
    # Split combine: M = max lse; o = sum 2^(lse_s - M) o_s / sum ...,
    # empty splits (lse = -inf) skipped, fixed order.
    lses = np.stack(parts_lse)
    big_m = lses.max(0)
    m_use = np.where(np.isfinite(big_m), big_m, 0.0)
    num = np.zeros_like(parts_o[0])
    den = np.zeros(R)
    for sp in range(S):
        fin = np.isfinite(lses[sp])
        with np.errstate(invalid="ignore"):
            w = np.where(fin, ctx.mufu.ex2(f32(lses[sp] - m_use)), 0.0)
        num = f32(w[:, None] * parts_o[sp] + num)  # FMA: one rounding
        den = f32(den + w)
    live = den > 0
    rcp = ctx.mufu.rcp(np.where(live, den, 1.0))
    return np.where(live[:, None], f32(num * rcp[:, None]), 0.0)


# --------------------------------------------------------------------------
# Emulator matrix (python -m pth_testing.emulate)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """One emulator-vs-bound case (see :func:`make_case`)."""

    name: str
    L: int
    q_len: int
    D: int = 256
    Hkv: int = 2
    G: int = 6
    v_scale: float = 1e-2  # typical stored V scale (absmax / 127)
    logit_std: float = 1.0
    peaks: int = 4
    q_spread_log2: float = 0.0  # Q element magnitudes spread over 2^-spread..1
    tiling: Tiling = Tiling()
    q_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.float32
    seed: int = 0
    window: int = 0  # sliding window, 0 = none
    causal: bool = True


PEAK_GAIN = 3.0  # peak tokens get K = 3 q / logit_std: logits ~3 sqrt(D) scale
V_SPREAD = 10.0  # per-token V absmax log-uniform within a factor of 10


def make_case(c: Case) -> tuple[torch.Tensor, list[DenseKV], list[int], float]:
    """Queries, quantised K/V, ``query_start_loc`` and scale for a :class:`Case`.

    K rows are N(0, 1); Q rows N(0, logit_std^2) so logits have standard
    deviation about ``logit_std`` at scale ``D^-1/2``; ``peaks`` tokens get K
    aligned with the last query of each KV head's first query head so the
    softmax is peaked. V rows have absmax near
    ``127 * v_scale`` (log-uniform within a factor 10), so the stored scales
    are ``v_scale``-sized; ``v_scale = 1e-6`` hits the writer's floor.
    ``q_spread_log2 > 0`` multiplies each Q element by ``2^-U(0, spread)`` so
    small elements underflow in fp16 after the prescale.
    """
    g = torch.Generator().manual_seed(c.seed)
    scale = c.D**-0.5
    Hq = c.Hkv * c.G
    k = torch.randn(c.L, c.Hkv, c.D, generator=g)
    q = torch.randn(c.q_len, Hq, c.D, generator=g) * c.logit_std
    if c.q_spread_log2:
        q = q * torch.exp2(-torch.rand(q.shape, generator=g) * c.q_spread_log2)
    if c.peaks:
        idx = torch.randperm(c.L, generator=g)[: c.peaks]
        for h in range(c.Hkv):
            k[idx, h] = q[-1, h * c.G] / c.logit_std * PEAK_GAIN
    lo = -math.log(V_SPREAD)
    spread = torch.exp(torch.empty(c.L, c.Hkv, 1).uniform_(lo, 0.0, generator=g))
    v = torch.rand(c.L, c.Hkv, c.D, generator=g) * 2.0 - 1.0
    v = v / v.abs().amax(-1, keepdim=True) * (127.0 * c.v_scale) * spread
    kv = [quantize_kv(k.to(torch.bfloat16), v.to(torch.bfloat16))]
    return q.to(c.q_dtype), kv, [0, c.q_len], scale


@dataclass
class CaseResult:
    """Result row of :func:`run_case`."""

    case: Case
    violations: int
    tightness: float
    rel_err: float  # max |o_hat - o| / max |o|
    bound_rel: float  # max bound / max |o|
    eps_max: float
    seconds: float


def run_case(
    c: Case,
    tc: TensorCoreModel | None = None,
    approx: ApproxModel | None = None,
) -> CaseResult:
    """Emulate a case, compute the bound, and compare element by element."""
    t0 = time.time()
    q, kv, qsl, scale = make_case(c)
    emu = emulate(
        q,
        kv,
        qsl,
        scale,
        tiling=c.tiling,
        out_dtype=c.out_dtype,
        tc=tc,
        approx=approx,
        seed=c.seed,
        window=c.window,
        causal=c.causal,
    )
    b = bound(
        q,
        kv,
        qsl,
        scale,
        tiling=c.tiling,
        out_dtype=c.out_dtype,
        window=c.window,
        causal=c.causal,
    )
    err = np.abs(emu.out.double().numpy() - b.ref)
    viol = int((err > b.bound)[b.valid].sum())
    omax = float(np.abs(b.ref).max()) or 1.0
    return CaseResult(
        c,
        viol,
        tightness(err, b.bound, b.valid),
        float(err[b.valid].max()) / omax,
        float(b.bound[b.valid].max()) / omax,
        b.eps_max,
        time.time() - t0,
    )


def default_matrix(tier: str = "small") -> list[Case]:
    """The emulator matrix (``small`` for CI, ``full`` adds long L).

    Tilings: decode WM=1 (WN=4, SUB=16), verify WM=2
    (WN=2, SUB=32), prefill WM=4 (WN=1, SUB=32 at D=256, 64 at D<=128),
    N = 64 throughout; ``bq`` is the row-tile query count.
    """
    n = 64
    dec = Tiling(n_tile=n, sub_tile=16, wn=4, num_splits=1, bq=1)
    ver = Tiling(n_tile=n, sub_tile=32, wn=2, num_splits=1, bq=4)
    pre = Tiling(n_tile=n, sub_tile=32, wn=1, num_splits=1, bq=10)
    small_d = Tiling(n_tile=n, sub_tile=64, wn=1, num_splits=2, bq=2)
    dec_s16 = Tiling(n_tile=n, sub_tile=16, wn=4, num_splits=16, bq=1)
    cases = [
        Case("decode-wn4", L=2048, q_len=1, tiling=dec),
        Case("decode-wn4-S3", L=2048, q_len=1, tiling=Tiling(n, 16, 4, 3, 1)),
        Case("verify-q4-wn2-S2", L=1024, q_len=4, tiling=Tiling(n, 32, 2, 2, 4)),
        Case("prefill-wm4", L=640, q_len=40, tiling=pre),
        Case("vs-floor-1e-6", L=1024, q_len=1, v_scale=1e-6, tiling=dec),
        Case("vs-1e-5", L=1024, q_len=1, v_scale=1e-5, tiling=dec),
        Case(
            "vs-1e-1-std0.3", L=1024, q_len=1, v_scale=1e-1, logit_std=0.3, tiling=dec
        ),
        Case("logit-std3", L=1024, q_len=1, logit_std=3.0, tiling=dec),
        Case("q-tiny", L=512, q_len=1, logit_std=1e-3, tiling=dec),
        Case("q-underflow", L=512, q_len=2, q_spread_log2=40.0, tiling=ver),
        Case("D128-gqa4", L=1024, q_len=2, D=128, G=4, tiling=small_d),
        Case("D64-gqa1", L=1024, q_len=3, D=64, G=1, tiling=Tiling(n, 64, 1, 1, 3)),
        Case("fp16-q", L=1024, q_len=1, q_dtype=torch.float16, tiling=dec),
        Case("bf16-out", L=1024, q_len=4, out_dtype=torch.bfloat16, tiling=ver),
        Case("fp16-out", L=1024, q_len=4, out_dtype=torch.float16, tiling=ver),
        Case("short-L15", L=15, q_len=1, tiling=dec),
        # Peaks ~200 log2 above the rest: most of the 16 splits lie entirely
        # deeper than DEEP_LOG2 below the row max (the combine flush term).
        Case("deep-splits-S16", L=2048, q_len=1, logit_std=3.0, tiling=dec_s16),
        # The DFlash2 drafter's TP=2 layer (D=128, G=4, q=8, window
        # 2048, non-causal; WM=2 -> WN=2, SUB=32) and a causal windowed
        # verify with splits whose range starts mid-tile.
        Case(
            "drafter-w2048-noncausal",
            L=2056,
            q_len=8,
            D=128,
            G=4,
            window=2048,
            causal=False,
            tiling=Tiling(n, 32, 2, 3, 8),
        ),
        Case(
            "window-w100-causal-S2",
            L=1000,
            q_len=4,
            D=128,
            G=4,
            window=100,
            tiling=Tiling(n, 16, 4, 2, 4),
        ),
    ]
    if tier == "full":
        cases += [
            Case("decode-16k", L=16384, q_len=1, tiling=dec),
            Case("decode-16k-S34", L=16384, q_len=1, tiling=Tiling(n, 16, 4, 34, 1)),
            Case("verify-q16-32k-S4", L=32768, q_len=16, tiling=Tiling(n, 32, 1, 4, 8)),
            Case("decode-128k-vs1e-5", L=131072, q_len=1, v_scale=1e-5, tiling=dec),
        ]
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    """Print the tightness table for the emulator matrix; exit 1 on a violation."""
    ap = argparse.ArgumentParser(description="emulator vs bound tightness matrix")
    ap.add_argument("--tier", choices=("small", "full"), default="small")
    ap.add_argument("--align", choices=("rz", "rd"), default="rz")
    ap.add_argument("--approx", choices=("random", "high", "low", "exact"))
    a = ap.parse_args(argv)
    tc = TensorCoreModel(align=a.align)
    mufu = ApproxModel(mode=a.approx or "random")
    print(
        "| case | L | q | D | G | N/sub/wn/S/bq | out | violations | tightness"
        " | max err / max abs(o) | max bound / max abs(o) | s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    worst = 0
    for c in default_matrix(a.tier):
        r = run_case(c, tc, mufu)
        t = c.tiling
        til = f"{t.n_tile}/{t.sub_tile}/{t.wn}/{t.num_splits}/{t.bq}"
        print(
            f"| {c.name} | {c.L} | {c.q_len} | {c.D} | {c.G} | {til} | "
            f"{str(c.out_dtype).removeprefix('torch.')} | {r.violations} | "
            f"{r.tightness:.3e} | {r.rel_err:.2e} | {r.bound_rel:.2e} | "
            f"{r.seconds:.1f} |",
            flush=True,
        )
        worst += r.violations
    return 1 if worst else 0


if __name__ == "__main__":
    sys.exit(main())
