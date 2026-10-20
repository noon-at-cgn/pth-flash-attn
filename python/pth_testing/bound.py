# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Data-dependent per-element error bound for the kernel.

``bound(...)`` returns, for every output element ``(token, head, d)``, a number
``B`` such that the kernel's output ``o_hat`` satisfies ``|o_hat - o| <= B``,
where ``o`` is fp64 attention over the exactly dequantised cache
(:func:`pth_testing.reference.ref_attention`). ``B`` is computed from the
actual data (scores, probabilities, V values, scales) and from the kernel's
tiling (load tile, compute sub-tile, warps per tile, splits, row tile), which
fix the accumulation lengths and the number of rescale events.

The bound holds for a kernel that follows the numeric contract: fp16 MMA with
fp32 accumulation, power-of-two Q prescale, per-sub-tile power-of-two V-scale
normalisation, one ``alpha`` applied to both O and l, log2-domain partials,
fixed-order combine.

This module shares no arithmetic with :mod:`pth_testing.emulate`, the
CPU model it is tested against, so a mistake cannot hide by appearing in both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from pth_testing.reference import DenseKV, visible_bounds

# --------------------------------------------------------------------------
# Unit roundoffs and hardware error constants
# --------------------------------------------------------------------------
U32 = 2.0**-24  # fp32 round-to-nearest unit roundoff
ULP32_REL = 2.0**-23  # one fp32 ulp relative to the value (upper bound)
U16 = 2.0**-11  # fp16 unit roundoff (normal range)
UBF16 = 2.0**-8  # bf16 unit roundoff
FP16_SUB_HALF_ULP = 2.0**-25  # max abs error of RN into the fp16 subnormal range
FP16_MIN_NORMAL = 2.0**-14
FP16_EMIN = -14
BF16_SUB_HALF_ULP = 2.0**-134
FP32_SUB_HALF_ULP = 2.0**-150
# ex2.approx.f32: "maximum ulp error is 2 ulp from correctly rounded result"
# (PTX ISA §9.7.3.21) -> <= 2.5 ulp from the exact value -> <= 5 * 2^-24.
EX2_REL = 5.0 * U32
# lg2.approx.f32: abs error <= 2^-22 for x in (0.5, 2), else relative 2^-22
# (PTX ISA §9.7.3.20).
LG2_ERR = 2.0**-22
# Normalisation o = O * rcp.approx(l): rcp is within 1 ulp of the correctly
# rounded value (<= 1.5 ulp = 3 * 2^-24 from exact) plus one multiply.
N_NORMALISE = 4
# Score s = fp32(fp32(acc * ks) * c_row), c_row = fp32(fp32(fp32(scale) *
# fp32(log2 e)) * 2^e): 2 multiplies + 3 roundings in c_row (2^e is exact).
N_SCORE_ROUNDINGS = 5
# Tensor-core fp16 x fp16 -> fp32 (Fasi et al. 2021; Khattak & Mikaitis 2025):
# products exact; a fused step adds >= NFMA_MIN products and the accumulator,
# aligned to the largest exponent keeping TC_FRAC_BITS fraction bits
# (truncated), then truncates the normalised sum to fp32.
TC_FRAC_BITS = 24
NFMA_MIN = 4
# If the largest exponent belongs to an fp16 subnormal factor, the alignment
# unit is not bounded by the operand's value; it is at most
# 2^(FP16_EMIN + INT8_EMAX - TC_FRAC_BITS) since the other factor is an int8.
INT8_EMAX = 7
TC_ABS_PER_TERM = 2.0 ** (FP16_EMIN + INT8_EMAX - TC_FRAC_BITS)
# Tokens whose exact score is more than DEEP_LOG2 below the maximum of their
# split (the row max when S = 1) can be flushed to zero by ex2.approx.ftz
# (results < 2^-126) somewhere on their in-split path (p, alpha, warp merge).
# They get a 100% weight error, which costs at most #tokens * 2^-96.
# The split combine is charged separately: its exp2f weight
# 2^(lse_s - max lse) may be flushed or subnormal once it is below 2^-126.
DEEP_LOG2 = 96.0
F32_MIN_NORMAL_EXP = -126
# The fp64 reference has its own (tiny) error: the same analysis with
# u = 2^-53 instead of 2^-24, i.e. 2^-29 of each term, plus PV/normalisation
# error <= (L + 4) 2^-53 (|o| + sum pbar |v|) <= 2^-32 (...) for L <= 2^20.
REF_REL_SLACK = 2.0**-20
REF_ABS_SLACK = 2.0**-32
# Q prescale: e = max(floor(log2 max|q|) - Q_TOP,
# Q_E_MIN), so max|q| 2^-e lies in [2^14, 2^15), the top binade of fp16 below
# 2^15, unless the clamp applies; the clamp keeps 2^-e and 2^e normal in fp32
# (rows with max|q| < 2^-98 then sit below the window, and every element that
# lands in fp16's subnormal range is charged 2^-25 as usual).
Q_TOP = 14
Q_E_MIN = -112
LOG2E = math.log2(math.e)
LN2 = math.log(2.0)


def gamma(n: float) -> float:
    """``n u / (1 - n u)``, ``u = 2^-24``: bound for n fp32 roundings (Higham §3.1)."""
    nu = n * U32
    return math.inf if nu >= 1.0 else nu / (1.0 - nu)


def gamma_tc(n: float) -> float:
    """Relative error of an n-term tensor-core dot product: ``|err| <= g sum|x|``.

    Each fused step with ``m >= NFMA_MIN`` products truncates at most ``m``
    operands by less than ``2^-24`` of the largest operand and truncates the
    sum by less than one fp32 ulp (``2^-23`` relative). With ``n / m`` steps,
    each bounded by the running ``sum|x|``:
    ``n (2^-24 + 2^-23 / m) (1 + n 2^-23)``.
    """
    per = 2.0**-TC_FRAC_BITS + ULP32_REL / NFMA_MIN
    return n * per * (1.0 + n * ULP32_REL)


# --------------------------------------------------------------------------
# Tiling (shared vocabulary with the emulator and the planner)
# --------------------------------------------------------------------------
MMA_K = 16  # m16n8k16: the k (token) dimension of one PV MMA


@dataclass(frozen=True)
class Tiling:
    """How one CTA walks its KV range.

    Attributes:
        n_tile: Load tile N in tokens (64).
        sub_tile: Compute sub-tile in tokens per warp step (32 at D=256 WM=4).
        wn: Warps that split each tile's tokens (``4 / WM``); each keeps its own
            online-softmax state, merged at the end of the CTA.
        num_splits: S, splits of the KV range.
        bq: Query tokens per CTA row tile (decode ``ceil(q / rt)``, prefill
            ``floor(64 / G)``); the CTA's KV range ends at its last row's limit.
    """

    n_tile: int = 64
    sub_tile: int = 32
    wn: int = 1
    num_splits: int = 1
    bq: int = 1

    def __post_init__(self) -> None:
        """Check the divisibility the kernel relies on."""
        if self.n_tile % self.wn:
            raise ValueError("n_tile must be a multiple of wn")
        per_warp = self.n_tile // self.wn
        if per_warp % self.sub_tile or self.sub_tile % MMA_K:
            raise ValueError("sub_tile must divide n_tile/wn and be a multiple of 16")
        if self.num_splits < 1 or self.bq < 1:
            raise ValueError("num_splits and bq must be >= 1")

    @property
    def subs_per_tile(self) -> int:
        """Sub-tiles each warp processes per load tile."""
        return self.n_tile // self.wn // self.sub_tile


@dataclass(frozen=True)
class TokenPlan:
    """Where every KV token of one CTA row tile goes.

    Arrays have length ``range_end`` (the CTA's KV range end).
    The CTA's tiles start at the tile holding ``range_begin`` (0 for causal
    attention; the window's first position otherwise); tokens of
    earlier tiles get ``split = -1`` (never loaded). ``split[t]`` is the split
    that owns token t (split s owns tiles ``[floor(sT/S), floor((s+1)T/S))`` of
    the CTA's T tiles); ``warp[t]`` is the warp that owns token t's slice of its
    tile; ``sub[t]`` a unique id of the (tile, warp, sub-tile) step.
    """

    range_end: int
    range_begin: int
    split: np.ndarray
    warp: np.ndarray
    sub: np.ndarray
    n_tiles_split: np.ndarray  # [S] tiles in each split


def token_plan(range_end: int, tiling: Tiling, range_begin: int = 0) -> TokenPlan:
    """Assign tokens ``[0, range_end)`` to (split, warp, sub-tile)."""
    N, S = tiling.n_tile, tiling.num_splits
    first = range_begin // N
    T = -(-range_end // N) - first
    tile_split = np.full(first + T, -1, dtype=np.int64)
    n_tiles_split = np.zeros(S, dtype=np.int64)
    for s in range(S):
        a, b = (s * T) // S, ((s + 1) * T) // S
        tile_split[first + a : first + b] = s
        n_tiles_split[s] = b - a
    t = np.arange(range_end, dtype=np.int64)
    tile, within = t // N, t % N
    per_warp = N // tiling.wn
    warp = within // per_warp
    sub_in = (within % per_warp) // tiling.sub_tile
    sub = (tile * tiling.wn + warp) * tiling.subs_per_tile + sub_in
    return TokenPlan(range_end, range_begin, tile_split[tile], warp, sub, n_tiles_split)


def pow2_ceil_exp(x: np.ndarray) -> np.ndarray:
    """``ceil(log2 x)`` for positive x, exactly (exponent bits via ``frexp``).

    ``frexp`` gives ``x = m 2^e`` with ``m`` in ``[0.5, 1)``; ``x`` is a power
    of two exactly when ``m == 0.5``, and then ``ceil(log2 x) = e - 1``.
    """
    m, e = np.frexp(x)
    return np.where(m == 0.5, e - 1, e).astype(np.int64)


def floor_log2(x: np.ndarray) -> np.ndarray:
    """``floor(log2 x)`` for positive x, exactly (``frexp`` exponent minus one)."""
    _, e = np.frexp(x)
    return (e - 1).astype(np.int64)


def out_rounding(dtype: torch.dtype) -> tuple[float, float]:
    """(relative, absolute) error of rounding an fp32 value into ``dtype``."""
    if dtype == torch.float32:
        return U32, FP32_SUB_HALF_ULP
    if dtype == torch.float16:
        return U16, FP16_SUB_HALF_ULP
    if dtype == torch.bfloat16:
        return UBF16, BF16_SUB_HALF_ULP
    raise ValueError(f"unsupported output dtype {dtype}")


TERM_NAMES = ("weights", "p_round", "pv_acc", "l_acc", "combine", "out_round")


@dataclass
class BoundResult:
    """Output of :func:`bound`.

    ``bound`` and ``ref`` are fp64 ``[T, Hq, D]`` (0 on invalid rows);
    ``valid`` is ``[T]``; ``terms`` holds the main named contributions
    (:data:`TERM_NAMES`, same shape) for the error report; ``eps_max``
    is the largest weight perturbation of a non-deep token (the derivation is
    first order and needs it well below 1).
    """

    bound: np.ndarray
    ref: np.ndarray
    valid: np.ndarray
    terms: dict[str, np.ndarray]
    eps_max: float


def bound(
    q: torch.Tensor,
    kv: Sequence[DenseKV],
    query_start_loc: torch.Tensor | Sequence[int],
    softmax_scale: float,
    *,
    tiling: Tiling,
    out_dtype: torch.dtype,
    window: int = 0,
    causal: bool = True,
) -> BoundResult:
    """Per-element bound on ``|kernel - fp64 reference|``.

    Args:
        q: ``[T, Hq, D]`` queries as given to the kernel (bf16/fp16/fp32).
        kv: One :class:`DenseKV` per request (stored ints and scales).
        query_start_loc: ``[R + 1]`` cumulative query counts.
        softmax_scale: The kernel's softmax scale.
        tiling: Load tile, sub-tile, warps per tile, splits and row tile. The
            accumulation lengths (``n_acc`` = token slots per warp and split,
            ``n_l`` = the same plus rescales and merges) and the number of
            ``alpha`` rescale events are derived from it per CTA.
        out_dtype: Output dtype (fp32 for the test build, bf16/fp16).
        window: Sliding window in tokens, 0 = none.
        causal: Bottom-right causal (True) or non-causal (False). The visible
            set of a row is :func:`pth_testing.reference.visible_bounds`; the
            bound's formulas depend only on which tokens are visible and how
            the CTA's range is split, so the same bound applies unchanged.

    Returns:
        :class:`BoundResult`.
    """
    qsl = [int(v) for v in query_start_loc]
    qn = q.detach().double().numpy()
    T, Hq, D = qn.shape
    res = BoundResult(
        np.zeros((T, Hq, D)),
        np.zeros((T, Hq, D)),
        np.zeros(T, dtype=bool),
        {k: np.zeros((T, Hq, D)) for k in TERM_NAMES},
        0.0,
    )
    for r, x in enumerate(kv):
        q0, q1 = qsl[r], qsl[r + 1]
        ql = q1 - q0
        if ql == 0:
            continue
        L = x.seq_len
        res.valid[q0:q1] = True
        Hkv = x.k_int.shape[1]
        G = Hq // Hkv
        for j0 in range(0, ql, tiling.bq):
            j1 = min(j0 + tiling.bq, ql)
            # The CTA's KV range: lo of its first token to hi of its last
            # token.
            lo, hi = visible_bounds(
                [L - ql + j for j in range(j0, j1)], L, window, causal
            )
            range_begin, range_end = lo[0], hi[-1] + 1
            tp = token_plan(range_end, tiling, range_begin)
            for h in range(Hkv):
                rows = (slice(q0 + j0, q0 + j1), slice(h * G, (h + 1) * G))
                blk = _Block(
                    qrows=qn[rows].reshape(-1, D),
                    lo=np.repeat(np.asarray(lo, dtype=np.int64), G),
                    hi=np.repeat(np.asarray(hi, dtype=np.int64), G),
                    kint=x.k_int[:range_end, h].double().numpy(),
                    ks=x.k_scale[:range_end, h].double().numpy(),
                    vint=x.v_int[:range_end, h].double().numpy(),
                    vs=x.v_scale[:range_end, h].double().numpy(),
                    tp=tp,
                    tiling=tiling,
                    score_scale=softmax_scale * LOG2E,
                    out_dtype=out_dtype,
                )
                b, o, terms, eps = _bound_block(blk)
                shape = (j1 - j0, G, D)
                res.bound[rows] = b.reshape(shape)
                res.ref[rows] = o.reshape(shape)
                for k in TERM_NAMES:
                    res.terms[k][rows] = terms[k].reshape(shape)
                res.eps_max = max(res.eps_max, eps)
    return res


@dataclass
class _Block:
    """Inputs for one (CTA row tile, KV head); rows are (token, head) pairs."""

    qrows: np.ndarray  # [R, D] queries as given (exact values)
    lo: np.ndarray  # [R] tokens in [lo, hi] are visible
    hi: np.ndarray  # [R]
    kint: np.ndarray  # [n, D] stored K ints (n = CTA range end)
    ks: np.ndarray  # [n] K scales
    vint: np.ndarray  # [n, D]
    vs: np.ndarray  # [n]
    tp: TokenPlan
    tiling: Tiling
    score_scale: float  # softmax_scale * log2(e)
    out_dtype: torch.dtype


def _score_error(b: _Block, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bound Delta[R, n] on the computed score error (log2 units), and e.

    Raw units are ``q~ . k_int`` with ``q~ = fp16(q 2^-e)``; the bound uses only
    ``q`` and the a-priori prescale error (not the emulator's q~).
    """
    D = b.qrows.shape[1]
    amax = np.abs(b.qrows).max(-1)
    nz = amax > 0
    e = np.where(
        nz, np.maximum(floor_log2(np.where(nz, amax, 1.0)) - Q_TOP, Q_E_MIN), 0
    )
    qs = np.ldexp(b.qrows, -e[:, None])  # q 2^-e, exact
    # Elements in fp16's subnormal range may lose half a subnormal ulp.
    und = np.where(np.abs(qs) < FP16_MIN_NORMAL, FP16_SUB_HALF_ULP, 0.0)
    absk = np.abs(b.kint)
    qabs = (np.abs(qs) + und) @ absk.T  # >= sum_d |q~_d k_d|
    under = und @ absk.T  # >= |sum_d (q~_d - q_d 2^-e) k_d|
    a_abs = np.abs(qs @ b.kint.T) + under  # >= |sum_d q~_d k_d|
    g_qk = gamma_tc(D)
    tc_abs = D * TC_ABS_PER_TERM
    acc_err = g_qk * qabs + tc_abs
    raw = acc_err + under + gamma(N_SCORE_ROUNDINGS) * (a_abs + acc_err)
    c0 = abs(b.score_scale) * np.exp2(e.astype(np.float64))  # raw -> log2 units
    delta = c0[:, None] * b.ks[None, :] * raw
    return np.where(allowed, delta, 0.0), e


def _bound_block(b: _Block) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """Bound, exact output, term breakdown and max eps for one block."""
    R, D = b.qrows.shape
    n = b.tp.range_end
    t = b.tiling
    S, wn = t.num_splits, t.wn
    pos = np.arange(n)[None, :]
    allowed = (pos >= b.lo[:, None]) & (pos <= b.hi[:, None])  # [R, n]

    # Exact scores (log2 units) and weights, fp64.
    vdeq = b.vint * b.vs[:, None]
    s2 = np.where(
        allowed, (b.qrows @ (b.kint * b.ks[:, None]).T) * b.score_scale, -np.inf
    )
    big_m = s2.max(-1)
    p = np.exp2(s2 - big_m[:, None])  # 0 where masked
    # Splits own contiguous token ranges (whole tiles), so each split works on
    # its own columns only: cost O(R n D) in total, independent of S.
    starts = np.searchsorted(b.tp.split, np.arange(S + 1))
    # Depth of each visible token below the exact maximum of its own split:
    # every ex2 argument inside a split is a difference of that split's scores
    # and running maxima; the combine is charged in xi.
    depth = np.zeros((R, n))
    for s in range(S):
        cs = slice(int(starts[s]), int(starts[s + 1]))
        s2_cs = s2[:, cs]  # -inf where not visible
        m_loc = s2_cs.max(-1, initial=-np.inf)
        with np.errstate(invalid="ignore"):  # -inf - -inf in rows without tokens
            depth[:, cs] = np.where(allowed[:, cs], m_loc[:, None] - s2_cs, 0.0)

    delta, _ = _score_error(b, allowed)
    dmax = delta.max(-1)

    # Weight perturbation eps_t. Each non-deep token's weight passes
    # through ex2 for p, every later alpha of its warp (<= the warp's step
    # count in the split) and the warp-merge factor (wn > 1). Every argument is
    # a difference of computed scores/maxima no larger than m_s - s_t + 2 dmax
    # (m_s = the token's split max), rounded once in fp32: three such
    # roundings per token.
    subs_split = b.tp.n_tiles_split * t.subs_per_tile  # steps per warp [S]
    n_ex = 1 + subs_split[b.tp.split] + (1 if wn > 1 else 0)  # [n]
    arg_mag = np.where(allowed, depth + 2.0 * dmax[:, None], 0.0)
    log_fac = delta * LN2 + n_ex[None, :] * math.log1p(EX2_REL)
    eps = np.expm1(log_fac + 3.0 * U32 * arg_mag * LN2)
    deep = depth > DEEP_LOG2
    eps = np.where(allowed, np.where(deep, np.maximum(eps, 1.0), eps), 0.0)
    eps_nondeep = float(np.where(allowed & ~deep, eps, 0.0).max()) if n else 0.0

    # Kernel contract: c per sub-tile over the sub-tile's tokens in range
    # (tokens below range_begin are zero-filled like those past range_end and
    # never reach a weight; their factor is 1 here).
    c_sub = np.full(int(b.tp.sub.max()) + 1 if n else 0, -(10**6), dtype=np.int64)
    inr = slice(b.tp.range_begin, n)
    np.maximum.at(c_sub, b.tp.sub[inr], pow2_ceil_exp(b.vs[inr]))
    in_range = np.arange(n) >= b.tp.range_begin
    pow_c = np.where(
        in_range, np.exp2(c_sub[b.tp.sub].astype(np.float64)), 1.0
    )  # 2^c per token

    absv = np.abs(vdeq)
    w_s = np.zeros((S, R))
    o_s = np.zeros((S, R, D))
    b_s = np.zeros((S, R, D))
    xi = np.zeros((S, R))
    parts = {k: np.zeros((S, R, D)) for k in ("weights", "p_round", "pv_acc", "l_acc")}
    rel_p = (1.0 + U32) * (1.0 + U16) - 1.0  # fp32 p*vs, then fp16
    for s in range(S):
        cs = slice(int(starts[s]), int(starts[s + 1]))
        in_s = allowed[:, cs]
        ps = np.where(in_s, p[:, cs], 0.0)
        ws = ps.sum(-1)
        live = ws > 0
        if not live.any():
            continue
        w_s[s] = ws
        pbar = ps / np.where(live, ws, 1.0)[:, None]
        eps_s, v_s = eps[:, cs], vdeq[cs]
        os_ = pbar @ v_s
        o_s[s] = os_
        # |o_w - o_s| <= sum pbar eps |v - o_s| / sum pbar (1 - eps).
        den = (pbar * (1.0 - eps_s)).sum(-1)
        den = np.where(live & (den > 0), den, np.nan)
        pe = pbar * eps_s
        dev = np.stack([pe[i] @ np.abs(v_s - os_[i]) for i in range(R)])
        b_w = dev / den[:, None]
        # sum over computed weights of |v|, normalised: <= sum pbar(1+eps)|v|/den
        sv = (pbar * (1.0 + eps_s)) @ absv[cs] / den[:, None]
        # fp16 subnormal P'' (absolute 2^-25 in units of the running
        # max; the split's own max is the largest such unit).
        n_w = float(subs_split[s] * t.sub_tile)  # token slots per warp: n_acc
        eps_l = gamma(n_w + 2 * subs_split[s] + 2 * wn)  # n_l
        s2_s = np.where(in_s, s2[:, cs], -np.inf)
        ms = np.where(live, s2_s.max(-1), 0.0)
        p_loc = np.exp2(s2_s - ms[:, None])
        pc = pow_c[cs]
        pdd = p_loc * (1.0 + eps_s) * b.vs[cs][None, :] / pc[None, :]
        a_abs = np.where(in_s, np.minimum(pdd, FP16_SUB_HALF_ULP) * (1.0 + eps_s), 0.0)
        norm = np.where(live, p_loc.sum(-1), 1.0) * den * (1.0 - eps_l)
        sub_abs = (a_abs * pc[None, :]) @ np.abs(b.vint[cs]) * np.exp2(dmax)[:, None]
        sub_abs = sub_abs / norm[:, None]
        # PV tensor-core accumulation over n_acc slots, O rescales
        # (one per step) and the warp merge (wn products, wn - 1 adds).
        rel_acc = (1.0 + gamma_tc(n_w)) * (1.0 + gamma(subs_split[s] + 2 * wn)) - 1.0
        e_num = ((1.0 + rel_p) * (1.0 + rel_acc) - 1.0) * sv + sub_abs
        # o = (sum w v + E_N) / (sum w (1 + lambda)), |lambda| <= eps_l.
        o_w_abs = np.abs(os_) + b_w
        l_part = o_w_abs * eps_l / (1.0 - eps_l)
        x = b_w + l_part + e_num / (1.0 - eps_l)
        b_s[s] = np.where(live[:, None], x + gamma(N_NORMALISE) * (np.abs(os_) + x), 0)
        for k, v in (
            ("weights", b_w),
            ("p_round", rel_p * sv + sub_abs),
            ("pv_acc", rel_acc * sv),
            ("l_acc", l_part),
        ):
            parts[k][s] = np.where(live[:, None], v, 0.0)
        # Split weight perturbation xi_s = error of 2^lse_s relative
        # to the exact split weight: token eps (weighted), l, lg2, the lse add
        # and the subtraction lse_s - max lse, and ex2 in the combine.
        eps_bar = (pbar * eps_s).sum(-1)
        lse_mag = np.abs(ms + np.log2(np.where(live, p_loc.sum(-1), 1.0)))
        lse_mag = lse_mag + 2.0 * dmax + 1.0
        gap = (big_m - ms) + np.log2(p.sum(-1)) + 2.0 * dmax + 1.0
        lg2 = LG2_ERR * max(1.0, math.log2(max(n_w * wn, 2.0)) + 1.0)
        # The combine weight 2^(lse_s - max lse) >= 2^-gap. Below 2^-126 exp2f
        # may return 0 or a subnormal: absolute error <= 2^-126, i.e. relative
        # <= 2^(gap - 126) (a whole split far below the row max).
        flush = np.where(
            live & (gap > -F32_MIN_NORMAL_EXP),
            np.exp2(gap + F32_MIN_NORMAL_EXP),
            0.0,
        )
        fac = np.exp2(lg2 + U32 * (lse_mag + gap)) * (1.0 + EX2_REL + flush)
        xi[s] = np.where(live, (1.0 + eps_bar) * (1.0 + eps_l) * fac - 1.0, 0.0)

    w_tot = w_s.sum(0)
    wbar = w_s / w_tot[None, :]
    o = np.einsum("sr,srd->rd", wbar, o_s)
    if S == 1:
        b_pre = b_s[0]
    else:
        # o' = sum w'_s o'_s / sum w'_s with w'_s = w_s (1 + eta_s),
        # |eta_s| <= xi_s, |o'_s - o_s| <= B_s; then the fp32 combine
        # arithmetic. w'_s >= 0 (exp2f), so a split with xi_s >= 1 (flushed
        # combine weight) adds nothing to the denominator's lower bound.
        den = (wbar * (1.0 - np.minimum(xi, 1.0))).sum(0)
        merge = np.einsum("sr,srd->rd", wbar * xi, np.abs(o_s - o[None]))
        merge = (merge + np.einsum("sr,srd->rd", wbar * (1.0 + xi), b_s)) / den[:, None]
        mag = np.einsum("sr,srd->rd", wbar * (1.0 + xi), np.abs(o_s) + b_s)
        mag = mag / den[:, None]
        n_live = (w_s > 0).sum(0)
        g_num = np.array([gamma(k) for k in n_live])[:, None]  # FMA chain
        g_den = np.array([gamma(k + N_NORMALISE) for k in n_live])[:, None]
        b_pre = merge + g_num * mag + g_den * (np.abs(o) + merge)
    pbar_all = p / p.sum(-1, keepdims=True)
    b_pre = b_pre * (1.0 + REF_REL_SLACK) + REF_ABS_SLACK * (
        np.abs(o) + pbar_all @ absv
    )
    u_out, abs_out = out_rounding(b.out_dtype)
    out_round = u_out * (np.abs(o) + b_pre) + abs_out
    terms = {k: np.einsum("sr,srd->rd", wbar, v) for k, v in parts.items()}
    terms["combine"] = b_pre - np.einsum("sr,srd->rd", wbar, b_s)
    terms["out_round"] = out_round
    return b_pre + out_round, o, terms, eps_nondeep


def tightness(
    err: np.ndarray, b: np.ndarray, valid_rows: np.ndarray | None = None
) -> float:
    """``max(err / bound)`` over elements (0/0 counts as 0; err > 0 = bound 0 is inf).

    ``valid_rows`` (``[T]`` bool) restricts the maximum to rows of requests.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b > 0, err / b, np.where(err > 0, np.inf, 0.0))
    if valid_rows is not None:
        ratio = ratio[valid_rows]
    return float(ratio.max()) if ratio.size else 0.0
