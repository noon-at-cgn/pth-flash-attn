# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Benchmark statistics for pth-flash-attn.

Pure Python + numpy, no GPU, no I/O except the small ``schedule`` CLI at the
bottom. Everything a benchmark verdict depends on is a pure function here so
it can be tested on the CPU with synthetic data whose truth is known.

The pieces, in the order a measurement uses them:

1. Design: ``mirrored_schedule`` / ``process_schedules`` give the ABBA-style
   interleaving of arms inside a process (rounds alternate A B C / C B A, which
   cancels drift that is linear in time) and reverse the arm order between
   fresh processes.
2. Noise rejection: ``block_rejection_reasons`` applies the rejection rules to
   one block of launches using the NVML samples that cover it: foreign GPU
   process, drift > 3 % of the watched clock(s) (SM and memory by default;
   memory only for arms measured to be insensitive to the SM clock),
   temperature > 83 C, thermal / hardware slowdown throttle, within-block
   IQR/median > 5 % (unless the IQR is within two timer steps), and "no
   NVML coverage" (fail closed). A software power-limit throttle alone (the
   220 W limit on the reference cards) is *not* a rejection: it is the
   operating condition being measured.
3. Pairing: ``paired_log_ratios`` turns blocks of two arms into one log-ratio
   of block medians per round. Rejecting either block drops the pair.
4. Inference: ``bootstrap_ci`` (percentile bootstrap, 10^4 resamples, seeded,
   stratified by process), ``minimum_detectable_effect`` from an A/A run.
5. Decisions: ``evaluate_gate`` passes only if the CI *lower* bound clears a
   threshold expressed against a measured hardware ``Ceiling``;
   ``evaluate_claim`` calls a difference only if the CI excludes
   1 +- max(2 %, k * MDE) (k = 1 for a claim, 2 for adopting an optimisation:
   an optimisation must beat twice the A/A MDE).
6. Reporting: ``summarize`` (median, IQR, n) and rejection counts.

No function here accepts a baseline kernel (Triton, FA2) as the reference of
a gate: ``Ceiling`` only admits hardware-ceiling kinds (a baseline is never
a ceiling). Comparison arms can still be compared with ``evaluate_claim``;
that produces a labelled comparison, never a pass/fail.
"""

from __future__ import annotations

import argparse
import bisect
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

# ---------------------------------------------------------------- constants
N_BOOTSTRAP = 10_000  # percentile-CI Monte-Carlo error ~0.2 % of width
CI_LEVEL = 0.95
DEFAULT_SEED = 0  # bootstrap RNG seed: a rerun of the analysis is bit-identical
MDE_POWER = 0.80  # conventional power for the minimum detectable effect
CLAIM_FLOOR = 0.02  # no claim below 2 % whatever the MDE
CLAIM_MDE_FACTOR = 1.0  # CI must exclude 1 +- max(2 %, MDE)
OPTIMISATION_MDE_FACTOR = 2.0  # an optimisation needs > 2x MDE
MAX_SM_CLOCK_DRIFT = 0.03  # applied to every watched clock (below)
# Which NVML clocks the drift rule watches (from the clock-noise analysis of
# the hardware-ceilings runs; see docs/benchmark-methodology.md). The default
# watches both: safe for any arm. A memory-bound arm whose time was measured
# not to depend on the SM clock (|elasticity| < MEMORY_DOMAIN_MAX_ELASTICITY)
# watches only the memory clock: at the 220 W power limit the governor moves
# the SM clock by up to 40 % under copy load with no measurable effect on copy
# time (median -0.02 % over 136 plateau changes, RTX 3080 20 GB, 2026-10), and
# the SM rule then rejected whole processes for nothing.
SM_CLOCK = "sm_clock_mhz"
MEM_CLOCK = "mem_clock_mhz"
CLOCK_DOMAINS: dict[str, tuple[str, ...]] = {
    "both": (SM_CLOCK, MEM_CLOCK),
    "sm": (SM_CLOCK,),
    "mem": (MEM_CLOCK,),
}
DEFAULT_CLOCK_DOMAIN = "both"
MEMORY_DOMAIN_MAX_ELASTICITY = 0.05  # 3 % clock move -> < 0.15 % time
# Timer resolution: an IQR of one timer step is not dispersion. Measured on
# the hardware-ceilings runs (RTX 3080 20 GB, 2026-10): CUDA-event graph
# timings are multiples of 1024 ns; CUPTI kernel timestamps step by 32 ns.
TIMER_QUANTUM_NS = {"cuda_event": 1024.0, "cupti": 32.0}
DISPERSION_MIN_QUANTA = 2  # IQR must exceed 2 timer steps to count
MAX_TEMP_C = 83.0
MAX_IQR_OVER_MEDIAN = 0.05  # kernel rule; recalibrated from the A/A runs
MIN_ACCEPTED_BLOCKS = 30  # >= 30 blocks per arm per process
SCHEDULED_BLOCKS = 36  # 30 / (1 - 1/6): survives a 1-in-6 rejection rate
N_PROCESSES = 3  # 3 fresh processes
MAX_RERUNS = 3  # at most 3 reruns, then "noisy"
NVML_INTERVAL_NS = 100_000_000  # nvml_log.py period (100 ms)
EXPECTED_COMPUTE_PROCS = 1  # the benchmark process itself
# Foreign compute processes. A block with a second
# compute process in its NVML window is always rejected; the run is invalidated
# only when the foreign process could have been a real competing workload or
# touched too many blocks. FOREIGN_MAX_MIB: a bare CUDA/NVML context of a
# tool is about 7 MiB (6.88 MiB observed on an RTX 3080 20 GB, 2026-10-12),
# the benchmark's own context 476-572 MiB; 64 MiB sits above every tool
# context seen and far below any workload that moves data.
# FOREIGN_MAX_BLOCK_FRACTION: half of the 1/6 rejection slack that
# SCHEDULED_BLOCKS keeps (36 scheduled, 30 needed), so foreign-process
# rejections alone can never push a process below MIN_ACCEPTED_BLOCKS.
FOREIGN_MAX_MIB = 64.0
FOREIGN_MAX_BLOCK_FRACTION = 1.0 / 12.0
FOREIGN_REASON = "foreign_process"  # real or unattributable: voids the run
FOREIGN_SMALL_REASON = "foreign_process:small"  # tiny context: block only

# NVML clocks-event (throttle) reason bits, from nvml.h / pynvml
# (nvmlClocksEventReason*). Values verified against nvidia-ml-py 13.615.71.
THROTTLE_GPU_IDLE = 0x1
THROTTLE_APP_CLOCKS = 0x2
THROTTLE_SW_POWER_CAP = 0x4
THROTTLE_HW_SLOWDOWN = 0x8
THROTTLE_SYNC_BOOST = 0x10
THROTTLE_SW_THERMAL = 0x20
THROTTLE_HW_THERMAL = 0x40
THROTTLE_HW_POWER_BRAKE = 0x80
THROTTLE_DISPLAY_CLOCKS = 0x100
THROTTLE_NAMES = {
    THROTTLE_GPU_IDLE: "gpu_idle",
    THROTTLE_APP_CLOCKS: "applications_clocks_setting",
    THROTTLE_SW_POWER_CAP: "sw_power_cap",
    THROTTLE_HW_SLOWDOWN: "hw_slowdown",
    THROTTLE_SYNC_BOOST: "sync_boost",
    THROTTLE_SW_THERMAL: "sw_thermal_slowdown",
    THROTTLE_HW_THERMAL: "hw_thermal_slowdown",
    THROTTLE_HW_POWER_BRAKE: "hw_power_brake_slowdown",
    THROTTLE_DISPLAY_CLOCKS: "display_clock_setting",
}
# The rejection rule is "thermal or HW-slowdown throttle". SW_POWER_CAP is the
# expected state at a 220 W power limit and must not reject (that would
# reject every block); GPU_IDLE shows up in samples taken between blocks.
REJECT_THROTTLE_MASK = (
    THROTTLE_HW_SLOWDOWN
    | THROTTLE_SW_THERMAL
    | THROTTLE_HW_THERMAL
    | THROTTLE_HW_POWER_BRAKE
)

# Kinds of reference a gate may be expressed against. Hardware ceilings only.
HARDWARE_CEILING_KINDS = frozenset({"sol_copy", "mma_ceiling", "graph_node_overhead"})

# One parsed line of bench/nvml_log.py output (results.schema.json
# $defs/nvml_sample). Values are JSON scalars or lists, hence Any.
Sample = Mapping[str, Any]


def throttle_names(mask: int) -> list[str]:
    """Names of the set bits of an NVML clocks-event mask, low bit first."""
    return [name for bit, name in sorted(THROTTLE_NAMES.items()) if mask & bit]


# ---------------------------------------------------------------- design
def mirrored_schedule(n_arms: int, rounds: int, reverse: bool = False) -> list[int]:
    """Arm order for one process: rounds alternate forward and mirrored.

    With arms A, B (and C ...), the order is A B | B A | A B | ... (ABBA when
    n_arms = 2). Any drift linear in time adds the same amount to every arm
    over each pair of rounds, so it cancels in the per-arm means; a fixed
    order (A A A ... B B B) would turn the same drift into a fake difference.
    ``reverse`` starts with the mirrored round; used for alternate processes
    so a residual first-position effect also cancels across processes.
    """
    if n_arms < 1 or rounds < 1:
        raise ValueError("n_arms and rounds must be >= 1")
    forward = list(range(n_arms))
    out: list[int] = []
    for r in range(rounds):
        mirrored = (r % 2 == 1) != reverse
        out.extend(forward[::-1] if mirrored else forward)
    return out


def process_schedules(
    n_arms: int, rounds: int = SCHEDULED_BLOCKS, n_processes: int = N_PROCESSES
) -> list[list[int]]:
    """One mirrored schedule per fresh process; odd processes start reversed."""
    return [
        mirrored_schedule(n_arms, rounds, reverse=(p % 2 == 1))
        for p in range(n_processes)
    ]


# ---------------------------------------------------------------- summaries
@dataclass(frozen=True)
class Summary:
    """Honest description of a sample: always report n and spread."""

    n: int
    median: float
    q1: float
    q3: float

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1

    @property
    def iqr_over_median(self) -> float:
        return self.iqr / self.median if self.median else math.inf

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "iqr_over_median": self.iqr_over_median,
        }


def summarize(values: Iterable[float]) -> Summary:
    """Median and quartiles (numpy 'linear' method, i.e. type 7)."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot summarize an empty sample")
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return Summary(n=int(x.size), median=float(med), q1=float(q1), q3=float(q3))


# ---------------------------------------------------------------- blocks
@dataclass(frozen=True)
class Block:
    """One timed block: ``replays`` launches of one arm, back to back.

    ``durations_ns`` are per-launch kernel times (CUPTI) or graph-replay
    times; the block statistic is their median, which ignores a single
    preempted launch without discarding the block.
    """

    process: int
    round: int
    seq: int
    arm: str
    t_start_ns: int
    t_end_ns: int
    durations_ns: tuple[float, ...]

    @property
    def key(self) -> tuple[int, int]:
        return (self.process, self.seq)

    @property
    def median_ns(self) -> float:
        return float(np.median(self.durations_ns))


def paired_log_ratios(
    blocks: Sequence[Block],
    numerator_arm: str,
    denominator_arm: str,
    rejected: Iterable[tuple[int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """log(median(numerator) / median(denominator)) per (process, round).

    Returns (log_ratios, strata) where strata is the process index of each
    pair, for a bootstrap stratified by process. A pair is dropped if either
    of its blocks is in ``rejected`` (keys are (process, seq)) so pairing,
    and with it the drift cancellation, is never broken.
    """
    bad = set(rejected)
    by_round: dict[tuple[int, int], dict[str, Block]] = {}
    for b in blocks:
        if b.arm in (numerator_arm, denominator_arm):
            slot = by_round.setdefault((b.process, b.round), {})
            if b.arm in slot:
                raise ValueError(
                    f"arm {b.arm} twice in round {b.round} of process {b.process}"
                )
            slot[b.arm] = b
    logs: list[float] = []
    strata: list[int] = []
    for (proc, _rnd), slot in sorted(by_round.items()):
        if numerator_arm not in slot or denominator_arm not in slot:
            continue
        num, den = slot[numerator_arm], slot[denominator_arm]
        if num.key in bad or den.key in bad:
            continue
        logs.append(math.log(num.median_ns / den.median_ns))
        strata.append(proc)
    return np.asarray(logs, dtype=np.float64), np.asarray(strata, dtype=np.int64)


# ---------------------------------------------------------------- bootstrap
@dataclass(frozen=True)
class CI:
    """A point estimate with a two-sided percentile-bootstrap interval."""

    estimate: float
    lo: float
    hi: float
    level: float
    n: int
    resamples: int
    seed: int
    method: str

    def exp(self) -> CI:
        """Map a CI on a log scale to the ratio scale (monotone, so exact)."""
        return CI(
            math.exp(self.estimate),
            math.exp(self.lo),
            math.exp(self.hi),
            self.level,
            self.n,
            self.resamples,
            self.seed,
            self.method + "+exp",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lo": self.lo,
            "hi": self.hi,
            "level": self.level,
            "n": self.n,
            "resamples": self.resamples,
            "seed": self.seed,
            "method": self.method,
        }


def _resample_index(
    rng: np.random.Generator, strata: np.ndarray, n_resamples: int
) -> np.ndarray:
    """(n_resamples, n) indices; each stratum is resampled within itself, so
    every bootstrap replicate keeps the number of pairs per process."""
    n = strata.size
    idx = np.empty((n_resamples, n), dtype=np.int64)
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        draws = rng.integers(0, members.size, size=(n_resamples, members.size))
        idx[:, members] = members[draws]
    return idx


def bootstrap_ci(
    x: Sequence[float] | np.ndarray,
    statistic: str = "mean",
    strata: Sequence[int] | np.ndarray | None = None,
    n_resamples: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = DEFAULT_SEED,
    cluster_robust: bool = False,
) -> CI:
    """Percentile bootstrap CI of the mean or median of ``x``.

    Stratified when ``strata`` is given (resampling within each process).
    Percentile, not BCa: it is the method a reader can check by hand, and the
    coverage simulation shows it within 1-2 points of nominal at
    our sample sizes for the mean of log-ratios *when processes do not differ*.

    The stratified bootstrap treats the processes as fixed: it says nothing
    about a fresh process. The hardware-ceilings runs measured per-process
    offsets (A/A mean log-ratio per process 0.36 / -0.04 / 0.55 %, one
    RTX 3080 20 GB, 2026-10), and on simulated runs calibrated on them the
    stratified CI covers the true ratio in only ~25 % of runs
    (docs/benchmark-methodology.md). ``cluster_robust=True`` widens the
    interval to at least estimate +- t_{P-1} * sd(process statistics)/sqrt(P)
    (P = number of processes), which restores coverage at the price of width
    when P is small (t_2 = 4.30). Off by default here (it changes every gate's CI);
    the gate and roofline functions below enable it.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("bootstrap needs at least 2 observations")
    if statistic not in ("mean", "median"):
        raise ValueError("statistic must be 'mean' or 'median'")
    st = np.zeros(arr.size, dtype=np.int64) if strata is None else np.asarray(strata)
    if st.shape != arr.shape:
        raise ValueError("strata must have the same length as x")
    rng = np.random.default_rng(seed)
    idx = _resample_index(rng, st, n_resamples)
    fn = np.mean if statistic == "mean" else np.median
    reps = fn(arr[idx], axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    est = float(fn(arr))
    method = f"percentile-bootstrap-{statistic}" + (
        "-stratified" if strata is not None else ""
    )
    if cluster_robust:
        groups = np.unique(st)
        if groups.size >= 2:
            per = np.array([fn(arr[st == g]) for g in groups])
            half = student_t_quantile(level, groups.size - 1) * float(
                np.std(per, ddof=1) / math.sqrt(groups.size)
            )
            lo, hi = min(lo, est - half), max(hi, est + half)
            method += "+cluster-t"
    return CI(
        estimate=est,
        lo=float(lo),
        hi=float(hi),
        level=level,
        n=int(arr.size),
        resamples=n_resamples,
        seed=seed,
        method=method,
    )


# Two-sided 95 % Student-t quantiles t_{0.975, df} for df = 1..30 (standard
# tables; checked against scipy.stats.t.ppf in tests/cpu/test_stats.py).
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def student_t_quantile(level: float, df: int) -> float:
    """Two-sided ``level`` quantile of Student's t (only level 0.95)."""
    if not math.isclose(level, 0.95):
        raise ValueError("only the 95 % level is tabulated")
    if df < 1:
        raise ValueError("df must be >= 1")
    if df <= len(_T975):
        return _T975[df - 1]
    return NormalDist().inv_cdf(0.975)  # within 2 % of t beyond df = 30


def ratio_to_rate_ceiling_ci(
    kernel_times_s: Sequence[float],
    work: float,
    ceiling_rates: Sequence[float],
    n_resamples: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = DEFAULT_SEED,
) -> CI:
    """eta = (work / median kernel time) / median ceiling rate, with a CI.

    Used when the ceiling is a rate measured in its own blocks (the MMA
    ceiling F in flop/s): eta_mma = executed flop / (F * T). The two samples
    are independent, so each is resampled on its own (two-sample bootstrap)
    and the CI carries the uncertainty of both the kernel and the ceiling.
    """
    t = np.asarray(kernel_times_s, dtype=np.float64)
    f = np.asarray(ceiling_rates, dtype=np.float64)
    if t.size < 2 or f.size < 2:
        raise ValueError("need at least 2 kernel blocks and 2 ceiling blocks")
    rng = np.random.default_rng(seed)
    t_rep = np.median(t[rng.integers(0, t.size, size=(n_resamples, t.size))], axis=1)
    f_rep = np.median(f[rng.integers(0, f.size, size=(n_resamples, f.size))], axis=1)
    reps = work / (t_rep * f_rep)
    alpha = 1.0 - level
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return CI(
        estimate=float(work / (np.median(t) * np.median(f))),
        lo=float(lo),
        hi=float(hi),
        level=level,
        n=int(t.size),
        resamples=n_resamples,
        seed=seed,
        method="two-sample-percentile-bootstrap-median",
    )


def eta_ci(
    kernel_s: Sequence[float] | np.ndarray,
    kernel_strata: Sequence[int] | np.ndarray,
    work: float,
    ceiling_rates: Sequence[float] | np.ndarray,
    ceiling_strata: Sequence[int] | np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = DEFAULT_SEED,
    cluster_robust: bool = True,
) -> CI:
    """eta = (work / median kernel time) / median ceiling rate, cluster-aware.

    The kernel (block medians, seconds) and the ceiling (per-block rates,
    work/s) are measured in different experiments, so they are independent
    samples: each is resampled on its own, within its processes (two-sample
    stratified percentile bootstrap of the medians). Both carry
    between-process offsets (copy-ceiling processes differ by up to ~1 %), so
    with ``cluster_robust`` the interval is widened to at least
    ``exp(log eta +- t_df sqrt(s_k^2 / P_k + s_c^2 / P_c))``, where s is the
    sd of the per-process log medians, P the process counts and
    ``df = min(P_k, P_c) - 1`` (conservative; Welch's df is never smaller).
    A side with one process contributes no between-process term (none can
    be estimated); the gate then rests on the bootstrap alone, and the
    record says so through ``method``.
    """
    t = np.asarray(kernel_s, dtype=np.float64)
    f = np.asarray(ceiling_rates, dtype=np.float64)
    ts, fs = np.asarray(kernel_strata), np.asarray(ceiling_strata)
    if t.size < 2 or f.size < 2:
        raise ValueError("need at least 2 kernel blocks and 2 ceiling blocks")
    if ts.shape != t.shape or fs.shape != f.shape:
        raise ValueError("strata must match their samples")
    rng = np.random.default_rng(seed)
    t_rep = np.median(t[_resample_index(rng, ts, n_resamples)], axis=1)
    f_rep = np.median(f[_resample_index(rng, fs, n_resamples)], axis=1)
    reps = work / (t_rep * f_rep)
    alpha = 1.0 - level
    lo, hi = (float(v) for v in np.quantile(reps, [alpha / 2, 1 - alpha / 2]))
    est = float(work / (np.median(t) * np.median(f)))
    method = "two-sample-stratified-percentile-bootstrap-median"
    if cluster_robust:
        var, groups = 0.0, []
        for x, s in ((t, ts), (f, fs)):
            g = np.unique(s)
            if g.size >= 2:
                per = np.log([np.median(x[s == k]) for k in g])
                var += float(np.var(per, ddof=1)) / g.size
                groups.append(int(g.size))
        if groups:
            half = student_t_quantile(level, min(groups) - 1) * math.sqrt(var)
            lo = min(lo, est * math.exp(-half))
            hi = max(hi, est * math.exp(half))
            method += "+cluster-t"
    return CI(
        estimate=est,
        lo=lo,
        hi=hi,
        level=level,
        n=int(t.size),
        resamples=n_resamples,
        seed=seed,
        method=method,
    )


@dataclass(frozen=True)
class RooflineEta:
    """``roofline_eta_ci`` result: the efficiency and the terms behind it."""

    ci: CI
    t_mem_s: float  # bytes / median copy ceiling (does not move with the SM clock)
    t_mma_s: float  # executed flop at the mma ceiling scaled to the kernel's clock
    t_ovh_s: float
    bound: str  # "mem" or "mma": which term of the max is larger at the estimate
    kernel_clock_mhz: float  # median over the kernel's own accepted blocks
    ceiling_clock_mhz: float  # median over the mma ceiling's own accepted blocks


def roofline_eta_ci(
    kernel_s: Sequence[float] | np.ndarray,
    kernel_clock_mhz: Sequence[float] | np.ndarray,
    kernel_strata: Sequence[int] | np.ndarray,
    bytes_read: float,
    flops: float,
    copy_rates: Sequence[float] | np.ndarray,
    copy_strata: Sequence[int] | np.ndarray,
    mma_rates: Sequence[float] | np.ndarray,
    mma_clock_mhz: Sequence[float] | np.ndarray,
    mma_strata: Sequence[int] | np.ndarray,
    t_ovh_s: float = 0.0,
    n_resamples: int = N_BOOTSTRAP,
    level: float = CI_LEVEL,
    seed: int = DEFAULT_SEED,
    cluster_robust: bool = True,
) -> RooflineEta:
    """Clock-adjusted roofline efficiency.

    ``eta_adj = (max(T_mem, T_mma(f_kernel)) + T_ovh) / T_measured`` with

    * ``T_mem = bytes / BW`` (BW: median copy-ceiling rate; DRAM streaming
      does not scale with the SM clock, so BW is used as measured);
    * ``T_mma(f) = flops / (F f / f_ceil)``: the tensor pipe issues a fixed
      number of flop per SM clock, so the mma ceiling ``F`` (median rate of
      its accepted blocks, measured at the median SM clock ``f_ceil`` of
      *those blocks*) is rescaled to ``f``, the median SM clock of the
      *kernel's own* accepted blocks (``kernel_clock_mhz``, one entry per
      kernel block, same order as ``kernel_s``). A kernel at 1200 MHz is thus
      judged against what the pipe can do at 1200 MHz, not at 1905.

    When the memory term is the larger one the result is the plain
    ``eta_mem`` (plus ``T_ovh / T``): the adjustment can only move a verdict
    for a row whose tensor time at its own clock exceeds its copy time.

    The interval is built like ``eta_ci`` (independent two-sample stratified
    bootstrap of the medians, kernel clock resampled with its block so a
    block's time and clock stay paired; the mma ceiling's rate and clock are
    resampled together), widened with the cluster-t term over processes:
    each of the three samples contributes the variance of the log of
    ``eta_adj`` recomputed with that sample's per-process medians in place of
    its pooled median (the same quantity ``eta_ci`` uses for its two).
    """
    t = np.asarray(kernel_s, dtype=np.float64)
    fk = np.asarray(kernel_clock_mhz, dtype=np.float64)
    cr = np.asarray(copy_rates, dtype=np.float64)
    mr = np.asarray(mma_rates, dtype=np.float64)
    fm = np.asarray(mma_clock_mhz, dtype=np.float64)
    ts, cs, ms = (np.asarray(s) for s in (kernel_strata, copy_strata, mma_strata))
    if min(t.size, cr.size, mr.size) < 2:
        raise ValueError("need at least 2 blocks of the kernel and of each ceiling")
    if fk.shape != t.shape or ts.shape != t.shape:
        raise ValueError("one SM clock and one stratum per kernel block")
    if fm.shape != mr.shape or ms.shape != mr.shape or cs.shape != cr.shape:
        raise ValueError("strata and clocks must match their samples")
    if not (np.all(fk > 0) and np.all(fm > 0)):
        raise ValueError("SM clocks must be positive (MHz)")

    def eta(t_m: Any, f_k: Any, bw: Any, f_mma: Any, f_c: Any) -> Any:
        t_mem = bytes_read / bw
        t_mma = flops * f_c / (f_mma * f_k)
        return (np.maximum(t_mem, t_mma) + t_ovh_s) / t_m

    med = (
        float(np.median(t)),
        float(np.median(fk)),
        float(np.median(cr)),
        float(np.median(mr)),
        float(np.median(fm)),
    )
    est = float(eta(*med))
    rng = np.random.default_rng(seed)
    ik = _resample_index(rng, ts, n_resamples)
    ic = _resample_index(rng, cs, n_resamples)
    im = _resample_index(rng, ms, n_resamples)
    reps = eta(
        np.median(t[ik], axis=1),
        np.median(fk[ik], axis=1),
        np.median(cr[ic], axis=1),
        np.median(mr[im], axis=1),
        np.median(fm[im], axis=1),
    )
    alpha = 1.0 - level
    lo, hi = (float(v) for v in np.quantile(reps, [alpha / 2, 1 - alpha / 2]))
    method = "two-sample-stratified-percentile-bootstrap-median-roofline"
    if cluster_robust:
        var, groups = 0.0, []
        for which, strata in ((0, ts), (2, cs), (3, ms)):
            g = np.unique(strata)
            if g.size < 2:
                continue
            per = []
            for k in g:
                m = list(med)
                if which == 0:
                    sel = ts == k
                    m[0], m[1] = float(np.median(t[sel])), float(np.median(fk[sel]))
                elif which == 2:
                    m[2] = float(np.median(cr[cs == k]))
                else:
                    sel = ms == k
                    m[3], m[4] = float(np.median(mr[sel])), float(np.median(fm[sel]))
                per.append(math.log(float(eta(*m))))
            var += float(np.var(per, ddof=1)) / g.size
            groups.append(int(g.size))
        if groups:
            half = student_t_quantile(level, min(groups) - 1) * math.sqrt(var)
            lo = min(lo, est * math.exp(-half))
            hi = max(hi, est * math.exp(half))
            method += "+cluster-t"
    t_mem = bytes_read / med[2]
    t_mma = flops * med[4] / (med[3] * med[1])
    return RooflineEta(
        ci=CI(est, lo, hi, level, int(t.size), n_resamples, seed, method),
        t_mem_s=t_mem,
        t_mma_s=t_mma,
        t_ovh_s=t_ovh_s,
        bound="mma" if t_mma > t_mem else "mem",
        kernel_clock_mhz=med[1],
        ceiling_clock_mhz=med[4],
    )


# ---------------------------------------------------------------- MDE
def standard_error(
    x: Sequence[float] | np.ndarray, strata: Sequence[int] | np.ndarray | None = None
) -> float:
    """Standard error of the mean of ``x``, cluster-aware.

    The naive s/sqrt(n) treats every pair as independent. If each process has
    its own offset (memory placement, clocks settling differently), pairs
    within a process are correlated and the naive SE is too small. With >= 2
    processes the cluster SE sd(process means)/sqrt(P) captures that; we take
    the larger of the two, so the estimate errs on the side of caution.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("need at least 2 observations")
    naive = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    if strata is None:
        return naive
    st = np.asarray(strata)
    groups = np.unique(st)
    if groups.size < 2:
        return naive
    means = np.array([arr[st == g].mean() for g in groups])
    cluster = float(np.std(means, ddof=1) / math.sqrt(groups.size))
    return max(naive, cluster)


def minimum_detectable_effect(
    aa_log_ratios: Sequence[float] | np.ndarray,
    strata: Sequence[int] | np.ndarray | None = None,
    level: float = CI_LEVEL,
    power: float = MDE_POWER,
) -> float:
    """Relative MDE from an A/A run (the same kernel against itself).

    MDE = exp((z_{1-alpha/2} + z_power) * SE) - 1, with SE from
    ``standard_error`` of the A/A log-ratios. The A/A run must use the same
    design (blocks x replays x processes) as the experiments it calibrates,
    so its SE is theirs. A true effect of exactly MDE is detected with
    probability ``power`` by a two-sided test at ``level``.
    """
    normal = NormalDist()
    z = normal.inv_cdf(1 - (1 - level) / 2) + normal.inv_cdf(power)
    return math.exp(z * standard_error(aa_log_ratios, strata)) - 1.0


# ---------------------------------------------------------------- ceilings
@dataclass(frozen=True)
class Ceiling:
    """A measured hardware ceiling a gate may reference.

    ``kind`` must be a hardware-ceiling kind (sol_copy, mma_ceiling,
    graph_node_overhead). A Triton or FA2 arm is a comparison row, never
    a ceiling; constructing one raises.
    """

    kind: str
    name: str
    unit: str
    value: CI
    source: str  # results file (and sha256) the value comes from

    def __post_init__(self) -> None:
        if self.kind not in HARDWARE_CEILING_KINDS:
            raise ValueError(
                f"ceiling kind {self.kind!r} is not a hardware ceiling "
                f"({sorted(HARDWARE_CEILING_KINDS)}); "
                "baselines are comparison rows only"
            )

    def as_ref(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "unit": self.unit,
            "source": self.source,
        }


# ---------------------------------------------------------------- decisions
@dataclass(frozen=True)
class Verdict:
    kind: str  # "gate" | "claim"
    # gate: pass/fail/inconclusive/noisy; claim: increase/decrease/no_claim/noisy
    result: str
    estimate: float
    lo: float
    hi: float
    threshold: float | None = None
    margin: float | None = None
    against: str = ""
    note: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "kind": self.kind,
            "result": self.result,
            "estimate": self.estimate,
            "lo": self.lo,
            "hi": self.hi,
            "against": self.against,
            "note": self.note,
        }
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.margin is not None:
            d["margin"] = self.margin
        d.update(self.extra)
        return d


def evaluate_gate(
    eta: CI,
    threshold: float,
    ceiling: Ceiling,
    n_accepted: int,
    min_accepted: int = MIN_ACCEPTED_BLOCKS,
) -> Verdict:
    """Gate on an efficiency eta = (ceiling time or rate) / (kernel), ratio scale.

    pass          CI lower bound >= threshold (the gate passes on the lower
                  bound, so noise can only make us fail, not pass)
    fail          CI upper bound < threshold (clearly below)
    inconclusive  the CI straddles the threshold: not a pass
    noisy         fewer accepted pairs than the design requires
    """
    if not 0.0 < threshold <= 1.5:
        raise ValueError("threshold is an efficiency against a hardware ceiling")
    if n_accepted < min_accepted:
        result = "noisy"
    elif eta.lo >= threshold:
        result = "pass"
    elif eta.hi < threshold:
        result = "fail"
    else:
        result = "inconclusive"
    return Verdict(
        kind="gate",
        result=result,
        estimate=eta.estimate,
        lo=eta.lo,
        hi=eta.hi,
        threshold=threshold,
        against=f"{ceiling.kind}:{ceiling.name}",
        extra={"n_accepted": n_accepted, "min_accepted": min_accepted},
    )


def evaluate_claim(
    log_ratio: CI,
    mde: float,
    floor: float = CLAIM_FLOOR,
    mde_factor: float = CLAIM_MDE_FACTOR,
    against: str = "",
    n_accepted: int | None = None,
    min_accepted: int = MIN_ACCEPTED_BLOCKS,
) -> Verdict:
    """Call a time ratio (arm / ``against``) different from 1, or not.

    The margin is m = max(floor, mde_factor * MDE). "increase" needs the
    whole ratio CI above 1 + m, "decrease" the whole CI below 1 - m;
    anything else is "no_claim". A CI that merely excludes 1 is not enough:
    effects inside the A/A noise floor or below 2 % are not reported as
    differences (an optimisation uses mde_factor = 2).
    """
    if mde < 0:
        raise ValueError("MDE must be >= 0")
    m = max(floor, mde_factor * mde)
    r = log_ratio.exp()
    if n_accepted is not None and n_accepted < min_accepted:
        result = "noisy"
    elif r.lo > 1.0 + m:
        result = "increase"
    elif r.hi < 1.0 - m:
        result = "decrease"
    else:
        result = "no_claim"
    return Verdict(
        kind="claim",
        result=result,
        estimate=r.estimate,
        lo=r.lo,
        hi=r.hi,
        margin=m,
        against=against,
        extra={"mde": mde, "mde_factor": mde_factor, "floor": floor},
    )


def evaluate_optimisation(log_ratio: CI, mde: float, against: str = "") -> Verdict:
    """An optimisation is adopted only beyond 2x the A/A MDE."""
    return evaluate_claim(
        log_ratio, mde, mde_factor=OPTIMISATION_MDE_FACTOR, against=against
    )


# ---------------------------------------------------------------- rejection
@dataclass(frozen=True)
class RejectionConfig:
    expected_procs: int = EXPECTED_COMPUTE_PROCS
    max_foreign_mib: float = FOREIGN_MAX_MIB
    max_clock_drift: float = MAX_SM_CLOCK_DRIFT
    max_temp_c: float = MAX_TEMP_C
    throttle_mask: int = REJECT_THROTTLE_MASK
    max_iqr_over_median: float = MAX_IQR_OVER_MEDIAN
    # NVML samples every 100 ms but a block can be a few ms long, so a block
    # is judged by the samples within one period either side of it.
    pad_ns: int = NVML_INTERVAL_NS
    pci_bus_id: str | None = None  # the GPU under test (NVML samples all)
    # NVML clock fields the drift rule watches; see CLOCK_DOMAINS.
    clock_fields: tuple[str, ...] = CLOCK_DOMAINS[DEFAULT_CLOCK_DOMAIN]
    # Step of the timer that produced the durations (0 = ignore).
    timer_quantum_ns: float = 0.0

    def __post_init__(self) -> None:
        if not self.clock_fields or not set(self.clock_fields) <= {SM_CLOCK, MEM_CLOCK}:
            raise ValueError(
                f"clock_fields must be a non-empty subset of "
                f"{(SM_CLOCK, MEM_CLOCK)}, got {self.clock_fields}"
            )


def rejection_config(
    *,
    clock_domain: str = DEFAULT_CLOCK_DOMAIN,
    timer: str | None = None,
    **kw: Any,
) -> RejectionConfig:
    """RejectionConfig from a clock-domain name and the timer kind."""
    if clock_domain not in CLOCK_DOMAINS:
        raise ValueError(f"clock_domain must be one of {sorted(CLOCK_DOMAINS)}")
    quantum = TIMER_QUANTUM_NS.get(timer, 0.0) if timer is not None else 0.0
    return RejectionConfig(
        clock_fields=CLOCK_DOMAINS[clock_domain], timer_quantum_ns=quantum, **kw
    )


def samples_in_window(
    samples: Iterable[Sample],
    t_start_ns: int,
    t_end_ns: int,
    pad_ns: int,
    pci_bus_id: str | None = None,
) -> list[Sample]:
    """NVML samples of the GPU under test with t in [start - pad, end + pad]."""
    lo, hi = t_start_ns - pad_ns, t_end_ns + pad_ns
    out = []
    for s in samples:
        if pci_bus_id is not None and _norm_bus(
            str(s.get("pci_bus_id", ""))
        ) != _norm_bus(pci_bus_id):
            continue
        t = int(s["t_mono_ns"])
        if lo <= t <= hi:
            out.append(s)
    return out


def _norm_bus(bus: str) -> str:
    """CUDA prints 0000:01:00.0, NVML 00000000:01:00.0: compare the tail."""
    return bus.lower().split(":", 1)[-1] if bus.count(":") == 2 else bus.lower()


def rule_coverage(samples: Sequence[Sample]) -> str | None:
    """No NVML sample near the block: we cannot vouch for it, so reject."""
    return None if samples else "no_nvml_coverage"


def foreign_processes(
    procs: Sequence[Mapping[str, Any]], expected_procs: int
) -> list[Mapping[str, Any]]:
    """The compute processes of one NVML sample that are not the benchmark.

    The benchmark is taken to be the ``expected_procs`` processes with the
    largest GPU memory (its CUDA context and buffers; foreign tool contexts
    are far smaller). A process whose memory NVML cannot attribute
    (``used_mib`` None) is never taken for the benchmark: it counts as
    foreign and unattributable, which fails closed.
    """
    known = sorted(
        (p for p in procs if p.get("used_mib") is not None),
        key=lambda p: -float(p["used_mib"]),
    )
    unknown = [p for p in procs if p.get("used_mib") is None]
    return known[expected_procs:] + unknown


def rule_foreign_process(
    samples: Sequence[Sample],
    expected_procs: int,
    max_foreign_mib: float = FOREIGN_MAX_MIB,
) -> str | None:
    """More compute processes on the GPU than the benchmark itself.

    ``foreign_process``: some foreign process held more than
    ``max_foreign_mib`` (or its memory is unknown): possibly a real competing
    workload. ``foreign_process:small``: only tiny contexts. Both reject the
    block; only the first voids a run (``foreign_verdict``).
    """
    small = False
    for s in samples:
        procs = s.get("compute_procs")
        if procs is None:
            return "nvml_missing:compute_procs"
        if len(procs) > expected_procs:
            extra = foreign_processes(procs, expected_procs)
            if any(
                p.get("used_mib") is None or float(p["used_mib"]) > max_foreign_mib
                for p in extra
            ):
                return FOREIGN_REASON
            small = True
    return FOREIGN_SMALL_REASON if small else None


def rule_temperature(samples: Sequence[Sample], max_c: float) -> str | None:
    for s in samples:
        t = s.get("temp_c")
        if t is None:
            return "nvml_missing:temp_c"
        if float(t) > max_c:
            return "temperature"
    return None


def rule_throttle(samples: Sequence[Sample], mask: int) -> str | None:
    for s in samples:
        r = s.get("throttle_reasons")
        if r is None:
            return "nvml_missing:throttle_reasons"
        if int(r) & mask:
            return "throttle:" + ",".join(throttle_names(int(r) & mask))
    return None


def block_clock(samples: Sequence[Sample], field: str = SM_CLOCK) -> float | None:
    """Median of one NVML clock field (MHz) over the samples covering a block."""
    vals = [float(s[field]) for s in samples if s.get(field) is not None]
    return float(np.median(vals)) if vals else None


def rule_clock_drift(
    samples: Sequence[Sample],
    reference_mhz: float,
    max_rel: float,
    field: str = SM_CLOCK,
) -> str | None:
    """The watched clock left its arm's reference by more than ``max_rel``.

    Reasons are named after the clock: ``sm_clock_drift`` or
    ``mem_clock_drift``.
    """
    clk = block_clock(samples, field)
    if clk is None:
        return f"nvml_missing:{field}"
    if abs(clk / reference_mhz - 1.0) > max_rel:
        return field.split("_", 1)[0] + "_clock_drift"
    return None


def rule_dispersion(
    durations_ns: Sequence[float],
    max_iqr_over_median: float,
    timer_quantum_ns: float = 0.0,
) -> str | None:
    """Launches inside one block disagree: something interfered mid-block.

    An IQR of at most DISPERSION_MIN_QUANTA timer steps is resolution, not
    disagreement: 2-us graph replays of the hardware-ceilings runs timed with
    CUDA events (1024-ns steps) had IQR = 1024 ns = 50 % of the median in 99
    of 108 blocks (RTX 3080 20 GB, 2026-10).
    """
    s = summarize(durations_ns)
    if s.iqr <= DISPERSION_MIN_QUANTA * timer_quantum_ns:
        return None
    if s.iqr_over_median > max_iqr_over_median:
        return "block_dispersion"
    return None


def block_rejection_reasons(
    block: Block,
    samples: Iterable[Sample],
    cfg: RejectionConfig,
    reference: Mapping[str, float],
) -> list[str]:
    """All reasons (possibly none) to reject one block. Pure function.

    ``reference`` maps each watched clock field to the arm's reference (MHz),
    as returned per arm by ``reference_clocks``.
    """
    near = samples_in_window(
        samples, block.t_start_ns, block.t_end_ns, cfg.pad_ns, cfg.pci_bus_id
    )
    reasons: list[str | None] = [rule_coverage(near)]
    if near:
        reasons += [
            rule_foreign_process(near, cfg.expected_procs, cfg.max_foreign_mib),
            rule_temperature(near, cfg.max_temp_c),
            rule_throttle(near, cfg.throttle_mask),
        ]
        reasons += [
            rule_clock_drift(near, reference.get(f, math.nan), cfg.max_clock_drift, f)
            for f in cfg.clock_fields
        ]
    reasons.append(
        rule_dispersion(
            block.durations_ns, cfg.max_iqr_over_median, cfg.timer_quantum_ns
        )
    )
    return [r for r in reasons if r is not None]


class SampleWindows:
    """NVML samples of the GPU under test, sorted, for O(log n) windows.

    Equivalent to ``samples_in_window`` (same bus filter, same closed
    interval) but without a scan of the whole log per block: the logs of a
    hardware-ceilings run hold ~23k samples per GPU and ~15k blocks.
    """

    def __init__(self, samples: Iterable[Sample], cfg: RejectionConfig) -> None:
        bus = None if cfg.pci_bus_id is None else _norm_bus(cfg.pci_bus_id)
        self._s = sorted(
            (
                s
                for s in samples
                if bus is None or _norm_bus(str(s.get("pci_bus_id", ""))) == bus
            ),
            key=lambda s: int(s["t_mono_ns"]),
        )
        self._t = [int(s["t_mono_ns"]) for s in self._s]
        self._pad = cfg.pad_ns

    def near(self, b: Block) -> list[Sample]:
        lo = bisect.bisect_left(self._t, b.t_start_ns - self._pad)
        hi = bisect.bisect_right(self._t, b.t_end_ns + self._pad)
        return self._s[lo:hi]


def reference_clocks(
    blocks: Sequence[Block],
    samples: Sequence[Sample],
    cfg: RejectionConfig,
) -> dict[str, dict[str, float]]:
    """Per-arm reference of each watched clock: median over the arm's blocks.

    Under a 220 W power limit the clock depends on the workload (tensor-core blocks
    draw more power and clock lower than copy blocks), so one reference for
    all arms would reject whole arms. The reference clock is therefore
    taken per arm; drift means a block's clock left its own arm's
    typical value.

    Resolution limit: NVML is sampled every 100 ms and a block lasts a few
    ms, so a block's window also contains samples taken during neighbouring
    blocks of other arms. Then every arm's reference describes the
    interleaved mix, and the rule detects drift of the whole run (heat
    soak, power-limit behaviour), not differences between arms. That is the intent.

    Limit (docs/benchmark-methodology.md): under copy load the SM clock sits
    on plateaus that the governor changes every 0.5 s, so its per-arm median
    can fall between plateaus and reject whole processes. That is why
    memory-bound arms watch the memory clock only (CLOCK_DOMAINS).
    """
    win = SampleWindows(samples, cfg)
    per_arm: dict[str, dict[str, list[float]]] = {}
    for b in blocks:
        near = win.near(b)
        for f in cfg.clock_fields:
            clk = block_clock(near, f)
            if clk is not None:
                per_arm.setdefault(b.arm, {}).setdefault(f, []).append(clk)
    return {
        arm: {f: float(np.median(v)) for f, v in fields.items()}
        for arm, fields in per_arm.items()
    }


def reject_blocks(
    blocks: Sequence[Block],
    samples: Sequence[Sample],
    cfg: RejectionConfig,
) -> dict[tuple[int, int], list[str]]:
    """Reasons for every rejected block, keyed by (process, seq)."""
    refs = reference_clocks(blocks, samples, cfg)
    win = SampleWindows(samples, cfg)
    out: dict[tuple[int, int], list[str]] = {}
    for b in blocks:
        # The window is pre-cut, so the bus filter inside is a no-op.
        reasons = block_rejection_reasons(b, win.near(b), cfg, refs.get(b.arm, {}))
        if reasons:
            out[b.key] = reasons
    return out


def choose_clock_domain(
    blocks: Sequence[Block],
    samples: Sequence[Sample],
    pci_bus_id: str | None,
    max_elasticity: float = MEMORY_DOMAIN_MAX_ELASTICITY,
) -> tuple[str, float | None]:
    """Clock domain for one experiment from its own data: ("mem" | "both", slope).

    Only the memory clock is watched for arms whose time was *measured* not to
    depend on the SM clock. The hardware-ceilings runs measured it for
    the copy ceilings; a kernel arm is measured here, on the same blocks the
    rule will judge: the slope of log(block median) on log(block SM clock),
    after removing each (process, arm) mean. If the SM clock never moved by
    more than the drift tolerance, the SM rule cannot fire and nothing is
    gained by dropping it ("both", slope None). Otherwise "mem" only if
    |slope| <= ``max_elasticity`` (then a 3 % clock move changes the time
    by < 0.15 %, below every A/A MDE of those runs).
    """
    win = SampleWindows(samples, RejectionConfig(pci_bus_id=pci_bus_id))
    xs, ys, keys = [], [], []
    for b in blocks:
        clk = block_clock(win.near(b), SM_CLOCK)
        if clk is not None and clk > 0 and b.median_ns > 0:
            xs.append(math.log(clk))
            ys.append(math.log(b.median_ns))
            keys.append((b.process, b.arm))
    if len(xs) < 3 or math.exp(max(xs) - min(xs)) < 1.0 + MAX_SM_CLOCK_DRIFT:
        return "both", None
    x, y = np.asarray(xs), np.asarray(ys)
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        x[m] -= x[m].mean()
        y[m] -= y[m].mean()
    var = float(np.dot(x, x))
    if var == 0.0:
        return "both", None
    slope = float(np.dot(x, y) / var)
    return ("mem" if abs(slope) <= max_elasticity else "both"), slope


def rejection_counts(
    rejected: Mapping[tuple[int, int], Sequence[str]],
) -> dict[str, int]:
    """How many blocks each reason hit (a block can have several reasons)."""
    c: Counter[str] = Counter()
    for reasons in rejected.values():
        c.update(r.split(":", 1)[0] for r in reasons)
    return dict(sorted(c.items()))


def foreign_summary(
    blocks: Sequence[Block],
    rejected: Mapping[tuple[int, int], Sequence[str]],
    max_fraction: float = FOREIGN_MAX_BLOCK_FRACTION,
) -> dict[str, Any]:
    """Foreign-process accounting of one arm's blocks.

    ``blocks``: the blocks of ONE arm over every process; ``rejected``: the
    reasons ``reject_blocks`` found. ``worst_fraction``: the largest share of
    blocks hit by any foreign-process reason within one process;
    ``large_blocks``: blocks hit by the voiding reason (a foreign process above
    the memory limit or of unknown memory). ``void``: the run cannot keep
    this arm: a large foreign process anywhere, or a hit share above
    ``max_fraction`` in some process.
    """
    hit: Counter[int] = Counter()
    total: Counter[int] = Counter()
    large = 0
    for b in blocks:
        total[b.process] += 1
        reasons = rejected.get(b.key, ())
        if any(r.split(":", 1)[0] == FOREIGN_REASON for r in reasons):
            hit[b.process] += 1
            large += FOREIGN_REASON in reasons
    worst = max((hit[p] / total[p] for p in total), default=0.0)
    return {
        "blocks": sum(hit.values()),
        "large_blocks": large,
        "worst_fraction": worst,
        "void": large > 0 or worst > max_fraction,
    }


def process_needs_rerun(
    blocks: Sequence[Block],
    rejected: Mapping[tuple[int, int], Sequence[str]],
    process: int,
    min_accepted: int = MIN_ACCEPTED_BLOCKS,
) -> bool:
    """True if any arm of ``process`` kept fewer than ``min_accepted`` blocks."""
    kept: Counter[str] = Counter()
    arms = set()
    for b in blocks:
        if b.process == process:
            arms.add(b.arm)
            if b.key not in rejected:
                kept[b.arm] += 1
    return any(kept[a] < min_accepted for a in arms)


def run_status(
    reruns_used: int, still_short: bool, max_reruns: int = MAX_RERUNS
) -> str:
    """'ok', 'rerun' (try a fresh process again) or 'noisy' (give up, report)."""
    if not still_short:
        return "ok"
    return "rerun" if reruns_used < max_reruns else "noisy"


# ---------------------------------------------------------------- CLI
def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
        + ". The only subcommand, `schedule`, prints the mirrored (ABBA-style) arm"
        " order of one process as comma-separated arm indices, the --schedule input"
        " of bench/sol_copy.cu and bench/mma_ceiling.cu. Pure computation: no GPU,"
        " no files."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schedule", help="print a mirrored schedule for one process")
    s.add_argument("--arms", type=int, required=True, help="number of arms")
    s.add_argument(
        "--rounds",
        type=int,
        default=SCHEDULED_BLOCKS,
        help="blocks per arm (default: the scheduled blocks, 36)",
    )
    s.add_argument(
        "--process", type=int, default=0, help="odd processes start reversed"
    )
    a = p.parse_args(argv)
    if a.cmd == "schedule":
        sched = mirrored_schedule(a.arms, a.rounds, reverse=(a.process % 2 == 1))
        print(",".join(str(i) for i in sched))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
