# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for bench/stats.py.

Every benchmark verdict of the project (decode eta_mem >= 0.90, prefill
eta_mma >= 0.70, "optimisation X is faster") is computed by these functions,
so each test states the property it protects. Synthetic data with a known
truth stands in for GPU measurements; seeds are fixed, so the tests are
deterministic, and the tolerances are justified from binomial error bars.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[2] / "bench"
sys.path.insert(0, str(BENCH))
import stats  # noqa: E402

CLEAN = 0x4  # SW power-limit throttle only: the normal state at a 220 W limit


# ---------------------------------------------------------------- helpers
def nvml_sample(
    t_ns,
    clock=1800.0,
    temp=70.0,
    throttle=CLEAN,
    procs=1,
    bus="00000000:01:00.0",
    mem=9251.0,
):
    return {
        "kind": "sample",
        "t_mono_ns": t_ns,
        "pci_bus_id": bus,
        "sm_clock_mhz": clock,
        "mem_clock_mhz": mem,
        "temp_c": temp,
        "throttle_reasons": throttle,
        "compute_procs": [{"pid": 100 + i, "used_mib": 500.0} for i in range(procs)],
    }


def make_blocks(
    true_ns,
    rounds,
    processes,
    sd_block,
    seed,
    replays=20,
    sd_within=0.002,
    gap_ns=10_000_000,
):
    """Blocks of several arms in the mirrored schedule, ``gap_ns`` apart.

    Each block's launches share a block-level log-normal factor (sd_block:
    the between-block noise that the paired design has to beat) plus a small
    within-block jitter.
    """
    rng = np.random.default_rng(seed)
    arms = list(true_ns)
    blocks = []
    t = 1_000_000_000
    for p in range(processes):
        for seq, a in enumerate(
            stats.mirrored_schedule(len(arms), rounds, reverse=p % 2 == 1)
        ):
            f = math.exp(rng.normal(0.0, sd_block))
            d = true_ns[arms[a]] * f * np.exp(rng.normal(0.0, sd_within, replays))
            blocks.append(
                stats.Block(
                    p, seq // len(arms), seq, arms[a], t, t + 4_000_000, tuple(d)
                )
            )
            t += gap_ns
    return blocks


def clean_samples(blocks):
    return [nvml_sample(b.t_start_ns + 2_000_000) for b in blocks]


# ---------------------------------------------------------------- design
def test_mirrored_schedule_rounds_are_mirrored_permutations():
    """Every round contains each arm once and consecutive rounds are mirror
    images (ABBA for two arms). The C++ programs rely on 'round = position //
    n_arms'; a schedule violating it would pair blocks from different times."""
    s = stats.mirrored_schedule(3, 4)
    assert s == [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0]
    assert stats.mirrored_schedule(2, 2) == [0, 1, 1, 0]
    assert stats.mirrored_schedule(2, 2, reverse=True) == [1, 0, 0, 1]


def test_process_schedules_reverse_odd_processes():
    """Arm order is reversed across fresh processes, so a
    first-position effect inside a process also cancels across processes."""
    p = stats.process_schedules(2, rounds=2, n_processes=3)
    assert p == [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]]


def test_mirrored_design_cancels_linear_drift():
    """Two identical arms under a 3 % linear clock drift: the mirrored design
    reports ratio 1, a sequential design (all A, then all B) reports a fake
    ~1.5 % difference. This is why blocks are interleaved.

    Negative control: the sequential design must show the bias, otherwise the
    test would not be able to see drift at all."""
    rounds = 36
    sched = stats.mirrored_schedule(2, rounds)
    n = len(sched)
    times = [100.0 * (1 + 0.03 * k / (n - 1)) for k in range(n)]
    a = [t for t, s in zip(times, sched, strict=True) if s == 0]
    b = [t for t, s in zip(times, sched, strict=True) if s == 1]
    mirrored = np.mean(np.log(np.array(b) / np.array(a)))
    seq_a, seq_b = np.array(times[:rounds]), np.array(times[rounds:])
    sequential = np.mean(np.log(seq_b / seq_a))
    assert abs(mirrored) < 1e-4
    assert sequential > 0.012  # half the drift shows up as a fake effect


# ---------------------------------------------------------------- summaries
def test_summarize_reports_n_median_iqr():
    """Honest reporting: every number travels with its n and spread."""
    s = stats.summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert (s.n, s.median, s.q1, s.q3) == (5, 3.0, 2.0, 4.0)
    assert s.iqr == 2.0 and s.iqr_over_median == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError):
        stats.summarize([])


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_ci_covers_true_ratio_at_nominal_rate():
    """The 95 % CI of the mean log-ratio must contain the true ratio about
    95 % of the time at the design size (3 processes x 30 pairs, stratified).
    If coverage were much lower, gates would pass on luck.

    300 seeds: binomial SE of a 0.95 rate is 0.013; a 1000-seed
    simulation measured 0.944 for this design. The band [0.92, 0.975] is
    about 2 SE either side of that."""
    true = math.log(1.05)
    st = np.repeat(np.arange(3), 30)
    hits = 0
    seeds = 300
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        x = true + rng.normal(0.0, 0.03, 90)
        ci = stats.bootstrap_ci(x, strata=st, seed=seed)
        hits += ci.lo <= true <= ci.hi
    assert 0.92 <= hits / seeds <= 0.975


def test_bootstrap_is_seeded_and_reproducible():
    """Re-running the analysis on the same data must give the same numbers;
    a different seed changes only the Monte-Carlo noise in the bounds."""
    x = np.random.default_rng(0).normal(0.0, 0.05, 40)
    a = stats.bootstrap_ci(x, seed=7)
    b = stats.bootstrap_ci(x, seed=7)
    c = stats.bootstrap_ci(x, seed=8)
    assert a == b
    assert a.estimate == c.estimate and (a.lo, a.hi) != (c.lo, c.hi)
    assert abs(a.lo - c.lo) < 0.003  # 10^4 resamples: bounds stable to <0.3 %


def test_stratified_ci_undercovers_with_process_offsets_cluster_t_fixes_it():
    """The hardware-ceilings runs measured per-process A/A offsets (mean
    log-ratio per process 0.36 / -0.04 / 0.55 %, one RTX 3080 20 GB,
    2026-10). With offsets of that size (sd 0.35 % per
    process and arm) and 3 processes, the stratified bootstrap CI treats the
    3 processes as the whole world and covers the true ratio far below 95 %;
    the cluster-t widening restores coverage.

    200 seeds: binomial SE ~ 0.03 at 0.5 and ~0.015 at 0.95."""
    true = math.log(1.05)
    st = np.repeat(np.arange(3), 30)
    plain = robust = 0
    seeds = 200
    for seed in range(seeds):
        rng = np.random.default_rng(5000 + seed)
        offsets = rng.normal(0.0, 0.0035 * math.sqrt(2), 3)  # two arms' offsets
        x = true + np.repeat(offsets, 30) + rng.normal(0.0, 0.0025, 90)
        a = stats.bootstrap_ci(x, strata=st, n_resamples=2000, seed=seed)
        b = stats.bootstrap_ci(
            x, strata=st, n_resamples=2000, seed=seed, cluster_robust=True
        )
        plain += a.lo <= true <= a.hi
        robust += b.lo <= true <= b.hi
        assert b.lo <= a.lo and b.hi >= a.hi  # only ever widens
    assert plain / seeds < 0.6
    assert robust / seeds >= 0.92


def test_cluster_t_is_a_no_op_without_strata():
    """One stratum carries no between-process information: the cluster-t
    option must not change the interval."""
    x = np.random.default_rng(1).normal(0.0, 0.01, 40)
    assert stats.bootstrap_ci(x, cluster_robust=True) == stats.bootstrap_ci(x)


def test_student_t_table_matches_scipy():
    """The tabulated t quantiles are the textbook values (checked against
    scipy where it is installed; the dev image has it)."""
    sp = pytest.importorskip("scipy.stats")
    for df in range(1, 31):
        assert stats.student_t_quantile(0.95, df) == pytest.approx(
            sp.t.ppf(0.975, df), abs=6e-4
        )
    assert stats.student_t_quantile(0.95, 1000) == pytest.approx(1.959964, abs=1e-6)
    with pytest.raises(ValueError):
        stats.student_t_quantile(0.9, 3)


def test_stratified_resampling_keeps_pairs_per_process():
    """A stratified replicate draws each process's pairs only from that
    process, so a replicate never consists mostly of one (possibly odd)
    process."""
    st = np.array([0] * 5 + [1] * 3 + [2] * 7)
    idx = stats._resample_index(np.random.default_rng(0), st, 200)
    for s in (0, 1, 2):
        cols = np.flatnonzero(st == s)
        assert np.all(np.isin(idx[:, cols], cols))


def test_ratio_to_rate_ceiling_ci_covers_truth():
    """eta_mma = flop / (F * T) with uncertainty from both the kernel and
    the ceiling F (two-sample bootstrap). Known truth: eta = 0.7."""
    rng = np.random.default_rng(3)
    F = 50e12
    work = 1e12
    t_true = work / (0.7 * F)
    t = t_true * np.exp(rng.normal(0, 0.01, 90))
    f = F * np.exp(rng.normal(0, 0.01, 90))
    ci = stats.ratio_to_rate_ceiling_ci(t, work, f)
    assert ci.lo < 0.7 < ci.hi
    assert ci.hi - ci.lo < 0.01  # 1 % noise, 90 blocks: a tight interval


# ---------------------------------------------------------------- A/A and MDE
def test_aa_ci_contains_one_and_makes_no_claim():
    """The same kernel against itself (A/A) must not produce a claim. Over 100
    seeds the ratio CI contains 1 in >= 90 % of runs (nominal 95 %, binomial
    SE 2.2 %), and evaluate_claim says 'no_claim' every time, because the
    2 % floor sits far outside A/A noise of 1 % per block."""
    contains, claims = 0, 0
    for seed in range(100):
        blocks = make_blocks({"a": 1e5, "b": 1e5}, 30, 3, 0.01, seed)
        lr, st = stats.paired_log_ratios(blocks, "b", "a")
        ci = stats.bootstrap_ci(lr, strata=st, seed=seed)
        contains += ci.exp().lo <= 1.0 <= ci.exp().hi
        mde = stats.minimum_detectable_effect(lr, st)
        claims += stats.evaluate_claim(ci, mde).result != "no_claim"
    assert contains >= 90
    assert claims == 0


def _claim_rate(sd, effect, seeds, use_mde=True):
    hits, mdes = 0, []
    st = np.repeat(np.arange(3), 30)
    for seed in range(seeds):
        rng = np.random.default_rng(20_000 + seed)
        aa = rng.normal(0.0, sd, 90)
        mde = stats.minimum_detectable_effect(aa, st) if use_mde else 0.0
        mdes.append(mde)
        x = math.log(1 + effect) + rng.normal(0.0, sd, 90)
        ci = stats.bootstrap_ci(x, strata=st, n_resamples=2000, seed=seed)
        floor = stats.CLAIM_FLOOR if use_mde else 0.0
        hits += stats.evaluate_claim(ci, mde, floor=floor).result == "increase"
    return hits / seeds, float(np.median(mdes))


def test_five_percent_slowdown_detected_when_mde_allows():
    """Low noise (1 % per pair): MDE ~0.3 %, so the margin is the 2 % floor
    and a true 5 % slowdown is claimed as an 'increase' in every run."""
    rate, mde = _claim_rate(sd=0.01, effect=0.05, seeds=50)
    assert mde < 0.02
    assert rate == 1.0


def test_five_percent_slowdown_not_claimed_when_mde_too_large():
    """High noise (20 % per pair): MDE ~6 % > 5 %, so a true 5 % slowdown is
    (correctly) not claimed: the measurement cannot resolve it. Allowed
    rate <= 5 % (expected ~1.4 % from the normal approximation).

    Negative control: the same data judged without the MDE margin (only
    'CI excludes 1') claims the effect in a large fraction of runs: it is
    the MDE rule, not luck, that keeps the claim honest."""
    rate, mde = _claim_rate(sd=0.20, effect=0.05, seeds=200)
    assert mde > 0.05
    assert rate <= 0.05
    naive, _ = _claim_rate(sd=0.20, effect=0.05, seeds=200, use_mde=False)
    assert naive >= 0.4


def test_mde_is_cluster_aware():
    """If each process has its own offset, pairs inside a process are not
    independent; the MDE must grow with the spread of process means, not
    only shrink with the number of pairs (without this, 85 % of
    A/A experiments with 1.5 % process offsets 'found' a difference)."""
    rng = np.random.default_rng(5)
    st = np.repeat(np.arange(3), 30)
    iid = rng.normal(0.0, 0.01, 90)
    offsets = iid + np.repeat(np.array([-0.03, 0.0, 0.03]), 30)
    assert stats.minimum_detectable_effect(
        offsets, st
    ) > 3 * stats.minimum_detectable_effect(iid, st)
    # Without strata the same data looks precise: that is the trap.
    assert stats.minimum_detectable_effect(offsets) < stats.minimum_detectable_effect(
        offsets, st
    )


def test_optimisation_needs_twice_the_mde():
    """A measured gain is a claim beyond max(2 %, MDE) but an
    optimisation is adopted only beyond max(2 %, 2 x MDE)."""
    ci = stats.CI(
        math.log(1.04), math.log(1.03), math.log(1.05), 0.95, 90, 10_000, 0, "t"
    )
    assert stats.evaluate_claim(ci, mde=0.025).result == "increase"
    assert stats.evaluate_optimisation(ci, mde=0.025).result == "no_claim"
    faster = stats.CI(
        math.log(0.9), math.log(0.88), math.log(0.92), 0.95, 90, 10_000, 0, "t"
    )
    assert stats.evaluate_optimisation(faster, mde=0.02).result == "decrease"


def test_claim_from_blocks_end_to_end():
    """Blocks -> pairs -> CI -> claim: an arm 5 % slower in 1 %-noise blocks is
    reported as 'increase' with a ratio CI around 1.05."""
    blocks = make_blocks({"base": 1e5, "slow": 1.05e5, "base#aa": 1e5}, 36, 3, 0.01, 11)
    aa, aa_st = stats.paired_log_ratios(blocks, "base#aa", "base")
    lr, st = stats.paired_log_ratios(blocks, "slow", "base")
    v = stats.evaluate_claim(
        stats.bootstrap_ci(lr, strata=st), stats.minimum_detectable_effect(aa, aa_st)
    )
    assert v.result == "increase"
    assert v.lo < 1.05 < v.hi


# ---------------------------------------------------------------- gates and ceilings
def _hw():
    ci = stats.CI(700.0, 690.0, 710.0, 0.95, 90, 10_000, 0, "t")
    return stats.Ceiling("sol_copy", "sep-nhd@128k", "GB/s", ci, "results/x.json")


def test_gate_passes_only_on_ci_lower_bound():
    """A gate passes on the CI lower bound. A point estimate of 0.92
    over a 0.90 threshold with a CI reaching 0.89 is 'inconclusive', not a
    pass; noise can make us fail a gate but never pass one."""
    ceil = _hw()

    def eta(est, lo, hi):
        return stats.CI(est, lo, hi, 0.95, 90, 10_000, 0, "t")

    assert (
        stats.evaluate_gate(eta(0.92, 0.89, 0.95), 0.90, ceil, 90).result
        == "inconclusive"
    )
    assert stats.evaluate_gate(eta(0.93, 0.91, 0.95), 0.90, ceil, 90).result == "pass"
    assert stats.evaluate_gate(eta(0.82, 0.80, 0.85), 0.90, ceil, 90).result == "fail"
    assert stats.evaluate_gate(eta(0.93, 0.91, 0.95), 0.90, ceil, 29).result == "noisy"
    v = stats.evaluate_gate(eta(0.93, 0.91, 0.95), 0.90, ceil, 90)
    assert v.against.startswith("sol_copy:")


def test_ceiling_refuses_baselines():
    """A baseline is never a ceiling: no gate relative to Triton or FA2. A Ceiling of
    such a kind cannot even be constructed, so no gate can reference one."""
    ci = stats.CI(1.0, 0.9, 1.1, 0.95, 90, 10_000, 0, "t")
    for kind in ("triton", "fa2", "flashinfer", "other_baseline"):
        with pytest.raises(ValueError):
            stats.Ceiling(kind, "x", "us", ci, "y")
    assert _hw().kind == "sol_copy"


# ---------------------------------------------------------------- rejection rules
CFG = stats.RejectionConfig(pci_bus_id="0000:01:00.0")
BLOCK = stats.Block(0, 0, 0, "a", 10_000_000_000, 10_004_000_000, tuple([1e5] * 20))


REF = {stats.SM_CLOCK: 1800.0, stats.MEM_CLOCK: 9251.0}


def reasons(samples, block=BLOCK, ref=REF, cfg=CFG):
    return stats.block_rejection_reasons(block, samples, cfg, ref)


def test_clean_block_is_accepted():
    """A block under the normal 220 W power limit, one process, 70 C, stable
    clock: no reason to reject. The SW power-limit bit alone must never reject
    (it is set during every loaded block at 220 W)."""
    assert reasons([nvml_sample(BLOCK.t_start_ns + 1)]) == []


def test_power_cap_bit_would_reject_everything_if_masked():
    """Negative control for the throttle mask: including SW_POWER_CAP in the
    mask rejects the clean block, which is why it is excluded."""
    cfg = stats.RejectionConfig(
        pci_bus_id="0000:01:00.0",
        throttle_mask=stats.REJECT_THROTTLE_MASK | stats.THROTTLE_SW_POWER_CAP,
    )
    assert reasons([nvml_sample(BLOCK.t_start_ns + 1)], cfg=cfg) == [
        "throttle:sw_power_cap"
    ]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"procs": 2}, "foreign_process"),
        ({"temp": 84.0}, "temperature"),
        ({"throttle": CLEAN | stats.THROTTLE_HW_SLOWDOWN}, "throttle:hw_slowdown"),
        ({"throttle": stats.THROTTLE_SW_THERMAL}, "throttle:sw_thermal_slowdown"),
        ({"throttle": stats.THROTTLE_HW_THERMAL}, "throttle:hw_thermal_slowdown"),
        (
            {"throttle": stats.THROTTLE_HW_POWER_BRAKE},
            "throttle:hw_power_brake_slowdown",
        ),
        ({"clock": 1800.0 * 1.04}, "sm_clock_drift"),
        ({"clock": 1800.0 * 0.96}, "sm_clock_drift"),
        ({"mem": 9251.0 * 0.96}, "mem_clock_drift"),
    ],
)
def test_each_rule_fires_on_crafted_sample(kwargs, expected):
    """Rejection rules, one at a time: foreign GPU process, > 83 C,
    thermal / HW-slowdown / power-brake throttle, > 3 % drift of a watched
    clock (both SM and memory by default)."""
    assert reasons([nvml_sample(BLOCK.t_start_ns + 1, **kwargs)]) == [expected]


def test_memory_domain_ignores_sm_clock_but_not_memory_clock():
    """Clock-noise analysis of the ceilings runs: memory-bound copy time does not
    depend on the SM clock (median -0.02 % over 136 plateau changes of 8-47 %,
    RTX 3080 20 GB, 2026-10; docs/benchmark-methodology.md), so a
    memory-domain arm is judged by the memory clock only. A 30 % SM-clock
    plateau is accepted; a memory-clock drop is still rejected.

    Negative control: the default (both clocks) and the SM domain reject
    the same SM plateau."""
    mem_cfg = stats.rejection_config(clock_domain="mem", pci_bus_id="0000:01:00.0")
    plateau = [nvml_sample(BLOCK.t_start_ns + 1, clock=1800.0 * 0.70)]
    assert reasons(plateau, cfg=mem_cfg) == []
    mem_dip = [nvml_sample(BLOCK.t_start_ns + 1, mem=5001.0)]
    assert reasons(mem_dip, cfg=mem_cfg) == ["mem_clock_drift"]
    assert reasons(plateau) == ["sm_clock_drift"]
    sm_cfg = stats.rejection_config(clock_domain="sm", pci_bus_id="0000:01:00.0")
    assert reasons(plateau, cfg=sm_cfg) == ["sm_clock_drift"]
    assert reasons(mem_dip, cfg=sm_cfg) == []


def test_clock_domain_names_are_checked():
    """A typo in the domain must not silently fall back to some default."""
    with pytest.raises(ValueError):
        stats.rejection_config(clock_domain="memory")
    with pytest.raises(ValueError):
        stats.RejectionConfig(clock_fields=("power_w",))


@pytest.mark.parametrize(
    "kwargs", [{"temp": 83.0}, {"clock": 1800 * 1.029}, {"clock": 1800 * 0.971}]
)
def test_rules_do_not_fire_at_the_boundary(kwargs):
    """Thresholds are strict: exactly 83 C and 2.9 % drift are accepted."""
    assert reasons([nvml_sample(BLOCK.t_start_ns + 1, **kwargs)]) == []


def test_block_dispersion_rule():
    """IQR/median > 5 % inside one block means something interfered while it
    ran."""
    d = tuple([1e5] * 10 + [1.2e5] * 10)
    b = stats.Block(0, 0, 0, "a", BLOCK.t_start_ns, BLOCK.t_end_ns, d)
    assert reasons([nvml_sample(b.t_start_ns + 1)], block=b) == ["block_dispersion"]


def test_dispersion_within_timer_resolution_is_not_dispersion():
    """2-us graph replays of the ceilings runs, timed with CUDA events in
    1024-ns steps (RTX 3080 20 GB, 2026-10), had
    IQR = one step = 50 % of the median in 99 of 108 blocks. That is the
    timer's resolution, not interference: with the event quantum known the
    block is accepted.

    Negative controls: without the quantum the same block is rejected, and a
    real 20 % spread on a 100-us kernel is still rejected with the quantum."""
    quantised = tuple([2048.0] * 14 + [3072.0] * 6)
    b = stats.Block(0, 0, 0, "a", BLOCK.t_start_ns, BLOCK.t_end_ns, quantised)
    ev = stats.rejection_config(timer="cuda_event", pci_bus_id="0000:01:00.0")
    near = [nvml_sample(b.t_start_ns + 1)]
    assert ev.timer_quantum_ns == 1024.0
    assert reasons(near, block=b, cfg=ev) == []
    assert reasons(near, block=b) == ["block_dispersion"]
    real = stats.Block(
        0, 0, 0, "a", BLOCK.t_start_ns, BLOCK.t_end_ns, tuple([1e5] * 10 + [1.2e5] * 10)
    )
    assert reasons(near, block=real, cfg=ev) == ["block_dispersion"]


def test_missing_nvml_data_rejects_fail_closed():
    """No sample near the block, a sample of another GPU only, or a field
    NVML could not read: we cannot vouch for the block, so it is rejected
    instead of being assumed clean."""
    assert reasons([]) == ["no_nvml_coverage"]
    other = nvml_sample(BLOCK.t_start_ns + 1, bus="00000000:02:00.0")
    assert reasons([other]) == ["no_nvml_coverage"]
    s = nvml_sample(BLOCK.t_start_ns + 1)
    s["temp_c"] = None
    assert reasons([s]) == ["nvml_missing:temp_c"]


def test_sample_window_is_one_period_either_side():
    """A 4 ms block between two 100 ms samples is judged by the samples within
    one period of it: 50 ms before counts, 150 ms before does not."""
    near = nvml_sample(BLOCK.t_start_ns - 50_000_000, temp=90.0)
    far = nvml_sample(BLOCK.t_start_ns - 150_000_000, temp=90.0)
    ok = nvml_sample(BLOCK.t_end_ns + 1)
    assert reasons([near, ok]) == ["temperature"]
    assert reasons([far, ok]) == []


def test_reference_clock_is_per_arm():
    """Under a power limit a tensor-core arm clocks lower than a copy arm. With a
    per-arm reference neither is rejected; with one shared reference (the
    negative control) every block of one arm would count as 'drift'.

    Blocks are 1 s apart here so each NVML window sees only its own block.
    Real blocks are a few ms apart and share 100 ms samples with their
    neighbours; then the per-arm references both describe the interleaved
    mix, and the drift rule catches drift of the run (thermal, power limit),
    which is its purpose."""
    blocks = make_blocks({"copy": 1e5, "mma": 2e6}, 4, 1, 0.0, 0, gap_ns=1_000_000_000)
    samples = [
        nvml_sample(b.t_start_ns + 1, clock=1900.0 if b.arm == "copy" else 1600.0)
        for b in blocks
    ]
    cfg = stats.RejectionConfig()
    assert stats.reject_blocks(blocks, samples, cfg) == {}
    refs = stats.reference_clocks(blocks, samples, cfg)
    assert {a: r[stats.SM_CLOCK] for a, r in refs.items()} == {
        "copy": 1900.0,
        "mma": 1600.0,
    }
    mma = next(b for b in blocks if b.arm == "mma")
    assert stats.block_rejection_reasons(mma, samples, cfg, refs["copy"]) == [
        "sm_clock_drift"
    ]


def test_reference_between_plateaus_rejects_good_blocks():
    """Why the SM rule fails for copies: under copy load the SM clock sits on
    plateaus (here 1770 and 1380 MHz, as in the copy-L128k ceiling runs) while the
    copy time is
    unchanged. The per-arm median falls between the plateaus and the SM rule
    rejects every block; the memory domain keeps them all."""
    blocks = make_blocks({"copy": 1.95e5}, 36, 1, 0.0, 0, gap_ns=1_000_000_000)
    samples = [
        nvml_sample(b.t_start_ns + 1, clock=1770.0 if b.round % 2 else 1380.0)
        for b in blocks
    ]
    sm = stats.reject_blocks(blocks, samples, stats.rejection_config(clock_domain="sm"))
    mem = stats.reject_blocks(
        blocks, samples, stats.rejection_config(clock_domain="mem")
    )
    assert len(sm) == 36 and len(mem) == 0


def test_fast_windows_match_the_reference_scan():
    """reject_blocks uses a sorted index of the samples; it must give exactly
    what the per-block linear scan (samples_in_window) gives, bus filter
    included."""
    rng = np.random.default_rng(4)
    blocks = make_blocks({"a": 1e5, "b": 1e5}, 30, 2, 0.01, 1, gap_ns=37_000_000)
    t0, t1 = blocks[0].t_start_ns, blocks[-1].t_end_ns
    samples = [
        nvml_sample(
            int(t),
            clock=float(rng.choice([1700.0, 1800.0, 1900.0])),
            temp=float(rng.choice([70.0, 85.0], p=[0.95, 0.05])),
            bus=str(rng.choice(["00000000:01:00.0", "00000000:02:00.0"])),
        )
        for t in np.arange(t0 - 2e8, t1 + 2e8, 5e7)
    ]
    cfg = stats.RejectionConfig(pci_bus_id="0000:01:00.0")
    refs = stats.reference_clocks(blocks, samples, cfg)
    slow = {
        b.key: r
        for b in blocks
        if (r := stats.block_rejection_reasons(b, samples, cfg, refs.get(b.arm, {})))
    }
    assert stats.reject_blocks(blocks, samples, cfg) == slow
    assert slow  # the fixture does reject something


def test_rejected_block_drops_its_whole_pair():
    """Rejecting one block of a round removes that round's pair, so the
    remaining pairs are still adjacent in time (the drift cancellation stays
    valid)."""
    blocks = make_blocks({"a": 1e5, "b": 1e5}, 4, 1, 0.0, 0)
    lr, _ = stats.paired_log_ratios(blocks, "b", "a")
    assert lr.size == 4
    victim = next(b for b in blocks if b.round == 2 and b.arm == "a")
    lr2, _ = stats.paired_log_ratios(blocks, "b", "a", rejected=[victim.key])
    assert lr2.size == 3


def test_rejection_counts_and_rerun_policy():
    """Report rejection counts; a process that kept fewer than 30
    blocks of an arm is rerun, at most 3 times, then the result is 'noisy'."""
    blocks = make_blocks({"a": 1e5, "b": 1e5}, 36, 1, 0.0, 0)
    a_blocks = [b for b in blocks if b.arm == "a"]
    rejected = {b.key: ["temperature"] for b in a_blocks[:6]}
    assert not stats.process_needs_rerun(blocks, rejected, 0)  # 30 of 36 kept
    rejected[a_blocks[6].key] = ["foreign_process", "temperature"]
    assert stats.process_needs_rerun(blocks, rejected, 0)  # 29 kept
    assert stats.rejection_counts(rejected) == {"foreign_process": 1, "temperature": 7}
    assert stats.run_status(0, still_short=False) == "ok"
    assert stats.run_status(2, still_short=True) == "rerun"
    assert stats.run_status(3, still_short=True) == "noisy"


def test_schedule_cli_matches_function():
    """ceilings_run.sh takes the arm order from this CLI; it must print exactly the
    mirrored schedule (process 1 reversed)."""
    out = subprocess.run(
        [
            sys.executable,
            str(BENCH / "stats.py"),
            "schedule",
            "--arms",
            "3",
            "--rounds",
            "2",
            "--process",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "2,1,0,0,1,2"


# ---------------------------------------------------------------- foreign processes
# A block with a second compute process is always rejected; the run is
# invalidated only by a large (or unattributable) foreign process or by too
# many hit blocks in one process.


def with_procs(t_ns, procs):
    s = nvml_sample(t_ns)
    s["compute_procs"] = procs
    return s


BENCH_PROC = {"pid": 100, "used_mib": 476.0}


def test_a_tiny_foreign_context_rejects_the_block_but_is_marked_small():
    """The 2026-10-12 17:27:51 case: a 6.88 MiB context next to the
    476 MiB benchmark."""
    s = with_procs(
        BLOCK.t_start_ns + 1, [BENCH_PROC, {"pid": 828838, "used_mib": 6.88}]
    )
    assert reasons([s]) == [stats.FOREIGN_SMALL_REASON]


@pytest.mark.parametrize("used", [64.5, 500.0, None])
def test_a_large_or_unattributable_foreign_process_is_the_voiding_reason(used):
    s = with_procs(BLOCK.t_start_ns + 1, [BENCH_PROC, {"pid": 7, "used_mib": used}])
    assert reasons([s]) == [stats.FOREIGN_REASON]


def test_the_memory_limit_is_inclusive_and_configurable():
    s = with_procs(BLOCK.t_start_ns + 1, [BENCH_PROC, {"pid": 7, "used_mib": 64.0}])
    assert reasons([s]) == [stats.FOREIGN_SMALL_REASON]
    cfg = stats.RejectionConfig(pci_bus_id="0000:01:00.0", max_foreign_mib=8.0)
    assert reasons([s], cfg=cfg) == [stats.FOREIGN_REASON]


def test_the_benchmark_is_the_largest_process_whatever_the_listing_order():
    """NVML lists processes in no promised order: the small context first
    must not make the benchmark look foreign."""
    s = with_procs(BLOCK.t_start_ns + 1, [{"pid": 9, "used_mib": 6.0}, BENCH_PROC])
    assert reasons([s]) == [stats.FOREIGN_SMALL_REASON]
    assert [p["pid"] for p in stats.foreign_processes(s["compute_procs"], 1)] == [9]


def test_a_worse_sample_wins_over_a_tiny_one_in_the_same_window():
    tiny = with_procs(BLOCK.t_start_ns + 1, [BENCH_PROC, {"pid": 9, "used_mib": 6.0}])
    big = with_procs(BLOCK.t_start_ns + 2, [BENCH_PROC, {"pid": 8, "used_mib": 900.0}])
    assert reasons([tiny, big]) == [stats.FOREIGN_REASON]


def _foreign_blocks(n_per_process=36, processes=2):
    return [
        stats.Block(
            p,
            i,
            i,
            "kernel",
            10**10 + i * 10**7,
            10**10 + i * 10**7 + 4 * 10**6,
            (1e5,),
        )
        for p in range(processes)
        for i in range(n_per_process)
    ]


def test_one_hit_block_of_36_keeps_the_run():
    blocks = _foreign_blocks()
    out = stats.foreign_summary(blocks, {(0, 5): [stats.FOREIGN_SMALL_REASON]})
    assert out == {
        "blocks": 1,
        "large_blocks": 0,
        "worst_fraction": 1 / 36,
        "void": False,
    }


def test_the_hit_fraction_limit_is_per_process_and_voids_above_one_twelfth():
    blocks = _foreign_blocks()
    three = {(0, i): [stats.FOREIGN_SMALL_REASON] for i in range(3)}  # 3/36 = 1/12
    assert stats.foreign_summary(blocks, three)["void"] is False
    four = {(0, i): [stats.FOREIGN_SMALL_REASON] for i in range(4)}
    out = stats.foreign_summary(blocks, four)
    assert out["void"] is True and out["worst_fraction"] == 4 / 36
    spread = {(p, i): [stats.FOREIGN_SMALL_REASON] for p in (0, 1) for i in range(3)}
    assert stats.foreign_summary(blocks, spread)["void"] is False


def test_one_block_with_a_large_foreign_process_voids_the_run():
    out = stats.foreign_summary(_foreign_blocks(), {(1, 0): [stats.FOREIGN_REASON]})
    assert out["large_blocks"] == 1 and out["void"] is True


def test_other_reasons_do_not_count_as_foreign():
    out = stats.foreign_summary(
        _foreign_blocks(), {(0, 0): ["temperature"], (0, 1): ["throttle:hw_slowdown"]}
    )
    assert out == {"blocks": 0, "large_blocks": 0, "worst_fraction": 0.0, "void": False}


def test_by_reason_counts_both_foreign_reasons_together():
    rejected = {(0, 0): [stats.FOREIGN_SMALL_REASON], (0, 1): [stats.FOREIGN_REASON]}
    assert stats.rejection_counts(rejected) == {"foreign_process": 2}
