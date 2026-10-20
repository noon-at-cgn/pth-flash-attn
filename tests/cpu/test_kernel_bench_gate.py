# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the kernel gates (bench/kernel_gate.py).

The decode decision ("decode eta_mem >= 0.90 at L >= 64k for q in {1,4},
b in {1,2}; verify q = 10 eta_mma >= 0.60") and the floor levels
(KD 0.75, KP 0.50) are computed by this program from the result records. These
tests feed it synthetic records whose true efficiency is known and check:

* a kernel at 0.95 of the copy ceiling passes; at 0.70 it fails and the KD
  floor level is reported as reached; at exactly 0.90 the result is
  inconclusive (the gate passes on the CI *lower* bound only);
* eta_mma is computed on executed flops, with the useful-flop
  eta reported beside it: a kernel at an executed eta of 0.62 passes
  VERIFY although its useful eta is below 0.60;
* a gate needs every one of its rows: one missing row or a missing ceiling
  means "missing", never "pass";
* records timed with the host clock, records with a foreign-process
  rejection, or records from two runs are refused (exit 2);
* the output validates against bench/kernel_gate.schema.json;
* the q = 4 rows of KD / DEC32K
  are gated on the roofline efficiency evaluated at the SM clock the kernel
  ran at, with the raw eta printed beside it, the floor level and the q = 1 /
  q = 10 gates raw. A tensor-bound kernel at a low clock passes the adjusted
  gate and fails the raw one; a memory-bound regression (and a kernel slower
  than its own tensor bound) still fails; the SM clocks come from the
  kernel arm's and the mma ceiling's own NVML windows and from no other arm.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
kg = importlib.import_module("kernel_gate")
kb = importlib.import_module("kernel_bench")

P, N = 6, 36  # processes x accepted blocks each (the kernel benchmark design)
BW = 700e9  # copy ceiling, B/s (~700 GB/s on the reference RTX 3080 20 GB)
F = 66.2e12  # mma ceiling, flop/s (66.2 TFLOPS on the reference RTX 3080 20 GB)
NUM_SMS = 68


def _blocks(rng, t_s, sd=0.002, offs_sd=0.002, foreign=False, clock=None):
    out, seq = [], 0
    offs = rng.normal(0, offs_sd, P)
    for p in range(P):
        for i in range(N):
            m = t_s * 1e9 * np.exp(offs[p] + rng.normal(0, sd))
            out.append(
                {
                    "process": p,
                    "round": i,
                    "seq": seq,
                    "t_start_mono_ns": seq,
                    "t_end_mono_ns": seq + 1,
                    "durations_ns": [m] * 3,
                    "median_ns": m,
                    "rejected": False,
                    "reasons": [],
                    "nvml": {"max_compute_procs": 1}
                    | ({} if clock is None else {"sm_clock_mhz_median": clock}),
                    "aux": {},
                }
            )
            seq += 1
    return out


def _rec(
    tool,
    exp,
    arm,
    role,
    shape,
    params,
    blocks,
    *,
    timer="cupti",
    session="2026-10-20-kernel-220w-gpu1",
    foreign=0,
    foreign_void=None,
    raw_files=(),
):
    return {
        "schema_version": 1,
        "record": "arm_result",
        "session": session,
        "experiment": exp,
        "tool": {"name": tool, "raw_files": list(raw_files), "argv": []},
        "git": {"sha": "a" * 40, "dirty": False},
        "env": {"pci_bus_id": "0000:04:00.0"},
        "arm": {"name": arm, "role": role, "params": params},
        "shape": shape,
        "timer": timer,
        "samples": {"blocks": blocks},
        "summary": {
            "n_blocks": len(blocks),
            "iqr_over_median": 0.01,
            "throughput": {"unit": "GB/s"},
        },
        "rejections": {
            "by_reason": {"foreign_process": foreign} if foreign else {},
            "status": "ok",
            # absent = a record written before the verdict field existed
            **(
                {}
                if foreign_void is None
                else {
                    "foreign": {
                        "blocks": foreign,
                        "large_blocks": int(foreign_void),
                        "worst_fraction": 0.03,
                        "void": foreign_void,
                    }
                }
            ),
        },
    }


def write_session(
    tmp_path,
    eta_mem=0.95,
    eta_mma=0.75,
    *,
    skip=(),
    no_copy=(),
    timer="cupti",
    foreign_in=None,
    foreign_void=None,
    seed=0,
    q4=None,
    kernel_clock=1500.0,
    aa_clock=None,
    mma_clock=1905.0,
    copy_clock=1400.0,
    no_clock=(),
):
    """Records for every quick-tier decode shape (the decode gate rows) at
    the given true efficiencies, their copy ceilings and the mma ceiling.

    ``q4 = (sm_clock_mhz, eta_adj)`` makes the q = 4 rows run at that SM clock
    with the true clock-adjusted efficiency ``eta_adj`` against the roofline
    at that clock (default: memory-bound at ``eta_mem`` like q = 1). The mma
    ceiling is measured at ``mma_clock``, the copies at ``copy_clock``; other
    kernel rows run at ``kernel_clock``; the A/A arm at ``aa_clock`` (default:
    the subject's). Experiments in ``no_clock`` have no clock in their
    subject's NVML windows."""
    rng = np.random.default_rng(seed)
    res = tmp_path / "res"
    res.mkdir(parents=True)
    for s in kb.select("decode", "quick"):
        if s.name in skip:
            continue
        (p,) = kb.plan_shape(s, NUM_SMS)
        B, L, q = len(s.q_lens), max(s.seq_lens), max(s.q_lens)
        # memory-bound rows at eta_mem; q = 10 rows at eta_mma (executed)
        clock = kernel_clock
        t = (
            p.bytes_kv / (eta_mem * BW)
            if q in (1, 4)
            else p.flops_executed / (eta_mma * F)
        )
        if q == 4 and q4 is not None:
            clock, eta_adj = q4
            t_mem = p.bytes_kv / BW
            t_mma = p.flops_executed / (F * clock / mma_clock)
            t = (max(t_mem, t_mma) + kg.T_OVH_S) / eta_adj
        shape = {
            "family": "decode",
            "q_max": q,
            "batch": B,
            "seq_len_max": L,
            "splits": str(p.num_splits),
        }
        params = {
            "bytes_per_launch": p.bytes_kv,
            "flops_executed": p.flops_executed,
            "flops_useful": p.flops_useful,
            "slices": [dataclasses.asdict(p)],
        }
        for arm, role in (("kernel", "subject"), ("kernel#aa", "aa_control")):
            arm_clock = clock if role == "subject" else (aa_clock or clock)
            if role == "subject" and s.name in no_clock:
                arm_clock = None
            r = _rec(
                "kernel_bench",
                s.name,
                arm,
                role,
                shape,
                params,
                _blocks(rng, t, clock=arm_clock),
                timer=timer,
                foreign=3 if foreign_in == s.name else 0,
                foreign_void=foreign_void if foreign_in == s.name else None,
            )
            (res / f"{s.name}__{arm.replace('#', '_')}.json").write_text(json.dumps(r))
        if (B, L) not in no_copy and not (res / f"ceil-copy-b{B}-L{L}.json").exists():
            byts = kb.kv_bytes(L, 2, 256) * B
            c = _rec(
                "sol_copy",
                f"ceil-copy-b{B}-L{L // 1024}k",
                "sep-nhd",
                "ceiling",
                {"batch": B, "seq_len": L},
                {"bytes_per_launch": byts},
                _blocks(rng, byts / BW, clock=copy_clock),
            )
            (res / f"ceil-copy-b{B}-L{L}.json").write_text(json.dumps(c))
    flop = 1e11
    raws = []
    for p in range(P):
        path = tmp_path / f"mma.p{p}.a0.json"
        path.write_text(
            json.dumps(
                {
                    "process": p,
                    "arms": [
                        {"name": kg.MMA_ARM, "flop_per_launch": flop * (1 + 0.001 * p)}
                    ],
                }
            )
        )
        raws.append({"path": str(path), "sha256": "0" * 64})
    mblocks = _blocks(rng, flop / F, clock=mma_clock)
    for b in mblocks:  # each process's own work (calibrated iterations)
        b["median_ns"] *= 1 + 0.001 * b["process"]
    m = _rec(
        "mma_ceiling",
        "ceil-mma",
        kg.MMA_ARM,
        "ceiling",
        {},
        {"flop_per_launch": flop},
        mblocks,
        raw_files=raws,
    )
    (res / "ceil-mma__f16f32.json").write_text(json.dumps(m))
    return res


def rule(rep, rid):
    return next(g for g in rep["rules"] if g["id"] == rid)


def test_fast_kernel_passes_and_report_is_schema_valid(tmp_path):
    """True eta_mem 0.95 and eta_mma 0.75: KD and VERIFY pass on every row,
    no floor level reached, gate verdict pass, exit 0, schema-valid output."""
    res = write_session(tmp_path)
    assert kg.main([str(res)]) == 0
    rep = json.loads((res / "k_gates.json").read_text())
    assert rep["gate_verdict"] == "pass"
    assert rule(rep, "KD")["overall"] == "pass" and len(rule(rep, "KD")["rows"]) == 16
    assert rule(rep, "VERIFY")["overall"] == "pass"
    assert rule(rep, "KD-floor")["overall"] == "not_reached"
    row = next(r for r in rep["experiments"] if r["experiment"] == "dec-q1-b1-L64k")
    assert row["eta_mem"]["estimate"] == pytest.approx(0.95, rel=0.01)
    assert row["eta_mem"]["lo"] > 0.9 and "cluster-t" in row["eta_mem"]["method"]
    assert row["mde"] is not None and row["mde"] < 0.02
    assert "GATES.txt" in {p.name for p in res.iterdir()}
    # which kernel was timed (q = 1 and 4: decode kernel, q = 10: 16-row
    # tile) and the spread are in the row and in the printed table
    q10 = next(r for r in rep["experiments"] if r["experiment"] == "dec-q10-b1-L64k")
    assert (row["kernels"], q10["kernels"]) == (kb.KERNEL_DECODE, kb.KERNEL_TILE16)
    assert row["iqr_over_median"] is not None
    assert "IQR/med" in (res / "GATES.txt").read_text()


def test_slow_kernel_fails_and_reaches_the_floor_level(tmp_path):
    """True eta_mem 0.70: KD fails, the KD floor level (0.75) is reached and
    reported, gate verdict fail, exit 1."""
    res = write_session(tmp_path, eta_mem=0.70)
    assert kg.main([str(res)]) == 1
    rep = json.loads((res / "k_gates.json").read_text())
    assert rule(rep, "KD")["overall"] == "fail"
    assert rule(rep, "KD-floor")["overall"] == "floor_level_reached"
    assert rep["gate_verdict"] == "fail"


def test_threshold_kernel_is_inconclusive_not_pass(tmp_path):
    """True eta_mem exactly 0.90: the CI straddles the threshold, so the
    gate is inconclusive: noise can make us fail, never pass."""
    res = write_session(tmp_path, eta_mem=0.90)
    rep = kg.build(res)
    assert rule(rep, "KD")["overall"] == "inconclusive"
    assert rep["gate_verdict"] == "not_passed"


def test_verify_gate_uses_executed_flops(tmp_path):
    """VERIFY credits the flops the tensor cores were issued ("on
    executed flops"); the useful-flop eta is reported beside it and is
    lower. Negative control: at a true executed eta of 0.62 the gate passes
    although the useful eta is far below 0.60."""
    res = write_session(tmp_path, eta_mma=0.62, seed=3)
    rep = kg.build(res)
    rows = [r for r in rep["experiments"] if r["q"] == 10]
    assert all(r["eta_mma_useful"]["estimate"] < r["eta_mma"]["estimate"] for r in rows)
    assert min(r["eta_mma_useful"]["estimate"] for r in rows) < 0.6
    assert rule(rep, "VERIFY")["overall"] == "pass"


def test_a_missing_row_or_ceiling_is_never_a_pass(tmp_path):
    """One gate row not measured, or one copy ceiling missing: the gate is
    "missing" and the gate verdict is not pass, even with every other row fast."""
    res = write_session(tmp_path, skip={"dec-q4-b2-L200k"})
    rep = kg.build(res)
    assert rule(rep, "KD")["overall"] == "missing"
    assert rep["gate_verdict"] == "not_passed"
    res2 = write_session(tmp_path / "b", no_copy={(1, 262144)})
    rep2 = kg.build(res2)
    miss = [c for c in rule(rep2, "KD")["rows"] if c["result"] == "missing"]
    assert {c["experiment"] for c in miss} == {"dec-q1-b1-L256k", "dec-q4-b1-L256k"}


@pytest.mark.parametrize("kind", ["host", "foreign", "two-sessions"])
def test_untrustworthy_records_are_refused(tmp_path, kind):
    """Host-clock timings, any block taken while a foreign process used the
    GPU, or a directory mixing runs: exit 2, no gate file."""
    if kind == "host":
        res = write_session(tmp_path, timer="host")
    elif kind == "foreign":
        res = write_session(tmp_path, foreign_in="dec-q1-b1-L64k")
    else:
        res = write_session(tmp_path)
        f = res / "dec-q1-b1-L64k__kernel.json"
        r = json.loads(f.read_text())
        r["session"] = "another"
        f.write_text(json.dumps(r))
    assert kg.main([str(res)]) == 2
    assert not (res / "k_gates.json").exists()


@pytest.mark.parametrize(("void", "exit_code"), [(True, 2), (False, 0)])
def test_the_gate_follows_the_records_foreign_verdict(tmp_path, void, exit_code):
    """Foreign-process rejections in a record whose own
    verdict is "not void" (tiny contexts, few blocks) are only dropped blocks;
    a void verdict refuses the run. Records without a verdict keep the old
    rule (test_untrustworthy_records_are_refused)."""
    res = write_session(tmp_path, foreign_in="dec-q1-b1-L64k", foreign_void=void)
    assert kg.main([str(res)]) == exit_code


def test_thresholds_are_the_documented_values():
    """Gates and floor levels have the documented numbers, each against a
    hardware ceiling metric."""
    t = {r.id: (r.kind, r.metric, r.threshold) for r in kg.RULES}
    assert t["KD"] == ("gate", "eta_mem", 0.90)
    assert t["KD-floor"] == ("floor", "eta_mem", 0.75)
    assert t["VERIFY"] == ("gate", "eta_mma", 0.60)
    assert t["PREFILL"] == ("gate", "eta_mma", 0.60)
    assert t["KP-floor"] == ("floor", "eta_mma", 0.50)
    assert t["PREFILL-target"][2] == t["VERIFY-target"][2] == 0.70


def test_the_verdict_covers_the_decode_gates_only():
    """Every rule has a scope; gate_verdict is decided by the decode-scope gates
    (KD, VERIFY), and the prefill rules (a gate reported beside it) do not
    enter it: a missing prefill row leaves a passing decode run passing."""
    scopes = {r.id: r.scope for r in kg.RULES}
    assert set(scopes.values()) == {"decode", "prefill"}
    assert {i for i, s in scopes.items() if s == "prefill"} == {
        "PREFILL",
        "PREFILL-target",
        "KP-floor",
    }
    assert {r.id for r in kg.RULES if r.kind == "gate" and r.scope == "decode"} == {
        "KD",
        "VERIFY",
    }


def test_the_balanced_policy_covers_exactly_the_q4_gate_rows():
    """KD and DEC32K judge q = 4 on eta_adj; the floor
    level, VERIFY, PREFILL and every threshold stay as they were."""
    adj = {r.id: r.adjusted_q for r in kg.RULES}
    assert adj["KD"] == adj["DEC32K"] == (4,)
    assert all(not q for rid, q in adj.items() if rid not in ("KD", "DEC32K"))
    assert kg.BALANCED_Q == 4 and abs(kg.T_OVH_S - 0.60e-6) < 1e-12


def _rows(rep, rid, tag):
    return [c for c in rule(rep, rid)["rows"] if f"-{tag}-" in c["experiment"]]


def test_tensor_bound_kernel_at_a_low_clock_passes_adjusted_fails_raw(tmp_path):
    """q = 4 rows at 800 MHz run at 0.95 of their roofline there: the tensor
    time of the 24 executed rows at 800 MHz is 1.19 x the copy time, so the raw
    eta_mem is 0.95 / 1.19 = 0.80 (raw gate: fail) while the adjusted gate
    passes. The floor level (raw 0.75) is not reached, q = 1 stays raw."""
    res = write_session(tmp_path, q4=(800.0, 0.95))
    rep = kg.build(res)
    rows = _rows(rep, "KD", "q4")
    assert len(rows) == 8
    for c in rows:
        assert (c["metric"], c["result"], c["bound"]) == ("eta_adj", "pass", "mma")
        assert c["sm_clock_mhz"] == 800.0
        assert c["estimate"] == pytest.approx(0.95, rel=0.01)
        assert c["raw"]["result"] == "fail"
        assert 0.76 < c["raw"]["lo"] < c["raw"]["estimate"] < 0.84
        assert c["against"].startswith("sol_copy:") and "+mma_ceiling:" in c["against"]
    assert all(c["metric"] == "eta_mem" for c in _rows(rep, "KD", "q1"))
    assert rule(rep, "KD")["overall"] == "pass"
    assert rule(rep, "KD-floor")["overall"] == "not_reached"
    assert rep["gate_verdict"] == "pass"
    exp = next(r for r in rep["experiments"] if r["experiment"] == "dec-q4-b1-L64k")
    assert exp["adj"]["bound"] == "mma"
    assert exp["adj"]["t_mma_us"] > 1.15 * exp["adj"]["t_mem_us"]
    # the raw eta is always printed beside the adjusted one
    text = kg.table(rep)
    assert "adj; raw 0.7" in text and "SM 800 MHz, mma-bound" in text


@pytest.mark.parametrize(
    "clock, eta_true, why",
    [
        (1700.0, 0.80, "memory-bound regression at a high clock"),
        (800.0, 0.80, "slower than its own tensor bound at a low clock"),
    ],
)
def test_regressions_still_fail_the_adjusted_gate(tmp_path, clock, eta_true, why):
    """Negative controls: a kernel at 0.80 of its roofline fails the adjusted
    gate whether memory-bound (1700 MHz: tensor time 0.56 x copy time, so
    eta_adj equals eta_mem up to T_ovh) or tensor-bound; the clock excuses
    only the tensor-bound share of a miss, never a kernel below its roofline."""
    res = write_session(tmp_path, q4=(clock, eta_true))
    rep = kg.build(res)
    rows = _rows(rep, "KD", "q4")
    assert {c["result"] for c in rows} == {"fail"}, why
    assert all(0.78 < c["estimate"] < 0.82 for c in rows)
    assert rule(rep, "KD")["overall"] == "fail"
    assert rep["gate_verdict"] == "fail"
    if clock > 1500:  # memory-bound: the adjusted eta is the raw one + T_ovh/T
        for c in rows:
            assert c["bound"] == "mem"
            assert c["raw"]["estimate"] <= c["estimate"] < c["raw"]["estimate"] * 1.01


def test_the_adjusted_gate_does_not_move_the_floor_level(tmp_path):
    """At 600 MHz a kernel at 0.95 of its roofline passes KD (adjusted) but
    its raw eta_mem is 0.60: the KD floor level (0.75 on the raw eta) is still
    reported as reached; the clock adjustment never hides it."""
    res = write_session(tmp_path, q4=(600.0, 0.95))
    rep = kg.build(res)
    assert rule(rep, "KD")["overall"] == "pass"
    assert rule(rep, "KD-floor")["overall"] == "floor_level_reached"
    reached = [c for c in rule(rep, "KD-floor")["rows"] if c["floor_level_reached"]]
    assert reached and all("-q4-" in c["experiment"] for c in reached)
    assert all(c["metric"] == "eta_mem" for c in rule(rep, "KD-floor")["rows"])


def test_sm_clocks_come_from_the_same_arms_nvml_window(tmp_path):
    """Guard: the kernel's SM clock is the median of the *kernel arm's own*
    accepted blocks, the mma ceiling is rescaled from the clock of *its own*
    blocks. Another arm's window (the A/A control at 1900 MHz, the copy
    ceilings at 1900 MHz) must not change a digit of eta_adj; the terms are
    the ones the formula gives for the arm's own clocks."""
    a = write_session(tmp_path / "a", q4=(800.0, 0.95), aa_clock=800.0, seed=5)
    b = write_session(
        tmp_path / "b",
        q4=(800.0, 0.95),
        aa_clock=1900.0,
        copy_clock=1900.0,
        mma_clock=1905.0,
        seed=5,
    )
    ra, rb = kg.build(a), kg.build(b)
    for name in ("dec-q4-b1-L64k", "dec-q4-b2-L256k"):
        xa = next(r for r in ra["experiments"] if r["experiment"] == name)
        xb = next(r for r in rb["experiments"] if r["experiment"] == name)
        assert xa["eta_adj"] == xb["eta_adj"] and xa["adj"] == xb["adj"]
        assert xa["adj"]["kernel_clock_mhz"] == 800.0
        assert xa["adj"]["mma_ceiling_clock_mhz"] == 1905.0
        assert xa["mma_ceiling_clock_mhz"] == xa["adj"]["mma_ceiling_clock_mhz"]
    # the mma ceiling's own clock scales its rate: measured at 1500 MHz the
    # same synthetic rate means a pipe that does F at 1500, so at 800 MHz the
    # tensor time is flops / (F * 800 / 1500), not / (F * 800 / 1905)
    c = write_session(
        tmp_path / "c", q4=(800.0, 0.95), mma_clock=1500.0, aa_clock=1905.0, seed=5
    )
    rc = kg.build(c)
    x = next(r for r in rc["experiments"] if r["experiment"] == "dec-q4-b1-L64k")
    s = next(s for s in kb.select("decode", "quick") if s.name == x["experiment"])
    (p,) = kb.plan_shape(s, NUM_SMS)
    assert x["adj"]["mma_ceiling_clock_mhz"] == 1500.0
    assert x["adj"]["t_mma_us"] == pytest.approx(
        p.flops_executed / (F * 800.0 / 1500.0) * 1e6, rel=0.01
    )


def test_rejected_blocks_do_not_contribute_a_clock(tmp_path):
    """A rejected block's NVML window is not evidence: adding one with a
    wild clock and a tiny time to the subject leaves eta_adj unchanged."""
    res = write_session(tmp_path, q4=(800.0, 0.95), seed=2)
    before = next(
        r for r in kg.build(res)["experiments"] if r["experiment"] == "dec-q4-b1-L64k"
    )
    f = res / "dec-q4-b1-L64k__kernel.json"
    rec = json.loads(f.read_text())
    bad = dict(
        rec["samples"]["blocks"][0],
        round=N + 1,  # its own round: the A/A pairing needs one block per round
        seq=10_000,
        rejected=True,
        reasons=["sm_clock_drift"],
    )
    bad["nvml"] = {"max_compute_procs": 1, "sm_clock_mhz_median": 100.0}
    bad["median_ns"] = 1.0
    rec["samples"]["blocks"].append(bad)
    f.write_text(json.dumps(rec))
    after = next(
        r for r in kg.build(res)["experiments"] if r["experiment"] == "dec-q4-b1-L64k"
    )
    assert after["eta_adj"] == before["eta_adj"] and after["adj"] == before["adj"]


def test_a_row_without_a_kernel_clock_is_missing_not_judged_raw(tmp_path):
    """No SM clock in the kernel arm's NVML windows: the q = 4 row is
    "missing" (with the reason), the gate is not passed, and nothing falls
    back to the raw eta or to another arm's clock."""
    res = write_session(tmp_path, no_clock={"dec-q4-b1-L64k"})
    rep = kg.build(res)
    (c,) = [c for c in rule(rep, "KD")["rows"] if c["experiment"] == "dec-q4-b1-L64k"]
    assert c["result"] == "missing" and "no SM clock" in c["note"]
    assert rule(rep, "KD")["overall"] == "missing"
    assert rep["gate_verdict"] == "not_passed"
    exp = next(r for r in rep["experiments"] if r["experiment"] == "dec-q4-b1-L64k")
    assert exp["eta_adj"] is None and exp["eta_mem"] is not None
    # the floor level is on the raw eta and does not need the clock
    assert all(c["result"] != "missing" for c in rule(rep, "KD-floor")["rows"])


def test_out_directory_leaves_the_sessions_own_verdict_files(tmp_path):
    """A re-evaluation (`--out`) writes its own k_gates.json and GATES.txt and
    never touches the session's."""
    res = write_session(tmp_path)
    out = tmp_path / "reeval"
    assert kg.main([str(res), "--out", str(out)]) == 0
    assert (out / "k_gates.json").exists() and (out / "GATES.txt").exists()
    assert not (res / "k_gates.json").exists() and not (res / "GATES.txt").exists()


# ---------------------------------------------------------------- stats
def _roofline(clock_k=1000.0, ovh=1e-6, t=130e-6, **kw):
    """Two processes, constant samples: T_mem 100 us (1e8 B at 1 TB/s), the
    tensor ceiling 100 Tflop/s at 2000 MHz, 6e9 flop issued."""
    stats = importlib.import_module("stats")
    n = 10
    return stats.roofline_eta_ci(
        np.full(n, t),
        np.full(n, clock_k),
        np.repeat([0, 1], n // 2),
        1e8,
        6e9,
        np.full(n, 1e12),
        np.repeat([0, 1], n // 2),
        np.full(n, 100e12),
        np.full(n, 2000.0),
        np.repeat([0, 1], n // 2),
        t_ovh_s=ovh,
        **kw,
    )


def test_roofline_formula_by_hand():
    """T = max(T_mem, T_mma(f)) + T_ovh with T_mma(f) = flop / (F f / f_ceil):
    6e9 flop at 100 Tflop/s (2000 MHz) is 60 us, at 1000 MHz 120 us, so the
    target is 121 us and eta = 121 / 130; at 2000 MHz the copy term, 100 us,
    is the larger one and eta = 101 / 130."""
    low = _roofline(1000.0)
    assert low.bound == "mma"
    assert low.t_mem_s == pytest.approx(100e-6) and low.t_mma_s == pytest.approx(120e-6)
    assert low.ci.estimate == pytest.approx(121 / 130)
    assert low.ci.lo == pytest.approx(121 / 130) == pytest.approx(low.ci.hi)
    high = _roofline(2000.0)
    assert high.bound == "mem" and high.t_mma_s == pytest.approx(60e-6)
    assert high.ci.estimate == pytest.approx(101 / 130)
    assert (low.kernel_clock_mhz, low.ceiling_clock_mhz) == (1000.0, 2000.0)


def test_roofline_reduces_to_eta_ci_when_memory_bound():
    """With T_mma below T_mem at the kernel's clock and no T_ovh, eta_adj is
    eta_mem: same estimate, same interval (same bootstrap draws)."""
    stats = importlib.import_module("stats")
    rng = np.random.default_rng(1)
    n, procs = 60, np.repeat(np.arange(6), 10)
    t = 120e-6 * np.exp(rng.normal(0, 0.01, n) + 0.01 * rng.normal(size=6)[procs])
    cr = 1e12 * np.exp(rng.normal(0, 0.005, n))
    mr = 100e12 * np.exp(rng.normal(0, 0.002, n))
    clk = np.full(n, 1900.0)
    ra = stats.roofline_eta_ci(
        t, clk, procs, 1e8, 6e9, cr, procs, mr, np.full(n, 1905.0), procs, t_ovh_s=0.0
    )
    rm = stats.eta_ci(t, procs, 1e8, cr, procs)
    assert ra.bound == "mem"
    assert ra.ci.estimate == pytest.approx(rm.estimate, rel=1e-12)
    assert ra.ci.lo == pytest.approx(rm.lo, rel=1e-9)
    assert ra.ci.hi == pytest.approx(rm.hi, rel=1e-9)


def test_roofline_is_never_below_raw_and_grows_as_the_clock_falls():
    """The roofline target is a maximum of T_mem and something: eta_adj >=
    eta_mem for every clock, and a lower kernel clock only raises the tensor
    term, so eta_adj is non-decreasing as the clock falls."""
    etas = [_roofline(c, ovh=0.0).ci.estimate for c in (2000, 1500, 1000, 700, 400)]
    raw = 100e-6 / 130e-6
    assert all(e >= raw - 1e-12 for e in etas)
    assert etas == sorted(etas)
    assert etas[0] == pytest.approx(raw) and etas[-1] > raw


def test_roofline_needs_clocks_per_block_and_positive():
    stats = importlib.import_module("stats")
    ok = dict(
        kernel_s=np.ones(4),
        kernel_clock_mhz=np.ones(4),
        kernel_strata=np.zeros(4),
        bytes_read=1.0,
        flops=1.0,
        copy_rates=np.ones(4),
        copy_strata=np.zeros(4),
        mma_rates=np.ones(4),
        mma_clock_mhz=np.ones(4),
        mma_strata=np.zeros(4),
    )
    assert stats.roofline_eta_ci(**ok).ci.estimate > 0
    for k, v in (
        ("kernel_clock_mhz", np.ones(3)),
        ("kernel_clock_mhz", np.zeros(4)),
        ("mma_clock_mhz", np.ones(2)),
    ):
        with pytest.raises(ValueError):
            stats.roofline_eta_ci(**{**ok, k: v})
