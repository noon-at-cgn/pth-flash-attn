# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for bench/results.schema.json and bench/bench_report.py.

The schema is a frozen interface: every published number is a
record that validates against it, carrying git sha, environment capture, arm,
shape, samples, median/IQR/n, a CI, rejection counts, a verdict and the
ceilings it references. These tests build synthetic raw output
of the C++ ceiling programs plus an NVML log, run the real report tool, and
check (a) its records validate and are right, and (b) the schema rejects the
mistakes it exists to catch (negative controls).
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "bench"
sys.path.insert(0, str(BENCH))
import stats  # noqa: E402

SCHEMA = json.loads((BENCH / "results.schema.json").read_text())
BUS = "0000:01:00.0"  # CUDA spelling; NVML writes 00000000:01:00.0
SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "ab" * 32


def validator():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    return jsonschema.Draft202012Validator(SCHEMA)


def fake_raw(process, arms_true_ns, rounds, t0, seed):
    """Raw output in the format sol_copy.cu writes (raw_format 1)."""
    rng = np.random.default_rng(seed)
    names = list(arms_true_ns)
    sched = stats.mirrored_schedule(len(names), rounds, reverse=process % 2 == 1)
    blocks, t = [], t0
    for seq, a in enumerate(sched):
        f = math.exp(rng.normal(0, 0.005))
        d = [
            int(arms_true_ns[names[a]] * f * math.exp(rng.normal(0, 0.001)))
            for _ in range(20)
        ]
        blocks.append(
            {
                "seq": seq,
                "round": seq // len(names),
                "arm": names[a],
                "t_start_mono_ns": t,
                "t_end_mono_ns": t + 5_000_000,
                "graph_ms": sum(d) / 1e6,
                "durations_ns": d,
            }
        )
        t += 300_000_000  # 300 ms apart: each block has its own NVML samples
    arms = [
        {
            "name": n,
            "variant": n.split("-")[0],
            "layout": "NHD",
            "bytes_per_launch": 136_314_880,
            "rotations": 1,
            "smem_bytes": 99_328,
            "ctas_per_sm": 1,
        }
        for n in names
    ]
    return {
        "tool": "sol_copy",
        "raw_format": 1,
        "argv": ["sol_copy", "--arms", ",".join(names)],
        "timer": "cupti",
        "duration_kind": "kernel",
        "process": process,
        "t_unix_ns": 1,
        "device": {
            "cuda_index": 0,
            "name": "NVIDIA GeForce RTX 3080",
            "pci_bus_id": BUS,
            "cc": "8.6",
            "sm_count": 68,
            "l2_bytes": 5_242_880,
            "runtime_version": 13000,
        },
        "shape": {
            "D": 256,
            "Hkv": 2,
            "block_size": 816,
            "batch": 1,
            "seq_len": 131072,
            "splits": 34,
        },
        "design": {
            "replays_per_block": 20,
            "warmup_s": 2.0,
            "n_arms": len(names),
            "n_blocks": len(sched),
        },
        "arms": arms,
        "blocks": blocks,
    }, t


def nvml_lines(raws, hot=lambda b: False):
    lines = [
        json.dumps(
            {"kind": "header", "log_format": 1, "interval_ms": 100, "devices": []}
        )
    ]
    tick = 0
    for raw in raws:
        for b in raw["blocks"]:
            for dt in (1_000_000, 4_000_000):
                lines.append(
                    json.dumps(
                        {
                            "kind": "sample",
                            "tick": tick,
                            "t_mono_ns": b["t_start_mono_ns"] + dt,
                            "t_unix_ns": 1,
                            "gpu_index": 0,
                            "pci_bus_id": "00000000:01:00.0",
                            "sm_clock_mhz": 1800.0,
                            "mem_clock_mhz": 9501.0,
                            "power_w": 219.0,
                            "temp_c": 90.0 if hot(b) else 70.0,
                            "throttle_reasons": 0x4,
                            "throttle_names": ["sw_power_cap"],
                            "util_gpu_pct": 100.0,
                            "compute_procs": [{"pid": 1, "used_mib": 900.0}],
                            "errors": {},
                        }
                    )
                )
                tick += 1
    return lines


@pytest.fixture(scope="module")
def ceilings_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ceilings")
    arms = {"sep-nhd": 180_000, "sep-nhd#aa": 180_000, "cell-nhd": 198_000}
    raws, t = [], 10_000_000_000
    for p in range(3):
        raw, t = fake_raw(p, arms, 36, t, seed=p)
        raws.append(raw)
        (d / f"p{p}.json").write_text(json.dumps(raw))
    (d / "nvml.jsonl").write_text("\n".join(nvml_lines(raws)) + "\n")
    (d / "smi.txt").write_text("nvidia-smi -q output\n")
    env = {
        "git_sha": SHA,
        "git_dirty": False,
        "image_digest": DIGEST,
        "nvcc_version": "13.0.88",
        "driver_version": "580.00",
        "power_limit_w": 220.0,
        "captures": [{"name": "nvidia-smi -q", "path": "smi.txt", "sha256": "0" * 64}],
    }
    (d / "env.json").write_text(json.dumps(env))
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "bench_report.py"),
            "report",
            "--raw",
            *[str(d / f"p{p}.json") for p in range(3)],
            "--nvml",
            str(d / "nvml.jsonl"),
            "--env",
            str(d / "env.json"),
            "--session",
            "2026-10-20-ceilings-220w-gpu0",
            "--experiment",
            "copy-128k",
            "--out-dir",
            str(d / "out"),
            "--aa",
            "sep-nhd",
            "sep-nhd#aa",
            "--compare",
            "cell-nhd",
            "sep-nhd",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return d


def records(d):
    return {
        p.name: json.loads(p.read_text()) for p in sorted((d / "out").glob("*.json"))
    }


def test_report_records_validate(ceilings_dir):
    """Every record bench_report.py writes validates against the schema."""
    v = validator()
    recs = records(ceilings_dir)
    assert set(recs) == {
        "copy-128k__sep-nhd.json",
        "copy-128k__sep-nhd_aa.json",
        "copy-128k__cell-nhd.json",
    }
    for name, rec in recs.items():
        errors = sorted(v.iter_errors(rec), key=str)
        assert not errors, (name, [e.message for e in errors])


def test_report_numbers_are_right(ceilings_dir):
    """Known truth: sep arm 180 us for 136,314,880 bytes = 757 GB/s; cell arm
    10 % slower; the A/A control is the same arm. The record must say so:
    throughput CI around 757 GB/s, the cell-vs-sep claim 'increase' with a
    ratio CI around 1.10, every block accepted (clean NVML), 3 x 36 blocks."""
    recs = records(ceilings_dir)
    sep = recs["copy-128k__sep-nhd.json"]
    tp = sep["summary"]["throughput"]
    assert tp["unit"] == "GB/s"
    assert tp["lo"] < 136_314_880 / 180_000 < tp["hi"]
    assert sep["summary"]["n_blocks"] == 108 and sep["summary"]["n_accepted"] == 108
    assert sep["rejections"]["total"] == 0 and sep["rejections"]["status"] == "ok"
    assert (
        sep["arm"]["role"] == "ceiling"
        and sep["verdict"]["overall"] == "not_applicable"
    )
    assert recs["copy-128k__sep-nhd_aa.json"]["arm"]["role"] == "aa_control"
    (claim,) = recs["copy-128k__cell-nhd.json"]["verdict"]["checks"]
    assert claim["kind"] == "claim" and claim["against"] == "sep-nhd"
    assert claim["result"] == "increase"
    assert claim["lo"] < 1.10 < claim["hi"]
    assert claim["margin"] == pytest.approx(
        0.02
    )  # MDE is tiny here: the 2 % floor applies


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (lambda r: r["git"].update(sha="HEAD"), "git sha must be a full 40-hex commit"),
        (lambda r: r.pop("env"), "environment capture is required"),
        (
            lambda r: r["summary"].pop("iqr_over_median"),
            "spread is required with every median",
        ),
        (lambda r: r["rejections"].pop("by_reason"), "rejection counts are required"),
        (
            lambda r: r["env"].update(image_digest="pth-flash-attn-dev:latest"),
            "image must be a digest, not a tag",
        ),
        (
            lambda r: r["arm"].update(role="comparison"),
            "a comparison row must carry its label",
        ),
        (lambda r: r.update(timer="wallclock"), "timer must be cupti or cuda_event"),
        (
            lambda r: r.update(session="run-1"),
            "the session is the result directory name, with the power limit in watts",
        ),
        (
            lambda r: r.update(session="2026-10-20-ceilings-gpu0"),
            "the power limit is part of the name",
        ),
        (lambda r: r["rejections"].update(reruns=4), "at most 3 reruns"),
    ],
)
def test_schema_negative_controls(ceilings_dir, mutate, why):
    """Each mutation breaks one schema requirement; the schema must refuse
    it. If any of these validated, the schema would not be doing its job."""
    rec = copy.deepcopy(records(ceilings_dir)["copy-128k__sep-nhd.json"])
    mutate(rec)
    assert list(validator().iter_errors(rec)), why


def test_schema_refuses_gate_against_a_baseline(ceilings_dir):
    """A baseline is never a ceiling, enforced in the data format as well as in
    stats.py: a gate check whose reference is not a hardware ceiling is
    invalid; the same check against sol_copy is valid."""
    rec = copy.deepcopy(records(ceilings_dir)["copy-128k__sep-nhd.json"])
    gate = {
        "kind": "gate",
        "result": "pass",
        "estimate": 0.93,
        "lo": 0.91,
        "hi": 0.95,
        "threshold": 0.9,
        "against": "sol_copy:sep-nhd@128k",
    }
    rec["verdict"]["checks"] = [gate]
    assert not list(validator().iter_errors(rec))
    rec["verdict"]["checks"] = [dict(gate, against="triton:unified_attention")]
    assert list(validator().iter_errors(rec))
    rec["verdict"]["checks"] = [dict(gate, result="increase")]
    assert list(validator().iter_errors(rec))  # a gate cannot have a claim result


def test_comparison_row_with_label_is_valid(ceilings_dir):
    """Comparison rows (Triton, FA2, ...) are allowed, labelled as such."""
    rec = copy.deepcopy(records(ceilings_dir)["copy-128k__sep-nhd.json"])
    rec["arm"].update(role="comparison", label="comparison, not a target")
    assert not list(validator().iter_errors(rec))


def test_status_requests_rerun_when_too_many_blocks_rejected(tmp_path):
    """ceilings_run.sh asks 'status' after every process: with the GPU above 83 C
    for 7 of 36 rounds an arm keeps only 29 accepted blocks (< 30) and the
    exit code asks for a rerun (3); with 6 hot rounds 30 remain: exit 0."""
    for hot_rounds, expected in ((range(7), 3), (range(6), 0)):
        raw, _ = fake_raw(
            0, {"sep-nhd": 180_000, "cell-nhd": 198_000}, 36, 10_000_000_000, 0
        )
        (tmp_path / "p0.json").write_text(json.dumps(raw))
        hot_seqs = {b["seq"] for b in raw["blocks"] if b["round"] in hot_rounds}
        lines = nvml_lines([raw], hot=lambda b, s=hot_seqs: b["seq"] in s)
        (tmp_path / "nvml.jsonl").write_text("\n".join(lines) + "\n")
        r = subprocess.run(
            [
                sys.executable,
                str(BENCH / "bench_report.py"),
                "status",
                "--raw",
                str(tmp_path / "p0.json"),
                "--nvml",
                str(tmp_path / "nvml.jsonl"),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == expected, r.stdout + r.stderr
        assert json.loads(r.stdout)["rejected"].get("temperature", 0) == 2 * len(
            hot_rounds
        )


def run_report(d, raws, experiment, extra=()):
    """Write raws + clean NVML + env into d and run 'bench_report.py report'."""
    for p, raw in enumerate(raws):
        (d / f"p{p}.json").write_text(json.dumps(raw))
    (d / "nvml.jsonl").write_text("\n".join(nvml_lines(raws)) + "\n")
    env = {
        "git_sha": SHA,
        "git_dirty": False,
        "image_digest": DIGEST,
        "nvcc_version": "13.0.88",
        "driver_version": "580.00",
        "power_limit_w": 220.0,
        "captures": [],
    }
    (d / "env.json").write_text(json.dumps(env))
    subprocess.run(
        [
            sys.executable,
            str(BENCH / "bench_report.py"),
            "report",
            "--raw",
            *[str(d / f"p{p}.json") for p in range(len(raws))],
            "--nvml",
            str(d / "nvml.jsonl"),
            "--env",
            str(d / "env.json"),
            "--session",
            "2026-10-20-ceilings-220w-gpu0",
            "--experiment",
            experiment,
            "--out-dir",
            str(d / "out"),
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return records(d)


def as_mma_ceiling(raw, mode, arms_meta, block_aux=None):
    """Re-dress a fake sol_copy raw file as mma_ceiling.cu writes it."""
    raw = copy.deepcopy(raw)
    raw.update(tool="mma_ceiling", mode=mode, arms=arms_meta)
    raw.pop("shape")
    if mode == "graph-overhead":
        raw.update(timer="cuda_event", duration_kind="graph_replay")
    for b in raw["blocks"]:
        b.pop("graph_ms")
        b.update(block_aux or {})
    return raw


def test_report_mma_mode(tmp_path):
    """mma_ceiling records: 1e11 flop in 2 ms is 50 TFLOPS (TOPS for s8), the
    in-kernel clock and cycles per mma are carried into the summary, and the
    records validate. Locks the ceilings code path that the copy fixture does not
    reach."""
    arms = {"f16f32:4:1:8": 2_000_000, "s8s32:4:1:8": 1_000_000}
    meta = [
        {"name": n, "kind": n.split(":")[0], "flop_per_launch": 1e11, "iters": 1}
        for n in arms
    ]
    aux = {"sm_clock_mhz_median": 1650.0, "cycles_per_mma_per_smsp": 32.5}
    raws = []
    t = 10_000_000_000
    for p in range(3):
        raw, t = fake_raw(p, arms, 36, t, seed=p)
        raws.append(as_mma_ceiling(raw, "mma", meta, aux))
    recs = run_report(tmp_path, raws, "mma-f16f32")
    v = validator()
    for rec in recs.values():
        assert not list(v.iter_errors(rec))
    f16 = recs["mma-f16f32__f16f32_4_1_8.json"]["summary"]
    assert f16["throughput"]["unit"] == "TFLOPS"
    assert f16["throughput"]["lo"] < 50.0 < f16["throughput"]["hi"]
    assert f16["clock"] == {
        "sm_clock_mhz_median": 1650.0,
        "cycles_per_mma_per_smsp_median": 32.5,
    }
    s8 = recs["mma-f16f32__s8s32_4_1_8.json"]["summary"]["throughput"]
    assert s8["unit"] == "TOPS" and s8["lo"] < 100.0 < s8["hi"]


def test_report_graph_overhead_mode(tmp_path):
    """Graph-node cost is the slope between node counts, from paired rounds:
    3 us for 1 node and 259 us for 129 nodes is 2 us per node."""
    arms = {"nodes:1": 3_000, "nodes:129": 259_000}
    meta = [{"name": n, "nodes": int(n.split(":")[1])} for n in arms]
    raws = []
    t = 10_000_000_000
    for p in range(3):
        raw, t = fake_raw(p, arms, 36, t, seed=p)
        raws.append(as_mma_ceiling(raw, "graph-overhead", meta))
    recs = run_report(tmp_path, raws, "graph-overhead")
    rec = recs["graph-overhead__nodes_129.json"]
    assert not list(validator().iter_errors(rec))
    tp = rec["summary"]["throughput"]
    assert tp["unit"] == "us/node" and tp["work_per_launch"] == 128
    assert tp["lo"] < 2.0 < tp["hi"]
    assert rec["timer"] == "cuda_event"
    assert "throughput" not in recs["graph-overhead__nodes_1.json"]["summary"]


def test_comparisons_use_time_per_unit_of_work(tmp_path):
    """mma_ceiling calibrates iterations per arm and process, so arms of one
    experiment do different work. The mma A/A pair of the ceilings runs
    differed by up to 1.55 %
    in iterations and the raw duration ratio showed that as an A/A 'effect'.
    Here an arm doing 3 % more work in 3 % more time has the same throughput:
    the claim must be 'no_claim' with a ratio of ~1, and the A/A MDE must
    reflect timing noise only.

    Negative control: the raw durations of the same blocks differ by 3 %,
    beyond the 2 % claim floor, so a raw-duration claim would say 'increase'."""
    arms = {"f16f32:4:1:8": 2_000_000, "f16f32:4:1:8#aa": 2_031_000, "other": 2_060_000}
    work = {"f16f32:4:1:8": 1e11, "f16f32:4:1:8#aa": 1.0155e11, "other": 1.03e11}
    meta = [
        {"name": n, "kind": "f16f32", "flop_per_launch": work[n], "iters": 1}
        for n in arms
    ]
    raws, t = [], 10_000_000_000
    for p in range(3):
        raw, t = fake_raw(p, arms, 36, t, seed=p)
        raws.append(as_mma_ceiling(raw, "mma", meta))
    recs = run_report(
        tmp_path,
        raws,
        "mma-work",
        extra=(
            "--aa",
            "f16f32:4:1:8",
            "f16f32:4:1:8#aa",
            "--compare",
            "other",
            "f16f32:4:1:8",
        ),
    )
    (claim,) = recs["mma-work__other.json"]["verdict"]["checks"]
    assert claim["result"] == "no_claim"
    assert claim["estimate"] == pytest.approx(1.0, abs=0.003)
    assert claim["mde"] < 0.01
    tp = recs["mma-work__other.json"]["summary"]["throughput"]
    assert tp["lo"] < 50.0 < tp["hi"]  # 1.03e11 flop in 2.06 ms = 50 TFLOPS
    raw_ratio = (
        recs["mma-work__other.json"]["summary"]["median_ns"]
        / recs["mma-work__f16f32_4_1_8.json"]["summary"]["median_ns"]
    )
    assert raw_ratio > 1.02


def test_cluster_robust_report_widens_and_validates(tmp_path):
    """--cluster-robust must only ever widen the throughput interval, say so in
    the CI method, and still produce schema-valid records. One process runs
    1 % slower, as the copy A/A processes of the ceilings runs did
    (+0.36 / -0.04 / +0.55 %): the stratified interval ignores that, the
    cluster-t interval covers it."""
    arms = {"sep-nhd": 180_000, "cell-nhd": 178_000}
    raws, t = [], 10_000_000_000
    for p in range(3):
        raw, t = fake_raw(p, arms, 36, t, seed=p)
        if p == 2:
            for b in raw["blocks"]:
                b["durations_ns"] = [int(d * 1.01) for d in b["durations_ns"]]
        raws.append(raw)
    (tmp_path / "plain").mkdir()
    (tmp_path / "robust").mkdir()
    plain = run_report(tmp_path / "plain", raws, "copy")
    robust = run_report(tmp_path / "robust", raws, "copy", extra=("--cluster-robust",))
    v = validator()
    for name, rec in robust.items():
        assert not list(v.iter_errors(rec)), name
        a = plain[name]["summary"]["throughput"]
        b = rec["summary"]["throughput"]
        assert b["lo"] <= a["lo"] and b["hi"] >= a["hi"]
        assert "cluster-t" in rec["summary"]["median_ci_ns"]["method"]
    a = plain["copy__sep-nhd.json"]["summary"]["throughput"]
    b = robust["copy__sep-nhd.json"]["summary"]["throughput"]
    assert (b["hi"] - b["lo"]) > 5 * (a["hi"] - a["lo"])
