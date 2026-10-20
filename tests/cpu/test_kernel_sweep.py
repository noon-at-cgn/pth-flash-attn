# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the sweeps (bench/k_sweep.py).

The sweeps decide two planner choices by measurement: the split count (is one
wave of splits right?) and the row tile (does the planner's row tile
beat the alternatives?). What
could mislead is not the timing but the *design*: arms that do not differ in
what the sweep says they differ in, or a flop model that describes another
launch than the one timed. Checked here on CPU:

* the S arms are waves of the planner's grid (same row tile, same bytes, same
  executed flops; only the split count moves) and the grid size of every arm
  is what its name claims;
* every WM arm fills the GPU once, with the split count the planner's own
  formula gives for ITS row-tile count, and its flop model is the brute-force
  executed_flops of its own tile;
* the A/A copy shares the planner's launch object, so the MDE measures noise
  of one launch, not of two input sets;
* a row-tile height the group size cannot fill is refused rather than timed;
* the whole CPU pipeline: k_sweep run on the host reference library -> raw
  files -> bench_report.py records with claims against ``planner`` that validate
  against results.schema.json -> the table.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ks = importlib.import_module("k_sweep")
kb = importlib.import_module("kernel_bench")
bench_report = importlib.import_module("bench_report")
sweep_table = importlib.import_module("sweep_table")
test_kernel_bench = importlib.import_module("test_kernel_bench")

NUM_SMS = 68


def _arms(name: str) -> dict[str, kb.ArmPlan]:
    (sw,) = [s for s in ks.sweeps() if s.name == name]
    return {a.name: a for a in ks.sweep_arms(sw, NUM_SMS)}


def _ctas(shape: kb.Shape, p: kb.SlicePlan) -> int:
    """Grid size: request row tiles x KV heads x splits."""
    q = max(shape.q_lens[r] for r in p.reqs)
    return len(p.reqs) * -(-q // p.bq) * shape.Hkv * p.num_splits


def test_sweep_experiments_are_unique_decode_shapes():
    """Names are the raw-file names, so they must be unique; every sweep is a
    one-slice decode shape (the planner call in sweep_arms unpacks one)."""
    sw = ks.sweeps()
    assert len({s.name for s in sw}) == len(sw)
    assert {s.kind for s in sw} == {"S", "BQ"}
    assert all(s.shape.family == "decode" and len(s.shape.slices) == 1 for s in sw)
    assert [s.name for s in ks.sweeps("BQ")][0] == "swp-BQ-q1-b1-L128k"  # control


def test_s_arms_are_waves_of_the_planners_grid():
    """q = 1, B = 1, L = 128k: the planner's S = 34 gives 68 CTAs = one wave
    of 68 SMs x 1 CTA/SM. The variants are 0.25 .. 4 waves; they keep the
    planner's row tile, so bytes and executed flops cannot differ and a
    timing difference is the split count."""
    arms = _arms("swp-S-q1-b1-L128k")
    assert list(arms) == [
        "planner",
        "planner#aa",
        "S9",
        "S17",
        "S51",
        "S68",
        "S102",
        "S136",
    ]
    shape = ks.sweeps()[1].shape
    (base,) = arms["planner"].plans
    assert base.num_splits == 34 and _ctas(shape, base) == 68
    for name, waves in (("S9", 0.25), ("S17", 0.5), ("S68", 2), ("S136", 4)):
        (p,) = arms[name].plans
        assert (p.wm, p.bq) == (base.wm, base.bq)
        assert p.bytes_kv == base.bytes_kv and p.flops_executed == base.flops_executed
        assert _ctas(shape, p) / 68 == pytest.approx(waves, abs=0.03)


def test_wm_arms_fill_one_wave_with_their_own_split_count():
    """q = 4 (G = 6): 24 rows, planner WM = 2, balanced BQ = 4, one row tile,
    S = 34, the decode kernel with three n-tiles. WM 1 holds 2 tokens, so 2
    row tiles and S = ceil(68 / (2 x 2)) = 17 (decode kernel, 12 rows -> 16
    issued); WM 2 is the planner's nominal BQ = 5 (30 rows, four n-tiles);
    WM 4 holds 10 tokens (60 rows, the 16-row-tile kernel). Every arm is 68
    CTAs, so only the tile differs, and the wasted rows show in the flops."""
    arms = _arms("swp-BQ-q4-b1-L128k")
    assert list(arms) == ["planner", "planner#aa", "wm1", "wm2", "wm4"]
    shape = [s for s in ks.sweeps() if s.name == "swp-BQ-q4-b1-L128k"][0].shape
    (base,) = arms["planner"].plans
    assert (base.wm, base.bq, base.num_splits) == (2, 4, 34)
    assert base.kernel == kb.KERNEL_DECODE
    want = {
        "wm1": (1, 2, 17, kb.KERNEL_DECODE, 16),
        "wm2": (2, 5, 34, kb.KERNEL_DECODE, 32),
        "wm4": (4, 10, 34, kb.KERNEL_TILE16, 64),
    }
    for name, (wm, bq, splits, kernel, issued) in want.items():
        (p,) = arms[name].plans
        assert (p.wm, p.bq, p.num_splits, p.kernel) == (wm, bq, splits, kernel)
        assert _ctas(shape, p) == 68
        assert p.flops_executed == sum(
            kb.executed_flops(q, L, shape.Hkv, shape.D, bq, issued)
            for q, L in zip(shape.q_lens, shape.seq_lens, strict=True)
        )
    assert arms["wm4"].plans[0].flops_executed > 1.9 * base.flops_executed


def test_q1_control_wastes_rows_not_bytes():
    """At q = 1 only one token needs 6 of the tile's rows: a taller tile
    cannot be faster except by accident, and it must execute more flops. The
    planner arm is the balanced bq = 1 (8 issued rows); the arms keep their
    nominal tiles bq = 2, 5, 10 (16, 32 and 64 issued rows: 2x, 4x, 8x). The
    control row exists to show a sweep that can say 'no'."""
    arms = _arms("swp-BQ-q1-b1-L128k")
    assert list(arms) == ["planner", "planner#aa", "wm1", "wm2", "wm4"]
    base = arms["planner"].plans[0]
    assert (base.wm, base.bq, base.kernel) == (1, 1, kb.KERNEL_DECODE)
    for name, ratio in (("wm1", 2), ("wm2", 4), ("wm4", 8)):
        p = arms[name].plans[0]
        assert p.bytes_kv == base.bytes_kv
        assert p.flops_executed == pytest.approx(base.flops_executed * ratio, rel=0.01)


def test_aa_copy_shares_the_planners_launch_object():
    """kb.run_experiment builds one set of input buffers per distinct plans
    object. The A/A copy must be the SAME object, or the MDE would include
    the difference between two output buffers and two workspaces."""
    arms = _arms("swp-S-q1-b1-L128k")
    assert arms["planner#aa"].plans is arms["planner"].plans
    assert arms["planner#aa"].role == "aa_control"
    assert arms["planner"].role == arms["S17"].role == "subject"
    assert arms["S17"].plans is not arms["planner"].plans


def test_gate_arms_still_share_their_launch():
    """Arms that carry plans must not change the gate
    benchmark: kernel, kernel#aa and ref share one plans tuple (buffers), the
    reference is labelled a comparison."""
    shape = next(s for s in kb.grid("decode") if s.name == "dec-q1-b1-L1k")
    (p,) = kb.plan_shape(shape, NUM_SMS)

    class _A:
        ref = True

    arms = kb.default_arm_plans(shape, _A(), [p])
    assert [a.name for a in arms] == ["kernel", "kernel#aa", "ref"]
    assert arms[0].plans is arms[1].plans is arms[2].plans
    assert [a.role for a in arms] == ["subject", "aa_control", "comparison"]
    assert [a.lib for a in arms] == ["kernel", "kernel", "ref"]


def test_replan_refuses_a_tile_the_group_cannot_fill():
    """G = 64 query heads per KV head: WM 1 holds 16 rows = 0 whole tokens.
    There is no such launch (bq >= 1), so the sweep must not invent one."""
    shape = kb.Shape(
        "x", "decode", (1,), (99,), (kb.Slice((0,), False),), "quick", G=64
    )
    with pytest.raises(ValueError, match="no row tile"):
        ks.tile_bq(shape, 1)


def test_waves_round_up_like_the_planner():
    """S = ceil(w x S_planner): 0.25 x 34 = 8.5 -> 9, never the banker's
    rounding of Python's round() (8), which would put the quarter wave off
    its name by a split."""
    assert ks.sweeps("S")
    assert math.ceil(0.25 * 34) == 9 and round(0.25 * 34) == 8


def test_cpu_pipeline_end_to_end(tmp_path):
    """k_sweep run on the host reference library -> raw files -> records with
    ``--compare VARIANT planner`` claims that validate against the schema ->
    the table lists every arm with its claim. Why: the GPU run must not
    be the first time these pieces meet."""
    raw_dir = tmp_path / "raw"
    for p in (0, 1):
        assert (
            ks.main(
                [
                    "run",
                    "--kind",
                    "selftest",
                    "--only",
                    "self-BQ",
                    "--process",
                    str(p),
                    "--out-dir",
                    str(raw_dir),
                    "--device",
                    "cpu",
                    "--kernel-kind",
                    "ref",
                    "--num-sms",
                    "4",
                    "--rounds",
                    "6",
                    "--replays",
                    "1",
                    "--warmup-s",
                    "0",
                ]
            )
            == 0
        )
    assert sorted(f.name for f in raw_dir.glob("*.json")) == [
        "self-BQ.p0.a0.json",
        "self-BQ.p1.a0.json",
    ]
    raws = {"self-BQ": sorted(str(f) for f in raw_dir.glob("self-BQ.p*.json"))}
    loaded = json.loads(Path(raws["self-BQ"][0]).read_text())
    names = [a["name"] for a in loaded["arms"]]
    assert names[:2] == ["planner", "planner#aa"] and len(names) > 2
    test_kernel_bench._nvml_log(
        tmp_path / "nvml.jsonl",
        [json.loads(Path(f).read_text()) for f in raws["self-BQ"]],
    )
    env = tmp_path / "env.json"
    env.write_text(
        json.dumps(
            {
                "git_sha": "0" * 40,
                "git_dirty": False,
                "image_digest": "sha256:" + "0" * 64,
                "nvcc_version": "13.0.88",
                "driver_version": "x",
                "power_limit_w": None,
                "captures": [],
            }
        )
    )
    out = tmp_path / "res"
    compare = [x for n in names[2:] for x in ("--compare", n, "planner")]
    assert (
        bench_report._main(
            [
                "report",
                "--raw",
                *raws["self-BQ"],
                "--nvml",
                str(tmp_path / "nvml.jsonl"),
                "--env",
                str(env),
                "--session",
                "2026-10-20-sweep-220w-gpu0",
                "--experiment",
                "self-BQ",
                "--out-dir",
                str(out),
                "--aa",
                "planner",
                "planner#aa",
                *compare,
                "--min-accepted",
                "1",
                "--cluster-robust",
            ]
        )
        == 0
    )
    schema = json.loads((ROOT / "bench" / "results.schema.json").read_text())
    recs = sweep_table.load_records(out)
    assert {r["arm"]["name"] for r in recs} == set(names)
    for r in recs:
        jsonschema.validate(r, schema)
    variant = next(r for r in recs if r["arm"]["name"] == names[2])
    (claim,) = [c for c in variant["verdict"]["checks"] if c["kind"] == "claim"]
    assert claim["against"] == "planner"
    table = sweep_table.format_table(recs)
    assert "== self-BQ" in table and names[2] in table and "vs" in table
