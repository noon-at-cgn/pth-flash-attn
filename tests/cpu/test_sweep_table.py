# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for bench/sweep_table.py, the side-by-side view of a
sweep result directory.

The table must say exactly what the records say; the mistakes it could add
are reading order (planner first, A/A copy next, then variants by split
count and tile height, not alphabetically: 'S102' sorts before 'S17') and
the direction of a ratio (below 1 = this arm is faster).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sweep_table = importlib.import_module("sweep_table")


def _rec(exp, name, median_ns, *, s=None, wm=None, claim=None):
    """The fields sweep_table reads, as bench_report.py writes them."""
    params = {"slices": [{"num_splits": s, "wm": wm}]} if s is not None else {}
    checks = (
        [
            {
                "kind": "claim",
                "result": claim[1],
                "estimate": claim[0],
                "lo": claim[0] - 0.01,
                "hi": claim[0] + 0.01,
                "against": "planner",
                "margin": 0.02,
            }
        ]
        if claim
        else []
    )
    return {
        "record": "arm_result",
        "experiment": exp,
        "arm": {"name": name, "params": params},
        "summary": {
            "median_ns": median_ns,
            "iqr_over_median": 0.01,
            "n_accepted": 214,
            "n_blocks": 216,
            "throughput": {"estimate": 100.0, "lo": 99.0, "hi": 101.0},
        },
        "verdict": {"checks": checks},
    }


def _records():
    return [
        _rec("e1", "S102", 90_000, s=102, wm=1, claim=(0.9, "decrease")),
        _rec("e1", "S17", 120_000, s=17, wm=1, claim=(1.2, "increase")),
        _rec("e1", "planner#aa", 100_100, s=34, wm=1),
        _rec("e1", "planner", 100_000, s=34, wm=1),
    ]


def test_planner_first_then_aa_then_variants_by_split_count():
    table = sweep_table.format_table(_records())
    rows = [ln.split()[0] for ln in table.splitlines()[2:] if ln.strip()]
    assert rows == ["planner", "planner#aa", "S17", "S102"]


def test_claims_print_the_ratio_the_record_holds_and_its_verdict():
    """Ratio = time(arm)/time(planner): S102 is faster (x0.9, decrease)."""
    table = sweep_table.format_table(_records())
    line = next(ln for ln in table.splitlines() if ln.startswith("S102"))
    assert "planner: x0.900" in line and "decrease" in line and "margin 2.0%" in line
    assert "90.00" in line  # median in microseconds
    assert "214/216" in line


def test_ceiling_arms_have_no_launch_columns():
    """sol_copy arms carry no slices: the S and wm cells stay empty."""
    table = sweep_table.format_table([_rec("c1", "sep-nhd", 195_000)])
    line = next(ln for ln in table.splitlines() if ln.startswith("sep-nhd"))
    assert "195.00" in line


def test_load_records_ignores_other_json(tmp_path):
    """A result directory also holds env.json, MANIFEST and raw/: only
    <experiment>__<arm>.json arm records are tables."""
    (tmp_path / "env.json").write_text("{}")
    (tmp_path / "e1__planner.json").write_text(json.dumps(_rec("e1", "planner", 1.0)))
    (tmp_path / "e1__other.json").write_text(json.dumps({"record": "something"}))
    assert [r["arm"]["name"] for r in sweep_table.load_records(tmp_path)] == ["planner"]


def test_main_exit_codes(tmp_path, capsys):
    assert sweep_table.main([]) == 64
    assert sweep_table.main([str(tmp_path)]) == 1  # nothing to show
    (tmp_path / "e1__planner.json").write_text(json.dumps(_rec("e1", "planner", 1e5)))
    assert sweep_table.main([str(tmp_path)]) == 0
    assert "== e1" in capsys.readouterr().out
