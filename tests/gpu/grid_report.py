# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Summarise one grid results directory into grid_report.json and SUMMARY.txt.

Inputs (written by tests/gpu/run_grid.sh):

* ``steps.tsv``: one row per step: name, exit code, expected exit code,
  seconds, number of tests in the step's JUnit file (``-`` if none);
* ``junit/<step>.xml``: pytest's JUnit XML (failures/errors counts);
* ``records/<step>.jsonl``: one JSON line per checked case (tests/gpu
  ``--grid-record``): grid cases (``test == "reference_grid"``) with
  violations, tightness and canary results; loader, graph and production
  records with ``passed``;
* ``logs/san-*.log``: compute-sanitizer output (error summary line).

Output: ``grid_report.json`` (validated against tests/gpu/grid_report.schema.json
before it is written) and ``SUMMARY.txt``. Exit 0 if the verdict is
``pass``, 1 if ``fail``, 2 if the directory is unusable.

The verdict is ``pass`` only if every step exited as expected, every pytest
step ran at least one test, every case of the grid
(tests/gpu/grid_cases.py) has a record and every record passed. "No tests
ran" is a failure, never a pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = Path(__file__).with_name("grid_report.schema.json")
sys.path.insert(0, str(ROOT / "tests" / "gpu"))

# Steps that run pytest (must have run >= 1 test); the others are scripts.
PYTEST_STEP = re.compile(r"^(quick|full|graph|san-.+|ctl-.+)$")
GRID_TEST = "reference_grid"
# "ERROR SUMMARY: 66 errors" (memcheck, synccheck, initcheck; a second line
# "ERROR SUMMARY: 63 errors were not printed" must not count) and
# "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)".
_SAN = re.compile(
    r"(?:ERROR SUMMARY: (\d+) errors?$|\((\d+) errors, \d+ warnings\))", re.MULTILINE
)
# Failures and missing cases listed by name in SUMMARY.txt (the JSON has all).
SUMMARY_LIST = 20


def expected_grid_cases() -> list[tuple[str, str]]:
    """(name, family) of every grid case, from the grid module itself."""
    import grid_cases

    return [(c.name, c.family) for c in grid_cases.cases("full")]


def _num(x: float) -> float | str:
    # JSON has no inf/nan; keep them visible as strings.
    return x if math.isfinite(x) else str(x)


def spread(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "median": None, "max": None}
    v = sorted(values)
    fin = [x for x in v if math.isfinite(x)]
    return {
        "min": fin[0] if fin else None,
        "median": statistics.median(fin) if fin else None,
        "max": _num(v[-1]),
    }


def read_steps(res: Path) -> list[dict[str, Any]]:
    path = res / "steps.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    rows = []
    for line in path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, rc, expect, secs, tests = line.split("\t")
        row: dict[str, Any] = {
            "name": name,
            "exit_code": int(rc),
            "expected_exit": int(expect),
            "seconds": int(secs),
            "tests": None if tests == "-" else int(tests),
            "failures": None,
            "sanitizer_errors": None,
        }
        junit = res / "junit" / f"{name}.xml"
        if junit.exists():
            suite = ET.parse(junit).getroot()
            if suite.tag == "testsuites":
                suite = suite[0]
            row["failures"] = int(suite.get("failures", 0)) + int(
                suite.get("errors", 0)
            )
        log = res / "logs" / f"{name}.log"
        if name.startswith(("san-", "ctl-")) and log.exists():
            m = _SAN.findall(log.read_text(errors="replace"))
            if m:
                row["sanitizer_errors"] = sum(int(a or b) for a, b in m)
        row["ok"] = row["exit_code"] == row["expected_exit"] and (
            not PYTEST_STEP.match(name) or bool(row["tests"])
        )
        rows.append(row)
    return rows


def read_records(res: Path) -> list[dict[str, Any]]:
    recs = []
    for path in sorted((res / "records").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_step"] = path.stem
                recs.append(r)
    return recs


def build_report(res: Path, grid: Sequence[tuple[str, str]]) -> dict[str, Any]:
    steps = read_steps(res)
    recs = read_records(res)
    reasons: list[str] = []
    for s in steps:
        if s["exit_code"] != s["expected_exit"]:
            reasons.append(
                f"step {s['name']} exited {s['exit_code']} "
                f"(expected {s['expected_exit']})"
            )
        if PYTEST_STEP.match(s["name"]) and not s["tests"]:
            reasons.append(f"step {s['name']} ran no tests")

    grid_recs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    other: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"recorded": 0, "failed": []}
    )
    device = None
    for r in recs:
        device = device or r.get("device")
        if r.get("test") == GRID_TEST:
            grid_recs[r["case"]].append(r)
        else:
            o = other[str(r.get("test"))]
            o["recorded"] += 1
            if not r.get("passed"):
                o["failed"].append(str(r.get("case")))

    names = [n for n, _ in grid]
    family = dict(grid)
    missing = [n for n in names if n not in grid_recs]
    unknown = sorted(set(grid_recs) - set(names))
    failed, tight, by_fam = [], [], defaultdict(list)
    worst: dict[str, Any] | None = None
    for name in names:
        rs = grid_recs.get(name)
        if not rs:
            continue
        ok = all(r["passed"] for r in rs)
        t = max(float(r["tightness"]) for r in rs)
        tight.append(t)
        by_fam[family[name]].append((ok, t))
        if worst is None or t > float(worst["tightness"]):
            w = max(rs, key=lambda r: float(r["tightness"]))
            worst = {"case": name, "tightness": _num(t), "worst": w.get("worst", {})}
        if not ok:
            bad = next(r for r in rs if not r["passed"])
            failed.append(
                {
                    "case": name,
                    "violations": int(bad["violations"]),
                    "nonfinite": int(bad.get("nonfinite", 0)),
                    "tightness": _num(float(bad["tightness"])),
                    "canary_out_ok": bool(bad["canary_out_ok"]),
                    "canary_ws_ok": bad.get("canary_ws_ok"),
                    "worst": bad.get("worst", {}),
                }
            )
    if missing:
        reasons.append(f"{len(missing)} of {len(names)} grid cases have no record")
    if failed:
        reasons.append(f"{len(failed)} grid cases failed the gate")
    if unknown:
        reasons.append(
            f"records for {len(unknown)} cases not in the grid: {unknown[:3]}"
        )
    for test, o in sorted(other.items()):
        if o["failed"]:
            reasons.append(f"{len(o['failed'])} {test} records failed")

    report = {
        "schema_version": 1,
        "record": "grid_report",
        "session": res.name,
        "verdict": "fail" if reasons else "pass",
        "reasons": reasons,
        "steps": steps,
        "grid": {
            "cases_expected": len(names),
            "cases_recorded": len(names) - len(missing),
            "cases_passed": len(names) - len(missing) - len(failed),
            "failed": failed,
            "missing": missing,
            "tightness": spread(tight),
            "by_family": {
                f: {
                    "cases": len(v),
                    "passed": sum(ok for ok, _ in v),
                    "tightness": spread([t for _, t in v]),
                }
                for f, v in sorted(by_fam.items())
            },
            "worst_case": worst,
        },
        "other_tests": dict(sorted(other.items())),
        "device": device,
    }
    jsonschema.validate(report, json.loads(SCHEMA.read_text()))
    return report


def _fmt(x: Any) -> str:
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def summary_text(rep: dict[str, Any]) -> str:
    g = rep["grid"]
    lines = [
        f"GRID {rep['session']}: {rep['verdict'].upper()}",
        *(f"  reason: {r}" for r in rep["reasons"]),
        "",
        "steps (name, exit/expected, seconds, tests, failures, sanitizer errors):",
    ]
    for s in rep["steps"]:
        lines.append(
            f"  {'ok  ' if s['ok'] else 'FAIL'} {s['name']:<15} "
            f"{s['exit_code']}/{s['expected_exit']}  {s['seconds']:>5} s  "
            f"tests {_fmt(s['tests'])}  failures {_fmt(s['failures'])}  "
            f"sanitizer {_fmt(s['sanitizer_errors'])}"
        )
    t = g["tightness"]
    lines += [
        "",
        (
            f"grid: {g['cases_passed']} passed / {g['cases_recorded']} recorded / "
            f"{g['cases_expected']} expected; tightness max(err/bound) min "
            f"{_fmt(t['min'])} median {_fmt(t['median'])} max {_fmt(t['max'])}"
        ),
    ]
    for f, v in g["by_family"].items():
        ft = v["tightness"]
        lines.append(
            f"  {f:<11} {v['passed']}/{v['cases']} passed, tightness median "
            f"{_fmt(ft['median'])} max {_fmt(ft['max'])}"
        )
    if g["worst_case"]:
        w = g["worst_case"]
        lines.append(
            f"  worst: {w['case']} tightness {_fmt(w['tightness'])} "
            f"at {w['worst'].get('index')}"
        )
    for f in g["failed"][:SUMMARY_LIST]:
        lines.append(
            f"  FAILED {f['case']}: {f['violations']} violations, tightness "
            f"{_fmt(f['tightness'])}, canary out {f['canary_out_ok']}, "
            f"workspace {f['canary_ws_ok']}, worst {f['worst']}"
        )
    for m in g["missing"][:SUMMARY_LIST]:
        lines.append(f"  MISSING {m}")
    for test, o in rep["other_tests"].items():
        lines.append(
            f"{test}: {o['recorded']} records, {len(o['failed'])} failed "
            f"{o['failed'][:SUMMARY_LIST]}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__ + "\nReads files only: no GPU is needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "results_dir",
        help="a tests/gpu/run_grid.sh result directory (read, then extended)",
    )
    a = ap.parse_args(argv)
    res = Path(a.results_dir)
    try:
        rep = build_report(res, expected_grid_cases())
    except (FileNotFoundError, ValueError) as e:
        print(f"grid_report: {e}", file=sys.stderr)
        return 2
    (res / "grid_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    text = summary_text(rep)
    (res / "SUMMARY.txt").write_text(text)
    print(text, end="")
    return 0 if rep["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
