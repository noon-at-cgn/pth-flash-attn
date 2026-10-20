# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""One table per experiment from the per-arm records of a result directory.

``bench/k_sweep.sh`` and ``bench/k_run.sh`` write one ``arm_result`` record
per arm (``bench/bench_report.py``); this prints them side by side, so a sweep
can be read on one screen instead of in SUMMARY.txt.
Nothing is computed here that the records do not already hold: the table is
the median, its cluster-aware CI, IQR/median, the accepted blocks, and each
claim exactly as ``stats.evaluate_claim`` decided it.

Columns: ``arm``; ``S wm`` (split count and row-tile warps from the arm's
launch metadata, kernel sweeps only); ``us`` median duration; ``GB/s`` with
its 95 % CI (bytes of KV read once per launch / duration); ``IQR`` as a share
of the median; ``n`` accepted of all blocks; ``vs``: for a claim, the arm it
is compared with, the time ratio arm/against with its CI, and the verdict
(``increase`` / ``decrease`` / ``no_claim`` / ``noisy``). A ratio below 1
means this arm is faster. ``margin`` is the larger of 2 x the A/A MDE and the
2 % floor: a claim exists only beyond it.

Usage: ``python3 bench/sweep_table.py results/<result dir>``. Reads files only:
no GPU is needed. Prints the tables to stdout; exit 1 if the directory holds
no arm records.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def load_records(res_dir: Path) -> list[dict[str, Any]]:
    """Every arm record of a result directory (files ``<exp>__<arm>.json``)."""
    out = []
    for p in sorted(res_dir.glob("*__*.json")):
        rec = json.loads(p.read_text())
        if rec.get("record") == "arm_result":
            out.append(rec)
    return out


def _launch(rec: dict[str, Any]) -> tuple[int | None, int | None]:
    """(S, WM) of the arm's first slice, or Nones for a ceiling arm."""
    slices = rec["arm"].get("params", {}).get("slices") or []
    if not slices:
        return None, None
    return slices[0].get("num_splits"), slices[0].get("wm")


def _order(rec: dict[str, Any]) -> tuple[int, int, int, str]:
    """planner (or the first ceiling) first, its A/A copy, then by (S, WM)."""
    name = rec["arm"]["name"]
    s, wm = _launch(rec)
    rank = 1 if name.endswith("#aa") else 0 if name == "planner" else 2
    return rank, s or 0, wm or 0, name


def format_table(records: Iterable[dict[str, Any]]) -> str:
    """The tables of every experiment, in the order of its first record."""
    by_exp: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_exp.setdefault(rec["experiment"], []).append(rec)
    lines: list[str] = []
    for exp, recs in by_exp.items():
        recs.sort(key=_order)
        lines.append(f"== {exp}")
        lines.append(
            f"{'arm':<14}{'S':>4}{'wm':>3}{'us':>10}{'GB/s':>8} {'95% CI':<15}"
            f"{'IQR':>7}{'n':>9}  vs"
        )
        for rec in recs:
            s, wm = _launch(rec)
            sm = rec["summary"]
            tp = sm["throughput"]
            checks = [c for c in rec["verdict"]["checks"] if c["kind"] == "claim"]
            claim = "; ".join(
                f"{c['against']}: x{c['estimate']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
                f" {c['result']} (margin {100 * c['margin']:.1f}%)"
                for c in checks
            )
            ci = f"[{tp['lo']:.1f}, {tp['hi']:.1f}]"
            lines.append(
                f"{rec['arm']['name']:<14}"
                f"{'' if s is None else s:>4}{'' if wm is None else wm:>3}"
                f"{sm['median_ns'] / 1e3:>10.2f}{tp['estimate']:>8.1f} {ci:<15}"
                f"{100 * sm['iqr_over_median']:>6.2f}%"
                f"{sm['n_accepted']:>5}/{sm['n_blocks']:<4} {claim}".rstrip()
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if len(args) != 1:
        print(__doc__.splitlines()[0], file=sys.stderr)
        print("usage: sweep_table.py RESULTS_DIR", file=sys.stderr)
        return 64
    records = load_records(Path(args[0]))
    if not records:
        print(f"no arm records in {args[0]}", file=sys.stderr)
        return 1
    print(format_table(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
