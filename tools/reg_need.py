#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Register need of each kernel: the smallest register cap without spills.

``ptxas -v`` reports how many registers a kernel *uses*, not how many it
*needs*: under a 255-register budget ptxas spends spare registers on
instruction scheduling, so two variants can report 254 and 217 registers
while needing far fewer. Comparing designs by the reported count is
therefore misleading; comparing by need is not.

This tool compiles the sources repeatedly with ``-D<MACRO>=R`` for caps
R = start, start - step, ... down to ``--stop`` (the source must map the
macro to ``__maxnreg__(R)``; the register probe does,
via ``PTH_PROBE_MAXNREG``), parses ptxas' report with ``ptxas_gate.py`` and
prints, per kernel:

* ``used@start``: registers used at the largest cap;
* ``need``: the smallest cap R such that R and every larger tested cap
  compile with zero spill bytes (resolution = ``--step``; 8 matches the
  register allocation unit, so a finer step cannot change occupancy);
* ``spill@need-step``: the spill bytes one step below the need, the evidence
  that the need is tight.

A kernel that never spills down to ``--stop`` gets ``need <= stop``.

Exit codes: 0 report produced, 2 a compile failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402
import ptxas_gate  # noqa: E402

REG_ALLOC_UNIT = 8  # registers per thread per allocation step (sm_8x)


def sweep(
    nvcc: str,
    sources: list[Path],
    arch: str,
    caps: list[int],
    macro: str,
    extra: list[str],
) -> dict[str, dict[int, ptxas_gate.KernelResources]]:
    """Compiles at every cap; returns {kernel: {cap: resources}}."""
    out: dict[str, dict[int, ptxas_gate.KernelResources]] = {}
    with tempfile.TemporaryDirectory(prefix="reg_need_") as tmp:
        for cap in caps:
            for src in sources:
                kernels, _ = ptxas_gate.compile_and_report(
                    nvcc, src, arch, Path(tmp), [f"-D{macro}={cap}", *extra]
                )
                for k in kernels:
                    out.setdefault(k.name, {})[cap] = k
    return out


def need_of(
    per_cap: dict[int, ptxas_gate.KernelResources],
) -> tuple[int | None, bool]:
    """(need, exact): smallest cap at and above which nothing spills.

    ``need`` is None when even the largest cap spills (need > start);
    ``exact`` is False when no tested cap spilled (need <= smallest cap).
    """
    need = None
    for cap in sorted(per_cap, reverse=True):
        k = per_cap[cap]
        if k.spill_stores or k.spill_loads:
            return need, need is not None
        need = cap
    return need, False


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--arch", default="86")
    ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    ap.add_argument("--macro", default="PTH_PROBE_MAXNREG")
    ap.add_argument("--start", type=int, default=ptxas_gate.MAX_REGISTERS)
    ap.add_argument("--stop", type=int, default=REG_ALLOC_UNIT * REG_ALLOC_UNIT)
    ap.add_argument("--step", type=int, default=REG_ALLOC_UNIT)
    ap.add_argument("--nvcc-flag", action="append", default=[])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    # The first cap is the start itself (255 is not a multiple of 8); then
    # multiples of the step below it.
    first_aligned = (args.start // args.step) * args.step
    if first_aligned == args.start:
        first_aligned -= args.step
    caps = [args.start, *range(first_aligned, args.stop - 1, -args.step)]
    try:
        res = sweep(args.nvcc, args.sources, args.arch, caps, args.macro,
                    args.nvcc_flag)  # fmt: skip
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for name, per_cap in res.items():
        need, exact = need_of(per_cap)
        below = per_cap.get(need - args.step) if need is not None else None
        rows.append(
            {
                "kernel": name,
                "arch": f"sm_{args.arch}",
                "used_at_start": per_cap[caps[0]].registers,
                "spill_at_start": per_cap[caps[0]].spill_stores
                + per_cap[caps[0]].spill_loads,
                "need": need,
                "need_exact": exact,
                "spill_below_need": (below.spill_stores, below.spill_loads)
                if below is not None and exact
                else None,
            }
        )
    print(f"caps tested: {caps}  ({' '.join(build_flags.gencode_flags(args.arch))})")
    print(f"{'kernel':<34} {'used@' + str(caps[0]):>9} {'need':>6}  spill@need-step")
    for r in rows:
        if r["need"] is None:
            need_txt = f">{caps[0]}"
        else:
            need_txt = f"{r['need']}" if r["need_exact"] else f"<={r['need']}"
        sp = r["spill_below_need"]
        sp_txt = f"{sp[0]} B st / {sp[1]} B ld" if sp else "-"
        print(f"{r['kernel']:<34} {r['used_at_start']:>9} {need_txt:>6}  {sp_txt}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"caps": caps, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
