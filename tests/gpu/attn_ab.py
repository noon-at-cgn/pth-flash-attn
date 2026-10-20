# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Interleaved A/B dev timing of attention libraries (kernel tuning).

Why interleaved: the GPU's clock and temperature drift over minutes (220 W
cap), so timing A for a minute and then B for a minute mixes drift into the
difference. Here the libraries alternate A B A B ... (``--rounds`` times each,
``--samples`` graph replays per run, ``attn_dev_timing.time_shape`` as the
timer), and the report gives, per library, the median of every run, the pooled
median and the A/A spread: the relative difference of the library's own runs.
A claimed gain has to exceed 2x the larger A/A spread (the project's rule:
no optimisation without a before/after row above 2x the A/A MDE).

Dev timing, not a benchmark gate: gate numbers come from bench/k_run.sh.

    python tests/gpu/attn_ab.py --libs build/attn build/attn_x --q 1 4 \\
        --B 1 --L 131072 --part attention --rounds 2 --samples 20

``LIB@N`` runs LIB with the C-ABI loader option N (0 shipped default, 1
separate streams, 2 whole cell), so one library can be its own A/B.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TIMER = Path(__file__).with_name("attn_dev_timing.py")


def time_shape(
    lib: Path,
    q: int,
    B: int,
    L: int,
    samples: int,
    part: str,
    loader: int = 0,
    prefill: bool = False,
    ceilings: str = "",
) -> dict:
    """One run of ``attn_dev_timing.py`` in its own process.

    A process per run because two attention libraries cannot share a process:
    they export the same template symbols and static cudart registrations
    (the second library's launch fails with "invalid argument").
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "r.json"
        subprocess.run(
            [sys.executable, str(TIMER), "--lib", str(lib), "--q", str(q),
             "--B", str(B), "--L", str(L), "--samples", str(samples),
             "--part", part, "--loader", str(loader), "--json", str(out),
             *(["--prefill"] if prefill else []),
             *(["--ceilings", ceilings] if ceilings else [])],
            check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        return json.loads(out.read_text())[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--libs", nargs="+", required=True, help="dirs under build/ or .so")
    ap.add_argument("--q", type=int, nargs="+", default=[1, 4, 10])
    ap.add_argument("--B", type=int, nargs="+", default=[1])
    ap.add_argument("--L", type=int, nargs="+", default=[131072])
    ap.add_argument(
        "--part", choices=("all", "attention", "combine"), default="attention"
    )
    ap.add_argument("--rounds", type=int, default=2, help="runs per library")
    ap.add_argument(
        "--prefill",
        action="store_true",
        help="--q are prefill chunks, --L the KV lengths",
    )
    ap.add_argument("--samples", type=int, default=20, help="replays per run")
    ap.add_argument("--ceilings", default="", help="attn_dev_timing.py --ceilings")
    ap.add_argument(
        "--out", default="",
        help="write every run (raw series, clock, power) and the per-library"
        " summary as JSON here (docs/benchmark-methodology.md: raw data)",
    )  # fmt: skip
    args = ap.parse_args(argv)

    libs = []
    for name in args.libs:
        path, _, loader = name.partition("@")
        p = Path(path)
        if p.is_dir() or not p.suffix:
            p = (p if p.is_absolute() else REPO / p) / "libpth_attn.so"
        libs.append((name, p, int(loader or 0)))
    report = []
    for q in args.q:
        for B in args.B:
            for L in args.L:
                runs: dict[str, list[dict]] = {n: [] for n, _, _ in libs}
                for _ in range(args.rounds):
                    for n, lib, loader in libs:
                        runs[n].append(
                            time_shape(
                                lib,
                                q,
                                B,
                                L,
                                args.samples,
                                args.part,
                                loader,
                                args.prefill,
                                args.ceilings,
                            )
                        )
                print(f"q={q} B={B} L={L} part={args.part} n={args.samples}/run "
                      f"x {args.rounds} runs per lib")  # fmt: skip
                base = None
                for n, _, _ in libs:
                    med = [r["median_us"] for r in runs[n]]
                    iqr = [100 * r["iqr_rel"] for r in runs[n]]
                    pooled = float(np.median(med))
                    aa = 100 * (max(med) - min(med)) / pooled if len(med) > 1 else 0.0
                    r0 = runs[n][0]
                    if base is None:
                        base = pooled
                    report.append({
                        "q": q, "B": B, "L": L, "part": args.part, "lib": n,
                        "runs_median_us": med, "pooled_us": pooled, "aa_pct": aa,
                        "d_pct": 100 * (pooled - base) / base,
                        "sm_mhz": [r["sm_mhz"] for r in runs[n]],
                        "power_w": [r["power_w"] for r in runs[n]],
                        "runs": runs[n],
                    })  # fmt: skip
                    print(f"  {n:<28} runs us {' / '.join(f'{m:8.1f}' for m in med)}"
                          f"  IQR% {' / '.join(f'{i:4.2f}' for i in iqr)}"
                          f"  pooled {pooled:8.1f}  A/A {aa:4.2f}%"
                          f"  d {100 * (pooled - base) / base:+6.2f}%"
                          f"  eta_mem {r0['eta_mem'] * r0['median_us'] / pooled:.3f}"
                          f"  eta_mma {r0['eta_mma'] * r0['median_us'] / pooled:.3f}"
                          f"  MHz {np.median([r['sm_mhz'] for r in runs[n]]):.0f}"
                          f"  W {np.median([r['power_w'] for r in runs[n]]):.0f}",
                          flush=True)  # fmt: skip
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ab.json").write_text(
            json.dumps({"argv": sys.argv, "rows": report}, indent=1)
        )
        cols = ("q", "B", "L", "part", "lib", "pooled_us", "aa_pct", "d_pct")
        lines = [",".join(cols + ("runs_median_us", "sm_mhz", "power_w"))]
        for r in report:
            lines.append(
                ",".join(str(r[c]) for c in cols)
                + "," + ",".join(
                    "/".join(f"{x:.4g}" for x in r[k])
                    for k in ("runs_median_us", "sm_mhz", "power_w")
                )
            )  # fmt: skip
        (out / "ab.csv").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
