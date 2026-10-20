# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Sweeps around the planner's launch: split count and row tile.

Two sweeps (S-waves and BQ) test the planner's two choices; each must show
the following:

* Split count: the planner picks S = ceil(C_target / (Hkv * row_tiles))
  so that the grid fills the GPU in ONE wave (68 SMs x 1 CTA/SM at D = 256).
  Its revisit condition is "2 waves beats 1 by more than 2 x MDE". The sweep
  therefore times the planner's S next to a quarter, half, 1.5, 2, 3 and 4
  times as many splits (waves of 0.25 .. 4), at the planner's row tile.
* Row tile: the planner takes the fewest warps whose 16 * WM rows hold
  the query tokens (WM = 1, 2, 4; BQ = 16 * WM // G tokens per tile at
  G = 6: 2, 5, 10). The sweep times every nominal WM tile at the same shape,
  each with the split count the planner's formula gives for ITS number of
  row tiles, so the GPU is filled once in every arm and only the tile height
  differs. The planner arm passes the balanced bq = min(planner bq, q)
  (kernel_bench.plan_slice); at row tiles of up to 32 rows the kernel is the
  swap-AB decode kernel, whose cost follows the rows in 8-row n-tiles, above
  that the 16-row-tile kernel (kernel_bench.issued_rows).

One experiment = one decode shape of the production geometry (D = 256,
Hkv = 2, G = 6, block 816) with arms ``planner``, ``planner#aa`` (a copy of
the planner's own launch: the A/A control that sets the MDE) and the
variants. Every arm is a launch of the SAME library over the SAME pool and
random page order. A claim "variant differs from the planner" is made only
beyond 2 x the A/A MDE (stats.evaluate_claim); a sweep finds
what to look at, it is not a gate.

Raw files have the format of the kernel benchmark (``bench/kernel_bench.py``,
``raw_format`` 1), so ``bench/bench_report.py`` makes the records and
``bench/k_sweep.sh`` runs the sweep with the same checks as ``k_run.sh``.
Splits above the planner's workspace cap are legal here because
the sweep allocates its own workspace; if a variant above the cap wins, the
cap (a memory cost) is what the result argues about.

Subcommands: ``list`` (experiment, kind, arms) and ``run``.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "python"))

import kernel_bench as kb  # noqa: E402
from pth_flash_attn import planner  # noqa: E402

# Waves of the planner's grid for the S sweep: the
# planner's own S is wave 1. 0.25 and 0.5 leave SMs idle (the cost of too few
# splits), 1.5 is a deliberately ragged second wave (68 + 34 CTAs on 68
# slots: the straggler case), 2 and above amortise launch and
# tail but cost combine traffic and workspace.
S_WAVES = (0.25, 0.5, 1.5, 2.0, 3.0, 4.0)
# Production decode shapes (q, B, L) per sweep kind. S:
# the L = 128k row of every gate (q, B) plus L = 64k and 256k at q = 1 B = 1
# (does the best S move with L? the planner says never), and the q = 10
# compute-leaning shape. BQ: q = 1 is the negative control (a taller tile
# than the query needs only wastes MMA rows), then the MTP verify q = 4, the
# q = 10 limit of one WM = 4 tile and q = 16 (two row tiles).
S_SHAPES = (
    (1, 1, 65536),
    (1, 1, 131072),
    (1, 1, 262144),
    (1, 2, 131072),
    (4, 1, 131072),
    (4, 2, 131072),
    (10, 1, 65536),
)
BQ_SHAPES = (
    (1, 1, 131072),
    (4, 1, 131072),
    (4, 2, 131072),
    (10, 1, 65536),
    (16, 1, 65536),
)
WM_CHOICES = (1, 2, 4)


@dataclass(frozen=True)
class Sweep:
    """One experiment: a decode shape and which launch parameter varies."""

    name: str
    kind: str  # "S" or "BQ"
    shape: kb.Shape


def sweeps(kind: str = "all") -> list[Sweep]:
    """The experiments in a fixed order. ``selftest``: two tiny experiments on
    the CPU tests' shape (host reference library); never part of ``all``."""
    if kind == "selftest":
        (shape,) = [
            s for s in kb.select("selftest", "quick") if s.name == "self-dec-b2"
        ]
        return [Sweep("self-S", "S", shape), Sweep("self-BQ", "BQ", shape)]
    out: list[Sweep] = []
    for k, shapes in (("S", S_SHAPES), ("BQ", BQ_SHAPES)):
        if kind not in ("all", k):
            continue
        for q, B, L in shapes:
            shape = kb.decode_shape(q, B, L, "quick")
            out.append(Sweep(f"swp-{k}-{shape.name[4:]}", k, shape))
    return out


def tile_bq(shape: kb.Shape, wm: int) -> int:
    """Tokens per row tile that fill 16 * WM rows (bq = 16 WM // G).

    The kernel derives its WM from the ``bq`` the caller passes
    (attn_launch.cu: wm = attn_wm_for_rows(bq * G)), so a WM variant is the
    bq that fills 16 * WM rows; the planner's wm_for_rows must agree or the
    arm's name would describe another tile.
    """
    bq = kb.MMA_ROWS_PER_WARP * wm // shape.G
    if bq < 1 or planner.wm_for_rows(bq * shape.G) != wm:
        raise ValueError(f"WM {wm} has no row tile for G = {shape.G}")
    return bq


def replan(
    shape: kb.Shape,
    base: kb.SlicePlan,
    *,
    bq: int,
    num_splits: int,
    variant_of: kb.VariantOf,
) -> kb.SlicePlan:
    """``base`` with another row tile (tokens) and split count; the kernel
    variant and the executed flops follow from the new tile (kb.slice_plan)."""
    return kb.slice_plan(
        shape,
        base.reqs,
        base.is_prefill,
        bq=bq,
        num_splits=num_splits,
        variant_of=variant_of,
    )


def sweep_arms(
    sw: Sweep, num_sms: int, variant_of: kb.VariantOf = kb.modelled_variant
) -> list[kb.ArmPlan]:
    """The arms of one experiment: planner, its A/A copy, the variants.

    The planner arm uses the balanced row tile (kb.plan_slice). A ``wm``
    arm keeps its nominal tile of 16 * WM rows even where the request has
    fewer tokens (bq > q): that is the unbalanced launch, and at q = 1 it is
    the negative control (more n-tiles of the decode kernel, or the 16-row-tile
    kernel, for the same bytes). An arm whose tile equals the planner's is
    not added (it would be a second A/A copy)."""
    shape = sw.shape
    (base,) = kb.plan_shape(shape, num_sms, variant_of)  # a decode shape is one slice
    c_target = planner.c_target(D=shape.D, num_sms=num_sms)
    planner_plans = (base,)
    arms = [
        kb.ArmPlan("planner", "subject", "kernel", planner_plans),
        kb.ArmPlan("planner#aa", "aa_control", "kernel", planner_plans),
    ]
    if sw.kind == "S":
        seen = {base.num_splits}
        for w in S_WAVES:
            s = max(1, math.ceil(w * base.num_splits))  # ceil, as the planner
            if s in seen:
                continue
            seen.add(s)
            arms.append(
                kb.ArmPlan(
                    f"S{s}",
                    "subject",
                    "kernel",
                    (
                        replan(
                            shape,
                            base,
                            bq=base.bq,
                            num_splits=s,
                            variant_of=variant_of,
                        ),
                    ),
                )
            )
    else:
        for wm in WM_CHOICES:
            try:
                bq = tile_bq(shape, wm)
            except ValueError:
                continue
            if bq == base.bq:
                continue
            row_tiles = -(-max(shape.q_lens) // bq)
            # The planner's formula for this tile count, without its
            # workspace cap: one wave in every arm.
            s = max(1, -(-c_target // (shape.Hkv * row_tiles)))
            arms.append(
                kb.ArmPlan(
                    f"wm{wm}",
                    "subject",
                    "kernel",
                    (
                        replan(
                            shape,
                            base,
                            bq=bq,
                            num_splits=s,
                            variant_of=variant_of,
                        ),
                    ),
                )
            )
    return arms


def cmd_list(a: argparse.Namespace) -> int:
    for sw in sweeps(a.kind):
        names = ",".join(ap.name for ap in sweep_arms(sw, a.num_sms))
        print(sw.name, sw.kind, names)
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    only = {s for s in a.only.split(",") if s}
    chosen = [s for s in sweeps(a.kind) if not only or s.name in only]
    unknown = only - {s.name for s in chosen}
    if unknown:
        raise SystemExit(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    # run_experiment names the raw file after the shape; the sweeps' names
    # differ (two kinds share a shape), so give each Shape its sweep's name.
    shapes = [dataclasses.replace(s.shape, name=s.name) for s in chosen]
    by_name = {s.name: s for s in chosen}
    return kb.cmd_run(
        a,
        shapes=shapes,
        arm_plans_for=lambda shape, num_sms, variant_of: sweep_arms(
            by_name[shape.name], num_sms, variant_of
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=__doc__
        + "\n`list` needs no GPU. `run` needs one NVIDIA GPU (sm80/86/89), a built "
        "kernel library and CUPTI; bench/k_sweep.sh is the driver that wraps it "
        "in a full measurement run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print experiment, kind and arms (no GPU)")
    ls.add_argument("--num-sms", type=int, required=True, help="the card's SM count")
    r = sub.add_parser("run", help="time the experiments (needs a GPU)")
    kb.add_run_args(r)
    for p in (ls, r):
        p.add_argument("--kind", default="all", choices=("all", "S", "BQ", "selftest"))
    a = ap.parse_args(argv)
    a.argv = ["bench/k_sweep.py", *argv]
    return cmd_list(a) if a.cmd == "list" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
