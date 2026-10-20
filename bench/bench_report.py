# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Turn raw ceiling-benchmark output into schema-valid results.

Inputs: the raw JSON written by ``sol_copy`` / ``mma_ceiling`` (one file per
fresh process), the NVML log from ``nvml_log.py`` and the environment capture
``env.json`` written by ``ceilings_run.sh``. Output: one ``results.schema.json``
record per arm. All statistics come from ``stats.py``; this file only parses,
joins and serialises.

Subcommands:

``status``  one process just finished: apply the rejection rules to its
            blocks and say whether it kept >= 30 accepted blocks per arm.
            Exit 0 = ok, 3 = rerun (ceilings_run.sh reruns at most 3 times).
``report``  all processes of one experiment: per-arm records with median, IQR,
            n, rejection counts, a bootstrap CI of the median, the throughput
            (GB/s, TFLOPS/TOPS or us/node) and any requested claims.

Roles: every ceilings arm is a hardware ceiling except the second arm of an A/A
pair (``--aa X Y``), which is the noise control. Claims between arms
(``--compare X Y``, repeatable) need an A/A pair in the same experiment,
because the MDE that sets the claim margin must come from the same design
and shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvml_log  # noqa: E402
import stats  # noqa: E402

EXIT_RERUN = 3  # shared with ceilings_run.sh
SCHEMA = Path(__file__).with_name("results.schema.json")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("raw_format") != 1:
        raise ValueError(f"{path}: unknown raw_format {raw.get('raw_format')}")
    return raw


def blocks_from_raw(raw: dict[str, Any]) -> list[stats.Block]:
    proc = int(raw["process"])
    return [
        stats.Block(
            process=proc,
            round=int(b["round"]),
            seq=int(b["seq"]),
            arm=str(b["arm"]),
            t_start_ns=int(b["t_start_mono_ns"]),
            t_end_ns=int(b["t_end_mono_ns"]),
            durations_ns=tuple(float(d) for d in b["durations_ns"]),
        )
        for b in raw["blocks"]
    ]


def work_per_launch(raw: dict[str, Any], arm: str) -> float | None:
    """Bytes (sol_copy) or flop/ops (mma_ceiling) one launch of ``arm`` does
    in this raw file's process; None for graph-overhead arms."""
    meta = next(m for m in raw["arms"] if m["name"] == arm)
    for key in ("bytes_per_launch", "flop_per_launch"):
        if key in meta:
            return float(meta[key])
    return None


def per_work_blocks(raws: Sequence[dict[str, Any]]) -> list[stats.Block]:
    """Blocks with durations divided by their own work per launch.

    mma_ceiling calibrates its iteration count per arm and per process, so
    two arms of one experiment, even an A/A pair, do slightly different
    work. The mma A/A copies of the hardware-ceilings runs differed by
    -0.21 .. +1.55 % in iterations and a raw duration ratio reported that as
    an A/A effect (MDE 0.60-0.68 % instead of 0.007-0.018 % per unit of work;
    RTX 3080 20 GB, 2026-10; docs/benchmark-methodology.md).
    Every comparison (A/A MDE, claims) and every throughput therefore uses
    time per unit of work. Block keys are unchanged, so rejections apply.
    """
    out = []
    for r in raws:
        for b in blocks_from_raw(r):
            w = work_per_launch(r, b.arm)
            out.append(
                b
                if w is None
                else replace(b, durations_ns=tuple(d / w for d in b.durations_ns))
            )
    return out


def load_nvml(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        return nvml_log.read_log(f.read().splitlines())


def rejection_config(
    raw: dict[str, Any], expected_procs: int, clock_domain: str
) -> stats.RejectionConfig:
    """Rules for one experiment: the clock domain is chosen per experiment
    (ceilings_run.sh: memory for copy arms at L >= 32k), the timer step comes from
    the raw file's own timer."""
    return stats.rejection_config(
        clock_domain=clock_domain,
        timer=str(raw["timer"]),
        expected_procs=expected_procs,
        pci_bus_id=str(raw["device"]["pci_bus_id"]),
    )


# ---------------------------------------------------------------- status
def cmd_status(a: argparse.Namespace) -> int:
    raw = load_raw(a.raw)
    _hdr, samples = load_nvml(a.nvml)
    blocks = blocks_from_raw(raw)
    cfg = rejection_config(raw, a.expected_procs, a.clock_domain)
    rejected = stats.reject_blocks(blocks, samples, cfg)
    short = stats.process_needs_rerun(
        blocks, rejected, int(raw["process"]), a.min_accepted
    )
    counts = stats.rejection_counts(rejected)
    print(json.dumps({"process": raw["process"], "rerun": short, "rejected": counts}))
    return EXIT_RERUN if short else 0


def cmd_domain(a: argparse.Namespace) -> int:
    """Print "mem" or "both" for the experiment (stats.choose_clock_domain)."""
    raws = [load_raw(p) for p in a.raw]
    _hdr, samples = load_nvml(a.nvml)
    blocks = [b for r in raws for b in blocks_from_raw(r)]
    domain, slope = stats.choose_clock_domain(
        blocks, samples, str(raws[0]["device"]["pci_bus_id"])
    )
    print(domain)
    print(f"elasticity d log t / d log f_sm = {slope}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- report helpers
def nvml_block_summary(near: Sequence[stats.Sample]) -> dict[str, Any]:
    def vals(k: str) -> list[float]:
        return [float(s[k]) for s in near if s.get(k) is not None]

    clk, temp, pw = vals("sm_clock_mhz"), vals("temp_c"), vals("power_w")
    masks = [
        int(s["throttle_reasons"])
        for s in near
        if s.get("throttle_reasons") is not None
    ]
    procs = [
        len(s["compute_procs"]) for s in near if s.get("compute_procs") is not None
    ]
    union = 0
    for m in masks:
        union |= m
    return {
        "n": len(near),
        "sm_clock_mhz_median": float(np.median(clk)) if clk else None,
        "temp_c_max": max(temp) if temp else None,
        "power_w_median": float(np.median(pw)) if pw else None,
        "throttle_mask_union": union if masks else None,
        "max_compute_procs": max(procs) if procs else None,
    }


def arm_meta(raws: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    for r in raws:
        for m in r["arms"]:
            if m["name"] == arm:
                return dict(m)
    raise KeyError(arm)


def work_and_unit(
    tool: str, mode: str, meta: dict[str, Any]
) -> tuple[float, str, float]:
    """(work per launch, throughput unit, scale from work/ns to the unit)."""
    if tool in ("sol_copy", "kernel_bench"):
        # kernel_bench: effective KV bytes read once
        return float(meta["bytes_per_launch"]), "GB/s", 1.0  # bytes/ns == GB/s
    if mode == "mma":
        unit = "TOPS" if meta["kind"] == "s8s32" else "TFLOPS"
        return float(meta["flop_per_launch"]), unit, 1e-3  # flop/ns = 1e9 flop/s
    raise ValueError("graph-overhead throughput is computed from a pair of arms")


def median_ci(
    blocks: Sequence[stats.Block], seed: int, cluster_robust: bool = False
) -> stats.CI:
    med = np.array([b.median_ns for b in blocks])
    strata = np.array([b.process for b in blocks])
    return stats.bootstrap_ci(
        med, statistic="median", strata=strata, seed=seed, cluster_robust=cluster_robust
    )


def per_node_cost(
    blocks: Sequence[stats.Block],
    small: str,
    big: str,
    n_small: int,
    n_big: int,
    seed: int,
) -> stats.CI:
    """Slope of graph replay time over node count, from paired rounds (us)."""
    by_round: dict[tuple[int, int], dict[str, stats.Block]] = {}
    for b in blocks:
        by_round.setdefault((b.process, b.round), {})[b.arm] = b
    diffs, strata = [], []
    for (proc, _r), slot in sorted(by_round.items()):
        if small in slot and big in slot:
            diffs.append(
                (slot[big].median_ns - slot[small].median_ns) / (n_big - n_small) / 1e3
            )
            strata.append(proc)
    return stats.bootstrap_ci(diffs, strata=strata, seed=seed)


# ---------------------------------------------------------------- report
def cmd_report(a: argparse.Namespace) -> int:
    raws = [load_raw(p) for p in a.raw]
    tool = raws[0]["tool"]
    mode = raws[0].get("mode", "copy")
    if any(r["tool"] != tool or r.get("mode", "copy") != mode for r in raws):
        raise ValueError(
            "all raw files of one report must come from the same tool and mode"
        )
    nvml_hdr, samples = load_nvml(a.nvml)
    with open(a.env, encoding="utf-8") as f:
        env = json.load(f)
    blocks = [b for r in raws for b in blocks_from_raw(r)]
    work_blocks = per_work_blocks(raws)
    cfg = rejection_config(raws[0], a.expected_procs, a.clock_domain)
    rejected = stats.reject_blocks(blocks, samples, cfg)
    refs = stats.reference_clocks(blocks, samples, cfg)
    windows = stats.SampleWindows(samples, cfg)
    dev = raws[0]["device"]
    arms = [m["name"] for m in raws[0]["arms"]]
    aa = tuple(a.aa) if a.aa else None
    mde = None
    if aa:
        lr, st = stats.paired_log_ratios(work_blocks, aa[1], aa[0], rejected)
        # Fewer than 2 surviving pairs (heavy rejection): no MDE, and any
        # --compare then fails loudly below; the records still get written
        # with status "noisy" instead of the whole report crashing.
        mde = stats.minimum_detectable_effect(lr, st) if lr.size >= 2 else None

    raw_refs = [{"path": str(p), "sha256": sha256_file(p)} for p in a.raw]
    nvml_ref = {
        "path": str(a.nvml),
        "sha256": sha256_file(a.nvml),
        "interval_ms": int(nvml_hdr["interval_ms"]),
        "n_samples": len(samples),
    }
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_nodes = (
        {m["name"]: int(m["nodes"]) for m in raws[0]["arms"]}
        if mode == "graph-overhead"
        else {}
    )

    written = []
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    for arm in arms:
        arm_blocks = [b for b in blocks if b.arm == arm]
        accepted = [b for b in arm_blocks if b.key not in rejected]
        meta = arm_meta(raws, arm)
        per_block: list[dict[str, Any]] = []
        for b in sorted(arm_blocks, key=lambda b: (b.process, b.seq)):
            near = windows.near(b)
            src = next(
                x
                for r in raws
                if int(r["process"]) == b.process
                for x in r["blocks"]
                if int(x["seq"]) == b.seq
            )
            aux = {
                k: src[k]
                for k in ("graph_ms", "sm_clock_mhz_median", "cycles_per_mma_per_smsp")
                if k in src
            }
            per_block.append(
                {
                    "process": b.process,
                    "round": b.round,
                    "seq": b.seq,
                    "t_start_mono_ns": b.t_start_ns,
                    "t_end_mono_ns": b.t_end_ns,
                    "durations_ns": list(b.durations_ns),
                    "median_ns": b.median_ns,
                    "rejected": b.key in rejected,
                    "reasons": rejected.get(b.key, []),
                    "nvml": nvml_block_summary(near),
                    "aux": aux,
                }
            )
        basis = accepted if len(accepted) >= 2 else arm_blocks
        summ = stats.summarize(b.median_ns for b in basis)
        mci = median_ci(basis, a.seed, a.cluster_robust)
        summary: dict[str, Any] = {
            "n_blocks": len(arm_blocks),
            "n_accepted": len(accepted),
            "median_ns": summ.median,
            "q1_ns": summ.q1,
            "q3_ns": summ.q3,
            "iqr_ns": summ.iqr,
            "iqr_over_median": summ.iqr_over_median,
            "median_ci_ns": mci.as_dict(),
        }
        if mode == "graph-overhead":
            small = min(graph_nodes, key=graph_nodes.__getitem__)
            if arm != small:
                c = per_node_cost(
                    blocks, small, arm, graph_nodes[small], graph_nodes[arm], a.seed
                )
                summary["throughput"] = {
                    "unit": "us/node",
                    "estimate": c.estimate,
                    "lo": c.lo,
                    "hi": c.hi,
                    "work_per_launch": float(graph_nodes[arm] - graph_nodes[small]),
                }
        else:
            work, unit, scale = work_and_unit(tool, mode, meta)
            # time per unit of work: each process's own work (see per_work_blocks)
            wkeys = {b.key for b in basis}
            wci = median_ci(
                [b for b in work_blocks if b.arm == arm and b.key in wkeys],
                a.seed,
                a.cluster_robust,
            )
            summary["throughput"] = {
                "unit": unit,
                "estimate": scale / wci.estimate,
                "lo": scale / wci.hi,
                "hi": scale / wci.lo,
                "work_per_launch": float(
                    np.median([work_per_launch(r, arm) or work for r in raws])
                ),
            }
        if mode == "mma":
            # rejected blocks and blocks without the in-kernel clock are left out
            clk = [
                x["aux"]["sm_clock_mhz_median"]
                for x in per_block
                if not x["rejected"] and "sm_clock_mhz_median" in x["aux"]
            ]
            cyc = [
                x["aux"]["cycles_per_mma_per_smsp"]
                for x in per_block
                if not x["rejected"] and "cycles_per_mma_per_smsp" in x["aux"]
            ]
            summary["clock"] = {
                "sm_clock_mhz_median": float(np.median(clk)) if clk else None,
                "cycles_per_mma_per_smsp_median": float(np.median(cyc))
                if cyc
                else None,
            }

        checks = []
        for x, y in a.compare or []:
            if x != arm:
                continue
            if mde is None:
                raise ValueError("--compare needs --aa in the same experiment (MDE)")
            lr, st = stats.paired_log_ratios(work_blocks, x, y, rejected)
            ci = stats.bootstrap_ci(
                lr, strata=st, seed=a.seed, cluster_robust=a.cluster_robust
            )
            v = stats.evaluate_claim(ci, mde, against=y, n_accepted=int(lr.size))
            d = v.as_dict()
            d["note"] = (
                "time per unit of work of this arm relative to the 'against' arm;"
                " claim, not a gate"
            )
            checks.append(d)

        if tool == "kernel_bench":
            # kernel_bench declares each arm's role (subject / aa_control /
            # comparison with its label) in the raw metadata.
            role = str(meta["role"])
        else:
            role = "aa_control" if aa and arm == aa[1] else "ceiling"
        n_proc = len({r["process"] for r in raws})
        status = "ok"
        for p in {r["process"] for r in raws}:
            if stats.process_needs_rerun(blocks, rejected, int(p), a.min_accepted):
                status = "noisy"
        rec = {
            "schema_version": 1,
            "record": "arm_result",
            "session": a.session,
            "experiment": a.experiment,
            "created_unix_ns": time.time_ns(),
            "tool": {
                "name": tool,
                "raw_files": raw_refs,
                "argv": list(raws[0]["argv"]),
            },
            "git": {"sha": env["git_sha"], "dirty": bool(env["git_dirty"])},
            "env": {
                "gpu_name": dev["name"],
                "pci_bus_id": dev["pci_bus_id"],
                "cc": dev["cc"],
                "sm_count": int(dev["sm_count"]),
                "l2_bytes": int(dev["l2_bytes"]),
                "driver_version": str(env["driver_version"]),
                "cuda_runtime": int(dev["runtime_version"]),
                "image_digest": env["image_digest"],
                "power_limit_w": env.get("power_limit_w"),
                "nvcc_version": env["nvcc_version"],
                "captures": env["captures"],
            },
            "arm": {
                "name": arm,
                "role": role,
                "params": meta,
                **({"label": str(meta["label"])} if "label" in meta else {}),
            },
            "shape": raws[0].get("shape", {}),
            "timer": raws[0]["timer"],
            "design": {
                "schedule": "mirrored",
                "blocks_per_arm_per_process": int(raws[0]["design"]["n_blocks"])
                // int(raws[0]["design"]["n_arms"]),
                "replays_per_block": int(raws[0]["design"]["replays_per_block"]),
                "processes": n_proc,
                "warmup_s": float(raws[0]["design"]["warmup_s"]),
                "min_accepted_blocks": a.min_accepted,
            },
            "samples": {
                "unit": "ns",
                "duration_kind": raws[0]["duration_kind"],
                "blocks": per_block,
            },
            "summary": summary,
            "rejections": {
                "total": sum(1 for b in arm_blocks if b.key in rejected),
                "by_reason": stats.rejection_counts(
                    {
                        k: v
                        for k, v in rejected.items()
                        if k in {b.key for b in arm_blocks}
                    }
                ),
                "foreign": stats.foreign_summary(
                    arm_blocks,
                    {
                        k: v
                        for k, v in rejected.items()
                        if k in {b.key for b in arm_blocks}
                    },
                ),
                "reruns": a.reruns,
                "status": status,
                "reference_sm_clock_mhz": refs.get(arm, {}).get(stats.SM_CLOCK),
                "clock_domain": a.clock_domain,
                "reference_clocks_mhz": refs.get(arm, {}),
            },
            "verdict": {
                "overall": "noisy" if status == "noisy" else "not_applicable",
                "checks": checks,
            },
            "ceilings_referenced": [],
            "nvml_trace": nvml_ref,
        }
        # A record that does not match the schema must never reach results/
        # (an earlier writer once emitted env captures without "name" and
        # nothing noticed until a later validation).
        jsonschema.validate(rec, schema)
        safe = arm.replace(":", "_").replace("#", "_")
        path = out_dir / f"{a.experiment}__{safe}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, allow_nan=False)
        written.append(path)
        tp = summary.get("throughput")
        tp_s = (
            f"{tp['estimate']:.4g} {tp['unit']} [{tp['lo']:.4g}, {tp['hi']:.4g}]"
            if tp
            else "-"
        )
        print(
            f"{a.experiment:24s} {arm:22s} median {summ.median / 1e3:10.3f} us  "
            f"IQR/med {summ.iqr_over_median:6.3%}  "
            f"n {len(accepted)}/{len(arm_blocks)}  "
            f"{tp_s}  status {status}"
        )
    if aa is not None and mde is not None:
        print(f"{a.experiment:24s} A/A {aa[0]} vs {aa[1]}: MDE {mde:.4%}")
    return 0


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__
        + "\nReads files only (raw JSON, NVML log, env.json): no GPU is needed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--raw", required=True)
    s.add_argument("--nvml", required=True)
    s.add_argument("--expected-procs", type=int, default=stats.EXPECTED_COMPUTE_PROCS)
    s.add_argument(
        "--clock-domain",
        choices=sorted(stats.CLOCK_DOMAINS),
        default=stats.DEFAULT_CLOCK_DOMAIN,
    )
    s.add_argument("--min-accepted", type=int, default=stats.MIN_ACCEPTED_BLOCKS)
    r = sub.add_parser("report")
    r.add_argument("--raw", required=True, nargs="+")
    r.add_argument("--nvml", required=True)
    r.add_argument("--env", required=True)
    r.add_argument(
        "--session",
        required=True,
        help="identifier written into every record (see results.schema.json)",
    )
    r.add_argument("--experiment", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument(
        "--aa", nargs=2, metavar=("X", "Y"), help="A/A pair: Y is a control copy of X"
    )
    r.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="claim on time(X)/time(Y); repeatable",
    )
    r.add_argument("--reruns", type=int, default=0)
    r.add_argument("--expected-procs", type=int, default=stats.EXPECTED_COMPUTE_PROCS)
    r.add_argument(
        "--clock-domain",
        choices=sorted(stats.CLOCK_DOMAINS),
        default=stats.DEFAULT_CLOCK_DOMAIN,
    )
    r.add_argument("--min-accepted", type=int, default=stats.MIN_ACCEPTED_BLOCKS)
    r.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    r.add_argument(
        "--cluster-robust",
        action="store_true",
        help="widen every CI to the cluster-t interval over processes"
        "; the CI method field records it",
    )
    d = sub.add_parser(
        "domain", help="clock domain measured from the experiment's own blocks"
    )
    d.add_argument("--raw", required=True, nargs="+")
    d.add_argument("--nvml", required=True)
    a = p.parse_args(argv)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "domain":
        return cmd_domain(a)
    return cmd_report(a)


if __name__ == "__main__":
    raise SystemExit(_main())
