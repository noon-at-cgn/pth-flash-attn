# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Turn a kernel-bench results directory into gate verdicts.

Inputs: the ``results.schema.json`` records ``bench/k_run.sh`` wrote into one
directory: kernel experiments (tool ``kernel_bench``) and the ceilings
measured in the same run on the same GPU (tool ``sol_copy``, arm
``sep-nhd``, one experiment per batch and KV length; tool ``mma_ceiling``,
arm ``f16f32:4:1:8``, our occupancy). Every gate is against those hardware
ceilings only (a baseline kernel is never a ceiling); the ``ref`` arm is a
comparison row.

Per kernel experiment (``kernel`` arm, accepted blocks only):

* ``eta_mem = (KV bytes read once / T) / BW_copy(B, L)``, the copy ceiling
  of the same batch and length (decode rows);
* ``eta_mma = flops_executed / (T * F)`` (on executed flops);
  ``eta_mma_useful`` with the causal useful flops is reported beside it;
* the A/A MDE of the experiment (``kernel`` vs ``kernel#aa``), which any
  later claim about a change needs (a change must beat twice
  that MDE);
* ``eta_adj = (max(T_mem, T_mma(f_kernel)) + T_ovh) / T`` for decode rows,
  the roofline with the tensor term evaluated at the SM clock the
  kernel actually ran at (``stats.roofline_eta_ci``). ``f_kernel``
  is the median of the NVML ``sm_clock_mhz_median`` of the *kernel arm's own*
  accepted blocks and the mma ceiling is rescaled from the clock of *its own*
  blocks, never from another arm's window; a row without those clocks is
  "missing", not silently judged raw.

Intervals: ``stats.eta_ci`` (two-sample stratified bootstrap of the medians,
widened by the cluster-t term over processes).
A gate passes on the CI lower bound (``stats.evaluate_gate``).

Gates:

* ``KD``      decode q in {1, 4}, B in {1, 2}, L >= 64k, >= 0.90: q = 1 rows on
              the raw ``eta_mem``; q = 4 (the balanced regime, R = 24) rows on
              the clock-adjusted ``eta_adj`` with the raw ``eta_mem`` printed
              beside it. Decode gate; ``KD-floor``: the same rows below 0.75
              of the *raw* ``eta_mem`` are the floor level;
* ``VERIFY``  decode q = 10, B in {1, 2}, L >= 64k: raw eta_mma >= 0.60
              (decode gate; 0.70 reported as the target);
* ``PREFILL`` prefill chunk 8192, every prefix: eta_mma >= 0.60 (prefill
              gate; 0.70 target; ``KP-floor`` below 0.50);
* ``DEC32K``  decode q in {1, 4}, B in {1, 2}, L = 32k: >= 0.80 (q = 4 on
              ``eta_adj`` like KD; a target row, not part of the decode gate
              verdict).

Refusals (exit 2): records timed with the CPU ``host`` timer, records with a
voiding foreign-process verdict (a timing taken while a real competing
process used the GPU is never evidence), mixed
runs or GPUs. Exit 0: every decode gate
row present and pass; 1 otherwise. Output: ``k_gates.json`` (schema
``bench/kernel_gate.schema.json``) and ``GATES.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import kernel_bench  # noqa: E402  (the grid: which rows each gate needs)
import stats  # noqa: E402

SCHEMA = Path(__file__).with_name("kernel_gate.schema.json")
COPY_ARM = "sep-nhd"  # the loader ceiling the gates use
MMA_ARM = "f16f32:4:1:8"  # our occupancy: 4 warps x 1 CTA/SM, chain 8
LONG_L = 65536  # "L >= 64k"
L_32K = 32768

# Thresholds.
KD_GATE, KD_FLOOR = 0.90, 0.75
VERIFY_GATE, MMA_TARGET = 0.60, 0.70
PREFILL_GATE, KP_FLOOR = 0.60, 0.50
DEC32K_TARGET = 0.80
PREFILL_GATE_Q = 8192
BALANCED_Q = 4  # the balanced regime: R = q G = 24 rows
# T_ovh of the roofline target: the CUDA-graph node cost measured by the
# hardware-ceilings runs; the pipeline-fill and
# combine parts are unmeasured, so this is the only part that can be added
# honestly. At 0.6 us against
# T_mem >= 97 us it moves eta_adj by < 0.7 %.
T_OVH_S = 0.60e-6


@dataclass(frozen=True)
class Rule:
    """One gate, target or floor row set."""

    id: str
    kind: str  # "gate" (decides gate_verdict) | "target" | "floor"
    scope: str  # "decode" (decode and verify rows) | "prefill"
    metric: str  # "eta_mem" | "eta_mma"
    threshold: float
    select: Callable[[dict[str, Any]], bool]
    text: str
    # q values whose rows are gated on the clock-adjusted eta instead of the
    # raw one (the raw eta is always reported beside it).
    adjusted_q: tuple[int, ...] = ()


def _dec(
    qs: Sequence[int], Bs: Sequence[int], cond: Callable[[int], bool]
) -> Callable[[dict[str, Any]], bool]:
    return lambda s: (
        s["family"] == "decode"
        and s["q_max"] in qs
        and s["batch"] in Bs
        and cond(int(s["seq_len_max"]))
    )


RULES = (
    Rule(
        "KD",
        "gate",
        "decode",
        "eta_mem",
        KD_GATE,
        _dec((1, 4), (1, 2), lambda L: L >= LONG_L),
        "decode q in {1,4}, B in {1,2}, L >= 64k: eta_mem >= 0.90 (q=4 on the "
        "clock-adjusted eta)",
        adjusted_q=(BALANCED_Q,),
    ),
    Rule(
        "KD-floor",
        "floor",
        "decode",
        "eta_mem",
        KD_FLOOR,
        _dec((1, 4), (1, 2), lambda L: L >= LONG_L),
        "floor level KD: the same rows below 0.75 of sol_copy",
    ),
    Rule(
        "VERIFY",
        "gate",
        "decode",
        "eta_mma",
        VERIFY_GATE,
        _dec((10,), (1, 2), lambda L: L >= LONG_L),
        "verify q=10, B in {1,2}, L >= 64k: eta_mma (executed) >= 0.60",
    ),
    Rule(
        "VERIFY-target",
        "target",
        "decode",
        "eta_mma",
        MMA_TARGET,
        _dec((10,), (1, 2), lambda L: L >= LONG_L),
        "verify q=10 target 0.70",
    ),
    Rule(
        "DEC32K",
        "target",
        "decode",
        "eta_mem",
        DEC32K_TARGET,
        _dec((1, 4), (1, 2), lambda L: L == L_32K),
        "decode at 32k >= 0.80 (q=4 on the clock-adjusted eta)",
        adjusted_q=(BALANCED_Q,),
    ),
    Rule(
        "PREFILL",
        "gate",
        "prefill",
        "eta_mma",
        PREFILL_GATE,
        lambda s: s["family"] == "prefill" and s["q_max"] == PREFILL_GATE_Q,
        "prefill chunk 8192: eta_mma (executed) >= 0.60 (prefill gate)",
    ),
    Rule(
        "PREFILL-target",
        "target",
        "prefill",
        "eta_mma",
        MMA_TARGET,
        lambda s: s["family"] == "prefill" and s["q_max"] == PREFILL_GATE_Q,
        "prefill target 0.70",
    ),
    Rule(
        "KP-floor",
        "floor",
        "prefill",
        "eta_mma",
        KP_FLOOR,
        lambda s: s["family"] == "prefill" and s["q_max"] == PREFILL_GATE_Q,
        "floor level KP: prefill below 0.50",
    ),
)


class Refused(Exception):
    """The directory cannot support any verdict."""


# ---------------------------------------------------------------- loading
def load_records(res: Path) -> list[dict[str, Any]]:
    recs = []
    for p in sorted(res.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise Refused(f"{p.name}: not JSON ({e})") from e
        if isinstance(r, dict) and r.get("record") == "arm_result":
            r["_file"] = p.name
            recs.append(r)
    if not recs:
        raise Refused(f"no arm_result records in {res}")
    for r in recs:
        if r["timer"] not in ("cupti", "cuda_event"):
            raise Refused(f"{r['_file']}: timer {r['timer']!r} is not a GPU timer")
        rej = r["rejections"]
        # The record's own verdict (a tiny foreign context only rejects its
        # blocks). Records without it predate that verdict: any
        # foreign_process rejection voids them.
        foreign = rej.get("foreign")
        void = (
            foreign["void"]
            if foreign is not None
            else bool(rej["by_reason"].get("foreign_process", 0))
        )
        if void:
            raise Refused(
                f"{r['_file']}: blocks taken while a foreign process used "
                "the GPU; this run is not evidence"
            )
    for key in ("session",):
        vals = {r[key] for r in recs}
        if len(vals) != 1:
            raise Refused(f"records from several {key}s: {sorted(vals)}")
    buses = {r["env"]["pci_bus_id"] for r in recs}
    if len(buses) != 1:
        raise Refused(f"records from several GPUs: {sorted(buses)}")
    return recs


def accepted(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in rec["samples"]["blocks"] if not b["rejected"]]


def raw_work_by_process(rec: dict[str, Any], key: str) -> dict[int, float]:
    """Per-process work of a record's arm from its raw files (mma_ceiling
    calibrates its iterations per process, so work differs by process)."""
    out: dict[int, float] = {}
    for ref in rec["tool"]["raw_files"]:
        path = Path(ref["path"])
        raw = json.loads((path if path.is_absolute() else ROOT / path).read_text())
        meta = next(m for m in raw["arms"] if m["name"] == rec["arm"]["name"])
        out[int(raw["process"])] = float(meta[key])
    return out


def _kernels(params: dict[str, Any]) -> str | None:
    """Kernel variants of an arm's slices, '+'-joined in slice order."""
    ks = [sl.get("kernel") for sl in params.get("slices", [])]
    return "+".join(ks) if ks and all(ks) else None


def block_clocks(blocks: Sequence[dict[str, Any]], key: str) -> np.ndarray | None:
    """One NVML per-block summary value (MHz) for each block, in order; None
    if any block lacks it (a clock-adjusted verdict needs every block's own
    clock, so a gap is reported, never filled)."""
    v = [b.get("nvml", {}).get(key) for b in blocks]
    if not v or any(x is None or x <= 0 for x in v):
        return None
    return np.array(v, dtype=np.float64)


def median_clock(blocks: Sequence[dict[str, Any]], key: str) -> float | None:
    """Median over blocks of one NVML per-block summary field (SM clock in
    MHz, power in W) from the records' ``nvml`` summaries; None if absent."""
    v = [b["nvml"][key] for b in blocks if b.get("nvml", {}).get(key) is not None]
    return float(np.median(v)) if v else None


@dataclass(frozen=True)
class CeilingSample:
    ceiling: stats.Ceiling
    rates: np.ndarray  # work per second, accepted blocks
    strata: np.ndarray
    file: str
    sm_clock_mhz: float | None = None  # median SM clock over accepted blocks
    clocks: np.ndarray | None = None  # per accepted block; None if any is missing


def ceiling_sample(
    rec: dict[str, Any], kind: str, rate_key: str, work: dict[int, float] | None
) -> CeilingSample:
    blocks = accepted(rec)
    if len(blocks) < 2:
        raise Refused(f"{rec['_file']}: fewer than 2 accepted ceiling blocks")
    w = work or {}
    rates = np.array(
        [
            (w.get(b["process"]) or float(rec["arm"]["params"][rate_key]))
            / (b["median_ns"] * 1e-9)
            for b in blocks
        ]
    )
    strata = np.array([b["process"] for b in blocks])
    tp = rec["summary"].get("throughput", {})
    c = stats.Ceiling(
        kind=kind,
        name=f"{rec['experiment']}:{rec['arm']['name']}",
        unit=str(tp.get("unit", "")),
        value=stats.CI(
            float(np.median(rates)),
            float(np.min(rates)),
            float(np.max(rates)),
            1.0,
            len(rates),
            0,
            0,
            "range",
        ),
        source=rec["_file"],
    )
    return CeilingSample(
        c,
        rates,
        strata,
        rec["_file"],
        median_clock(blocks, "sm_clock_mhz_median"),
        block_clocks(blocks, "sm_clock_mhz_median"),
    )


def copy_ceilings(
    recs: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], CeilingSample]:
    out = {}
    for r in recs:
        if r["tool"]["name"] == "sol_copy" and r["arm"]["name"] == COPY_ARM:
            key = (int(r["shape"]["batch"]), int(r["shape"]["seq_len"]))
            if key in out:
                raise Refused(f"two copy ceilings for B, L = {key}")
            out[key] = ceiling_sample(r, "sol_copy", "bytes_per_launch", None)
    return out


def mma_ceiling(recs: Sequence[dict[str, Any]]) -> CeilingSample | None:
    found = [
        r
        for r in recs
        if r["tool"]["name"] == "mma_ceiling" and r["arm"]["name"] == MMA_ARM
    ]
    if len(found) > 1:
        raise Refused("more than one mma ceiling record")
    if not found:
        return None
    r = found[0]
    return ceiling_sample(
        r, "mma_ceiling", "flop_per_launch", raw_work_by_process(r, "flop_per_launch")
    )


# ---------------------------------------------------------------- per experiment
def aa_mde(subject: dict[str, Any], aa: dict[str, Any] | None) -> float | None:
    """MDE of the experiment from kernel vs kernel#aa block pairs."""
    if aa is None:
        return None
    blocks = []
    for rec in (subject, aa):
        for b in rec["samples"]["blocks"]:
            blocks.append(
                stats.Block(
                    b["process"],
                    b["round"],
                    b["seq"],
                    rec["arm"]["name"],
                    b["t_start_mono_ns"],
                    b["t_end_mono_ns"],
                    tuple(b["durations_ns"]),
                )
            )
    bad = [
        (b["process"], b["seq"])
        for rec in (subject, aa)
        for b in rec["samples"]["blocks"]
        if b["rejected"]
    ]
    lr, st = stats.paired_log_ratios(
        blocks, aa["arm"]["name"], subject["arm"]["name"], bad
    )
    return stats.minimum_detectable_effect(lr, st) if lr.size >= 2 else None


def _ci(c: stats.CI | None) -> dict[str, Any] | None:
    return (
        None
        if c is None
        else {"estimate": c.estimate, "lo": c.lo, "hi": c.hi, "method": c.method}
    )


def slice_flops(row: dict[str, Any]) -> float:
    """Formula flops of a decode row, 4 D 16 ceil(R/16) Hkv L per
    request (R = q G rows, idle 16-row slices skipped): the tensor work a
    16-row-slice kernel would issue. Informational reading of the balanced
    regime; the gate uses the executed flops."""
    rows = row["q"] * kernel_bench.G
    slices = -(-rows // 16)
    return float(
        4
        * kernel_bench.D
        * 16
        * slices
        * kernel_bench.HKV
        * row["seq_len"]
        * row["batch"]
    )


def experiment_row(
    subject: dict[str, Any],
    aa: dict[str, Any] | None,
    ref: dict[str, Any] | None,
    copies: dict[tuple[int, int], CeilingSample],
    mma: CeilingSample | None,
    seed: int,
    t_ovh_s: float = T_OVH_S,
) -> dict[str, Any]:
    shape = subject["shape"]
    params = subject["arm"]["params"]
    blocks = accepted(subject)
    t = np.array([b["median_ns"] * 1e-9 for b in blocks])
    st = np.array([b["process"] for b in blocks])
    row: dict[str, Any] = {
        "experiment": subject["experiment"],
        "family": shape["family"],
        "q": int(shape["q_max"]),
        "batch": int(shape["batch"]),
        "seq_len": int(shape["seq_len_max"]),
        "splits": str(shape["splits"]),
        "n_accepted": len(blocks),
        "n_blocks": subject["summary"]["n_blocks"],
        "status": subject["rejections"]["status"],
        "median_us": float(np.median(t) * 1e6) if t.size else None,
        "eta_mem": None,
        "eta_mma": None,
        "eta_mma_useful": None,
        "gbps": None,
        "tflops_executed": None,
        "tflops_useful": None,
        "mde": aa_mde(subject, aa),
        "ceilings": [],
        "ref_median_us": None,
        "file": subject["_file"],
        # Informational, never a gate: the SM clock the kernel actually ran at
        # under the power cap, and eta_mma rescaled to the clock of the mma
        # ceiling (eta * ceiling clock / kernel clock; only meaningful for
        # tensor-core-bound rows: a copy-bound row does not scale with it).
        "sm_clock_mhz": median_clock(blocks, "sm_clock_mhz_median"),
        "power_w": median_clock(blocks, "power_w_median"),
        "mma_ceiling_clock_mhz": None if mma is None else mma.sm_clock_mhz,
        "eta_mma_clock_adj": None,
        # Spread of the accepted blocks (IQR / median, informational) and the
        # kernel variant(s) the launch ran (kernel_bench.KERNEL_*; records of
        # results from before the decode kernel have none).
        "iqr_over_median": subject["summary"].get("iqr_over_median"),
        "kernels": _kernels(params),
        # The clock-adjusted roofline efficiency of
        # decode rows, its terms, and why it is absent when it is.
        "eta_adj": None,
        "adj": None,
        "eta_adj_16row": None,
        "eta_adj_note": None,
    }
    if ref is not None and accepted(ref):
        row["ref_median_us"] = float(
            np.median([b["median_ns"] for b in accepted(ref)]) / 1e3
        )
    if t.size < 2:
        return row
    by = float(params["bytes_per_launch"])
    fe, fu = float(params["flops_executed"]), float(params["flops_useful"])
    med = float(np.median(t))
    row["gbps"] = by / med / 1e9
    row["tflops_executed"] = fe / med / 1e12
    row["tflops_useful"] = fu / med / 1e12
    cp = (
        copies.get((row["batch"], row["seq_len"]))
        if row["family"] == "decode"
        else None
    )
    if cp is not None:
        c = stats.eta_ci(t, st, by, cp.rates, cp.strata, seed=seed)
        row["eta_mem"] = _ci(c)
        row["_eta_mem"] = (c, cp)
        row["ceilings"].append(cp.ceiling.as_ref())
    if mma is not None:
        c = stats.eta_ci(t, st, fe, mma.rates, mma.strata, seed=seed)
        cu = stats.eta_ci(t, st, fu, mma.rates, mma.strata, seed=seed)
        row["eta_mma"], row["eta_mma_useful"] = _ci(c), _ci(cu)
        row["_eta_mma"] = (c, mma)
        if mma.sm_clock_mhz and row["sm_clock_mhz"]:
            row["eta_mma_clock_adj"] = (
                c.estimate * mma.sm_clock_mhz / row["sm_clock_mhz"]
            )
        row["ceilings"].append(mma.ceiling.as_ref())
    if row["family"] == "decode" and row["q"] == BALANCED_Q:
        # The kernel's clock is the median of its own accepted blocks' NVML
        # windows (paired with their times in the bootstrap), the mma
        # ceiling's that of its own blocks: no other arm's window enters.
        kclk = block_clocks(blocks, "sm_clock_mhz_median")
        if cp is None:
            row["eta_adj_note"] = "no copy ceiling for this batch and length"
        elif mma is None:
            row["eta_adj_note"] = "no mma ceiling in the run"
        elif kclk is None:
            row["eta_adj_note"] = "a kernel block has no SM clock in its NVML window"
        elif mma.clocks is None:
            row["eta_adj_note"] = "an mma ceiling block has no SM clock (NVML window)"
        else:

            def roofline(flops: float) -> stats.RooflineEta:
                return stats.roofline_eta_ci(
                    t,
                    kclk,
                    st,
                    by,
                    flops,
                    cp.rates,
                    cp.strata,
                    mma.rates,
                    mma.clocks,
                    mma.strata,
                    t_ovh_s=t_ovh_s,
                    seed=seed,
                )

            # The gate's tensor term is what the kernel issues (executed
            # flops, like VERIFY), so a kernel that removed tensor work (the
            # decode kernel, 24 rows instead of 32 at q = 4) is not excused
            # by a bound it does not have. The formula flops (16-row
            # slices) give the other reading; it is reported, never gated.
            ra = roofline(fe)
            row["eta_adj"] = _ci(ra.ci)
            row["_eta_adj"] = (ra.ci, cp, mma)
            row["eta_adj_16row"] = _ci(roofline(slice_flops(row)).ci)
            row["adj"] = {
                "t_mem_us": ra.t_mem_s * 1e6,
                "t_mma_us": ra.t_mma_s * 1e6,
                "t_ovh_us": ra.t_ovh_s * 1e6,
                "bound": ra.bound,
                "kernel_clock_mhz": ra.kernel_clock_mhz,
                "mma_ceiling_clock_mhz": ra.ceiling_clock_mhz,
            }
    return row


def expected_shapes(rule: Rule) -> list[tuple[str, int]]:
    """(experiment name, q) of the full grid (kernel_bench.grid) the rule
    covers: a gate is passed only with every one of them measured, never on a
    subset that happened to run."""
    out = []
    for s in kernel_bench.grid("all"):
        shape = {
            "family": s.family,
            "q_max": max(s.q_lens),
            "batch": len(s.q_lens),
            "seq_len_max": max(s.seq_lens),
        }
        if rule.select(shape):
            out.append((s.name, int(shape["q_max"])))
    return out


def expected_rows(rule: Rule) -> list[str]:
    """Experiment names of ``expected_shapes``."""
    return [name for name, _ in expected_shapes(rule)]


def overall_of(kind: str, checks: Sequence[dict[str, Any]]) -> str:
    if kind == "floor":
        return (
            "floor_level_reached"
            if any(c.get("floor_level_reached") for c in checks)
            else "not_reached"
        )
    results = {c["result"] for c in checks}
    if not checks:
        return "missing"
    if results == {"pass"}:
        return "pass"
    for worst in ("fail", "missing", "noisy"):
        if worst in results:
            return worst
    return "inconclusive"


def _check(rule: Rule, name: str, q: int, r: dict[str, Any] | None) -> dict[str, Any]:
    """One row of one rule. Rows of ``rule.adjusted_q`` are judged on the
    clock-adjusted eta with the raw eta_mem and what
    the raw gate would say recorded beside it; all others on ``rule.metric``."""
    adjusted = q in rule.adjusted_q
    metric = "eta_adj" if adjusted else rule.metric
    pair = None if r is None else r.get(f"_{metric}")
    if r is None or pair is None:
        if r is None:
            note = "not measured"
        elif adjusted:
            why = r.get("eta_adj_note") or "fewer than 2 accepted blocks"
            note = f"no eta_adj: {why}"
        else:
            note = f"no {metric}: ceiling not measured or fewer than 2 accepted blocks"
        return {
            "experiment": name,
            "metric": metric,
            "result": "missing",
            "note": note,
        }
    ci, ceil = pair[0], pair[1]
    v = stats.evaluate_gate(ci, rule.threshold, ceil.ceiling, r["n_accepted"])
    d = v.as_dict()
    d["experiment"] = name
    d["metric"] = metric
    if adjusted:
        mma = pair[2]
        d["against"] = (
            f"{ceil.ceiling.kind}:{ceil.ceiling.name}"
            f"+{mma.ceiling.kind}:{mma.ceiling.name}"
        )
        rci, rceil = r["_eta_mem"]
        rv = stats.evaluate_gate(rci, rule.threshold, rceil.ceiling, r["n_accepted"])
        d["raw"] = {
            "estimate": rci.estimate,
            "lo": rci.lo,
            "hi": rci.hi,
            "result": rv.result,
        }
        d["bound"] = r["adj"]["bound"]
        d["sm_clock_mhz"] = r["adj"]["kernel_clock_mhz"]
    if rule.kind == "floor":
        # The floor level is reached only if the kernel is clearly below it
        # (CI upper bound under the level).
        d["floor_level_reached"] = v.result == "fail"
    return d


def apply_rules(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {r["experiment"]: r for r in rows}
    out = []
    for rule in RULES:
        checks = [
            _check(rule, name, q, by_name.get(name))
            for name, q in expected_shapes(rule)
        ]
        out.append(
            {
                "id": rule.id,
                "kind": rule.kind,
                "scope": rule.scope,
                "metric": rule.metric,
                "adjusted_q": list(rule.adjusted_q),
                "threshold": rule.threshold,
                "text": rule.text,
                "overall": overall_of(rule.kind, checks),
                "rows": checks,
            }
        )
    return out


def build(
    res: Path, seed: int = stats.DEFAULT_SEED, t_ovh_s: float = T_OVH_S
) -> dict[str, Any]:
    recs = load_records(res)
    copies = copy_ceilings(recs)
    mma = mma_ceiling(recs)
    kb = [r for r in recs if r["tool"]["name"] == "kernel_bench"]
    by_exp: dict[str, dict[str, dict[str, Any]]] = {}
    for r in kb:
        by_exp.setdefault(r["experiment"], {})[r["arm"]["role"]] = r
    rows = []
    for exp, arms in sorted(by_exp.items()):
        if "subject" not in arms:
            raise Refused(f"{exp}: no kernel (subject) record")
        rows.append(
            experiment_row(
                arms["subject"],
                arms.get("aa_control"),
                arms.get("comparison"),
                copies,
                mma,
                seed,
                t_ovh_s,
            )
        )
    rules = apply_rules(rows)
    decode_gates = [g for g in rules if g["kind"] == "gate" and g["scope"] == "decode"]
    verdict = (
        "pass"
        if decode_gates and all(g["overall"] == "pass" for g in decode_gates)
        else (
            "fail"
            if any(g["overall"] == "fail" for g in decode_gates)
            else "not_passed"
        )
    )
    for r in rows:
        r.pop("_eta_mem", None)
        r.pop("_eta_mma", None)
        r.pop("_eta_adj", None)
    # Which kernel was timed (k_run.sh's env.json: kind and source commit).
    env_path = res / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    rep = {
        "schema_version": 1,
        "record": "kernel_gates",
        "session": kb[0]["session"] if kb else recs[0]["session"],
        "gpu": recs[0]["env"]["pci_bus_id"],
        "git_sha": kb[0]["git"]["sha"] if kb else recs[0]["git"]["sha"],
        "kernel": env.get("kernel"),
        "kernel_source": env.get("kernel_source"),
        "balanced_policy": {
            "adjusted_q": [BALANCED_Q],
            "t_ovh_us": t_ovh_s * 1e6,
        },
        "gate_verdict": verdict,
        "ceilings": {
            "copy": {
                f"B{b}-L{L}": {
                    "median_gbps": float(np.median(c.rates)) / 1e9,
                    "n": int(c.rates.size),
                    "file": c.file,
                }
                for (b, L), c in sorted(copies.items())
            },
            "mma": None
            if mma is None
            else {
                "median_tflops": float(np.median(mma.rates)) / 1e12,
                "n": int(mma.rates.size),
                "file": mma.file,
            },
        },
        "rules": rules,
        "experiments": rows,
    }
    jsonschema.validate(rep, json.loads(SCHEMA.read_text()))
    return rep


def _f(x: Any, fmt: str = ".3f") -> str:
    return "-" if x is None else format(x, fmt)


def _ci_text(c: dict[str, Any] | None) -> str:
    return "-" if c is None else f"{c['estimate']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def table(rep: dict[str, Any]) -> str:
    head = (
        f"kernel gates {rep['session']} ({rep['gpu']}, {rep['git_sha'][:10]}): "
        f"gate verdict {rep['gate_verdict'].upper()}"
    )
    pol = rep["balanced_policy"]
    policy = (
        f"policy: q={pol['adjusted_q']} rows are gated on the clock-adjusted eta "
        f"(T_ovh {pol['t_ovh_us']:.2f} us), raw eta beside it; "
        "floor levels and q=1 / q=10 gates are raw"
    )
    lines = [head, policy, ""]
    for g in rep["rules"]:
        lines.append(
            f"{g['id']:<15} {g['kind']:<6} {g['scope']:<7} "
            f"{g['overall']:<20} {g['text']}"
        )
        for c in g["rows"]:
            if c["result"] == "missing":
                lines.append(f"    {c['experiment']:<26} missing: {c.get('note', '')}")
                continue
            line = (
                f"    {c['experiment']:<26} {c['result']:<12} "
                f"{c['estimate']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"
            )
            if c.get("metric") == "eta_adj":
                raw = c["raw"]
                line += (
                    f"  adj; raw {raw['estimate']:.3f} [{raw['lo']:.3f}, "
                    f"{raw['hi']:.3f}] ({raw['result']}); "
                    f"SM {c['sm_clock_mhz']:.0f} MHz, {c['bound']}-bound"
                )
            lines.append(line)
    columns = (
        f"{'experiment':<26} {'S':>9} {'n':>7} {'us':>10} {'GB/s':>7} "
        f"{'eta_mem [lo, hi]':>22} {'TF ex':>6} {'eta_mma [lo, hi]':>22} "
        f"{'MDE':>7} {'ref us':>9} {'SM MHz':>7} {'W':>6} {'eta_mma*clk':>11} "
        f"{'IQR/med':>8} {'kernel':>8} {'eta_adj [lo, hi]':>22} {'bound':>5} "
        f"{'eta_adj_16row':>22}"
    )
    lines += ["", columns]
    for r in rep["experiments"]:
        iqr = None if r["iqr_over_median"] is None else 100 * r["iqr_over_median"]
        lines.append(
            f"{r['experiment']:<26} {r['splits']:>9} "
            f"{r['n_accepted']:>3}/{r['n_blocks']:<3} {_f(r['median_us'], '.1f'):>10} "
            f"{_f(r['gbps'], '.1f'):>7} {_ci_text(r['eta_mem']):>22} "
            f"{_f(r['tflops_executed'], '.2f'):>6} {_ci_text(r['eta_mma']):>22} "
            f"{_f(None if r['mde'] is None else 100 * r['mde'], '.2f'):>6}% "
            f"{_f(r['ref_median_us'], '.1f'):>9} "
            f"{_f(r['sm_clock_mhz'], '.0f'):>7} {_f(r['power_w'], '.0f'):>6} "
            f"{_f(r['eta_mma_clock_adj'], '.3f'):>11} "
            f"{_f(iqr, '.2f'):>7}% "
            f"{r['kernels'] or '-':>8} {_ci_text(r['eta_adj']):>22} "
            f"{(r['adj'] or {}).get('bound', '-'):>5} "
            f"{_ci_text(r['eta_adj_16row']):>22}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__
        + "\nReads files only: no GPU is needed. Exit codes: 0 gate_verdict pass; "
        "1 not passed or fail; 2 refused.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "results_dir", help="bench/k_run.sh result directory (per-arm records)"
    )
    ap.add_argument(
        "--seed", type=int, default=stats.DEFAULT_SEED, help="bootstrap seed"
    )
    ap.add_argument(
        "--out",
        help="write k_gates.json and GATES.txt here instead of into results_dir "
        "(a re-evaluation never overwrites the run's own verdict files)",
    )
    a = ap.parse_args(argv)
    res = Path(a.results_dir)
    out = Path(a.out) if a.out else res
    try:
        rep = build(res, a.seed)
    except Refused as e:
        print(f"kernel_gate: REFUSED: {e}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / "k_gates.json").write_text(json.dumps(rep, indent=1, allow_nan=False) + "\n")
    text = table(rep)
    (out / "GATES.txt").write_text(text)
    print(text, end="")
    return 0 if rep["gate_verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
