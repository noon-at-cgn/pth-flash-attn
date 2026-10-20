#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Render the tables of docs/benchmarks.md from the curated CSV files in this directory.

    python bench/results/render_tables.py            # print every table block
    python bench/results/render_tables.py --check    # exit 1 on any difference
    python bench/results/render_tables.py --write    # rewrite the blocks in place

``--check`` and ``--write`` cover docs/benchmarks.md and the headline block of the
repository README.md.

Every CSV row holds one published number: ``table``, ``row``, ``metric``, ``unit``,
``arm``, ``value``, ``ci95_low``, ``ci95_high``, ``n``, ``date``, ``hardware``,
``power_w``, ``software``, ``config``, ``source_url``, ``notes`` and ``bold``. A table
is the set of rows that share a ``table`` key: rows of the table are the distinct
``row`` values in file order, columns are the ``arm`` values named in SPECS below.
``bold`` is 1 where the rule of docs/benchmarks.md ("How to read the tables") marks
the better value. The caption above every table is built from the ``hardware``,
``power_w``, ``software``, ``date``, ``config`` and ``source_url`` columns of the same
rows, so that no table can lose its label.

``source_url`` is a public pull-request URL or a path of this repository (optionally
with a ``#anchor``) that names the method that produced the number.
tests/cpu/test_benchmark_tables.py asserts that the blocks of the page equal the
output of this script and that the headline block of README.md equals the one of the
page.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE.parents[1] / "docs" / "benchmarks.md"
README = HERE.parents[1] / "README.md"
COLUMNS = [
    "table",
    "row",
    "metric",
    "unit",
    "arm",
    "value",
    "ci95_low",
    "ci95_high",
    "n",
    "date",
    "hardware",
    "power_w",
    "software",
    "config",
    "source_url",
    "notes",
    "bold",
]

# id, csv file, caption title, header of the first column, [(arm, header, suffix)]
SPECS = [
    (
        "k1",
        "k1_drafter_microbench.csv",
        "K1. Sliding-window drafter attention against Triton",
        "shape, cache state",
        [
            ("pth_2056", "this kernel, L=2,056 (µs)", ""),
            ("pth_131080", "this kernel, L=131,080 (µs)", ""),
            ("triton_2056", "Triton, L=2,056 (µs)", ""),
            ("triton_131080", "Triton, L=131,080 (µs)", ""),
            ("ratio_2056", "Triton ÷ kernel, L=2,056", "×"),
            ("ratio_131080", "Triton ÷ kernel, L=131,080", "×"),
        ],
    ),
    (
        "k2p",
        "k2_public_harness_20261020.csv",
        (
            "K2a. Causal decode, verify and prefill at D=256 against Triton, "
            "public harness"
        ),
        "batch (per layer)",
        [
            ("op_cold", "this op, cold L2 (µs)", ""),
            ("op_warm", "this op, warm L2 (µs)", ""),
            ("tri_cold", "Triton int8, cold L2 (µs)", ""),
            ("ratio", "Triton ÷ op", "×"),
            ("op_dev", "this op in K2b (µs)", ""),
            ("delta", "this op against K2b", ""),
        ],
    ),
    (
        "k2",
        "k2_causal_d256_vs_triton.csv",
        (
            "K2b. Causal decode, verify and prefill at D=256 against Triton, "
            "development measurement"
        ),
        "batch (per layer)",
        [
            ("triton_bf16", "Triton, bf16 cache", ""),
            ("triton_int8", "Triton, int8 cache", ""),
            ("op_int8", "this op, int8 cache", ""),
            ("ratio", "Triton int8 ÷ op", "×"),
        ],
    ),
    (
        "k3_decode",
        "k3_roofline.csv",
        "K3a. Decode efficiency against the copy ceiling (η_mem)",
        "shape",
        [
            ("median_us", "median (µs)", ""),
            ("eta", "η_mem", ""),
            ("eta_lo", "η_mem, CI lower bound", ""),
            ("kept", "blocks kept", ""),
            ("sm_mhz", "SM clock (MHz)", ""),
        ],
    ),
    (
        "k3_mma",
        "k3_roofline.csv",
        "K3b. Verify and prefill efficiency against the tensor-core ceiling (η_mma)",
        "shape",
        [
            ("median_us", "median (µs)", ""),
            ("eta", "η_mma", ""),
            ("eta_lo", "η_mma, CI lower bound", ""),
            ("kept", "blocks kept", ""),
            ("sm_mhz", "SM clock (MHz)", ""),
        ],
    ),
    (
        "e1_ms",
        "e1_dflash2_3080_20261017.csv",
        "E1a. DFlash2: time per decode pass (ms, median)",
        "cell",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth_t", "int8 + pth, target only", ""),
            ("pth_td", "int8 + pth, target + drafter", ""),
        ],
    ),
    (
        "e1_ms_ratio",
        "e1_dflash2_3080_20261017.csv",
        "E1b. DFlash2: ratios of time per decode pass [95 % CI]",
        "cell",
        [
            ("td_over_fa2", "target+drafter ÷ FA2", ""),
            ("td_over_t", "target+drafter ÷ target only", ""),
            ("t_over_fa2", "target only ÷ FA2", ""),
            ("fa2_over_triton", "FA2 ÷ Triton", ""),
            ("t_over_triton", "target only ÷ Triton", ""),
        ],
    ),
    (
        "e1_tps",
        "e1_dflash2_3080_20261017.csv",
        "E1c. DFlash2: decode tokens per second (median, per stream)",
        "cell",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth_t", "int8 + pth, target only", ""),
            ("pth_td", "int8 + pth, target + drafter", ""),
        ],
    ),
    (
        "e1_tps_ratio",
        "e1_dflash2_3080_20261017.csv",
        "E1d. DFlash2: ratios of decode tokens per second [95 % CI]",
        "cell",
        [
            ("td_over_fa2", "target+drafter ÷ FA2", ""),
            ("t_over_fa2", "target only ÷ FA2", ""),
            ("fa2_over_triton", "FA2 ÷ Triton", ""),
            ("t_over_triton", "target only ÷ Triton", ""),
        ],
    ),
    (
        "e1_other",
        "e1_dflash2_3080_20261017.csv",
        "E1e. DFlash2: prefill, KV pool, accuracy, acceptance",
        "measure",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth_t", "int8 + pth, target only", ""),
            ("pth_td", "int8 + pth, target + drafter", ""),
        ],
    ),
    (
        "e1_ttft_ratio",
        "e1_dflash2_3080_20261017.csv",
        "E1f. DFlash2: ratios of the N=1 ladder TTFT medians [95 % CI]",
        "cell",
        [
            ("td_over_fa2", "target+drafter ÷ FA2", ""),
            ("t_over_fa2", "target only ÷ FA2", ""),
            ("td_over_t", "target+drafter ÷ target only", ""),
            ("fa2_over_triton", "FA2 ÷ Triton", ""),
        ],
    ),
    (
        "e2_main",
        "e2_mtp_3080_20261019.csv",
        "E2. MTP: default (int8 + pth) against PTH_FA=0 (fp8 + FlashInfer)",
        "cell",
        [
            ("fp8fi", "fp8 + FlashInfer (PTH_FA=0)", ""),
            ("pth", "int8 + pth (default)", ""),
            ("ratio", "ratio, both boots pooled [95 % CI]", ""),
        ],
    ),
    (
        "a1_dflash2",
        "a1_logprob.csv",
        "A1a. DFlash2 checkpoint: prompt logprobs against a bf16-KV reference",
        "measure",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth", "int8 + pth (target, and target + drafter)", ""),
        ],
    ),
    (
        "a1_dflash2_ni",
        "a1_logprob.csv",
        (
            "A1b. DFlash2 checkpoint: paired difference of mean |Δ logprob|, "
            "new minus old [95 % CI]"
        ),
        "comparison (new vs old)",
        [
            ("d32k", "32k document", ""),
            ("d128k", "128k document", ""),
            ("verdict", "non-inferiority (32k / 128k)", ""),
        ],
    ),
    (
        "a1_mtp",
        "a1_logprob.csv",
        "A1c. MTP checkpoint: prompt logprobs against a bf16-KV reference",
        "measure",
        [("fp8fi", "fp8 + FlashInfer", ""), ("pth", "int8 + pth", "")],
    ),
    (
        "a1_mtp_ni",
        "a1_logprob.csv",
        (
            "A1d. MTP checkpoint: paired difference of mean |Δ logprob|, "
            "new minus old [95 % CI]"
        ),
        "comparison (new vs old)",
        [
            ("d32k", "32k document", ""),
            ("d128k", "128k document", ""),
            ("verdict", "non-inferiority (32k / 128k)", ""),
        ],
    ),
    (
        "t1_dflash2",
        "t1_battery.csv",
        "T1a. T4 battery, SPEC=dflash2",
        "measure",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth", "int8 + pth (target + drafter)", ""),
        ],
    ),
    (
        "t1_mtp",
        "t1_battery.csv",
        "T1b. T4 battery, SPEC=mtp",
        "measure",
        [
            ("fp8fi", "fp8 + FlashInfer (PTH_FA=0)", ""),
            ("pth", "int8 + pth (default)", ""),
        ],
    ),
    (
        "m1_mtp",
        "m1_hq_3090.csv",
        "M1a. RTX 3090, SPEC=mtp (reported by the HyperQwen maintainer)",
        "measure",
        [
            ("fp8fi", "fp8 + FlashInfer", ""),
            ("pth", "int8 + pth", ""),
            ("delta", "difference stated in the source", ""),
        ],
    ),
    (
        "m1_mtp_ab",
        "m1_hq_3090.csv",
        (
            "M1c. RTX 3090, SPEC=mtp, A/B on the merged default "
            "(reported by the HyperQwen maintainer)"
        ),
        "measure",
        [
            ("fp8fi", "fp8 + FlashInfer (PTH_FA=0)", ""),
            ("pth", "int8 + pth (default)", ""),
            ("delta", "difference stated in the source", ""),
        ],
    ),
    (
        "m1_dflash2",
        "m1_hq_3090.csv",
        "M1b. RTX 3090, SPEC=dflash2 (reported by the HyperQwen maintainer)",
        "measure",
        [
            ("triton", "int8 + Triton", ""),
            ("fa2", "fp8 + FA2", ""),
            ("pth", "int8 + pth (target + drafter)", ""),
            ("delta", "int8 + pth against FA2, as stated in the source", ""),
        ],
    ),
]
SPEC_BY_ID = {s[0]: s for s in SPECS}
# tables whose setup (hardware, power limit, software, date, configuration) is that
# of an earlier table
SAME_AS = {
    "e1_ms_ratio": "e1_ms",
    "e1_tps": "e1_ms",
    "e1_tps_ratio": "e1_ms",
    "e1_other": "e1_ms",
    "e1_ttft_ratio": "e1_ms",
    "a1_dflash2_ni": "a1_dflash2",
    "a1_mtp_ni": "a1_mtp",
}
PR_RE = re.compile(
    r"^https://github\.com/syv-ai/HyperQwen/pull/(\d+)(?:#issuecomment-\d+)?$"
)


def load(name: str) -> list[dict]:
    with open(HERE / name, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{name}: header {reader.fieldnames} != {COLUMNS}")
        return list(reader)


def table_rows(tid: str) -> list[dict]:
    return [r for r in load(SPEC_BY_ID[tid][1]) if r["table"] == tid]


def esc(text: str) -> str:
    """Escape the pipe characters of a Markdown table cell."""
    return text.replace("|", "\\|")


def cell(rec: dict | None, suffix: str = "") -> str:
    if rec is None:
        return "–"
    value = rec["value"]
    if rec["unit"] == "tokens" and re.fullmatch(r"\d+", value):
        value = f"{int(value):,}"
    text = value + suffix
    if rec["bold"] == "1":
        text = f"**{text}**"
    lo, hi = rec["ci95_low"], rec["ci95_high"]
    if lo and hi:
        sep = ", " if lo[0] in "+-" else "–"
        text += f" [{lo}{sep}{hi}]"
    return text


def uniq(values) -> list[str]:
    out: list[str] = []
    for v in values:
        if v and v not in out:
            out.append(v)
    return out


def source_link(src: str) -> str:
    """One source: a pull-request link, or a repository path relative to docs/."""
    m = PR_RE.match(src)
    if m:
        label = f"#{m.group(1)}" + (" comment" if "#issuecomment" in src else "")
        return f"[{label}]({src})"
    parts = []
    for item in (p.strip() for p in src.split(";")):
        path, _, anchor = item.partition("#")
        href = path[len("docs/") :] if path.startswith("docs/") else "../" + path
        href += ("#" + anchor) if anchor else ""
        parts.append(f"[`{item}`]({href})")
    return ", ".join(parts)


def setup_of(recs: list[dict]) -> tuple:
    return tuple(
        tuple(uniq(r[k] for r in recs))
        for k in ("hardware", "power_w", "software", "date", "config")
    )


def caption(tid: str, title: str, recs: list[dict]) -> str:
    srcs = ", ".join(source_link(s) for s in uniq(r["source_url"] for r in recs))
    if tid in SAME_AS:
        ref = SAME_AS[tid]
        if setup_of(recs) != setup_of(table_rows(ref)):
            raise SystemExit(f"table {tid}: setup differs from {ref}")
        label = SPEC_BY_ID[ref][2].split(".")[0]
        return (
            f"**{title}.** Setup as {label} (hardware, power limit, software, date, "
            f"configuration). Source: {srcs}."
        )
    hw = "; ".join(uniq(r["hardware"] for r in recs))
    pw = "/".join(uniq(r["power_w"] for r in recs))
    sw = "; ".join(uniq(r["software"] for r in recs))
    dates = "; ".join(uniq(r["date"] for r in recs))
    cfg = "; ".join(uniq(r["config"] for r in recs))
    return (
        f"**{title}.** Hardware: {hw}, {pw} W power limit. Software: {sw}. "
        f"Date: {dates}. Configuration: {cfg}. Source: {srcs}."
    )


def render_table(tid: str) -> str:
    _, _, title, first, cols = SPEC_BY_ID[tid]
    recs = table_rows(tid)
    if not recs:
        raise SystemExit(f"no rows for table {tid}")
    rows: list[str] = []
    for r in recs:
        if r["row"] not in rows:
            rows.append(r["row"])
    index = {(r["row"], r["arm"]): r for r in recs}
    lines = [
        caption(tid, title, recs),
        "",
        "| " + " | ".join(esc(h) for h in [first] + [c[1] for c in cols]) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in cols]) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [esc(row)] + [esc(cell(index.get((row, c[0])), c[2])) for c in cols]
            )
            + " |"
        )
    return "\n".join(lines)


def value(tid: str, row: str, arm: str) -> str:
    for r in table_rows(tid):
        if r["row"] == row and r["arm"] == arm:
            return r["value"]
    raise KeyError((tid, row, arm))


def _pr(number: int) -> str:
    return f"[#{number}](https://github.com/syv-ai/HyperQwen/pull/{number})"


def render_headline() -> str:
    v = value
    k1 = [float(r["value"]) for r in table_rows("k1") if r["arm"].startswith("ratio")]
    K1_ROW = "16 q / 4 kv heads per rank (TP=2 shape), cold L2"
    e1 = "e1_ms_ratio"
    ctx = ("N=1, 128k", "N=1, 32k", "N=1, 8k")
    ms_fa2 = " / ".join(v(e1, c, "td_over_fa2") for c in ctx)
    ms_fp8 = " / ".join(
        v("e2_main", f"time per decode pass (ms), N=1, {c}", "ratio")
        for c in ("128k", "32k", "8k")
    )
    # Row keys of the CSV files, named once.
    tps_128k = "decode tok/s, N=1, 128k"
    lp_32k = "mean |Δ logprob| (nats), 32k document"
    lp_128k = "mean |Δ logprob| (nats), 128k document"
    dec_66k = "decode at ~66k (tok/s)"
    dec_100k = "decode at ~100k (tok/s)"
    ttft_128k = "cold TTFT, 128k prompt (s; one request per arm)"
    dec_66k_ab = "decode at ~66k (tok/s), per boot"
    dec_100k_ab = "decode at ~100k (tok/s), per boot"
    rows = [
        (
            "K1",
            (
                "Windowed drafter attention, D=128, one RTX 3080 20 GB, "
                "220 W power limit, median of 200 CUDA-graph samples"
            ),
            (
                f"{min(k1):.2f}–{max(k1):.2f}× faster than vLLM's Triton "
                "`unified_attention` over four shapes "
                f"({v('k1', K1_ROW, 'pth_2056')} µs against "
                f"{v('k1', K1_ROW, 'triton_2056')} µs per layer at the TP=2 shape, "
                "cold cache)"
            ),
            _pr(338),
        ),
        (
            "E2",
            (
                "`SPEC=mtp`, two RTX 3080 20 GB, TP=2, 220 W power limit, "
                "vLLM 0.31.0, one sitting, N=1"
            ),
            (
                f"Time per decode pass is {ms_fp8} of fp8 + FlashInfer's at "
                "128k / 32k / 8k "
                f"(decode {v('e2_main', tps_128k, 'pth')} against "
                f"{v('e2_main', tps_128k, 'fp8fi')} tok/s at 128k). "
                "Kernel and CUDA-graph mode are not separated."
            ),
            _pr(345),
        ),
        (
            "E1",
            (
                "`SPEC=dflash2`, two RTX 3080 20 GB, TP=2, 220 W power limit, "
                "vLLM 0.31.0, one sitting, N=1"
            ),
            (
                f"Time per decode pass is {ms_fa2} of fp8 + FA2's at "
                "128k / 32k / 8k "
                f"(decode {v('e1_tps', 'N=1, 128k', 'pth_td')} against "
                f"{v('e1_tps', 'N=1, 128k', 'fa2')} tok/s at 128k)."
            ),
            f"{_pr(337)}, {_pr(338)}",
        ),
        (
            "A1",
            (
                "Teacher-forced prompt logprobs against a bf16-KV reference, "
                "32k / 128k documents"
            ),
            (
                f"Mean |Δ logprob| {v('a1_dflash2', lp_32k, 'pth')} / "
                f"{v('a1_dflash2', lp_128k, 'pth')} (int8 + pth) against "
                f"{v('a1_dflash2', lp_32k, 'fa2')} / "
                f"{v('a1_dflash2', lp_128k, 'fa2')} (fp8 + FA2); "
                f"{v('a1_mtp', lp_32k, 'pth')} / "
                f"{v('a1_mtp', lp_128k, 'pth')} against "
                f"{v('a1_mtp', lp_32k, 'fp8fi')} / "
                f"{v('a1_mtp', lp_128k, 'fp8fi')} (fp8 + FlashInfer) "
                "on the MTP checkpoint"
            ),
            f"{_pr(338)}, {_pr(345)}",
        ),
        (
            "M1",
            (
                "Independent runs by the HyperQwen maintainer, one RTX 3090, TP=1, "
                "250 W, vLLM 0.31.0, one boot per arm (M1a, M1b) or two (M1c)"
            ),
            (
                f"`SPEC=mtp` decode at ~66k {v('m1_mtp', dec_66k, 'pth')} against "
                f"{v('m1_mtp', dec_66k, 'fp8fi')} tok/s "
                f"({v('m1_mtp', dec_66k, 'delta')}); "
                f"`SPEC=dflash2` {v('m1_dflash2', dec_66k, 'pth')} against "
                f"{v('m1_dflash2', dec_66k, 'fa2')} tok/s "
                f"({v('m1_dflash2', dec_66k, 'delta')}) at ~66k and "
                f"{v('m1_dflash2', dec_100k, 'pth')} against "
                f"{v('m1_dflash2', dec_100k, 'fa2')} tok/s "
                f"({v('m1_dflash2', dec_100k, 'delta')}) at ~100k; merged `SPEC=mtp` "
                f"default, two boots per arm: {v('m1_mtp_ab', dec_66k_ab, 'pth')} "
                f"against {v('m1_mtp_ab', dec_66k_ab, 'fp8fi')} tok/s at ~66k "
                f"({v('m1_mtp_ab', dec_66k_ab, 'delta')}) and "
                f"{v('m1_mtp_ab', dec_100k_ab, 'pth')} against "
                f"{v('m1_mtp_ab', dec_100k_ab, 'fp8fi')} tok/s at ~100k "
                f"({v('m1_mtp_ab', dec_100k_ab, 'delta')}). Kernel and CUDA-graph "
                "mode are not separated in M1c"
            ),
            (
                "[#325 comment](https://github.com/syv-ai/HyperQwen/pull/325"
                "#issuecomment-6250055160), "
                "[#339 comment](https://github.com/syv-ai/HyperQwen/pull/339"
                "#issuecomment-6262787058), "
                "[#345 comment](https://github.com/syv-ai/HyperQwen/pull/345"
                "#issuecomment-6290008416)"
            ),
        ),
    ]
    lines = ["| Label | Measurement | Result | Source |", "|---|---|---|---|"]
    lines += [f"| {a} | {esc(b)} | {esc(c)} | {d} |" for a, b, c, d in rows]
    pool = (
        int(v("e1_other", "KV pool (tokens)", "fa2")),
        int(v("e1_other", "KV pool (tokens)", "pth_td")),
    )
    lines += [
        "",
        (
            "The comparison arms are ahead in places: with `SPEC=dflash2` the "
            "fp8 + FA2 build prefills a cold 128k prompt "
            f"faster ({float(v('e1_other', ttft_128k, 'fa2')):.1f} s against "
            f"{float(v('e1_other', ttft_128k, 'pth_td')):.1f} s, one request per "
            f"arm, {_pr(338)}) and holds a larger KV pool "
            f"({pool[0]:,} against {pool[1]:,} tokens). Every row is a single "
            "machine, a single model family and, for M1, "
            'one or two boots per arm; the section "Read before the numbers" of the '
            "benchmarks page lists the limits."
        ),
    ]
    return "\n".join(lines)


def marker(tid: str) -> str:
    return "HEADLINE_TABLE" if tid == "headline" else f"TABLE:{tid}"


def render_all() -> dict[str, str]:
    out = {"headline": render_headline()}
    for spec in SPECS:
        out[spec[0]] = render_table(spec[0])
    return out


def block(tid: str, text: str) -> str:
    return f"<!-- {marker(tid)}:start -->\n{text}\n<!-- {marker(tid)}:end -->"


def page_blocks(page: str) -> dict[str, str]:
    found = {}
    for m in re.finditer(
        r"<!-- (HEADLINE_TABLE|TABLE:[a-z0-9_]+):start -->\n(.*?)\n<!-- \1:end -->",
        page,
        re.S,
    ):
        tid = (
            "headline"
            if m.group(1) == "HEADLINE_TABLE"
            else m.group(1)[len("TABLE:") :]
        )
        found[tid] = m.group(2)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the page or the README headline differs",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="rewrite the page tables and the README headline",
    )
    ap.add_argument("--page", type=Path, default=PAGE)
    ap.add_argument("--readme", type=Path, default=README)
    args = ap.parse_args(argv)
    rendered = render_all()
    if not (args.check or args.write):
        print("\n\n".join(block(t, x) for t, x in rendered.items()))
        return 0
    page = args.page.read_text(encoding="utf-8")
    readme = args.readme.read_text(encoding="utf-8")
    have = page_blocks(page)
    bad = sorted(t for t in rendered if have.get(t) != rendered[t])
    bad += sorted(set(have) - set(rendered))
    if page_blocks(readme) != {"headline": rendered["headline"]}:
        bad.append("headline (README.md)")
    if args.check:
        for t in bad:
            print(f"differs: {t}", file=sys.stderr)
        return 1 if bad else 0
    for t, x in rendered.items():
        page = _replace_block(page, t, x, args.page)
    args.page.write_text(page, encoding="utf-8")
    readme = _replace_block(readme, "headline", rendered["headline"], args.readme)
    args.readme.write_text(readme, encoding="utf-8")
    return 0


def _replace_block(text: str, tid: str, body: str, path: Path) -> str:
    pattern = re.compile(
        rf"<!-- {re.escape(marker(tid))}:start -->\n.*?"
        rf"\n<!-- {re.escape(marker(tid))}:end -->",
        re.S,
    )
    if not pattern.search(text):
        raise SystemExit(f"{path}: no block for {tid}")
    return pattern.sub(lambda _m: block(tid, body), text)


if __name__ == "__main__":
    sys.exit(main())
