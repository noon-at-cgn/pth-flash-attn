# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Per-kernel timeline of decode launches: where the fixed per-launch cost goes.

The kernel benchmark (bench/kernel_bench.py) times one launch unit as a
single span, first kernel start to last kernel end, and its eta_mem compares
that span with the copy ceiling T_mem measured in the same run. At q = 4 the
span exceeds T_mem by a roughly constant 14-17 us per launch (RTX 3080 20 GB,
220 W power limit), which is what holds the L = 64k and DEC32K rows
below their targets. A span cannot say which
part of the launch that is, so this tool keeps every CUPTI record of every
launch (same timer, inputs, planner launch, rotations and warm-up as the
gate) and splits the span into

    attn  attention kernel, start to end
    gap   combine start - attention end (the inter-kernel gap in the graph)
    comb  combine kernel, start to end (only when the planner splits, S > 1)
    next  next launch's first start - this launch's last end (replay to
          replay; vLLM replays one graph per step, so this is reported, not
          charged to the kernel)

and, with ``--ceilings-from`` a ``bench/k_run.sh`` result directory of the
same GPU and power limit, the excess over the memory floor: attn - T_mem and
span - T_mem. Fitting attn
against L over a sweep separates the L-proportional part from the intercept
(the fixed part per launch).

Usage (needs an NVIDIA GPU, CUPTI and a built kernel library; see
``--help`` of bench/kernel_bench.py for the library paths):
    python bench/k_timeline.py --kernel-lib build/k/libpth_attn.so \\
        --q 4 --B 1 2 --L 32768 65536 131072 --blocks 60 \\
        --ceilings-from results/<k_run result dir> --out-dir <out dir>/timeline
Outputs: timeline_raw.csv.gz (one row per launch), timeline.json (per
shape: n, median, IQR of every part, T_mem, the fit), timeline.csv (flat).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import kernel_bench as kb  # noqa: E402  (sets the source tree from --kernel-lib)
import numpy as np
import torch

# 60 blocks x 20 replays = 1200 launches per shape: medians of every part to
# well under 0.1 us.
BLOCKS = 60


def quantiles(x: np.ndarray) -> dict[str, float]:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "median_ns": float(med),
        "q1_ns": float(q1),
        "q3_ns": float(q3),
    }


def copy_floor_ns(ceil_dir: Path | None, B: int, L: int) -> float | None:
    """T_mem of the gate: the sol_copy median of the k_run run for the same (B, L)."""
    if ceil_dir is None:
        return None
    p = ceil_dir / f"ceil-copy-b{B}-L{L // 1024}k__sep-nhd.json"
    if not p.is_file():
        return None
    return float(json.loads(p.read_text())["summary"]["median_ns"])


def time_shape(
    lib: Any,
    timer: kb.CuptiTimer,
    dev: dict[str, Any],
    q: int,
    B: int,
    L: int,
    blocks: int,
    replays: int,
    warmup_s: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[list[float]]]:
    shape = kb.decode_shape(q, B, L, "full")
    plans = kb.plan_shape(shape, dev["sm_count"], kb.library_variant(lib))
    scale = shape.D**-0.5
    ios0 = kb.make_io(shape, plans, seed, device)
    n_rot = kb.rotations(kb.working_set_bytes(ios0), dev["l2_bytes"])
    rot = [ios0] + [
        kb.make_io(shape, plans, seed + 7 * r, device) for r in range(1, n_rot)
    ]
    launches = [kb.make_launch(lib, io, plans, scale, False) for io in rot]
    stream = torch.cuda.Stream(device)
    graphs = []
    with torch.cuda.stream(stream):
        for fn in launches:
            fn(stream.cuda_stream)  # load modules outside the capture
        torch.cuda.synchronize(device)
        for fn in launches:
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=stream):
                fn(stream.cuda_stream)
            graphs.append(gr)
    torch.cuda.synchronize(device)
    timer.drain()
    graphs[0].replay()
    torch.cuda.synchronize(device)
    k = len(timer.drain())
    t_end = time.monotonic() + warmup_s  # power-cap equilibrium, as the gate
    i = 0
    while time.monotonic() < t_end:
        # block-sized bursts, drained each time: the CUPTI buffer is finite
        for _ in range(replays):
            graphs[i % n_rot].replay()
            i += 1
        torch.cuda.synchronize(device)
        timer.drain()
    rows: list[list[float]] = []
    for b in range(blocks):
        for r in range(replays):
            graphs[(b * replays + r) % n_rot].replay()
        torch.cuda.synchronize(device)
        rec = timer.drain()
        if len(rec) != replays * k:
            raise RuntimeError(f"expected {replays * k} kernel records, got {len(rec)}")
        for r in range(replays):
            grp = rec[r * k : (r + 1) * k]
            a0, a1 = grp[0]
            c0, c1 = grp[-1] if k > 1 else (a1, a1)
            nxt = rec[(r + 1) * k][0] - c1 if r + 1 < replays else float("nan")
            rows.append([b, r, a1 - a0, c0 - a1, c1 - c0, c1 - a0, nxt])
    del graphs, launches, rot, ios0
    torch.cuda.empty_cache()
    meta = {
        "experiment": shape.name,
        "q": q,
        "B": B,
        "L": L,
        "kernels_per_launch": k,
        "splits": plans[0].num_splits,
        "bq": plans[0].bq,
        "rotations": n_rot,
        "bytes_kv": plans[0].bytes_kv,
    }
    return meta, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kernel-lib", required=True, help="path of libpth_attn.so (bench/k_build.sh)"
    )
    ap.add_argument(
        "--q", type=int, nargs="+", default=[4], help="query tokens per request"
    )
    ap.add_argument(
        "--B", type=int, nargs="+", default=[1, 2], help="requests per launch"
    )
    ap.add_argument(
        "--L", type=int, nargs="+", default=[32768, 65536, 131072], help="KV lengths"
    )
    ap.add_argument("--blocks", type=int, default=BLOCKS, help="blocks per shape")
    ap.add_argument(
        "--replays",
        type=int,
        default=kb.REPLAYS_PER_BLOCK,
        help="graph replays per block",
    )
    ap.add_argument(
        "--warmup-s", type=float, default=kb.WARMUP_S, help="warm-up seconds per shape"
    )
    ap.add_argument("--seed", type=int, default=0, help="input seed")
    ap.add_argument(
        "--ceilings-from",
        type=Path,
        default=None,
        help="bench/k_run.sh result directory whose sol_copy records give T_mem",
    )
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="directory for the outputs"
    )
    a = ap.parse_args(argv)
    device = torch.device("cuda")
    dev = kb.device_info(device)
    lib = kb.load_lib("attn", a.kernel_lib, device)
    timer = kb.CuptiTimer(Path(kb.TIMER_LIB))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["block", "replay", "attn_ns", "gap_ns", "comb_ns", "span_ns", "next_ns"]
    shapes, flat = [], []
    with gzip.open(a.out_dir / "timeline_raw.csv.gz", "wt", newline="") as fz:
        w = csv.writer(fz)
        w.writerow(["experiment", *cols])
        for q in a.q:
            for B in a.B:
                for L in a.L:
                    meta, rows = time_shape(
                        lib,
                        timer,
                        dev,
                        q,
                        B,
                        L,
                        a.blocks,
                        a.replays,
                        a.warmup_s,
                        a.seed,
                        device,
                    )
                    for row in rows:
                        w.writerow([meta["experiment"], *row])
                    arr = np.array(rows, dtype=float)
                    parts = {
                        c: quantiles(arr[:, j]) for j, c in enumerate(cols) if j >= 2
                    }
                    nxt = arr[:, 6]
                    parts["next_ns"] = quantiles(nxt[~np.isnan(nxt)])
                    t_mem = copy_floor_ns(a.ceilings_from, B, L)
                    if t_mem is not None:
                        meta["t_mem_ns"] = t_mem
                        meta["attn_minus_tmem_ns"] = (
                            parts["attn_ns"]["median_ns"] - t_mem
                        )
                        meta["span_minus_tmem_ns"] = (
                            parts["span_ns"]["median_ns"] - t_mem
                        )
                        meta["eta_mem_span"] = t_mem / parts["span_ns"]["median_ns"]
                    meta["parts"] = parts
                    shapes.append(meta)
                    flat.append(
                        {k: v for k, v in meta.items() if k != "parts"}
                        | {
                            f"{c}_{s}": parts[c][s]
                            for c in parts
                            for s in ("median_ns", "q1_ns", "q3_ns")
                        }
                        | {"n": parts["span_ns"]["n"]}
                    )
                    print(
                        f"{meta['experiment']}: S={meta['splits']} attn "
                        f"{parts['attn_ns']['median_ns'] / 1e3:.2f} gap "
                        f"{parts['gap_ns']['median_ns'] / 1e3:.2f} comb "
                        f"{parts['comb_ns']['median_ns'] / 1e3:.2f} span "
                        f"{parts['span_ns']['median_ns'] / 1e3:.2f} us"
                        + (
                            f", T_mem {t_mem / 1e3:.2f} us" if t_mem is not None else ""
                        ),
                        flush=True,
                    )
    fits = []
    for q in a.q:
        for B in a.B:
            sel = [s for s in shapes if s["q"] == q and s["B"] == B]
            if len(sel) < 2:
                continue
            x = np.array([s["L"] for s in sel], dtype=float)
            for part in ("attn_ns", "span_ns"):
                y = np.array([s["parts"][part]["median_ns"] for s in sel])
                slope, icpt = np.polyfit(x, y, 1)
                fits.append(
                    {
                        "q": q,
                        "B": B,
                        "part": part,
                        "L": x.astype(int).tolist(),
                        "intercept_ns": float(icpt),
                        "ns_per_token": float(slope),
                    }
                )
    out = {
        "tool": "k_timeline",
        "argv": ["bench/k_timeline.py", *(sys.argv[1:] if argv is None else argv)],
        "timer": "cupti",
        "device": dev,
        "ceilings_from": str(a.ceilings_from) if a.ceilings_from else None,
        "design": {"blocks": a.blocks, "replays": a.replays, "warmup_s": a.warmup_s},
        "shapes": shapes,
        "linear_fits": fits,
    }
    (a.out_dir / "timeline.json").write_text(json.dumps(out, indent=1))
    with open(a.out_dir / "timeline.csv", "w", newline="") as f:
        # shapes without a copy floor (no ceiling record for that L) lack the
        # t_mem columns: the header is the union, in first-seen order
        fields = list(dict.fromkeys(k for row in flat for k in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)
    timer.lib.pth_timer_stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
