# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Dev timing of the tensor-core kernel on production shapes.

NOT the gate measurement: the gate numbers come from the CUPTI/ABBA harness
(bench/). This script gives quick, honest numbers while tuning,
labelled "dev timing":

* production geometry (D=256, Hkv=2, G=6, block 816, NHD, bf16 q/out), the
  planner's launch (bq, S) for the production workspace;
* one launch unit = attention (+ combine if S > 1), captured 20 times in a
  CUDA graph; a sample = one graph replay timed with CUDA events, divided
  by 20; 2 s of warm-up first (power-limit equilibrium);
* KV working set >= 67 MB per launch, far above the 5 MB L2, so no
  rotation is needed at L >= 64k;
* reports median and IQR over n samples, GB/s = KV bytes read once /
  time, eta_mem against the separate-stream hardware ceiling, and executed
  TFLOPS with eta_mma against the f16f32 hardware ceiling
  (docs/benchmark-methodology.md). Executed
  flops are bench/kernel_bench.py's model: 4 D x issued rows x 64 tokens per KV
  tile up to each row tile's causal limit, with the issued rows of the kernel
  that runs (16 WM for the 16-row kernel, 8 ceil(rows / 8) for the decode
  kernel at D = 256, rows <= 32);
* ``--prefill``: the query length is a prefill chunk (planner prefill launch,
  bq = 10 at G = 6), ``--L`` is the KV length (prefix + chunk).

Run on an otherwise idle GPU:
    docker run --rm --gpus device=0 --network none -v $PWD:/w -w /w \\
      -e PYTHONPATH=/w/python:/w/tests/gpu --entrypoint /app/venv/bin/python \\
      pth-flash-attn-dev tests/gpu/attn_dev_timing.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import types
from pathlib import Path

import grid_cases as grid
import numpy as np
import torch
from attn_lib import ATTN_LIB, AttnLib
from grid_cuda import cache_to
from pth_flash_attn.planner import plan_launch, wm_for_rows
from pth_testing import ref_capi
from pth_testing.reference import make_int8_cache, make_queries

# Hardware ceilings measured at a 220 W power limit (docs/benchmark-methodology.md):
# separate-stream loader ceiling and the f16-in/f32-acc mma.sync ceiling.
# A run at another power limit passes that
# limit's ceilings with --ceilings, so its eta columns never mix power limits.
SEP_CEILING_GBS = {65536: 688.4, 131072: 698.8, 262144: 699.6}
MMA_CEILING_TFLOPS = 66.2
CEILINGS_POWER_LIMIT_W: float | None = 220.0
D, HKV, G, BLOCK = 256, 2, 6, 816
CELL_BYTES = 2 * D + 8  # K, V int8 + two f32 scales per (token, KV head)
REPLAYS = 20  # launches per graph replay
WARMUP_S = 2.0


def executed_flops(q: int, L: int, Hkv: int, d: int, bq: int, group: int) -> int:
    """Tensor-core flops of one request (bench/kernel_bench.py executed_flops).

    Each row tile of ``bq`` tokens multiplies every 64-token KV tile up to its
    last row's causal limit L - q + j1 with the kernel's issued rows.
    """
    rows = bq * group
    if d == 256 and rows <= 8 * 4:  # the decode kernel (attn_decode.cuh)
        issued = 8 * math.ceil(rows / 8)
    else:
        issued = 16 * wm_for_rows(rows)
    tiles = sum(-(-(L - q + min(j0 + bq, q)) // 64) for j0 in range(0, q, bq))
    return 4 * d * issued * 64 * tiles * Hkv


def time_shape(
    lib: AttnLib,
    q_len: int,
    B: int,
    L: int,
    samples: int,
    part: str = "all",
    layout: str = "inline",
    direct: int = 0,
    warmup_s: float = WARMUP_S,
    prefill: bool = False,
) -> dict:
    dev = torch.device("cuda", 0)
    T = q_len * B
    cache = cache_to(
        make_int8_cache([L] * B, Hkv=HKV, D=D, block_size=BLOCK, seed=1), dev
    )
    if layout == "decoupled":
        # vLLM #56164's opt-in layout: K, V and the scales as separate dense
        # tensors (K/V rows 16-B aligned, so the loader uses 16-B cp.async).
        cache = types.SimpleNamespace(
            k_cache=cache.k_cache.contiguous(),
            v_cache=cache.v_cache.contiguous(),
            k_scale=cache.k_scale.contiguous(),
            v_scale=cache.v_scale.contiguous(),
            block_table=cache.block_table,
            seq_lens=cache.seq_lens,
            block_size=cache.block_size,
            Hkv=cache.Hkv,
            D=cache.D,
        )
    q, qsl = make_queries([q_len] * B, G * HKV, D, seed=2, dtype=torch.bfloat16)
    q, qsl = q.to(dev), qsl.to(dev)
    plan = plan_launch(
        num_tokens=T, num_reqs=B, max_query_len=q_len, G=G, Hkv=HKV, D=D,
        num_sms=grid.NUM_SMS, is_prefill=prefill,
        ws_token_splits=grid.production_ws_token_splits(),
    )  # fmt: skip
    S = plan.num_splits
    # Balanced row tile: no more tokens per tile than the request has. The
    # planner's bq = floor(16 wm / G) is 2 at q = 1 (balanced tiles
    # use one row); the decode kernel's n-tiles follow bq * G,
    # so q = 1 would otherwise pay for 12 rows.
    bq = min(plan.bq, q_len)
    out = torch.empty(T, G * HKV, D, dtype=torch.bfloat16, device=dev)
    ws_o = torch.empty(S, T, G * HKV, D, device=dev) if S > 1 else None
    ws_lse = torch.empty(S, T, G * HKV, device=dev) if S > 1 else None
    args = ref_capi.attn_args(
        cache, q, qsl, None if S > 1 else out, softmax_scale=D**-0.5,
        bq=bq, num_splits=S, ws_o=ws_o, ws_lse=ws_lse,
    )  # fmt: skip
    cargs = ref_capi.combine_args(ws_o, ws_lse, out, qsl, B, S) if S > 1 else None
    stream = torch.cuda.Stream(dev)

    def unit(s: int) -> None:
        if part != "combine":
            rc = lib.attention(args, host=False, stream=s)
            assert rc == 0, lib.last_error()
        if cargs is not None and part != "attention":
            rc = lib.combine(cargs, host=False, stream=s)
            assert rc == 0, lib.last_error()

    with torch.cuda.stream(stream):
        unit(stream.cuda_stream)  # sets the smem attribute before capture
    torch.cuda.synchronize(dev)
    if direct:
        # Profiler mode (ncu): plain launches, no graph, no timing.
        with torch.cuda.stream(stream):
            for _ in range(direct):
                unit(stream.cuda_stream)
        torch.cuda.synchronize(dev)
        return {"label": "direct launches", "q": q_len, "B": B, "L": L, "S": S}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(REPLAYS):
            unit(stream.cuda_stream)
    t_end = time.monotonic() + warmup_s
    while time.monotonic() < t_end:
        graph.replay()
        torch.cuda.synchronize(dev)
    # The card is held at its 220 W cap: record the SM clock and
    # power while sampling, because a kernel that burns more power per byte
    # runs the SMs slower.
    mhz: list[float] = []
    watts: list[float] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            mhz.append(float(torch.cuda.clock_rate(dev)))
            watts.append(torch.cuda.power_draw(dev) / 1e3)
            stop.wait(0.05)

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    ts = []
    for _ in range(samples):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3 / REPLAYS)
    stop.set()
    poller.join()
    t = np.array(ts)
    med = float(np.median(t))
    q1, q3 = np.percentile(t, [25, 75])
    kv_bytes = B * L * HKV * CELL_BYTES
    rows_exec = executed_flops(q_len, L, HKV, D, bq, G) * B
    flops = rows_exec
    gbs = kv_bytes / med / 1e9
    ceil = SEP_CEILING_GBS.get(L, SEP_CEILING_GBS[131072])
    return {
        "label": "dev timing",
        "part": part,
        "layout": layout,
        "q": q_len, "B": B, "L": L, "S": S, "wm": plan.wm, "bq": bq,
        "prefill": prefill,
        "n": samples, "median_us": med * 1e6,
        "series_us": [float(x) * 1e6 for x in t],
        "sm_mhz": float(np.median(mhz)) if mhz else 0.0,
        "power_w": float(np.median(watts)) if watts else 0.0,
        "iqr_rel": float((q3 - q1) / med),
        "gb_s": gbs, "eta_mem": gbs / ceil, "ceiling_gb_s": ceil,
        "tflops_exec": flops / med / 1e12,
        "eta_mma": flops / med / 1e12 / MMA_CEILING_TFLOPS,
        "ceilings_power_limit_w": CEILINGS_POWER_LIMIT_W,
        "mma_ceiling_tflops": MMA_CEILING_TFLOPS,
    }  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--q", type=int, nargs="+", default=[1, 4, 10])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--L", type=int, nargs="+", default=[65536, 131072])
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--lib", help="library path (experiments; default build/attn)")
    ap.add_argument(
        "--part",
        choices=("all", "attention", "combine"),
        default="all",
        help="time only one kernel of the unit (diagnostics)",
    )
    ap.add_argument("--layout", choices=("inline", "decoupled"), default="inline")
    ap.add_argument(
        "--prefill", action="store_true",
        help="--q are prefill chunks and --L the KV lengths (prefix + chunk)",
    )  # fmt: skip
    ap.add_argument(
        "--loader", type=int, default=0, help="0 default, 1 separate, 2 whole cell"
    )
    ap.add_argument(
        "--direct", type=int, default=0,
        help="only launch the unit this many times, untimed (for ncu)",
    )  # fmt: skip
    ap.add_argument(
        "--warmup-s", type=float, default=WARMUP_S,
        help="seconds of replays before sampling (0: sample from a cold clock)",
    )  # fmt: skip
    ap.add_argument(
        "--ceilings", default="",
        help='JSON {"power_limit_w": .., "copy_gbs": {"L": GB/s}, "mma_tflops": ..}'
        " written by bench/ceilings.py for the run's power limit (default: the"
        " hardware ceilings measured at a 220 W power limit)",
    )  # fmt: skip
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    if args.ceilings:
        global SEP_CEILING_GBS, MMA_CEILING_TFLOPS, CEILINGS_POWER_LIMIT_W  # noqa: PLW0603
        c = json.loads(Path(args.ceilings).read_text())
        SEP_CEILING_GBS = {int(k): float(v) for k, v in c["copy_gbs"].items()}
        MMA_CEILING_TFLOPS = float(c["mma_tflops"])
        CEILINGS_POWER_LIMIT_W = c["power_limit_w"]
    lib = AttnLib(Path(args.lib) if args.lib else ATTN_LIB, loader=args.loader)
    print(f"library: {lib.path} loader={args.loader} layout={args.layout}")
    limit = (
        "unknown power limit"
        if CEILINGS_POWER_LIMIT_W is None
        else f"{CEILINGS_POWER_LIMIT_W:g} W power limit"
    )
    print(f"ceilings: {limit}")
    rows = []
    heads = ("q", "B", "L", "S", "wm", "us", "IQR%", "GB/s", "eta_mem", "TFLOPS",
             "eta_mma", "MHz", "W")  # fmt: skip
    widths = (3, 2, 7, 3, 2, 9, 6, 7, 7, 7, 7, 5, 5)
    print(" ".join(f"{h:>{w}}" for h, w in zip(heads, widths)))
    for q_len in args.q:
        for B in args.B:
            for L in args.L:
                r = time_shape(
                    lib,
                    q_len,
                    B,
                    L,
                    args.samples,
                    args.part,
                    args.layout,
                    args.direct,
                    args.warmup_s,
                    args.prefill,
                )
                rows.append(r)
                if args.direct:
                    continue
                print(
                    f"{q_len:>3} {B:>2} {L:>7} {r['S']:>3} {r['wm']:>2} "
                    f"{r['median_us']:>9.1f} {100 * r['iqr_rel']:>6.2f} "
                    f"{r['gb_s']:>7.1f} {r['eta_mem']:>7.3f} "
                    f"{r['tflops_exec']:>7.2f} {r['eta_mma']:>7.3f} "
                    f"{r['sm_mhz']:>5.0f} {r['power_w']:>5.0f}",
                    flush=True,
                )
                torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
