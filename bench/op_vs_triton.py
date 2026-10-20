#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Time ``pth_flash_attn_fwd`` against vLLM's Triton ``unified_attention`` on one shape.

Both kernels read the SAME int8_per_token_head KV cache tensor, in the page layout
vLLM allocates: ``[num_blocks, num_kv_heads, block_size, 2 * (head_size + 4)]`` over
physical NHD storage, each (slot, head) cell holding ``K | k_scale | V | v_scale``,
with the float32 scale views carved from the head padding exactly as vLLM's
``TritonAttentionImpl._ensure_scale_caches`` does. The cache content is synthetic
(random int8 and scales): kernel speed does not depend on the values.

Arms (per regime):

* ``pth``: ``torch.ops.pth_flash_attn.pth_flash_attn_fwd`` with the requested mask
  (``--window``, ``--non-causal``), once with the planner's launch (``wm`` and split
  count from ``pth_flash_attn.planner``) and once per fixed split count of
  ``--sweep-splits``;
* ``triton``: ``vllm.v1.attention.ops.triton_unified_attention.unified_attention`` with
  the same mask, ``kv_quant_mode=INT8_PER_TOKEN_HEAD``: a causal layer without window
  runs ``causal=True`` with no window; a single-token decode (query length 1) gets the
  split-KV (3D) buffers that ``TritonAttentionImpl`` passes, so that Triton runs its 3D
  kernel as vLLM does (without them it falls back to the 2D kernel and the ratio at q=1
  is about ten times too large); a windowed layer runs ``window_size=(W - 1, 0)``,
  the value vLLM passes for a layer with sliding window W (with ``--non-causal``:
  ``causal=False``). The two masks of a windowed non-causal layer are not identical
  (the op sees keys up to ``p + W - 1``, Triton up to ``p``), so that comparison is a
  cost comparison for the same layer, not a numerical one;
* ``triton_causal`` (only when the requested mask is not causal-without-window and
  every block is real): the Triton kernel on the plain causal mask, as a reference row.

Timing, as ``tests/gpu/attn_dev_timing.py``: one *unit* of ``--replays`` launches is
captured in a CUDA graph; a sample is one graph replay timed with CUDA events divided by
``--replays``; ``--warmup-s`` seconds of replays come first; ``--samples`` samples give
the median and IQR; the SM clock and power draw are polled meanwhile. Two cache regimes:

* ``warm``: every launch of the unit reads the same blocks, so the working set can
  sit in L2;
* ``cold``: each launch reads its own set of blocks, so the unit touches
  ``replays x`` the per-launch bytes, far more than L2, and a launch finds its KV in
  HBM, as a layer does in a real step.

A plumbing check runs first (when every block is real): on the plain causal mask both
kernels must agree on the same cache (max / mean absolute difference are recorded).
Windowed layers keep only
the blocks that overlap the window real; earlier blocks map to the null block 0 (zeros,
scale 1.0), as vLLM does for evicted window positions, so a 128k-token windowed layer
needs only a few blocks of memory per launch.

Needs: one NVIDIA GPU (sm80, sm86 or sm89) that nothing else is using; PyTorch with
CUDA; the built op library (``pth_flash_attn``); vLLM (0.31.0 is the tested version)
for the Triton arm. Output: a line per row on stdout, a ratio table, and ``--json``
FILE with every row (including all samples). Example (the decode shape of the
Qwen3.8-27B layers at TP=2: D=256, 12 query heads, 2 KV heads, one query token, 128k
context)::

    python bench/op_vs_triton.py --head-dim 256 --num-q-heads 12 --num-kv-heads 2 \\
        --query-len 1 --seq-len 131072 --json out.json

The windowed drafter layer (D=128, 16 query heads, 4 KV heads, 8 query tokens, window
2048, block 864)::

    python bench/op_vs_triton.py --head-dim 128 --num-q-heads 16 --num-kv-heads 4 \\
        --query-len 8 --seq-len 2056 --block-size 864 --window 2048 --non-causal \\
        --json drafter.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pth_flash_attn import ops  # noqa: E402
from pth_flash_attn import planner as pl  # noqa: E402

SCALE_PAD = 4  # one float32 scale per (token, head) per side: the head padding
DEFAULT_BLOCK = {256: 816, 128: 864}  # page sizes of the two supported cache shapes
# One launch's key window: the last q positions see keys in [L - W - q, L); blocks
# wholly before that are never read by a windowed layer and map to the null block.
NULL_BLOCK = 0


# ---------------------------------------------------------------- pure helpers
def default_block_size(head_size: int) -> int:
    """The page size vLLM uses for this head size (256: 816, 128: 864)."""
    if head_size not in DEFAULT_BLOCK:
        raise SystemExit(f"--block-size is required for head size {head_size}")
    return DEFAULT_BLOCK[head_size]


def triton_mask(window: int, causal: bool) -> tuple[bool, tuple[int, int]]:
    """(causal, window_size) arguments of ``unified_attention`` for the same layer.

    No window: ``(causal, (-1, -1))``. Window W: ``(causal, (W - 1, 0))``, the sliding
    window a decoder-type layer carries in vLLM.
    """
    if window <= 0:
        return causal, (-1, -1)
    return causal, (window - 1, 0)


def real_block_range(
    seq_len: int, query_len: int, window: int, block_size: int
) -> tuple[int, int]:
    """(first, count) of the blocks of one request that must hold real data.

    Without a window every block is real. With window W the query rows sit in the last
    ``query_len`` positions, so keys before ``seq_len - window - query_len`` are never
    read; blocks wholly before that position are null.
    """
    n_blocks = math.ceil(seq_len / block_size)
    if window <= 0:
        return 0, n_blocks
    first = max(0, seq_len - window - query_len) // block_size
    return first, n_blocks - first


def build_block_tables(
    seq_len: int,
    query_len: int,
    window: int,
    block_size: int,
    batch: int,
    rotations: int,
    device: torch.device | str,
) -> tuple[list[torch.Tensor], int]:
    """``rotations`` independent block tables ``[batch, n_blocks]`` and the block count.

    Table ``r`` gives request ``b`` its own run of real blocks (ids from 1; block 0 is
    the null block), so launches that use different tables share no cache line. The
    returned count is the number of blocks the cache needs (including the null block).
    """
    first, n_real = real_block_range(seq_len, query_len, window, block_size)
    n_blocks = math.ceil(seq_len / block_size)
    tables = []
    for r in range(rotations):
        bt = torch.zeros(batch, n_blocks, dtype=torch.int32)
        for b in range(batch):
            base = 1 + (r * batch + b) * n_real
            bt[b, first:] = torch.arange(base, base + n_real, dtype=torch.int32)
        tables.append(bt.to(device).contiguous())
    return tables, 1 + rotations * batch * n_real


def cache_bytes(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int
) -> int:
    """Bytes of the packed cache: ``2 * (head_size + 4)`` per (slot, head)."""
    return num_blocks * block_size * num_kv_heads * 2 * (head_size + SCALE_PAD)


def quantiles(t_us: np.ndarray) -> dict[str, float]:
    """Median, mean, min, max and IQR (absolute and relative) of the samples."""
    q1, q3 = np.percentile(t_us, [25, 75])
    med = float(np.median(t_us))
    return {
        "median_us": med,
        "mean_us": float(t_us.mean()),
        "min_us": float(t_us.min()),
        "max_us": float(t_us.max()),
        "q1_us": float(q1),
        "q3_us": float(q3),
        "iqr_rel": float((q3 - q1) / med),
    }


# ---------------------------------------------------------------- cache
def build_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """The packed int8 cache and its views: ``(raw, key, value, k_scale, v_scale)``.

    ``key`` and ``value`` are ``[num_blocks, block_size, num_kv_heads, head_size + 4]``
    views (the op takes ``[..., :head_size]``); the scale views are float32
    ``[num_blocks, block_size, num_kv_heads]`` carved from the padding.
    """
    phs = head_size + SCALE_PAD
    raw = torch.empty(
        num_blocks, block_size, num_kv_heads, 2 * phs, dtype=torch.int8, device=device
    )
    raw.random_(-127, 128)
    kv = raw.permute(
        0, 2, 1, 3
    )  # logical (nb, nkv, block, content), strides as in vLLM
    base = torch.tensor([], dtype=torch.float32, device=device).set_(
        kv.untyped_storage()
    )
    st = kv.stride()

    def f32(elems: int) -> int:  # int8: bytes == elements
        return elems // 4

    blk, head, slot = f32(st[0]), f32(st[1]), f32(st[2])
    k_off = f32(kv.storage_offset()) + f32(head_size)
    v_off = f32(kv.storage_offset()) + f32(phs + head_size)
    shape = (num_blocks, block_size, num_kv_heads)
    ks = torch.as_strided(base, shape, (blk, slot, head), k_off)
    vs = torch.as_strided(base, shape, (blk, slot, head), v_off)
    ks.copy_(torch.rand(ks.shape, device=device) * 0.015 + 0.005)
    vs.copy_(torch.rand(vs.shape, device=device) * 0.015 + 0.005)
    # block 0 is vLLM's null block: zeros, scale 1.0 (the scale caches start at 1.0)
    raw[NULL_BLOCK].zero_()
    ks[NULL_BLOCK].fill_(1.0)
    vs[NULL_BLOCK].fill_(1.0)
    key, value = kv.transpose(1, 2).split(phs, dim=-1)
    return raw, key, value, ks, vs


# ---------------------------------------------------------------- device information
def device_info(dev: torch.device) -> dict[str, Any]:
    """Best-effort NVML facts (power limit, driver); ``None`` where NVML cannot say."""
    info: dict[str, Any] = {
        "name": torch.cuda.get_device_name(dev),
        "sm_count": torch.cuda.get_device_properties(dev).multi_processor_count,
        "capability": list(torch.cuda.get_device_capability(dev)),
        "power_limit_w": None,
        "driver_version": None,
    }
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(dev.index or 0)
        info["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
        drv = pynvml.nvmlSystemGetDriverVersion()
        info["driver_version"] = drv.decode() if isinstance(drv, bytes) else str(drv)
    except Exception:  # noqa: BLE001 (NVML is optional here)
        pass
    return info


# ---------------------------------------------------------------- timing
def time_unit(
    launch: Any,
    args: argparse.Namespace,
    dev: torch.device,
    label: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Capture ``args.replays`` launches in a CUDA graph and time its replays.

    ``launch(i)`` issues launch ``i`` of the unit (it reads rotation ``i % rotations``).
    """
    replays = args.replays
    stream = torch.cuda.Stream(dev)
    with torch.cuda.stream(stream):
        for i in range(replays):
            launch(i)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(replays):
            launch(i)
    t_end = time.monotonic() + args.warmup_s
    while time.monotonic() < t_end:
        graph.replay()
        torch.cuda.synchronize(dev)
    mhz: list[float] = []
    watts: list[float] = []
    stop = threading.Event()

    def poll() -> None:
        try:
            while not stop.is_set():
                mhz.append(float(torch.cuda.clock_rate(dev)))
                watts.append(torch.cuda.power_draw(dev) / 1e3)
                stop.wait(0.05)
        except Exception:  # noqa: BLE001 (NVML bindings missing: no clock or power rows)
            return

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    ts = []
    for _ in range(args.samples):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        graph.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) / 1e3 / replays)
    stop.set()
    th.join()
    t_us = np.array(ts) * 1e6
    r: dict[str, Any] = {
        "label": label,
        "n": args.samples,
        "replays": replays,
        **quantiles(t_us),
        "sm_mhz": float(np.median(mhz)) if mhz else None,
        "power_w": float(np.median(watts)) if watts else None,
        "series_us": [float(x) for x in t_us],
    }
    r.update(extra)
    clock = (
        "" if r["sm_mhz"] is None else f"  {r['sm_mhz']:.0f} MHz {r['power_w']:.0f} W"
    )
    print(
        f"{label:<56} median {r['median_us']:9.2f} us  "
        f"IQR {100 * r['iqr_rel']:5.2f} %{clock}",
        flush=True,
    )
    return r


# ---------------------------------------------------------------- command line
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shape = ap.add_argument_group(
        "shape (default: the decode shape, D=256, 12/2 heads, 1 token, 128k)"
    )
    shape.add_argument(
        "--head-size",
        "--head-dim",
        dest="head_size",
        type=int,
        default=256,
        choices=(64, 128, 256),
        help="head size D",
    )
    shape.add_argument(
        "--num-q-heads", type=int, default=12, help="query heads on this rank"
    )
    shape.add_argument(
        "--num-kv-heads", type=int, default=2, help="KV heads on this rank"
    )
    shape.add_argument(
        "--query-len",
        type=int,
        default=1,
        help="query tokens per request (decode: 1; verify: 4-16)",
    )
    shape.add_argument("--batch", type=int, default=1, help="requests per launch")
    shape.add_argument(
        "--seq-len",
        type=int,
        default=131072,
        help="KV length per request, including the query tokens",
    )
    shape.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="page size (default: 816 for D=256, 864 for D=128)",
    )
    shape.add_argument(
        "--window",
        type=int,
        default=0,
        help="sliding window W in tokens (0 = none; needs D=128)",
    )
    shape.add_argument(
        "--non-causal", action="store_true", help="non-causal mask (needs D=128)"
    )
    run = ap.add_argument_group("timing")
    run.add_argument(
        "--samples", type=int, default=200, help="graph replays timed per row"
    )
    run.add_argument(
        "--replays",
        type=int,
        default=20,
        help="launches per graph (and cold rotations)",
    )
    run.add_argument(
        "--warmup-s", type=float, default=2.0, help="seconds of replays before sampling"
    )
    run.add_argument(
        "--regimes", nargs="+", default=["cold", "warm"], choices=("cold", "warm")
    )
    run.add_argument(
        "--sweep-splits",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 6, 8, 12, 16],
        help="fixed split counts timed next to the planner's (empty: planner only)",
    )
    run.add_argument("--device", type=int, default=0, help="CUDA device index")
    run.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--json", required=True, help="output file with every row and all samples"
    )
    a = ap.parse_args(argv)
    if a.block_size is None:
        a.block_size = default_block_size(a.head_size)
    if a.num_q_heads % a.num_kv_heads:
        ap.error("--num-q-heads must be a multiple of --num-kv-heads")
    if (a.window > 0 or a.non_causal) and a.head_size != 128:
        ap.error(
            "--window and --non-causal need --head-size 128 (the only windowed kernel)"
        )
    if a.seq_len < a.query_len or a.query_len < 1 or a.batch < 1:
        ap.error("need 1 <= --query-len <= --seq-len and --batch >= 1")
    if a.replays < 1 or a.samples < 1:
        ap.error("--replays and --samples must be >= 1")
    return a


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(argv)
    # Imported late so that --help works without vLLM and without a GPU.
    import vllm
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    from vllm.v1.kv_cache_interface import KVQuantMode

    if not torch.cuda.is_available():
        raise SystemExit("op_vs_triton needs a CUDA GPU")
    if not ops._LIBRARY_LOADED:
        raise SystemExit(
            f"the op library {ops.LIBRARY_PATH} is not built (tools/build_in_image.sh)"
        )
    dev = torch.device("cuda", a.device)
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed)
    D, HKV, Q, B, L = a.head_size, a.num_kv_heads, a.query_len, a.batch, a.seq_len
    HQ, G, BLOCK, WINDOW = (
        a.num_q_heads,
        a.num_q_heads // a.num_kv_heads,
        a.block_size,
        a.window,
    )
    causal = not a.non_causal
    scale = D**-0.5
    info = device_info(dev)
    num_sms = info["sm_count"]
    rotations = max(a.replays if "cold" in a.regimes else 1, 1)
    tables, num_blocks = build_block_tables(L, Q, WINDOW, BLOCK, B, rotations, dev)
    need = cache_bytes(num_blocks, BLOCK, HKV, D)
    free, _ = torch.cuda.mem_get_info(dev)
    if need > 0.9 * free:
        raise SystemExit(
            f"the cache needs {need / 2**30:.1f} GiB, {free / 2**30:.1f} GiB are free: "
            "lower --replays or --batch (or use --regimes warm)"
        )
    _, key, value, ks, vs = build_cache(num_blocks, BLOCK, HKV, D, dev)
    q = torch.randn(B * Q, HQ, D, dtype=torch.bfloat16, device=dev)
    out_p = torch.zeros_like(q)
    out_t = torch.zeros_like(q)
    qsl = torch.arange(0, B * Q + 1, Q, dtype=torch.int32, device=dev)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=dev)

    w_ts = pl.workspace_token_splits(
        max_num_seqs=max(2, B), q_dec_max=Q, Hkv=HKV, D=D, num_sms=num_sms
    )
    launch_plan = pl.plan_launch(
        num_tokens=B * Q,
        num_reqs=B,
        max_query_len=Q,
        G=G,
        Hkv=HKV,
        D=D,
        num_sms=num_sms,
        is_prefill=False,
        ws_token_splits=w_ts,
        window=WINDOW,
    )
    print(
        f"device {info['name']} ({num_sms} SMs, power limit "
        f"{info['power_limit_w']} W); planner decode launch: wm={launch_plan.wm} "
        f"bq={launch_plan.bq} S={launch_plan.num_splits} (W_ts={w_ts})"
    )

    sweep = sorted(set(a.sweep_splits) | {launch_plan.num_splits})
    workspaces: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for s in sweep:
        if s <= 1:
            z = torch.empty(0, dtype=torch.float32, device=dev)
            workspaces[s] = (z, z)
        else:
            workspaces[s] = (
                torch.empty(s, B * Q, HQ, D, dtype=torch.float32, device=dev),
                torch.empty(s, B * Q, HQ, dtype=torch.float32, device=dev),
            )

    def run_pth(
        out: torch.Tensor, bt: torch.Tensor, splits: int, window: int, is_causal: bool
    ) -> None:
        ws_o, ws_lse = workspaces[splits]
        ops.pth_flash_attn_fwd(
            out,
            q,
            key[..., :D],
            value[..., :D],
            ks,
            vs,
            bt,
            seq_lens,
            qsl,
            Q,
            launch_plan.wm,
            splits,
            scale,
            ws_o,
            ws_lse,
            window,
            is_causal,
        )

    # vLLM's TritonAttentionImpl passes the buffers of the split-KV (3D) kernel to
    # unified_attention; without them a single-token decode (max_seqlen_q == 1)
    # runs the slower 2D kernel. Mirror its sizes (triton_attn.py, 0.31.0):
    # seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads. vLLM then snaps
    # it to the nearest CUDA-graph capture size; this harness has no engine
    # configuration and uses the unsnapped value.
    par_segments = triton_attn.NUM_PAR_SOFTMAX_SEGMENTS
    seg_threshold = triton_attn.MIN_LAUNCH_GRID_SIZE_2D // HKV
    seg_3d = {
        "seq_threshold_3D": seg_threshold,
        "num_par_softmax_segments": par_segments,
        "softmax_segm_output": torch.empty(
            seg_threshold,
            HQ,
            par_segments,
            1 << (D - 1).bit_length(),
            dtype=torch.float32,
            device=dev,
        ),
        "softmax_segm_max": torch.empty(
            seg_threshold, HQ, par_segments, dtype=torch.float32, device=dev
        ),
        "softmax_segm_expsum": torch.empty(
            seg_threshold, HQ, par_segments, dtype=torch.float32, device=dev
        ),
    }

    def run_tri(
        out: torch.Tensor, bt: torch.Tensor, window: int, is_causal: bool
    ) -> None:
        t_causal, t_window = triton_mask(window, is_causal)
        unified_attention(
            q=q,
            k=key,
            v=value,
            out=out,
            cu_seqlens_q=qsl,
            max_seqlen_q=Q,
            seqused_k=seq_lens,
            max_seqlen_k=L,
            softmax_scale=scale,
            causal=t_causal,
            window_size=t_window,
            block_table=bt,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            k_scale_cache=ks,
            v_scale_cache=vs,
            **seg_3d,
        )

    results: dict[str, Any] = {
        "shape": {
            "head_size": D,
            "num_q_heads": HQ,
            "num_kv_heads": HKV,
            "group": G,
            "query_len": Q,
            "batch": B,
            "seq_len": L,
            "block_size": BLOCK,
            "window": WINDOW,
            "causal": causal,
        },
        "device": info,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "planner_plan": {
            "wm": launch_plan.wm,
            "bq": launch_plan.bq,
            "num_splits": launch_plan.num_splits,
            "W_ts": w_ts,
        },
        "args": {k: v for k, v in vars(a).items() if k != "json"},
        "rows": [],
    }

    # ---- 1. plumbing: plain causal mask on the same cache; the kernels must agree.
    # Only meaningful when every block is real: with a null prefix (a windowed layer
    # much longer than its window) the causal mask would read mostly null blocks.
    all_real = real_block_range(L, Q, WINDOW, BLOCK)[0] == 0
    if all_real:
        run_pth(out_p, tables[0], launch_plan.num_splits, 0, True)
        run_tri(out_t, tables[0], 0, True)
        torch.cuda.synchronize(dev)
        diff = (out_p.float() - out_t.float()).abs()
        results["plumbing_check"] = {
            "what": "plain causal mask, same cache: pth (planner launch) vs "
            "unified_attention",
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "out_abs_max": float(out_t.float().abs().max()),
        }
    else:
        results["plumbing_check"] = {
            "skipped": "null-prefix cache (window much shorter than seq-len)"
        }
    print("plumbing check:", results["plumbing_check"])

    # ---- 2. timings
    def pth_call(splits: int) -> Callable[[torch.Tensor], None]:
        return lambda bt: run_pth(out_p, bt, splits, WINDOW, causal)

    def tri_call(window: int, is_causal: bool) -> Callable[[torch.Tensor], None]:
        return lambda bt: run_tri(out_t, bt, window, is_causal)

    def unit(call: Callable[[torch.Tensor], None], bts: list[torch.Tensor]) -> Any:
        """Launch ``i`` of a unit reads block table ``i % len(bts)``."""
        return lambda i: call(bts[i % len(bts)])

    for regime in a.regimes:
        bts = tables[: a.replays if regime == "cold" else 1]
        extra = {"regime": regime, "seq_len": L, "query_len": Q, "batch": B}
        arms: list[tuple[str, Callable[[torch.Tensor], None], dict[str, Any]]] = []
        for s in sweep:
            tag = "planner" if s == launch_plan.num_splits else "sweep"
            arms.append(
                (
                    f"pth S={s} ({tag})",
                    pth_call(s),
                    {"kernel": "pth", "S": s, "kind": tag},
                )
            )
        arms.append(
            (
                "triton unified_attention",
                tri_call(WINDOW, causal),
                {"kernel": "triton", "S": 0, "kind": "same mask"},
            )
        )
        if all_real and (WINDOW > 0 or not causal):
            arms.append(
                (
                    "triton unified_attention, plain causal",
                    tri_call(0, True),
                    {"kernel": "triton_causal", "S": 0, "kind": "reference"},
                )
            )
        for label, call, kind in arms:
            row = time_unit(
                unit(call, bts), a, dev, f"{label} {regime}", dict(extra, **kind)
            )
            results["rows"].append(row)

    # ---- 3. summary: Triton / pth(planner) per regime
    print("\nTriton time / pth time (pth = planner launch):")
    for regime in a.regimes:
        rows = [r for r in results["rows"] if r["regime"] == regime]
        p = next(r for r in rows if r["kernel"] == "pth" and r["kind"] == "planner")
        t = next(r for r in rows if r["kernel"] == "triton")
        ratio = t["median_us"] / p["median_us"]
        print(
            f"  {regime}: {t['median_us']:.2f} us / {p['median_us']:.2f} us "
            f"= {ratio:.2f}x"
        )
        results.setdefault("ratios", {})[regime] = ratio
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
