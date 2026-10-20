# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Time the attention kernel on the benchmark grid.

One process of this program runs a list of *experiments* (one shape each)
and writes one raw file per experiment in the format ``bench/bench_report.py``
already reads for the hardware ceilings (``raw_format`` 1), so the statistics,
rejection rules and result records are those of the ceilings, unchanged:

* **arms**: ``kernel`` (the library under test), ``kernel#aa`` (the same
  kernel again: the A/A control that gives the MDE) and, on small shapes
  only, ``ref`` (the scalar reference kernel: a sanity row, never a target);
* **design**: mirrored ABBA rounds (``stats.mirrored_schedule``), 20 graph
  replays per block, 2 s warm-up per experiment, odd processes reversed (both
  the arm order and the experiment order);
* **timer**: the CUPTI kernel timer (``build/k/libpth_cupti_timer.so``,
  ``bench/cupti_timer.cu``). One launch of a shape is every kernel of every
  slice (attention plus combine when S > 1, two slices for a mixed batch);
  its duration is the span from the first kernel's start to the last
  kernel's end (``kernel_span``; ``kernel`` when it is one kernel).
  ``--timer events``: CUDA events around each graph replay
  (``graph_replay``), recorded as such;
* **cold L2**: when one launch touches less than 4 x L2, the inputs are
  rotated over enough independent copies that 4 x L2 bytes pass between two
  uses of the same copy;
* **launch geometry**: from ``pth_flash_attn.planner`` with the SM count of
  the device (never a hard-coded 68), exactly as vLLM will launch it.

Work per launch, written into each arm's metadata for ``kernel_gate.py``:
``bytes_per_launch`` = KV bytes read once (``(2D + 8)`` per token
per KV head), ``flops_useful`` = ``4 D`` per (query row, visible key) and
``flops_executed`` = what the tensor cores are issued with the kernel's
tiling. Two kernels exist (``kernel_variant``): the 16-row-tile kernel pads
the rows of a row tile to ``16 wm`` per MMA, the swap-AB decode kernel
(attn_decode.cuh, row tiles of at most 32 rows at D = 256) puts the rows on
the MMA's N and pads them to whole 8-row n-tiles; both pad the keys to whole
64-token tiles up to the row tile's causal limit. The row tile is the
balanced ``bq = min(planner bq, q)``: the planner's bq is 2 at q = 1 (12
rows, two n-tiles), the harness passes one token.
The cache content is synthetic (random
int8 and scales on the device): the kernel's speed does not depend on the
values, and building full-size caches with the CPU reference writer
would take minutes per shape.

Subcommands: ``list`` (experiment names for bench/k_run.sh) and ``run``.
``--device cpu`` runs the host build of the reference library with a
wall-clock timer (``timer: host``): only for the CPU tests of the whole
pipeline; kernel_gate.py refuses such records.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def kernel_source_root(argv: Sequence[str]) -> Path:
    """The source tree whose ``python/`` and ``tests/gpu`` this process uses.

    A run times ``build/k/libpth_attn.so``, built from a named commit
    (bench/k_build.sh keeps that commit's tree in ``build/k/src``). The
    Python side that fills the C structs and plans the launch must be the
    same commit's, or an edit of ``tests/gpu/attn_lib.py`` or
    ``ref_capi.py`` in the working tree meanwhile changes (or breaks) a
    running benchmark. With
    ``--kernel-lib <dir>/libpth_attn.so`` and ``<dir>/src`` present that tree
    is used; otherwise (stand-in kernels, CPU tests) the working tree.
    """
    lib = None
    for i, a in enumerate(argv):
        if a == "--kernel-lib" and i + 1 < len(argv):
            lib = argv[i + 1]
        elif a.startswith("--kernel-lib="):
            lib = a.split("=", 1)[1]
    if lib is not None:
        src = Path(lib).resolve().parent / "src"
        if (src / "python" / "pth_flash_attn").is_dir():
            return src
    return ROOT


SOURCE_ROOT = kernel_source_root(sys.argv)
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(SOURCE_ROOT / "python"))
import stats  # noqa: E402
from pth_flash_attn import planner  # noqa: E402

# The C-struct side must come from the kernel's own tree (see above): importing the
# working tree's pth_testing against another tree's attn_lib would hand it a foreign
# ctypes struct (ArgumentError).
from pth_testing import ref_capi  # noqa: E402
from pth_testing.reference import packed_views  # noqa: E402

RAW_FORMAT = 1  # bench_report.load_raw
TIMER_LIB = ROOT / "build" / "k" / "libpth_cupti_timer.so"
TIMER_ABI = 1
ATTN_LIB = ROOT / "build" / "attn" / "libpth_attn.so"

# ---------------------------------------------------------------- constants
# Design values are those of the ceilings runs.
REPLAYS_PER_BLOCK = 20
ROUNDS = stats.SCHEDULED_BLOCKS  # 36 rounds per arm per process
WARMUP_S = 2.0
COLD_L2_FACTOR = 4  # >= 4 x L2 between reuses
SCALE_BYTES = 4  # one f32 scale per (token, head) per K and V
LOAD_TILE = 64  # tile_traits.cuh kLoadTileTokens (ref and kernel)
MMA_ROWS_PER_WARP = planner.MMA_ROWS_PER_WARP
# The swap-AB decode kernel (csrc/pth_attn/attn_decode.cuh: kDecRowsPerNt,
# kDecMaxRows): the MMA's N is 8 rows of the row tile, up to 4 n-tiles.
DEC_ROWS_PER_NT = 8
DEC_MAX_ROWS = 32
KERNEL_TILE16, KERNEL_DECODE = "tile16", "decode"
# The quick tier's prefill chunks run with prefixes up to this; the 200k
# prefix (one process took ~800 s with 36 rounds x 3 arms, RTX 3080 20 GB,
# 2026-10-11) stays in the full tier.
PREFILL_QUICK_MAX_PREFIX = 131072
# Target geometry per rank.
D, HKV, G, BLOCK = 256, 2, 6, 816
# KV lengths and query shapes.
L_DECODE = (1024, 4096, 16384, 32768, 65536, 131072, 204800, 262144)
Q_DECODE = (1, 2, 4, 8, 10, 16)
B_DECODE = (1, 2, 4, 8)
Q_PREFILL = (16, 33, 128, 512, 2048, 8192)
PREFIX_PREFILL = (0, 32768, 131072, 204800)
MIXED = ((8188, 0), (4, 65536))  # (q, prefix): one prefill chunk + one MTP verify
# The ref arm (scalar fp32 kernel, ~10 GB/s) is timed only where one launch
# stays short: sum over requests of q * L <= this (L=64k decode: 6.6 ms,
# RTX 3080 20 GB, 2026-10-11).
REF_MAX_WORK = 65536
# Synthetic cache: int8 in [-127, 127] (the writer never stores -128) and
# scales uniform in [0.01, 0.1] (bf16 rows of magnitude 1-13).
INT8_MAX = 127
SCALE_RANGE = (0.01, 0.1)
# Spare pages in the pool beyond the requests' pages (any block id may be
# touched by a wrong kernel; one spare is enough for the random placement).
SPARE_BLOCKS = 1
MAX_RECORDS = 1 << 16  # CUPTI records drained per block (20 x a few kernels)
PROD_MAX_NUM_SEQS, PROD_Q_DEC_MAX = 2, 16  # workspace floor


# ---------------------------------------------------------------- shapes
@dataclass(frozen=True)
class Slice:
    """Requests launched together: one planner call, one kernel launch."""

    reqs: tuple[int, ...]
    is_prefill: bool


@dataclass(frozen=True)
class Shape:
    """One experiment: requests ``(q, prefix)`` with KV length q + prefix."""

    name: str
    family: str
    q_lens: tuple[int, ...]
    prefix: tuple[int, ...]
    slices: tuple[Slice, ...]
    tier: str = "full"
    D: int = D
    Hkv: int = HKV
    G: int = G
    block_size: int = BLOCK
    layout: str = "NHD"

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(q + p for q, p in zip(self.q_lens, self.prefix))

    @property
    def Hq(self) -> int:
        return self.G * self.Hkv

    @property
    def has_ref(self) -> bool:
        return sum(q * L for q, L in zip(self.q_lens, self.seq_lens)) <= REF_MAX_WORK


def decode_shape(q: int, B: int, L: int, tier: str) -> Shape:
    """B requests of q query tokens over KV length L (the gate-grid rows)."""
    return Shape(
        f"dec-q{q}-b{B}-L{L // 1024}k",
        "decode",
        (q,) * B,
        (L - q,) * B,
        (Slice(tuple(range(B)), False),),
        tier,
    )


def grid(family: str = "all") -> list[Shape]:
    """The shape grid, in a fixed order; ``quick`` marks the gate rows."""
    out: list[Shape] = []
    if family in ("all", "decode"):
        for q in Q_DECODE:
            for B in B_DECODE:
                for L in L_DECODE:
                    gate = q in (1, 4, 10) and B in (1, 2) and L >= 65536
                    out.append(decode_shape(q, B, L, "quick" if gate else "full"))
    if family in ("all", "prefill"):
        for q in Q_PREFILL:
            for p in PREFIX_PREFILL:
                out.append(
                    Shape(
                        f"pre-q{q}-P{p // 1024}k",
                        "prefill",
                        (q,),
                        (p,),
                        (Slice((0,), True),),
                        "quick"
                        if q == 8192 and p <= PREFILL_QUICK_MAX_PREFIX
                        else "full",
                    )
                )
    if family in ("all", "mixed"):
        (qp, pp), (qd, pd) = MIXED
        out.append(
            Shape(
                f"mix-q{qp}-q{qd}-L{(qd + pd) // 1024}k",
                "mixed",
                (qp, qd),
                (pp, pd),
                (Slice((0,), True), Slice((1,), False)),
                "quick",
            )
        )
    if family == "smoke":
        # A few short shapes from the grid (decode, verify, one prefill): an
        # end-to-end check of a run in minutes, e.g. with the reference
        # library as stand-in kernel. Contains no gate row.
        out = [
            decode_shape(1, 1, 4096, "quick"),
            decode_shape(4, 2, 4096, "quick"),
            decode_shape(10, 1, 1024, "quick"),
            Shape(
                "pre-q128-P0k", "prefill", (128,), (0,), (Slice((0,), True),), "quick"
            ),
        ]
    if family == "selftest":
        # Tiny shapes for the CPU tests (host reference library).
        out = [
            Shape(
                "self-dec-b2",
                "selftest",
                (1, 1),
                (129, 40),
                (Slice((0, 1), False),),
                "quick",
                D=64,
                G=2,
                block_size=16,
            ),
            Shape(
                "self-mix",
                "selftest",
                (20, 2),
                (0, 70),
                (Slice((0,), True), Slice((1,), False)),
                "quick",
                D=64,
                G=2,
                block_size=16,
            ),
        ]
    if not out:
        raise ValueError(f"unknown family {family!r}")
    return out


def select(family: str, tier: str, only: Sequence[str] = ()) -> list[Shape]:
    shapes = [s for s in grid(family) if tier == "full" or s.tier == "quick"]
    if only:
        names = {s.name for s in shapes}
        missing = sorted(set(only) - names)
        if missing:
            raise ValueError(f"unknown experiments: {missing}")
        shapes = [s for s in shapes if s.name in set(only)]
    return shapes


# ---------------------------------------------------------------- planning
@dataclass(frozen=True)
class SlicePlan:
    """The planner's launch for one slice, plus the work it does."""

    reqs: tuple[int, ...]
    is_prefill: bool
    num_tokens: int
    wm: int
    bq: int
    num_splits: int
    q_tiles_upper: int
    kernel: str  # KERNEL_TILE16 | KERNEL_DECODE: which kernel the launch runs
    bytes_kv: int
    flops_useful: int
    flops_executed: int


def kv_bytes(L: int, Hkv: int, D: int) -> int:
    """KV bytes read once for one request: K, V and two scales."""
    return L * Hkv * (2 * D + 2 * SCALE_BYTES)


def useful_flops(q: int, L: int, Hq: int, D: int) -> int:
    """4 D flops (QK and PV multiply-adds) per query row and visible key.

    Query token i of q sees the keys j <= L - q + i (bottom-right causal):
    sum_i (L - q + i + 1) = q (L - q) + q (q + 1) / 2.
    """
    return 4 * D * Hq * (q * (L - q) + q * (q + 1) // 2)


VariantOf = Callable[[int, int, int], str]  # (D, bq, G) -> KERNEL_*


def modelled_variant(D: int, bq: int, G: int) -> str:
    """Which kernel a row tile of ``bq`` tokens x ``G`` heads runs, from the
    dispatch rule of csrc/pth_attn/instantiate_d256.cu (decode kernel for
    D = 256 and rows <= 32). Used where no library is loaded (CPU tests, the
    sweep's ``list``); a run takes the answer from the library itself."""
    if D == 256 and bq * G <= DEC_MAX_ROWS:
        return KERNEL_DECODE
    return KERNEL_TILE16


def library_variant(lib: Any) -> VariantOf:
    """The variant function that asks the loaded library (``pth_attn_tiling``)
    instead of assuming the dispatch rule: a library without the decode
    kernel (a build without it) then models the 16-row-tile kernel,
    and a load tile other than 64 tokens is an error here, not a silently
    wrong eta_mma. The decode kernel is the only one that reports wn = 1 at
    up to 32 rows (the 16-row-tile kernel has wn = 4 / wm >= 2 there)."""
    tiling = getattr(lib, "tiling", None)
    if tiling is None:
        return modelled_variant

    def variant(D: int, bq: int, G: int) -> str:
        n_tile, _sub, wn = tiling(D, bq, G)
        if n_tile != LOAD_TILE:
            raise RuntimeError(
                f"kernel tile {n_tile} != {LOAD_TILE} tokens: update executed_flops"
            )
        decode = D == 256 and bq * G <= DEC_MAX_ROWS and wn == 1
        return KERNEL_DECODE if decode else KERNEL_TILE16

    return variant


def issued_rows(kernel: str, rows: int, wm: int) -> int:
    """Rows of the row tile that one pass over a 64-token KV tile multiplies
    (the M of the 16-row-tile kernel, the N of the decode kernel).

    tile16: ``16 wm`` (all warps' m16 slices, padded rows included).
    decode: ``8 ceil(rows / 8)`` (S^T = K Q^T has the rows on N, in whole
    m16n8k16 n-tiles, so q = 1 at G = 6 issues 8 rows, q = 4 issues 24).
    """
    if kernel == KERNEL_DECODE:
        return DEC_ROWS_PER_NT * -(-rows // DEC_ROWS_PER_NT)
    if kernel == KERNEL_TILE16:
        return MMA_ROWS_PER_WARP * wm
    raise ValueError(f"unknown kernel variant {kernel!r}")


def executed_flops(
    q: int, L: int, Hkv: int, D: int, bq: int, rows: int, n_tile: int = LOAD_TILE
) -> int:
    """Tensor-core flops issued by the kernel for one request.

    Each row tile (``bq`` query tokens; ``rows`` = ``issued_rows(...)`` MMA
    rows after padding) of each KV head multiplies with every whole
    ``n_tile``-token tile up to its last row's causal limit L - q + j1:
    4 D x rows x n_tile flops per tile (QK and PV), whatever the
    masking inside the tile. Splits only partition the tiles; the combine's
    few flops are not counted.
    """
    total = 0
    for j0 in range(0, q, bq):
        j1 = min(j0 + bq, q)
        tiles = -(-(L - q + j1) // n_tile)
        total += tiles
    return 4 * D * rows * n_tile * total * Hkv


def slice_plan(
    shape: Shape,
    reqs: tuple[int, ...],
    is_prefill: bool,
    *,
    bq: int,
    num_splits: int,
    variant_of: VariantOf,
) -> SlicePlan:
    """The launch of one slice with row tile ``bq`` and ``num_splits``, and the
    work it does. The kernel derives its tile height from ``bq * G`` (wm =
    wm_for_rows, attn_launch.cu), so ``wm`` here is that, not the planner's."""
    qs = [shape.q_lens[r] for r in reqs]
    Ls = [shape.seq_lens[r] for r in reqs]
    T = sum(qs)
    rows = bq * shape.G
    wm = planner.wm_for_rows(rows)
    kernel = variant_of(shape.D, bq, shape.G)
    issued = issued_rows(kernel, rows, wm)
    return SlicePlan(
        reqs=reqs,
        is_prefill=is_prefill,
        num_tokens=T,
        wm=wm,
        bq=bq,
        num_splits=num_splits,
        q_tiles_upper=T // bq + len(reqs),  # planner.plan_launch's bound
        kernel=kernel,
        bytes_kv=sum(kv_bytes(L, shape.Hkv, shape.D) for L in Ls),
        flops_useful=sum(useful_flops(q, L, shape.Hq, shape.D) for q, L in zip(qs, Ls)),
        flops_executed=sum(
            executed_flops(q, L, shape.Hkv, shape.D, bq, issued) for q, L in zip(qs, Ls)
        ),
    )


def plan_slice(
    shape: Shape,
    sl: Slice,
    num_sms: int,
    variant_of: VariantOf = modelled_variant,
) -> SlicePlan:
    """Planner launch for one slice (workspace sized for this batch, never
    below the production reservation), with the balanced row tile
    ``bq = min(planner bq, q)``: at q = 1 the planner's bq is 2, which would
    hand the decode kernel 12 rows (2 n-tiles) for one token. The split count
    is the planner's: both bq give one row tile there (q <= planner bq)."""
    qs = [shape.q_lens[r] for r in sl.reqs]
    T, R, qmax = sum(qs), len(qs), max(qs)
    w_ts = planner.workspace_token_splits(
        max_num_seqs=max(R, PROD_MAX_NUM_SEQS),
        q_dec_max=max(qmax, PROD_Q_DEC_MAX),
        Hkv=shape.Hkv,
        D=shape.D,
        num_sms=num_sms,
    )
    p = planner.plan_launch(
        num_tokens=T,
        num_reqs=R,
        max_query_len=qmax,
        G=shape.G,
        Hkv=shape.Hkv,
        D=shape.D,
        num_sms=num_sms,
        is_prefill=sl.is_prefill,
        ws_token_splits=w_ts,
    )
    return slice_plan(
        shape,
        sl.reqs,
        sl.is_prefill,
        bq=min(p.bq, qmax),
        num_splits=p.num_splits,
        variant_of=variant_of,
    )


def plan_shape(
    shape: Shape,
    num_sms: int,
    variant_of: VariantOf = modelled_variant,
) -> list[SlicePlan]:
    return [plan_slice(shape, sl, num_sms, variant_of) for sl in shape.slices]


# ---------------------------------------------------------------- inputs
def synthetic_cache(shape: Shape, seed: int, device: torch.device) -> SimpleNamespace:
    """A paged int8_per_token_head pool holding every request of ``shape``.

    Same packed layout and views as vLLM (reference.packed_views: strides
    from the packed tensor, scale views inside the cells), random pages
    (shuffled block ids, so no kernel can rely on consecutive pages), random
    int8 content and positive scales, generated on the device.
    """
    D_, Hkv, bs = shape.D, shape.Hkv, shape.block_size
    C = 2 * (D_ + SCALE_BYTES)
    page = Hkv * bs * C
    nblk = [-(-L // bs) for L in shape.seq_lens]
    nb = sum(nblk) + SPARE_BLOCKS
    g = torch.Generator(device=device).manual_seed(seed)
    storage = torch.randint(
        -INT8_MAX,
        INT8_MAX + 1,
        (nb * page,),
        dtype=torch.int8,
        device=device,
        generator=g,
    )
    strides = (page, bs * C, C, 1) if shape.layout == "HND" else (page, C, Hkv * C, 1)
    kv_packed = torch.as_strided(storage, (nb, Hkv, bs, C), strides)
    k_cache, v_cache, k_scale, v_scale = packed_views(kv_packed)
    lo, hi = SCALE_RANGE
    for s in (k_scale, v_scale):
        s.copy_(torch.rand(s.shape, device=device, generator=g) * (hi - lo) + lo)
    perm = torch.randperm(nb, generator=torch.Generator().manual_seed(seed))
    bt = torch.zeros(len(nblk), max(nblk), dtype=torch.int32)
    k = 0
    for r, n in enumerate(nblk):
        bt[r, :n] = perm[k : k + n].to(torch.int32)
        k += n
    return SimpleNamespace(
        storage=storage,
        k_cache=k_cache[..., :D_],
        v_cache=v_cache[..., :D_],
        k_scale=k_scale,
        v_scale=v_scale,
        block_table=bt.to(device),
        seq_lens=torch.tensor(shape.seq_lens, dtype=torch.int32, device=device),
        block_size=bs,
        Hkv=Hkv,
        D=D_,
    )


def _slice_cache(cache: SimpleNamespace, reqs: Sequence[int]) -> SimpleNamespace:
    idx = torch.tensor(list(reqs), device=cache.block_table.device)
    return SimpleNamespace(
        **{
            **vars(cache),
            "block_table": cache.block_table[idx].contiguous(),
            "seq_lens": cache.seq_lens[idx].contiguous(),
        }
    )


@dataclass
class SliceIO:
    cache: SimpleNamespace
    q: torch.Tensor
    qsl: torch.Tensor
    out: torch.Tensor
    ws_o: torch.Tensor | None
    ws_lse: torch.Tensor | None


def make_io(
    shape: Shape,
    plans: Sequence[SlicePlan],
    seed: int,
    device: torch.device,
    cache: SimpleNamespace | None = None,
) -> list[SliceIO]:
    """One cache (all requests) and per-slice q/out/workspace.
    ``cache``: reuse a pool built by ``synthetic_cache`` (bench/k_sweep.py times
    several launch configurations over the same pool; building one per
    configuration would multiply the memory and give each its own page order).
    """
    if cache is None:
        cache = synthetic_cache(shape, seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    ios = []
    for p in plans:
        qs = [shape.q_lens[r] for r in p.reqs]
        qsl = torch.zeros(len(qs) + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(qs).cumsum(0)
        q = torch.randn(p.num_tokens, shape.Hq, shape.D, device=device, generator=g).to(
            torch.bfloat16
        )
        out = torch.empty(
            p.num_tokens, shape.Hq, shape.D, device=device, dtype=torch.bfloat16
        )
        ws_o = ws_lse = None
        if p.num_splits > 1:
            ws_o = torch.empty(
                p.num_splits, p.num_tokens, shape.Hq, shape.D, device=device
            )
            ws_lse = torch.empty(p.num_splits, p.num_tokens, shape.Hq, device=device)
        ios.append(
            SliceIO(_slice_cache(cache, p.reqs), q, qsl.to(device), out, ws_o, ws_lse)
        )
    return ios


def working_set_bytes(ios: Sequence[SliceIO]) -> int:
    """Bytes one launch touches: the cache pool plus q/out/workspace."""
    n = ios[0].cache.storage.numel()
    for io in ios:
        for t in (io.q, io.out, io.ws_o, io.ws_lse):
            if t is not None:
                n += t.numel() * t.element_size()
    return n


def rotations(working_set: int, l2_bytes: int) -> int:
    """Independent input copies so 4 x L2 bytes pass between reuses."""
    return max(1, math.ceil(COLD_L2_FACTOR * l2_bytes / max(working_set, 1)))


# ---------------------------------------------------------------- kernels
def load_lib(kind: str, path: str | None, device: torch.device) -> Any:
    """``ref`` (ref_capi.RefLib) or ``attn`` (tests/gpu/attn_lib.AttnLib)."""
    host = device.type == "cpu"
    if kind == "ref":
        # The reference library is built in the working tree (build/ref) by
        # k_build.sh, not by the kernel commit: its path is the harness's,
        # whichever tree's ref_capi is imported.
        default = ref_capi.HOST_LIB if host else ref_capi.CUDA_LIB
        return ref_capi.RefLib(
            Path(path) if path else ROOT / "build" / "ref" / default.name
        )
    if kind == "attn":
        if host:
            raise ValueError("the attention library has no host build")
        sys.path.insert(0, str(SOURCE_ROOT / "tests" / "gpu"))
        from attn_lib import AttnLib  # type: ignore[import-not-found]

        return AttnLib(Path(path) if path else ATTN_LIB)
    raise ValueError(f"unknown kernel kind {kind!r}")


def make_launch(
    lib: Any,
    ios: Sequence[SliceIO],
    plans: Sequence[SlicePlan],
    scale: float,
    host: bool,
) -> Callable[[int], None]:
    """``launch(stream)``: every slice's attention (+ combine when S > 1)."""
    calls = []
    for io, p in zip(ios, plans):
        a = ref_capi.attn_args(
            io.cache,
            io.q,
            io.qsl,
            io.out,
            softmax_scale=scale,
            bq=p.bq,
            num_splits=p.num_splits,
            ws_o=io.ws_o,
            ws_lse=io.ws_lse,
        )
        c = (
            ref_capi.combine_args(
                io.ws_o, io.ws_lse, io.out, io.qsl, len(p.reqs), p.num_splits
            )
            if io.ws_o is not None and io.ws_lse is not None
            else None
        )
        calls.append((a, c))

    def launch(stream: int) -> None:
        for a, c in calls:
            rc = lib.attention(a, host=host, stream=stream)
            if rc != ref_capi.OK:
                raise RuntimeError(f"attention returned {rc}: {lib.last_error()}")
            if c is not None:
                rc = lib.combine(c, host=host, stream=stream)
                if rc != ref_capi.OK:
                    raise RuntimeError(f"combine returned {rc}: {lib.last_error()}")

    return launch


# ---------------------------------------------------------------- timers
class CuptiTimer:
    """build/k/libpth_cupti_timer.so: per-kernel GPU start/end (ns)."""

    name, kind = "cupti", "kernel"

    def __init__(self, path: Path = TIMER_LIB) -> None:
        self.lib = ctypes.CDLL(str(path))
        self.lib.pth_timer_collect.restype = ctypes.c_int64
        if self.lib.pth_timer_abi_version() != TIMER_ABI:
            raise RuntimeError(f"{path}: timer ABI differs")
        self.start = (ctypes.c_uint64 * MAX_RECORDS)()
        self.end = (ctypes.c_uint64 * MAX_RECORDS)()
        self.lib.pth_timer_start()

    def drain(self) -> list[tuple[int, int]]:
        n = self.lib.pth_timer_collect(self.start, self.end, MAX_RECORDS)
        if n < 0:
            raise RuntimeError("more CUPTI records than MAX_RECORDS in one block")
        return [(self.start[i], self.end[i]) for i in range(n)]


def spans(
    records: Sequence[tuple[int, int]], launches: int, per_launch: int
) -> list[float]:
    """Durations of ``launches`` launches of ``per_launch`` kernels each.

    Records arrive sorted by start; launch i owns records [i k, (i+1) k).
    A wrong count means a kernel record was lost or a foreign kernel ran in
    the block: the block is invalid and the process stops (exit 2), as in the
    ceiling programs.
    """
    if len(records) != launches * per_launch:
        raise RuntimeError(
            f"expected {launches * per_launch} kernel records, got {len(records)}"
        )
    out = []
    for i in range(launches):
        grp = records[i * per_launch : (i + 1) * per_launch]
        out.append(float(max(e for _, e in grp) - grp[0][0]))
    return out


# ---------------------------------------------------------------- one experiment
@dataclass
class Arm:
    name: str
    lib: Any
    role: str
    graphs: list[Any] = field(default_factory=list)
    launches: list[Callable[[int], None]] = field(default_factory=list)


def device_info(device: torch.device) -> dict[str, Any]:
    if device.type == "cpu":
        return {
            "cuda_index": -1,
            "name": "cpu",
            "pci_bus_id": "cpu",
            "cc": "0.0",
            "sm_count": 1,
            "l2_bytes": 0,
            "runtime_version": 0,
        }
    p = torch.cuda.get_device_properties(device)
    return {
        "cuda_index": device.index or 0,
        "name": p.name,
        # torch's type stubs lack the PCI fields; same format as CUDA's
        # cudaDeviceGetPCIBusId (and sol_copy's device.json).
        "pci_bus_id": "{:04x}:{:02x}:{:02x}.0".format(
            *(getattr(p, k) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
        ),
        "cc": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "runtime_version": int(torch._C._cuda_getCompiledVersion()),
    }


@dataclass(frozen=True)
class ArmPlan:
    """One timed arm: its name and role, the library that runs it (a key of
    ``libs``) and the launch it makes. ``bench/k_sweep.py`` gives arms of one
    shape different plans (split count, row tile); the kernel benchmark gives the
    kernel, its A/A copy and the reference the planner's launch."""

    name: str
    role: str
    lib: str
    plans: tuple[SlicePlan, ...]


def default_arm_plans(
    shape: Shape, a: argparse.Namespace, plans: Sequence[SlicePlan]
) -> list[ArmPlan]:
    """``kernel``, its A/A copy and, on short shapes, the ``ref`` comparison."""
    names = ["kernel", "kernel#aa"] + (["ref"] if shape.has_ref and a.ref else [])
    shared = tuple(plans)  # one object: the arms share their input buffers
    return [
        ArmPlan(
            n,
            "comparison"
            if n == "ref"
            else ("aa_control" if n.endswith("#aa") else "subject"),
            "ref" if n == "ref" else "kernel",
            shared,
        )
        for n in names
    ]


def work_of(plans: Sequence[SlicePlan]) -> dict[str, Any]:
    """Work per launch of one arm, written into its metadata for the gates."""
    return {
        "bytes_per_launch": sum(p.bytes_kv for p in plans),
        "flops_useful": sum(p.flops_useful for p in plans),
        "flops_executed": sum(p.flops_executed for p in plans),
        "slices": [asdict(p) for p in plans],
    }


def run_experiment(
    shape: Shape,
    a: argparse.Namespace,
    process: int,
    libs: dict[str, Any],
    timer: CuptiTimer | None,
    dev: dict[str, Any],
    device: torch.device,
    arm_plans: Sequence[ArmPlan] | None = None,
) -> dict[str, Any]:
    """Measure one shape in this process; returns the raw record.
    ``arm_plans`` None: the gate arms (kernel, kernel#aa, ref) with the
    planner's launch. Arms that share one ``plans`` tuple share their input
    buffers, as the gate arms always did."""
    host = device.type == "cpu"
    num_sms = a.num_sms or dev["sm_count"]
    plans = plan_shape(shape, num_sms, library_variant(libs["kernel"]))
    if arm_plans is None:
        arm_plans = default_arm_plans(shape, a, plans)
    scale = shape.D**-0.5
    arms = [Arm(ap.name, libs[ap.lib], ap.role) for ap in arm_plans]
    seed = a.seed * 1000 + process
    # One pool per rotation, shared by every arm; one set of q/out/workspace
    # buffers per distinct plans tuple.
    distinct = {id(ap.plans): ap.plans for ap in arm_plans}

    def build_inputs(seed_r: int) -> dict[int, list[SliceIO]]:
        cache = synthetic_cache(shape, seed_r, device)
        return {
            k: make_io(shape, pl, seed_r, device, cache=cache)
            for k, pl in distinct.items()
        }

    ios0 = build_inputs(seed)
    n_rot = (
        1
        if host
        else rotations(max(map(working_set_bytes, ios0.values())), dev["l2_bytes"])
    )
    rot = [ios0] + [build_inputs(seed + 7 * r) for r in range(1, n_rot)]
    for arm, ap in zip(arms, arm_plans):
        arm.launches = [
            make_launch(arm.lib, r[id(ap.plans)], ap.plans, scale, host) for r in rot
        ]
    stream = None if host else torch.cuda.Stream(device)
    if not host:
        assert stream is not None
        with torch.cuda.stream(stream):
            for arm in arms:
                for fn in arm.launches:
                    fn(stream.cuda_stream)  # load modules outside the capture
            torch.cuda.synchronize(device)
            for arm in arms:
                for fn in arm.launches:
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr, stream=stream):
                        fn(stream.cuda_stream)
                    arm.graphs.append(gr)
        torch.cuda.synchronize(device)

    counter = [0]

    def replay(arm: Arm) -> None:
        i = counter[0] % n_rot
        counter[0] += 1
        if host:
            arm.launches[i](0)
        else:
            arm.graphs[i].replay()

    # Kernels per launch, from one timed replay of each arm (the ref and the
    # kernel may differ, e.g. a fused combine).
    per_launch: dict[str, int] = {}
    if timer is not None:
        timer.drain()
        for arm in arms:
            replay(arm)
            torch.cuda.synchronize(device)
            per_launch[arm.name] = len(timer.drain())
    # Warm-up: power-limit equilibrium, arms alternating.
    t_end = time.monotonic() + a.warmup_s
    while time.monotonic() < t_end:
        for arm in arms:
            replay(arm)
        if not host:
            torch.cuda.synchronize(device)
    if timer is not None:
        timer.drain()

    order = stats.mirrored_schedule(len(arms), a.rounds, reverse=process % 2 == 1)
    blocks = []
    for seq, ai in enumerate(order):
        arm = arms[ai]
        t0 = time.monotonic_ns()
        durs: list[float]
        if host:
            durs = []
            for _ in range(a.replays):
                s0 = time.perf_counter_ns()
                replay(arm)
                durs.append(float(time.perf_counter_ns() - s0))
        elif timer is not None:
            for _ in range(a.replays):
                replay(arm)
            torch.cuda.synchronize(device)
            durs = spans(timer.drain(), a.replays, per_launch[arm.name])
        else:
            assert stream is not None
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(a.replays + 1)]
            with torch.cuda.stream(stream):
                ev[0].record(stream)
                for i in range(a.replays):
                    replay(arm)
                    ev[i + 1].record(stream)
            torch.cuda.synchronize(device)
            durs = [ev[i].elapsed_time(ev[i + 1]) * 1e6 for i in range(a.replays)]
        t1 = time.monotonic_ns()
        blocks.append(
            {
                "seq": seq,
                "round": seq // len(arms),
                "arm": arm.name,
                "t_start_mono_ns": t0,
                "t_end_mono_ns": t1,
                "durations_ns": durs,
            }
        )

    k_any = max(per_launch.values(), default=1)
    if host:
        timer_name, kind = "host", "host_wallclock"
    elif timer is not None:
        timer_name, kind = "cupti", "kernel" if k_any == 1 else "kernel_span"
    else:
        timer_name, kind = "cuda_event", "graph_replay"
    shape_d = {
        "D": shape.D,
        "Hkv": shape.Hkv,
        "G": shape.G,
        "Hq": shape.Hq,
        "block_size": shape.block_size,
        "layout": shape.layout,
        "family": shape.family,
        "batch": len(shape.q_lens),
        "q_lens": ",".join(map(str, shape.q_lens)),
        "seq_lens": ",".join(map(str, shape.seq_lens)),
        "q_max": max(shape.q_lens),
        "seq_len_max": max(shape.seq_lens),
        "splits": ",".join(str(p.num_splits) for p in plans),
        "rotations": n_rot,
        "num_sms": num_sms,
    }
    arms_meta = []
    for arm, ap in zip(arms, arm_plans):
        m: dict[str, Any] = {
            "name": arm.name,
            "role": arm.role,
            "lib": str(getattr(getattr(arm.lib, "lib", None), "_name", "")),
            "kernels_per_launch": per_launch.get(arm.name),
            **work_of(ap.plans),
        }
        if arm.role == "comparison":
            m["label"] = "comparison, not a target"
        arms_meta.append(m)
    return {
        "tool": "kernel_bench",
        "raw_format": RAW_FORMAT,
        "argv": list(a.argv),
        "timer": timer_name,
        "duration_kind": kind,
        "process": process,
        "device": dev,
        "shape": shape_d,
        "design": {
            "replays_per_block": a.replays,
            "warmup_s": a.warmup_s,
            "n_arms": len(arms),
            "n_blocks": len(order),
        },
        "arms": arms_meta,
        "blocks": blocks,
    }


def write_raw(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, allow_nan=False))
    os.replace(tmp, path)  # a killed process never leaves a half file


def cmd_run(
    a: argparse.Namespace,
    shapes: Sequence[Shape] | None = None,
    arm_plans_for: Callable[[Shape, int, VariantOf], Sequence[ArmPlan]] | None = None,
) -> int:
    """Run this process's experiments. Defaults: the gate grid selected by
    the arguments and the gate arms; ``bench/k_sweep.py`` passes its own
    shapes and ``arm_plans_for(shape, num_sms, variant_of)``."""
    device = torch.device(a.device if a.device == "cpu" else "cuda", 0)
    timer = None
    if device.type == "cuda" and a.timer == "cupti":
        timer = CuptiTimer(Path(a.timer_lib))  # before the first kernel
    dev = device_info(device)
    # The ref arm (and the ref stand-in kernel) come from --ref-lib when the
    # run passes its own hashed copy: build/ref is rebuilt by whoever runs
    # tools/build_ref_lib.sh, and a rebuild mid-run voids it.
    kernel_lib = a.kernel_lib or (a.ref_lib if a.kernel_kind == "ref" else None)
    libs = {"kernel": load_lib(a.kernel_kind, kernel_lib, device)}
    libs["ref"] = (
        libs["kernel"]
        if a.kernel_kind == "ref" and not a.kernel_lib
        else load_lib("ref", a.ref_lib, device)
    )
    shapes = list(
        shapes
        if shapes is not None
        else select(a.family, a.tier, [s for s in a.only.split(",") if s])
    )
    if a.process % 2 == 1:
        shapes = shapes[::-1]  # experiment order mirrored too
    out_dir = Path(a.out_dir)
    for shape in shapes:
        t0 = time.monotonic()
        num_sms = a.num_sms or dev["sm_count"]
        raw = run_experiment(
            shape,
            a,
            a.process,
            libs,
            timer,
            dev,
            device,
            arm_plans_for(shape, num_sms, library_variant(libs["kernel"]))
            if arm_plans_for
            else None,
        )
        raw["t_unix_ns"] = time.time_ns()
        write_raw(out_dir / f"{shape.name}.p{a.process}.a{a.attempt}.json", raw)
        print(
            f"{shape.name} p{a.process} a{a.attempt}: {time.monotonic() - t0:.1f} s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if timer is not None:
        timer.lib.pth_timer_stop()
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    shapes = select(a.family, a.tier, [s for s in a.only.split(",") if s])
    if a.copy_ceilings:
        # (B, L) of every decode shape: one sol_copy ceiling each (eta_mem).
        for B, L in sorted(
            {(len(s.q_lens), max(s.seq_lens)) for s in shapes if s.family == "decode"}
        ):
            print(B, L)
        return 0
    for s in shapes:
        print(s.name, s.family, s.tier, int(s.has_ref))
    return 0


def add_run_args(r: argparse.ArgumentParser) -> None:
    """The options of every ``run`` subcommand (shared with bench/k_sweep.py)."""
    r.add_argument(
        "--process",
        type=int,
        required=True,
        help="index of this fresh process (odd ones run in reversed order)",
    )
    r.add_argument(
        "--attempt", type=int, default=0, help="rerun counter, in file names"
    )
    r.add_argument("--out-dir", required=True, help="directory for the raw files")
    r.add_argument("--only", default="", help="comma-separated experiment names")
    r.add_argument(
        "--kernel-kind",
        default="attn",
        choices=("attn", "ref"),
        help="attn = the kernel library; ref = the reference library as stand-in",
    )
    r.add_argument(
        "--kernel-lib",
        default=None,
        help="libpth_attn.so to time (default build/attn/libpth_attn.so)",
    )
    r.add_argument(
        "--ref-lib", default=None, help="the ref library (default build/ref)"
    )
    r.add_argument(
        "--no-ref",
        dest="ref",
        action="store_false",
        help="leave the ref comparison arm out",
    )
    r.add_argument(
        "--timer",
        default="cupti",
        choices=("cupti", "events"),
        help="CUPTI kernel timestamps (default) or CUDA events",
    )
    r.add_argument(
        "--timer-lib",
        default=str(TIMER_LIB),
        help="libpth_cupti_timer.so (bench/k_build.sh)",
    )
    r.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="cpu: host build of the reference library, for pipeline tests only",
    )
    r.add_argument("--num-sms", type=int, default=0, help="0 = the device's")
    r.add_argument("--rounds", type=int, default=ROUNDS, help="blocks per arm")
    r.add_argument(
        "--replays", type=int, default=REPLAYS_PER_BLOCK, help="graph replays per block"
    )
    r.add_argument(
        "--warmup-s",
        type=float,
        default=WARMUP_S,
        help="warm-up seconds per experiment",
    )
    r.add_argument("--seed", type=int, default=0, help="input seed")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description=__doc__
        + "\n`list` needs no GPU. `run` needs one NVIDIA GPU (sm80/86/89), a built "
        "kernel library and the CUPTI timer library; bench/k_run.sh is the driver "
        "that wraps it in a full measurement run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print the experiment names (no GPU)")
    r = sub.add_parser("run", help="time the experiments (needs a GPU)")
    for p in (ls, r):
        p.add_argument(
            "--family",
            default="all",
            choices=("all", "decode", "prefill", "mixed", "smoke", "selftest"),
        )
        p.add_argument("--tier", default="full", choices=("quick", "full"))
    ls.add_argument(
        "--copy-ceilings",
        action="store_true",
        help="print the 'B L' pairs that need a sol_copy ceiling",
    )
    ls.add_argument(
        "--only", default="", help="comma-separated experiment names (a subset)"
    )
    add_run_args(r)
    a = ap.parse_args(argv)
    a.argv = ["bench/kernel_bench.py", *argv]
    return cmd_list(a) if a.cmd == "list" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
