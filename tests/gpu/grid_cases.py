# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The correctness grid: cases, inputs, expected values and the gate.

This module is shared by two suites and therefore has no CUDA code:

* ``tests/gpu/`` runs every case through the scalar reference kernel and the
  combine (``libpth_ref.so``) in the GPU correctness grid;
* ``tests/cpu/test_grid.py`` runs the same harness with the numpy emulator
  of the kernel's arithmetic (``pth_testing.emulate``) in place of the
  kernel, plus deliberately broken "mutant" kernels that the gate must catch.

The gate: for every valid output element
``|out - ref| <= bound`` where ``ref`` is fp64 attention over the exactly
dequantised cache and ``bound`` is ``pth_testing.bound.bound`` for the
tiling the kernel used. Two buffer rules come from the launch structs
(``csrc/pth_attn/attn_args.h``): rows that belong to no request
(``token >= qsl[num_reqs]``) are never written, in ``out`` or in the
workspace, and nothing outside the views is written. Both are checked with
canary bytes.

Grid composition: the test matrix's axes as families, one
factor or one known-risky interaction per family, instead of the Cartesian
product (about 10^6 cases); boundary families carry *sentinel* keys so an
off-by-one at long L is visible (``test_off_by_one_at_long_L_needs_
a_sentinel``). Every case has a fixed seed derived from its name, so a
failure reproduces from its id alone.

The fp64 reference and the bound cost seconds to minutes of CPU per case and
do not depend on the kernel, so they are computed once per (case, code
version) and cached in ``build/grid_expected/``; ``python tests/gpu/grid_cases.py
expected --tier full --jobs N`` fills the cache in parallel before a GPU run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pth_flash_attn.planner import (
    MAX_ROWS_PER_CTA,
    max_tokens_per_tile,
    plan_launch,
    s_alone,
    workspace_token_splits,
)
from pth_testing import bound as bound_mod
from pth_testing import reference as reference_mod
from pth_testing.bound import Tiling, bound
from pth_testing.emulate import emulate
from pth_testing.reference import (
    DenseKV,
    Int8Cache,
    gen_rows,
    make_int8_cache,
    make_queries,
    visible_bounds,
)

REPO = Path(__file__).resolve().parents[2]
EXPECTED_DIR = REPO / "build" / "grid_expected"

# ---------------------------------------------------------------- constants
# SM count of the RTX 3080: planner input for the S_max and production rows.
NUM_SMS = 68
# The reference kernel walks 64-token tiles (ref_capi.h) and has no sub-tiles
# or warps along N; bound() covers it with Tiling(64, 64, 1, S, bq).
REF_N_TILE = 64
# S_max: the largest split count the planner can ask for on this GPU, S_alone
# at the smallest Hkv (1) and the most CTAs/SM (D=64: 3) = ceil(68*3/1) = 204.
S_MAX = s_alone(Hkv=1, D=64, num_sms=NUM_SMS)
# Sentinel key = GAIN * sqrt(D) * u, with every query of the KV head close to
# sqrt(D) * u: its logit is ~GAIN * sqrt(D) (32 at D=64, 64 at D=256), far
# above the O(1-10) logits of random keys, so the sentinels carry the weight
# (same gain as tests/cpu/test_bound.py SENTINEL_GAIN).
SENTINEL_GAIN = 4.0
# Per-row spread of the sentinel queries around sqrt(D) * u (std, units of
# one query element), so rows and heads are not identical.
SENTINEL_Q_NOISE = 0.25
# Canary: every byte of out / workspace buffers starts as 0xFF (NaN in f32,
# f16 and bf16); a byte that still reads 0xFF afterwards was not written.
CANARY_BYTE = 255
# Strided I/O (stride fuzz for q, out and workspace): extra heads and extra
# elements per head row around the view, so no stride equals the shape's.
STRIDE_PAD_HEADS = 1
STRIDE_PAD_ELEMS = 8
# Production descriptor: the workspace is reserved for max_num_seqs=2 and q_dec_max=16.
PROD_MAX_NUM_SEQS = 2
PROD_Q_DEC_MAX = 16

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "f32": torch.float32}


# ---------------------------------------------------------------- cases
@dataclass(frozen=True)
class GridCase:
    """One correctness case: a batch, a cache layout, a launch and its tier.

    ``q_lens[r] <= seq_lens[r]`` (the new tokens are part of the KV length,
    bottom-right alignment). ``window`` / ``causal`` select the visible range
    (sliding window and/or non-causal, D = 128 only).
    ``pad_rows`` extra query rows after the last request model CUDA-graph
    padding (``num_tokens > qsl[num_reqs]``). ``out_dtype`` is ``"f32"``
    (test build) or ``"q"`` (the query dtype, as in production).
    ``ws_token_splits > 0`` carves the workspace from buffers of the
    planner's size instead of exact-size buffers.
    """

    name: str
    family: str
    q_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    G: int = 6
    Hkv: int = 2
    D: int = 256
    block_size: int = 816
    layout: str = "NHD"
    pad_page_bytes: int = 0
    storage_offset_bytes: int = 0
    high_block_ids: bool = False
    scale_mode: str | tuple[str, ...] = "normal"
    q_dtype: str = "bf16"
    out_dtype: str = "f32"
    num_splits: int = 1
    pad_rows: int = 0
    strided_io: bool = False
    sentinel: bool = False
    ws_token_splits: int = 0
    # Cases that must see identical data (one production descriptor at
    # several S) share ``data_name`` (seed source, default: the name) and
    # ``sentinel_splits`` (split counts whose edges get sentinels, default:
    # this case's S), so their outputs can be compared with each other.
    data_name: str = ""
    sentinel_splits: tuple[int, ...] = ()
    tier: str = "full"
    window: int = 0
    causal: bool = True

    @property
    def windowed(self) -> bool:
        """True for the windowed kernels."""
        return self.window > 0 or not self.causal

    @property
    def Hq(self) -> int:
        return self.G * self.Hkv

    @property
    def num_tokens(self) -> int:
        return sum(self.q_lens) + self.pad_rows

    @property
    def seed(self) -> int:
        # Stable across processes and Python versions (hash() is salted).
        return int(
            hashlib.sha256((self.data_name or self.name).encode()).hexdigest()[:8], 16
        )

    @property
    def bq(self) -> int:
        """Query tokens per row tile: the optimised kernel's rule, from the
        planner (decode launch if every q fits one tile, else prefill), so the
        reference runs the row tiling the tensor-core kernel will run."""
        q_max = max(self.q_lens)
        return plan_launch(
            num_tokens=sum(self.q_lens),
            num_reqs=len(self.q_lens),
            max_query_len=q_max,
            G=self.G,
            Hkv=self.Hkv,
            D=self.D,
            num_sms=NUM_SMS,
            is_prefill=q_max > max_tokens_per_tile(self.G),
            ws_token_splits=0,
        ).bq

    @property
    def tiling(self) -> Tiling:
        return Tiling(REF_N_TILE, REF_N_TILE, 1, self.num_splits, self.bq)

    @property
    def torch_q_dtype(self) -> torch.dtype:
        return DTYPES[self.q_dtype]

    @property
    def torch_out_dtype(self) -> torch.dtype:
        return self.torch_q_dtype if self.out_dtype == "q" else DTYPES[self.out_dtype]

    def describe(self) -> dict[str, Any]:
        """JSON-able description (results records, cache key)."""
        d = dataclasses.asdict(self)
        d.update(bq=self.bq, seed=self.seed, num_tokens=self.num_tokens)
        return d


def _check_case(c: GridCase) -> GridCase:
    if len(c.q_lens) != len(c.seq_lens):
        raise ValueError(f"{c.name}: one q_len per seq_len")
    if any(q > L for q, L in zip(c.q_lens, c.seq_lens)):
        raise ValueError(f"{c.name}: q_len > seq_len")
    # One GQA group must fit a 64-row CTA tile.
    if c.G > MAX_ROWS_PER_CTA or c.D not in D_VALUES:
        raise ValueError(f"{c.name}: unsupported head geometry")
    if c.ws_token_splits and c.num_splits * c.num_tokens > c.ws_token_splits:
        raise ValueError(f"{c.name}: S * T exceeds the workspace")
    if c.windowed and c.D not in W_D_VALUES:
        raise ValueError(f"{c.name}: no windowed kernel at D = {c.D}")
    return c


# Grid axes.
L_PAGING = (1, 15, 16, 63, 64, 65, 815, 816, 817, 4096)
BLOCK_SIZES = (16, 64, 528, 816)
Q_ROWS = (1, 2, 5, 10, 11, 16, 37)
G_VALUES = (1, 4, 6, 8, 16)
D_VALUES = (64, 128, 256)
S_VALUES = (1, 2, 34, S_MAX)
L_SPLITS = (65, 817, 4096, 65536)
L_ROWS = 817  # one token past a production page (block 816): two pages
L_LONG = 262144
FUZZ_CASES = 12
# Window axes: windows at a tile edge (64,
# 65), around the DFlash2 drafter's 2048, and 4096; L at the window edges,
# around the production page (816) and at 64k; q in {1, 8, 16}; G in
# {4, 6}; D = 128 (the windowed kernels' head size).
W_VALUES = (64, 65, 2047, 2048, 4096)
W_Q = (1, 8, 16)
W_G = (4, 6)
W_D_VALUES = (128,)
W_FUZZ_CASES = 12
DRAFTER_BLOCK = 864  # the drafter group's uniform block size (measured deployment)


def _paging() -> list[GridCase]:
    """Decode over every length edge x block size x layout, with sentinels.

    Why: page and tile boundaries (L = 15/16/17-style edges around 16, 64,
    816) are where paged address math breaks; sentinels at first/last token,
    tile edge and page edge make one wrong token fail the gate.
    """
    out = []
    i = 0
    for L in L_PAGING:
        for bs in BLOCK_SIZES:
            for layout in ("NHD", "HND"):
                D = D_VALUES[i % len(D_VALUES)]
                G = G_VALUES[i % len(G_VALUES)]
                quick = bs in (16, 816) and L in (1, 16, 64, 65, 817)
                out.append(
                    GridCase(
                        f"page-L{L}-bs{bs}-{layout}-D{D}-gqa{G}",
                        "paging",
                        (1,),
                        (L,),
                        G=G,
                        D=D,
                        block_size=bs,
                        layout=layout,
                        sentinel=True,
                        tier="quick" if quick else "full",
                    )
                )
                i += 1
    return out


def _rows() -> list[GridCase]:
    """Query length x GQA group x head size: row tiling and causal alignment.

    Why: bq = floor(16 wm / G) query tokens per row tile, rows of one tile
    have different causal limits (L - q + i), and G packs heads into rows;
    q in {1, 2, 5, 10, 11, 16, 37} crosses every bq for G in {1..16}.
    """
    out = []
    quick_gd = {(6, 256), (1, 64), (16, 128)}
    for q in Q_ROWS:
        for G in G_VALUES:
            for D in D_VALUES:
                out.append(
                    GridCase(
                        f"rows-q{q}-gqa{G}-D{D}",
                        "rows",
                        (q,),
                        (L_ROWS,),
                        G=G,
                        D=D,
                        tier="quick" if (G, D) in quick_gd else "full",
                    )
                )
    for q, L, G, D in (
        (512, 512, 1, 64),
        (512, 4096, 8, 128),
        (512, 512 + L_ROWS, 6, 256),
        (2048, 2048, 4, 128),
        (2048, 2048 + L_ROWS, 6, 256),
    ):
        out.append(
            GridCase(f"rows-q{q}-L{L}-gqa{G}-D{D}", "rows", (q,), (L,), G=G, D=D)
        )
    return out


def _batches() -> list[GridCase]:
    """Mixed batches B in {1, 2, 3, 7}: per-request offsets, empty requests,
    graph padding rows (never written), per-request scale modes."""
    return [
        GridCase(
            "batch-b2-decode-S2",
            "batch",
            (1, 1),
            (L_ROWS, 65),
            num_splits=2,
            tier="quick",
        ),
        GridCase(
            "batch-b3-mixed-HND-bs64",
            "batch",
            (1, 5, 10),
            (16, 4096, L_ROWS),
            G=4,
            D=128,
            block_size=64,
            layout="HND",
            scale_mode=("normal", "floor", "extreme"),
            tier="quick",
        ),
        GridCase(
            "batch-b7-mixed-pad5-bs16",
            "batch",
            (1, 2, 5, 10, 11, 0, 37),
            (1, 15, 63, 815, 64, 0, 4096),
            G=1,
            D=64,
            block_size=16,
            pad_rows=5,
            tier="quick",
        ),
        GridCase(
            "batch-b7-decode-pad9-S34",
            "batch",
            (1,) * 7,
            (1, 16, 65, 816, 817, 4096, 63),
            num_splits=34,
            pad_rows=9,
        ),
        GridCase(
            "batch-b2-empty-first-pad4",
            "batch",
            (0, 4),
            (0, L_ROWS),
            G=8,
            D=128,
            pad_rows=4,
        ),
        GridCase(
            "batch-b3-q-equals-L",
            "batch",
            (5, 10, 11),
            (5, 10, 11),
            G=4,
            D=64,
            block_size=16,
        ),
        GridCase(
            "batch-b1-pad3-S2",
            "batch",
            (10,),
            (4096,),
            num_splits=2,
            pad_rows=3,
            sentinel=True,
        ),
    ]


def _splits() -> list[GridCase]:
    """S in {1, 2, 34, S_max} x L: empty splits (S > tiles), split edges, combine."""
    out = []
    for L in L_SPLITS:
        for S in S_VALUES:
            quick = L == L_ROWS or (L == 65 and S == S_MAX)
            out.append(
                GridCase(
                    f"split-L{L}-S{S}",
                    "splits",
                    (1,),
                    (L,),
                    num_splits=S,
                    sentinel=True,
                    tier="quick" if quick else "full",
                )
            )
    out.append(
        GridCase(
            "split-L65536-q10-S34",
            "splits",
            (10,),
            (65536,),
            num_splits=34,
            sentinel=True,
        )
    )
    return out


def _long() -> list[GridCase]:
    """L = 262144 (B = 1 only) and 64k with B = 2: int32-safe
    token math and the longest split ranges."""
    return [
        GridCase("long-L262144-S1", "long", (1,), (L_LONG,), sentinel=True),
        GridCase(
            "long-L262144-S34", "long", (1,), (L_LONG,), num_splits=34, sentinel=True
        ),
        GridCase("long-L262144-q10-S34", "long", (10,), (L_LONG,), num_splits=34),
        GridCase("long-b2-L65536-S34", "long", (1, 1), (65536, 65536), num_splits=34),
    ]


def _scales() -> list[GridCase]:
    """Scale extremes: every |x| at the 1e-6 floor, +-127 outliers, mixed."""
    out = []
    for mode in ("floor", "extreme", "mixed"):
        for D, G in ((64, 4), (256, 6)):
            out.append(
                GridCase(
                    f"scale-{mode}-D{D}",
                    "scales",
                    (5,),
                    (L_ROWS,),
                    G=G,
                    D=D,
                    scale_mode=mode,
                    tier="quick" if D == 256 else "full",
                )
            )
    return out


def _dtypes() -> list[GridCase]:
    """q in bf16/fp16 x out in fp32 / q dtype, for decode and verify."""
    out = []
    for qd in ("bf16", "fp16"):
        for od in ("f32", "q"):
            out.append(
                GridCase(
                    f"dtype-q{qd}-out{od}-decode",
                    "dtypes",
                    (1,),
                    (4096,),
                    q_dtype=qd,
                    out_dtype=od,
                    tier="quick",
                )
            )
            out.append(
                GridCase(
                    f"dtype-q{qd}-out{od}-verify",
                    "dtypes",
                    (10,),
                    (L_ROWS,),
                    q_dtype=qd,
                    out_dtype=od,
                    num_splits=2,
                )
            )
    return out


def _fuzz() -> list[GridCase]:
    """Stride fuzz: odd page padding, unaligned storage offsets, block ids
    beyond 2^31 bytes, strided q/out/workspace views, random shapes.

    Deterministic: the draws come from random.Random(0) and are part of the
    case names, so the list never changes between runs.
    """
    rng = random.Random(0)
    out = []
    for i in range(FUZZ_CASES):
        B = rng.choice((1, 2, 3))
        q = tuple(rng.choice((1, 3, 10, 16)) for _ in range(B))
        L = tuple(rng.randint(ql, 5000) for ql in q)
        c = GridCase(
            f"fuzz-{i:02d}",
            "fuzz",
            q,
            L,
            G=rng.choice((1, 2, 4, 6, 8)),
            Hkv=rng.choice((1, 2, 4)),
            D=rng.choice(D_VALUES),
            block_size=rng.choice((16, 32, 64, 528, 816)),
            layout=rng.choice(("NHD", "HND")),
            pad_page_bytes=rng.choice((0, 4, 72, 4100)),
            storage_offset_bytes=rng.choice((0, 4, 12, 20)),
            # >2 GiB of address space each; two cases are enough for int64.
            high_block_ids=i in (3, 9),
            q_dtype=rng.choice(("bf16", "fp16")),
            out_dtype=rng.choice(("f32", "q")),
            num_splits=rng.choice((1, 2, 3, 34)),
            pad_rows=rng.choice((0, 0, 3)),
            strided_io=True,
            tier="quick" if i < 2 else "full",
        )
        out.append(c)
    return out


def _window() -> list[GridCase]:
    """Sliding window and non-causal, with window-edge sentinels.

    Why: the window's first key (p - W + 1), the key just outside it, the
    non-causal block's later tokens, the CTA's first tile (which may start
    below the range) and the page/tile edges inside a window are where a
    windowed kernel breaks; sentinels on both sides of every row's window
    edge make an off-by-one an O(1) error. Families: window x L edges x
    causal (q, G, D and S cycle), the production page edge, 64k, the
    drafter's own shapes (TP=2 16/4, TP=1 32/8, block 864, a batch with an
    empty request and padding rows) and a seeded stride fuzz.
    """
    out = []
    i = 0
    for W in W_VALUES:
        for L in (W - 1, W, W + 1, 2 * W + 8):
            for causal in (True, False):
                G, D = W_G[(i // 2) % len(W_G)], W_D_VALUES[0]
                q = W_Q[i % len(W_Q)]
                S = (1, 3)[i % 2]
                quick = W in (64, 65) and L == W + 1
                out.append(
                    GridCase(
                        f"win-W{W}-L{L}-{'c' if causal else 'nc'}"
                        f"-q{q}-gqa{G}-D{D}-S{S}",
                        "window",
                        (q,),
                        (L,),
                        G=G,
                        D=D,
                        num_splits=S,
                        sentinel=True,
                        window=W,
                        causal=causal,
                        tier="quick" if quick else "full",
                    )
                )
                i += 1
    for W in (64, 2048):
        for L in (815, 816, 817):
            for causal in (True, False):
                out.append(
                    GridCase(
                        f"win-page-W{W}-L{L}-{'c' if causal else 'nc'}",
                        "window",
                        (8,),
                        (L,),
                        G=4,
                        D=128,
                        sentinel=True,
                        window=W,
                        causal=causal,
                    )
                )
    out += [
        GridCase(
            "win-L65536-W2048-nc-q8-S34",
            "window",
            (8,),
            (65536,),
            G=4,
            D=128,
            num_splits=34,
            sentinel=True,
            window=2048,
            causal=False,
        ),
        GridCase(
            "win-L65536-W4096-c-q1-S2",
            "window",
            (1,),
            (65536,),
            G=6,
            D=128,
            num_splits=2,
            sentinel=True,
            window=4096,
        ),
        # The DFlash2 drafter (window 2048, non-causal, q = 8 = bonus + 7
        # mask tokens, D = 128): TP=2 16/4 and TP=1 32/8, block 864.
        GridCase(
            "win-drafter-tp2-b2",
            "window",
            (8, 8),
            (2056, 8000),
            G=4,
            Hkv=4,
            D=128,
            block_size=DRAFTER_BLOCK,
            sentinel=True,
            window=2048,
            causal=False,
            tier="quick",
        ),
        GridCase(
            "win-drafter-tp2-b2-S3",
            "window",
            (8, 8),
            (2056, 8000),
            G=4,
            Hkv=4,
            D=128,
            block_size=DRAFTER_BLOCK,
            num_splits=3,
            sentinel=True,
            window=2048,
            causal=False,
        ),
        GridCase(
            "win-drafter-tp1-b1",
            "window",
            (8,),
            (130000,),
            G=4,
            Hkv=8,
            D=128,
            block_size=DRAFTER_BLOCK,
            num_splits=2,
            window=2048,
            causal=False,
        ),
        GridCase(
            "win-drafter-b3-empty-pad3",
            "window",
            (8, 0, 8),
            (9, 0, 3000),
            G=4,
            Hkv=4,
            D=128,
            block_size=DRAFTER_BLOCK,
            pad_rows=3,
            out_dtype="q",
            window=2048,
            causal=False,
            tier="quick",
        ),
    ]
    rng = random.Random(30)
    for i in range(W_FUZZ_CASES):
        B = rng.choice((1, 2, 3))
        q = tuple(rng.choice((1, 3, 8, 16)) for _ in range(B))
        L = tuple(rng.randint(ql, 5000) for ql in q)
        out.append(
            GridCase(
                f"win-fuzz-{i:02d}",
                "window",
                q,
                L,
                G=rng.choice((1, 2, 4, 6, 8)),
                Hkv=rng.choice((1, 2, 4)),
                D=rng.choice(W_D_VALUES),
                block_size=rng.choice((16, 32, 64, 528, 816, DRAFTER_BLOCK)),
                layout=rng.choice(("NHD", "HND")),
                pad_page_bytes=rng.choice((0, 4, 72, 4100)),
                storage_offset_bytes=rng.choice((0, 4, 12, 20)),
                q_dtype=rng.choice(("bf16", "fp16")),
                out_dtype=rng.choice(("f32", "q")),
                num_splits=rng.choice((1, 2, 3, 34)),
                pad_rows=rng.choice((0, 0, 3)),
                strided_io=True,
                window=rng.choice((0, 1, 17, 64, 65, 300, 2048, 6000)),
                causal=rng.choice((True, False)),
                tier="quick" if i < 2 else "full",
            )
        )
    return out


def production_descriptors() -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    """Production decode/verify batches (D=256, Hkv=2, G=6): name, q, L."""
    return [
        ("decode-b1", (1,), (65536,)),
        ("mtp-b2", (4, 4), (65536, L_ROWS)),
        ("dflash-b1", (16,), (4096,)),
        ("dflash-b2", (16, 16), (4096, 300)),
    ]


def production_ws_token_splits() -> int:
    """W_ts for the production config: 2 x 16 x 34 = 1088."""
    return workspace_token_splits(
        max_num_seqs=PROD_MAX_NUM_SEQS,
        q_dec_max=PROD_Q_DEC_MAX,
        Hkv=2,
        D=256,
        num_sms=NUM_SMS,
    )


def production_splits(q: Sequence[int]) -> int:
    """The planner's S for one production descriptor."""
    return plan_launch(
        num_tokens=sum(q),
        num_reqs=len(q),
        max_query_len=max(q),
        G=6,
        Hkv=2,
        D=256,
        num_sms=NUM_SMS,
        is_prefill=False,
        ws_token_splits=production_ws_token_splits(),
    ).num_splits


def _production() -> list[GridCase]:
    """Each production descriptor at the planner's S and at S in {1, 2, 34},
    with the workspace carved from the planner-sized buffers.
    All S of one descriptor share their data (seed and sentinels at the
    split edges of every S), so test_production_shapes can compare them."""
    w_ts = production_ws_token_splits()
    out = []
    for name, q, L in production_descriptors():
        splits = tuple(sorted({production_splits(q), 1, 2, 34}))
        for S in splits:
            out.append(
                GridCase(
                    f"prod-{name}-S{S}",
                    "production",
                    q,
                    L,
                    num_splits=S,
                    ws_token_splits=w_ts,
                    sentinel=True,
                    data_name=f"prod-{name}",
                    sentinel_splits=splits,
                )
            )
    return out


def all_cases() -> list[GridCase]:
    """The full grid (quick cases included), in a fixed order."""
    cases = [
        _check_case(c)
        for fam in (
            _paging,
            _rows,
            _batches,
            _splits,
            _long,
            _scales,
            _dtypes,
            _fuzz,
            _production,
            _window,
        )
        for c in fam()
    ]
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate case names")
    return cases


def cases(tier: str) -> list[GridCase]:
    """``"quick"`` (minutes) or ``"full"`` (every case, quick ones included)."""
    if tier not in ("quick", "full"):
        raise ValueError("tier must be quick or full")
    return [c for c in all_cases() if tier == "full" or c.tier == "quick"]


def case_by_name(name: str) -> GridCase:
    for c in all_cases():
        if c.name == name:
            return c
    raise KeyError(name)


# Small shapes for compute-sanitizer (every tool slows the kernel 10-100x):
# padding rows, empty requests, empty splits, HND, padded pages, unaligned
# storage offsets, strided I/O and fp16 output, and the windowed kernels
# all at L <= 5000.
SANITIZER_CASES = (
    "page-L65-bs16-NHD-D128-gqa1",
    "page-L817-bs816-HND-D256-gqa4",
    "batch-b7-mixed-pad5-bs16",
    "batch-b2-decode-S2",
    "split-L65-S204",
    "fuzz-00",
    "fuzz-01",
    "dtype-qfp16-outq-decode",
    "win-W64-L65-nc-q16-gqa4-D128-S3",
    "win-drafter-b3-empty-pad3",
    "win-fuzz-00",
    "win-fuzz-01",
)


# ---------------------------------------------------------------- inputs
def sentinel_positions(
    L: int,
    block_size: int,
    splits: Sequence[int],
    q_len: int = 1,
    window: int = 0,
    causal: bool = True,
) -> list[int]:
    """KV positions that get a sentinel key in a request of length L.

    First and last token, both sides of the first tile edge (63/64) and of the
    first page edge (BS-1/BS), and the last token before and the first token
    of every split edge for every split count in ``splits`` (split
    s of S owns tiles [floor(sT/S), floor((s+1)T/S)) of the T = ceil(L/64)
    tiles of the request's last row tile). Windowed: also both
    sides of the window's edges, lo - 1 / lo and hi / hi + 1, for the
    request's first and last query token.
    """
    pos = {0, L - 1, REF_N_TILE - 1, REF_N_TILE, block_size - 1, block_size}
    tiles = -(-L // REF_N_TILE)
    for num_splits in splits:
        for s in range(1, num_splits):
            start = s * tiles // num_splits * REF_N_TILE
            pos.update((start - 1, start))
    if q_len and (window > 0 or not causal):
        lo, hi = visible_bounds([L - q_len, L - 1], L, window, causal)
        for a, b in zip(lo, hi):
            pos.update((a - 1, a, b, b + 1))
    return sorted(p for p in pos if 0 <= p < L)


@dataclass
class CaseInputs:
    """Host tensors of one case, exactly as the kernel would receive them."""

    case: GridCase
    cache: Int8Cache
    q: torch.Tensor  # [T, Hq, D] view of q_buf (strided if case.strided_io)
    q_buf: torch.Tensor
    qsl: torch.Tensor  # int32 [R + 1]
    softmax_scale: float
    kv: list[DenseKV] = field(default_factory=list)


def build_inputs(case: GridCase) -> CaseInputs:
    """Cache, queries and dense KV of a case (CPU, deterministic)."""
    c = case
    Hq, D = c.Hq, c.D
    sources = None
    unit = None
    if c.sentinel:
        g = torch.Generator().manual_seed(c.seed + 2)
        unit = torch.randn(c.Hkv, D, generator=g, dtype=torch.float64)
        unit /= unit.norm(dim=-1, keepdim=True)
        modes = (
            [c.scale_mode] * len(c.seq_lens)
            if isinstance(c.scale_mode, str)
            else list(c.scale_mode)
        )
        sources = []
        for L, ql, mode in zip(c.seq_lens, c.q_lens, modes):
            k = gen_rows(g, L, c.Hkv, D, mode)
            v = gen_rows(g, L, c.Hkv, D, mode)
            splits = c.sentinel_splits or (c.num_splits,)
            for p in sentinel_positions(
                L, c.block_size, splits, ql, c.window, c.causal
            ):
                k[p] = (SENTINEL_GAIN * math.sqrt(D) * unit).float()
            sources.append((k, v))
    cache = make_int8_cache(
        c.seq_lens,
        Hkv=c.Hkv,
        D=D,
        block_size=c.block_size,
        layout=c.layout,
        pad_page_bytes=c.pad_page_bytes,
        seed=c.seed,
        scale_mode=c.scale_mode,
        high_block_ids=c.high_block_ids,
        storage_offset_bytes=c.storage_offset_bytes,
        sources=sources,
    )
    q_real, qsl = make_queries(c.q_lens, Hq, D, seed=c.seed + 1, dtype=torch.float32)
    g = torch.Generator().manual_seed(c.seed + 3)
    q_all = torch.cat([q_real, torch.randn(c.pad_rows, Hq, D, generator=g)])
    if unit is not None:
        # Every query head of KV head h points along sqrt(D) * unit[h].
        base = (math.sqrt(D) * unit).float().repeat_interleave(c.G, dim=0)
        q_all = base.unsqueeze(0) + SENTINEL_Q_NOISE * q_all
    T = c.num_tokens
    if c.strided_io:
        q_buf = torch.zeros(
            T, Hq + STRIDE_PAD_HEADS, D + STRIDE_PAD_ELEMS, dtype=c.torch_q_dtype
        )
        q = q_buf[:, :Hq, :D]
        q.copy_(q_all)
    else:
        q_buf = q_all.to(c.torch_q_dtype).contiguous()
        q = q_buf
    return CaseInputs(c, cache, q, q_buf, qsl, D**-0.5, cache.dense_all())


@dataclass
class Buffers:
    """Output and workspace buffers with their views, canary-filled.

    ``*_buf`` own the memory; the views are what the kernel gets. Views are
    taken by slicing, so their strides differ from the shape's when
    ``strided_io`` (the kernel must use view strides).
    """

    out_buf: torch.Tensor
    out: torch.Tensor  # [T, Hq, D]
    ws_o_buf: torch.Tensor | None = None
    ws_o: torch.Tensor | None = None  # [S, T, Hq, D] f32
    ws_lse_buf: torch.Tensor | None = None
    ws_lse: torch.Tensor | None = None  # [S, T, Hq] f32


def _canary(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    t = torch.empty(tuple(shape), dtype=dtype)
    t.view(torch.uint8).fill_(CANARY_BYTE)
    return t


def make_buffers(case: GridCase) -> Buffers:
    """Host buffers for one launch (a GPU kernel copies them to the device
    and rebuilds the views with :func:`rebase_view`)."""
    c = case
    T, Hq, D, S = c.num_tokens, c.Hq, c.D, c.num_splits
    hp, dp = (STRIDE_PAD_HEADS, STRIDE_PAD_ELEMS) if c.strided_io else (0, 0)
    out_buf = _canary((T, Hq + hp, D + dp), c.torch_out_dtype)
    b = Buffers(out_buf, out_buf[:, :Hq, :D])
    if S > 1:
        if c.ws_token_splits:
            # [W_ts, Hq, D] and [W_ts, Hq] reserved at load; a launch
            # uses the first S*T token-splits as [S, T, ...].
            w = c.ws_token_splits
            b.ws_o_buf = _canary((w, Hq, D), torch.float32)
            b.ws_lse_buf = _canary((w, Hq), torch.float32)
            b.ws_o = b.ws_o_buf[: S * T].view(S, T, Hq, D)
            b.ws_lse = b.ws_lse_buf[: S * T].view(S, T, Hq)
        else:
            b.ws_o_buf = _canary(
                (S, T + 1 if hp else T, Hq + hp, D + dp), torch.float32
            )
            b.ws_lse_buf = _canary((S, T + 1 if hp else T, Hq + hp), torch.float32)
            b.ws_o = b.ws_o_buf[:, :T, :Hq, :D]
            b.ws_lse = b.ws_lse_buf[:, :T, :Hq]
    return b


def rebase_view(
    view: torch.Tensor, base: torch.Tensor, new_base: torch.Tensor
) -> torch.Tensor:
    """The view of ``new_base`` that ``view`` is of ``base`` (same dtype, same
    element offsets and strides; used to move a buffer and its view)."""
    if view.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
        raise ValueError("view is not a view of base")
    off = view.storage_offset() - base.storage_offset()
    return new_base.as_strided(
        view.shape, view.stride(), new_base.storage_offset() + off
    )


# ---------------------------------------------------------------- expected
@dataclass
class Expected:
    """fp64 reference, bound and valid rows of one case (kernel-independent)."""

    ref: np.ndarray  # [T, Hq, D] f64
    bound: np.ndarray  # [T, Hq, D] f64
    valid: np.ndarray  # [T] bool: rows that belong to a request
    eps_max: float
    seconds: float


def _code_key() -> str:
    h = hashlib.sha256()
    for mod in (bound_mod, reference_mod):
        h.update(Path(str(mod.__file__)).read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(f"{torch.__version__} {np.__version__}".encode())
    return h.hexdigest()


def expected_path(case: GridCase, code_key: str | None = None) -> Path:
    key = hashlib.sha256(
        (
            json.dumps(case.describe(), sort_keys=True) + (code_key or _code_key())
        ).encode()
    ).hexdigest()[:16]
    return EXPECTED_DIR / f"{case.name}-{key}.npz"


def compute_expected(case: GridCase, inp: CaseInputs | None = None) -> Expected:
    inp = inp or build_inputs(case)
    t0 = time.perf_counter()
    b = bound(
        inp.q,
        inp.kv,
        inp.qsl,
        inp.softmax_scale,
        tiling=case.tiling,
        out_dtype=case.torch_out_dtype,
        window=case.window,
        causal=case.causal,
    )
    return Expected(b.ref, b.bound, b.valid, float(b.eps_max), time.perf_counter() - t0)


def load_expected(
    case: GridCase, inp: CaseInputs | None = None, *, use_cache: bool = True
) -> Expected:
    """Expected values from the cache, computing (and storing) them if absent.

    The cache key covers the case and the source of bound.py, reference.py and
    this file, so a code change can never reuse stale values.
    """
    path = expected_path(case)
    if use_cache and path.exists():
        with np.load(path) as z:
            return Expected(
                z["ref"],
                z["bound"],
                z["valid"],
                float(z["eps_max"]),
                float(z["seconds"]),
            )
    e = compute_expected(case, inp)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            ref=e.ref,
            bound=e.bound,
            valid=e.valid,
            eps_max=e.eps_max,
            seconds=e.seconds,
        )
        tmp.replace(path)
    return e


# ---------------------------------------------------------------- the gate
@dataclass
class KernelRun:
    """What a kernel left in the host copies of its buffers."""

    bufs: Buffers
    seconds: float
    checks_workspace: bool  # False for the CPU emulator (it has no workspace)


Kernel = Callable[[CaseInputs, Buffers], KernelRun]


@dataclass
class CaseResult:
    """Outcome of one case; ``passed`` is the gate."""

    case: str
    family: str
    tier: str
    passed: bool
    violations: int
    checked: int
    nonfinite: int
    tightness: float
    max_err: float
    worst: dict[str, Any]
    canary_out_ok: bool
    canary_ws_ok: bool | None
    seconds_kernel: float
    seconds_expected: float
    eps_max: float
    shape: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def message(self) -> str:
        w = self.worst
        return (
            f"case {self.case}: {self.violations}/{self.checked} elements over the "
            f"bound, {self.nonfinite} non-finite, tightness max(err/bound) = "
            f"{self.tightness:.3g}; worst at (token, head, d) = {w.get('index')}: "
            f"err {w.get('err', 0):.3g}, bound {w.get('bound', 0):.3g}, "
            f"ref {w.get('ref', 0):.3g}; canary out ok={self.canary_out_ok}, "
            f"workspace ok={self.canary_ws_ok}"
        )


def _outside_rows_untouched(
    buf: torch.Tensor, view: torch.Tensor, token_dim: int, rows: slice
) -> bool:
    """True if every byte of ``buf`` is canary except ``view``'s token ``rows``.

    ``token_dim`` is 0 for ``out`` ``[T, Hq, D]`` and 1 for the workspace
    views ``[S, T, ...]``. The bytes allowed to change are exactly the
    elements the C ABI says the kernel writes; everything else (padding rows
    ``>= qsl[num_reqs]``, the stride padding around the view, workspace
    token-splits beyond ``S * T``) must still hold the canary.
    """
    allowed = torch.zeros(buf.shape, dtype=torch.bool)
    idx = (slice(None),) * token_dim + (rows,)
    rebase_view(view, buf, allowed)[idx] = True
    by = buf.view(torch.uint8).reshape(*buf.shape, -1)  # [..., bytes per element]
    return bool((by[~allowed] == CANARY_BYTE).all())


def check(case: GridCase, inp: CaseInputs, run: KernelRun, exp: Expected) -> CaseResult:
    """Apply the gate to one kernel run (see the module docstring)."""
    rows = slice(0, int(inp.qsl[-1]))
    o = run.bufs.out.double().numpy()
    v = exp.valid
    err = np.abs(o - exp.ref)
    ok = err <= exp.bound  # NaN/inf compare False: they count as violations
    viol = int((~ok)[v].sum())
    checked = int(v.sum()) * case.Hq * case.D
    nonfinite = int((~np.isfinite(o))[v].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        # err == 0 is exact (ratio 0 even where the bound is 0); a NaN output
        # gives NaN err and ratio inf, so it is the worst element.
        ratio = np.where(err == 0, 0.0, err / exp.bound)
    ratio = np.where(np.isnan(ratio), np.inf, ratio)
    ratio = np.where(v[:, None, None], ratio, 0.0)
    worst: dict[str, Any] = {}
    tight = 0.0
    if checked:
        idx = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        tight = float(ratio[idx])
        worst = {
            "index": [int(i) for i in idx],
            "err": float(err[idx]),
            "bound": float(exp.bound[idx]),
            "ref": float(exp.ref[idx]),
            "out": float(o[idx]),
        }
    max_err = float(np.nanmax(np.where(v[:, None, None], err, 0.0))) if checked else 0.0
    canary_out = _outside_rows_untouched(run.bufs.out_buf, run.bufs.out, 0, rows)
    canary_ws: bool | None = None
    if run.checks_workspace and run.bufs.ws_o is not None:
        assert run.bufs.ws_o_buf is not None and run.bufs.ws_lse_buf is not None
        assert run.bufs.ws_lse is not None
        canary_ws = _outside_rows_untouched(
            run.bufs.ws_o_buf, run.bufs.ws_o, 1, rows
        ) and _outside_rows_untouched(run.bufs.ws_lse_buf, run.bufs.ws_lse, 1, rows)
    passed = viol == 0 and nonfinite == 0 and canary_out and canary_ws is not False
    return CaseResult(
        case=case.name,
        family=case.family,
        tier=case.tier,
        passed=passed,
        violations=viol,
        checked=checked,
        nonfinite=nonfinite,
        tightness=tight,
        max_err=max_err,
        worst=worst,
        canary_out_ok=canary_out,
        canary_ws_ok=canary_ws,
        seconds_kernel=run.seconds,
        seconds_expected=exp.seconds,
        eps_max=exp.eps_max,
        shape=case.describe(),
    )


def run_case(case: GridCase, kernel: Kernel, *, use_cache: bool = True) -> CaseResult:
    """Build inputs, run ``kernel``, apply the gate."""
    inp = build_inputs(case)
    exp = load_expected(case, inp, use_cache=use_cache)
    run = kernel(inp, make_buffers(case))
    return check(case, inp, run, exp)


# ---------------------------------------------------------------- CPU kernel
def emulator_kernel(
    mutate_kv: Callable[[CaseInputs], list[DenseKV]] | None = None,
    mutate_window: Callable[[GridCase], tuple[int, bool]] | None = None,
) -> Kernel:
    """The numpy emulator of the kernel's arithmetic as a :data:`Kernel`.

    It models the optimised kernel's numerics (fp16 MMA, ex2.approx, ...),
    which are coarser than the scalar reference's, with the reference's
    tiling, so a pass here says the grid's gate is satisfiable by a correct
    design. It writes only the request rows of ``out`` (the C ABI contract)
    and has no workspace. ``mutate_kv`` feeds it wrong K/V, ``mutate_window``
    a wrong (window, causal) (mutants).
    """

    def run(inp: CaseInputs, bufs: Buffers) -> KernelRun:
        t0 = time.perf_counter()
        kv = mutate_kv(inp) if mutate_kv else inp.kv
        c = inp.case
        window, causal = mutate_window(c) if mutate_window else (c.window, c.causal)
        res = emulate(
            inp.q,
            kv,
            inp.qsl,
            inp.softmax_scale,
            tiling=inp.case.tiling,
            out_dtype=inp.case.torch_out_dtype,
            window=window,
            causal=causal,
        )
        n = int(inp.qsl[-1])
        bufs.out[:n] = res.out[:n]
        return KernelRun(bufs, time.perf_counter() - t0, checks_workspace=False)

    return run


# ---------------------------------------------------------------- CLI / pool
def _init_worker() -> None:
    # One thread per worker: the pool provides the parallelism.
    torch.set_num_threads(1)


def _expected_worker(name: str) -> tuple[str, float]:
    c = case_by_name(name)
    return name, load_expected(c).seconds


def _emulate_worker(name: str) -> dict[str, Any]:
    return run_case(case_by_name(name), emulator_kernel()).record()


def pool_map(fn: Callable[[str], Any], names: Sequence[str], jobs: int) -> list[Any]:
    """Map over case names in ``jobs`` spawned processes (longest first)."""
    # spawn, not fork: torch's thread pools are not fork-safe.
    ctx = mp.get_context("spawn")
    with ctx.Pool(jobs, initializer=_init_worker) as pool:
        return list(pool.imap_unordered(fn, names))


def _cost(c: GridCase) -> int:
    # Rough work estimate for scheduling: rows x KV length.
    return sum(max(q, 1) * L for q, L in zip(c.q_lens, c.seq_lens)) * c.Hq * c.D


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("list", "expected", "emulate"))
    ap.add_argument("--tier", choices=("quick", "full"), default="full")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    sel = sorted(cases(a.tier), key=_cost, reverse=True)
    names = [c.name for c in sel]
    if a.command == "list":
        for c in cases(a.tier):
            print(c.name, c.family, c.tier)
        print(f"{len(names)} cases", file=sys.stderr)
        return 0
    t0 = time.perf_counter()
    if a.command == "expected":
        for name, s in pool_map(_expected_worker, names, a.jobs):
            print(f"{name}: {s:.1f} s", flush=True)
        print(
            f"{len(names)} cases in {time.perf_counter() - t0:.0f} s", file=sys.stderr
        )
        return 0
    recs = pool_map(_emulate_worker, names, a.jobs)
    bad = [r for r in recs if not r["passed"]]
    for r in sorted(recs, key=lambda r: r["case"]):
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "case",
                        "passed",
                        "violations",
                        "tightness",
                        "seconds_kernel",
                    )
                }
            )
        )
    print(
        f"{len(recs)} cases, {len(bad)} failed, {time.perf_counter() - t0:.0f} s",
        file=sys.stderr,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
