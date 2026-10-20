# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""One production-geometry attention launch (D=256, Hkv=2, G=6, block 816, bf16).

Shared by ``test_attn_graph.py`` and ``attn_stress.py``: builds the cache,
queries, planner launch and the C-ABI arguments once, and runs "the unit" (the
attention kernel, plus the combine when there are splits) on a stream.
"""

from __future__ import annotations

import types

import grid_cases as grid
import torch
from grid_cuda import cache_to
from pth_flash_attn.planner import plan_launch
from pth_testing import ref_capi
from pth_testing.reference import make_int8_cache, make_queries

D, HKV, G, BLOCK = 256, 2, 6, 816


def build(q_len: int, B: int, L: int, num_splits: int | None = None):
    """Launch for ``B`` requests of ``q_len`` query tokens and ``L`` KV tokens.

    ``num_splits`` overrides the planner's S (1 gives the direct-output path).
    The row tile is the balanced ``min(planner bq, q_len)``.

    The C-ABI structs hold raw device pointers, so the returned namespace owns
    every tensor they point to (q and qsl included). A freed tensor whose pages
    the allocator hands out again breaks a captured graph (the replay reads the
    new tenant's bytes) while eager launches seem fine.
    """
    dev = torch.device("cuda", 0)
    T = q_len * B
    cache = cache_to(
        make_int8_cache([L] * B, Hkv=HKV, D=D, block_size=BLOCK, seed=1), dev
    )
    q, qsl = make_queries([q_len] * B, G * HKV, D, seed=2, dtype=torch.bfloat16)
    q, qsl = q.to(dev), qsl.to(dev)
    plan = plan_launch(
        num_tokens=T, num_reqs=B, max_query_len=q_len, G=G, Hkv=HKV, D=D,
        num_sms=grid.NUM_SMS, is_prefill=False,
        ws_token_splits=grid.production_ws_token_splits(),
    )  # fmt: skip
    S = num_splits or plan.num_splits
    bq = min(plan.bq, q_len)
    out = torch.zeros(T, G * HKV, D, dtype=torch.bfloat16, device=dev)
    ws_o = torch.zeros(S, T, G * HKV, D, device=dev) if S > 1 else None
    ws_lse = torch.zeros(S, T, G * HKV, device=dev) if S > 1 else None
    args = ref_capi.attn_args(
        cache, q, qsl, None if S > 1 else out, softmax_scale=D**-0.5,
        bq=bq, num_splits=S, ws_o=ws_o, ws_lse=ws_lse,
    )  # fmt: skip
    cargs = ref_capi.combine_args(ws_o, ws_lse, out, qsl, B, S) if S > 1 else None
    return types.SimpleNamespace(
        cache=cache, q=q, qsl=qsl, out=out, ws_o=ws_o, ws_lse=ws_lse, args=args,
        cargs=cargs, S=S, bq=bq, rows=bq * G, T=T,
    )  # fmt: skip


def run(lib, c, stream: int) -> None:
    """Attention (and combine) of ``c`` on a CUDA stream handle."""
    rc = lib.attention(c.args, host=False, stream=stream)
    assert rc == 0, lib.last_error()
    if c.cargs is not None:
        rc = lib.combine(c.cargs, host=False, stream=stream)
        assert rc == 0, lib.last_error()


def clear(c) -> None:
    """Zero every output so a launch that writes nothing cannot pass."""
    c.out.zero_()
    if c.ws_o is not None:
        c.ws_o.zero_()
        c.ws_lse.zero_()
