# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""``PthFlashAttnImpl.forward`` on a real int8 cache, builder to kernel.

The CPU tests prove the slicing and the planner glue with a recording stand-in
for the op; this file runs the real thing on the GPU: the builder splits a
mixed batch into its decode-like and prefill slices, the Impl runs the real op
on each, and the result is compared with the fp64 reference over the exact
dequantised cache (a mixed batch: two MTP verifies of 4 tokens, a 1-token
decode, a 300-token chunk with a prefix and a 20-token chunk). Then:

* the same Impl is captured in a CUDA graph for a uniform decode batch (the
  workspace views are requested inside the capture) and the replay equals the
  eager call after ``seq_lens`` changed on the device;
* the forward and the build run with ``torch.cuda.set_sync_debug_mode("error")``:
  any host synchronisation (``.item()``, ``.cpu()``, a sync inside the op)
  raises (the in-process twin of ``VLLM_GPU_SYNC_CHECK=error``);
* the Impl allocates nothing in forward (``memory_allocated`` and the workspace
  buffer are unchanged after the first call).

Production geometry: D = 256, Hq = 12, Hkv = 2 (G = 6), page 816, bf16.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from grid_cuda import cache_to
from pth_flash_attn import planner, vllm_backend
from pth_flash_attn.vllm_backend import (
    PthFlashAttnImpl,
    PthFlashAttnMetadataBuilder,
)
from pth_testing.reference import (
    make_int8_cache,
    make_queries,
    ref_attention_dense,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker import workspace as vllm_workspace

pytestmark = [pytest.mark.attn, pytest.mark.backend]

HQ, HKV, D, BLOCK = 12, 2, 256, 816
MAX_SEQS = 2 + 4  # production max_num_seqs is 2; a mixed batch of 5 needs more


def make_config(max_num_seqs: int = MAX_SEQS, k: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            rswa_window=None, get_num_attention_heads=lambda pc: HQ
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=False, num_ubatches=1, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=k, parallel_drafting=False
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.fixture
def workspace(device):
    vllm_workspace.reset_workspace_manager()
    vllm_workspace.init_workspace_manager(device)
    yield vllm_workspace.current_workspace_manager()
    vllm_workspace.reset_workspace_manager()


def make_impl() -> PthFlashAttnImpl:
    return PthFlashAttnImpl(
        num_heads=HQ,
        head_size=D,
        scale=D**-0.5,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="int8_per_token_head",
    )


def make_builder(device, cfg) -> PthFlashAttnMetadataBuilder:
    spec = SimpleNamespace(num_kv_heads=HKV, head_size=D)
    return PthFlashAttnMetadataBuilder(spec, ["layers.3.attn"], cfg, device)


def device_cam(q_lens, seq_lens, cache, device) -> CommonAttentionMetadata:
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    tokens = int(qsl[-1])
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(q_lens),
        num_actual_tokens=tokens,
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=cache.block_table,
        slot_mapping=torch.zeros(tokens, dtype=torch.int64, device=device),
    )


def prepared_impl(cache) -> PthFlashAttnImpl:
    """An Impl whose scale views are the cache's own.

    ``_ensure_scale_caches`` fills the scale slots with 1.0 on first use, which
    is right for an empty cache and would erase the scales of a cache that
    already holds data, so the views the helper would build are set directly.
    """
    impl = make_impl()
    impl._k_scale_cache = cache.k_scale
    impl._v_scale_cache = cache.v_scale
    return impl


def mixed_case(device):
    q_lens = [4, 4, 1, 300, 20]
    seq_lens = [3000, 5000, 100, 2000, 400]
    host = make_int8_cache(seq_lens, Hkv=HKV, D=D, block_size=BLOCK, seed=3)
    q, qsl = make_queries(q_lens, HQ, D, seed=4, dtype=torch.bfloat16)
    return q_lens, seq_lens, cache_to(host, device), q.to(device), qsl, host


def test_mixed_batch_matches_the_fp64_reference(device, workspace) -> None:
    """Builder + Impl + op on a mixed batch equal the exact attention, per row."""
    cfg = make_config()
    q_lens, seq_lens, cache, q, qsl, host = mixed_case(device)
    impl = prepared_impl(cache)
    impl._reserve_workspace(cfg)
    b = make_builder(device, cfg)
    md = b.build(0, device_cam(q_lens, seq_lens, cache, device))
    assert md.decode is not None and md.prefill is not None
    out = torch.full_like(q, float("nan"))
    impl.forward(None, q, None, None, cache.kv_packed, md, out)
    torch.cuda.synchronize()
    ref = ref_attention_dense(q.cpu(), host.dense_all(), qsl, D**-0.5)
    got = out.float().cpu()
    want = ref.out.float()
    assert not torch.isnan(got).any(), "a row was not written"
    scale = want.abs().max()
    assert (got - want).abs().max() <= 3e-2 * scale, (got - want).abs().max() / scale


def test_forward_and_build_make_no_host_sync(device, workspace) -> None:
    """set_sync_debug_mode('error'): no .item()/.cpu()/sync from build to launch."""
    cfg = make_config()
    q_lens, seq_lens, cache, q, *_ = mixed_case(device)
    impl = prepared_impl(cache)
    impl._reserve_workspace(cfg)
    b = make_builder(device, cfg)
    cam = device_cam(q_lens, seq_lens, cache, device)
    out = torch.empty_like(q)
    impl.forward(None, q, None, None, cache.kv_packed, b.build(0, cam), out)  # warm
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        md = b.build(0, cam)
        impl.forward(None, q, None, None, cache.kv_packed, md, out)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_forward_allocates_nothing(device, workspace) -> None:
    """After the first call a forward adds no allocation and keeps the workspace."""
    cfg = make_config()
    q_lens, seq_lens, cache, q, *_ = mixed_case(device)
    impl = prepared_impl(cache)
    impl._reserve_workspace(cfg)
    b = make_builder(device, cfg)
    md = b.build(0, device_cam(q_lens, seq_lens, cache, device))
    out = torch.empty_like(q)
    impl.forward(None, q, None, None, cache.kv_packed, md, out)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    buf = workspace._current_workspaces[0]
    ptr, size = buf.data_ptr(), buf.numel()
    for _ in range(3):
        impl.forward(None, q, None, None, cache.kv_packed, md, out)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    buf = workspace._current_workspaces[0]
    assert (buf.data_ptr(), buf.numel()) == (ptr, size)


@pytest.mark.graph
def test_uniform_decode_graph_replay_equals_eager(device, workspace) -> None:
    """Capture the Impl for two MTP verifies; replay after seq_lens changed."""
    cfg = make_config()
    q_lens, seq_lens = [4, 4], [30000, 20000]
    cache = cache_to(
        make_int8_cache(seq_lens, Hkv=HKV, D=D, block_size=BLOCK, seed=5), device
    )
    q, _ = make_queries(q_lens, HQ, D, seed=6, dtype=torch.bfloat16)
    q = q.to(device)
    impl = prepared_impl(cache)
    impl._reserve_workspace(cfg)
    vllm_workspace.lock_workspace()
    b = make_builder(device, cfg)
    cam = device_cam(q_lens, seq_lens, cache, device)
    md = b.build(0, cam)
    assert md.decode is not None and md.decode.plan.num_splits > 1
    out_graph = torch.zeros_like(q)
    out_eager = torch.zeros_like(q)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        impl.forward(None, q, None, None, cache.kv_packed, md, out_graph)  # warm-up
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        impl.forward(None, q, None, None, cache.kv_packed, md, out_graph)
    base = cam.seq_lens.clone()
    for new in (base, base // 3, base - 100):
        cam.seq_lens.copy_(new)
        out_graph.zero_()
        graph.replay()
        impl.forward(None, q, None, None, cache.kv_packed, md, out_eager)
        torch.cuda.synchronize()
        assert torch.equal(out_graph.view(torch.int16), out_eager.view(torch.int16))
        assert not torch.isnan(out_graph).any()


def test_workspace_reservation_is_counted_at_construction(device) -> None:
    """The reservation is allocated when the Impl is built.

    The model-load window is where vLLM's memory profile counts allocations
    (``Model loading took`` and the KV budget). The Impl's ``__init__`` is in
    that window, a builder is not, so the 3,355,392 B of production partials
    must appear in ``memory_allocated`` right after the Impl reserves them, and
    a second Impl of the same geometry (the MTP drafter) must add nothing.
    """
    vllm_workspace.reset_workspace_manager()
    vllm_workspace.init_workspace_manager(device)
    try:
        cfg = make_config(max_num_seqs=2)
        want = planner.workspace_bytes(ws_token_splits=272, Hq=HQ, D=D)
        assert want == 3_355_392
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        alloc0 = torch.cuda.memory_allocated()
        reserved0 = torch.cuda.memory_reserved()
        first = make_impl()
        first._reserve_workspace(cfg)
        torch.cuda.synchronize()
        alloc1 = torch.cuda.memory_allocated()
        reserved1 = torch.cuda.memory_reserved()
        assert abs((alloc1 - alloc0) - want) <= 2**20
        # What the device gives up is the reserved pool. A fresh 3.2 MiB request
        # takes a whole 20 MiB segment of the caching allocator (requests of
        # 1-10 MiB are carved from 20 MiB segments); the rest of the segment
        # serves later allocations, so the lasting cost is the request itself.
        assert want <= reserved1 - reserved0 <= 20 * 2**20
        second = make_impl()
        second._reserve_workspace(cfg)
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == alloc1
        assert first._ws_token_splits == second._ws_token_splits == 272
        assert (
            vllm_backend.reserved_token_splits(
                cfg, num_kv_heads=HKV, head_size=D, num_sms=68
            )
            == 272
        )
    finally:
        vllm_workspace.reset_workspace_manager()
