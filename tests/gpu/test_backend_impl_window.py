# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The DFlash2 drafter's layer on ``CUSTOM`` against vLLM's Triton backend.

``test_attn_window.py`` compares the windowed kernels with vLLM's Triton
``unified_attention`` through the C ABI on a cache the test builds. This
module runs the production path end to end at the layer level, as vLLM 0.31
runs the drafter:

* one ``int8_per_token_head`` cache (NHD pages of 864 tokens, block 0 the
  null block) is written through ``do_kv_cache_update``, the writer both
  backends share (Triton's); pages wholly below a request's window point to
  the null block and are never written, as vLLM's evicted window pages;
* ``PthFlashAttnImpl.forward`` (our builder, planner and op, with the group's
  ``SlidingWindowSpec`` and ``causal=False``) and vLLM's
  ``TritonAttentionImpl.forward`` (the backend that serves the drafter
  today) read it for the same queries: q = 8 per request (bonus + 7), window
  2048, non-causal, D = 128, 16/4 and 32/8 heads, contexts 8 to 33,000,
  batch 1 and 3;
* gates, per element over the fp64 reference of the dequantised cache: ours
  within the kernel's bound (``pth_testing.bound``, the op's tiling and the
  builder's S); Triton within ``TRITON_TOL`` of the largest |v| (its own
  arithmetic, ``test_attn_window.py``); and ours vs Triton within bound +
  tolerance. Triton is the semantic oracle;
* the null block is then filled with NaN-scale bytes: our output must keep
  every byte (block 0 is never read below the window). Control: the same
  Impl without a window reads it, so its output is outside its own bound
  against the fp64 reference of the clean cache.

Needs vLLM 0.31 (the version this project pins) and the built op: without
either the module skips; with both, nothing here skips. Run in the dev image
(vLLM 0.31) with the worktree's ``python/`` (and its ``_C.abi3.so``) first on
``PYTHONPATH`` and ``build/attn/libpth_attn.so`` (the op's tiling for the
bound).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

vllm = pytest.importorskip("vllm")
from test_attn_window import ORACLE_VLLM, TRITON_TOL  # noqa: E402

if not vllm.__version__.startswith(ORACLE_VLLM):
    pytest.skip(
        f"the Triton oracle is vLLM {ORACLE_VLLM}, this is {vllm.__version__}",
        allow_module_level=True,
    )

from pth_flash_attn import ops  # noqa: E402

if not ops.is_available():
    pytest.skip("op library _C.abi3.so not built", allow_module_level=True)

import attn_grid  # noqa: E402
import grid_cases as grid  # noqa: E402
from pth_flash_attn.vllm_backend import (  # noqa: E402
    PthFlashAttnImpl,
    PthFlashAttnMetadataBuilder,
)
from pth_testing.bound import Tiling, bound  # noqa: E402
from pth_testing.reference import DenseKV, cdiv, make_queries, visible_bounds  # noqa: E402
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends.triton_attn import (  # noqa: E402
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec  # noqa: E402
from vllm.v1.worker import workspace as vllm_workspace  # noqa: E402

pytestmark = [pytest.mark.attn, pytest.mark.backend]

D, G, Q, W = 128, 4, 8, 2048  # head size, group, query block (1 + 7), window
BLOCK = grid.DRAFTER_BLOCK  # 864: the drafter group's page in the measured deployment
NULL_BLOCK = 0  # vLLM's null block id
SCALE_BYTES = 4  # one f32 scale after each head's data (inline per-token-head)
DRAFTER_HEADS = ((16, 4), (32, 8))  # Hq/Hkv per rank: TP=2, TP=1
BATCHES = (
    (8,),
    (2047,),
    (2048,),
    (2049,),
    (5000,),
    (33000,),
    (8, 2049, 33000),
    (2047, 2048, 5000),
)


def _params():
    for hq, hkv in DRAFTER_HEADS:
        for lens in BATCHES:
            name = "-".join(map(str, lens))
            yield pytest.param(hq, hkv, lens, id=f"H{hq}-{hkv}-L{name}")


def make_config(hq: int, max_num_seqs: int) -> SimpleNamespace:
    """The ``VllmConfig`` fields the backend reads; DFlash2 k = 7 (q = 8)."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            rswa_window=None, get_num_attention_heads=lambda pc: hq
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=False, num_ubatches=1, decode_context_parallel_size=1
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=Q - 1, parallel_drafting=False
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


@pytest.fixture
def workspace(device):
    vllm_workspace.reset_workspace_manager()
    vllm_workspace.init_workspace_manager(device)
    yield vllm_workspace.current_workspace_manager()
    vllm_workspace.reset_workspace_manager()


def _impl_args(hq: int, hkv: int, window: int | None) -> dict:
    return dict(
        num_heads=hq, head_size=D, scale=D**-0.5, num_kv_heads=hkv,
        alibi_slopes=None, sliding_window=window,
        kv_cache_dtype="int8_per_token_head",
    )  # fmt: skip


def _block_table(seq_lens, device) -> tuple[torch.Tensor, int, int]:
    """Pages per request: the null block below the window, fresh ids above.

    Returns (block_table, number of blocks, null pages). Ids are handed out
    from 1 in reverse request order so pages are not contiguous per request.
    """
    pages = [cdiv(L, BLOCK) for L in seq_lens]
    bt = torch.full((len(seq_lens), max(pages)), NULL_BLOCK, dtype=torch.int32)
    next_id, nulls = 1, 0
    for r in reversed(range(len(seq_lens))):
        lo, _ = visible_bounds([seq_lens[r] - Q], seq_lens[r], W, False)
        evicted = lo[0] // BLOCK  # pages wholly below the first query's window
        nulls += evicted
        for p in range(evicted, pages[r]):
            bt[r, p] = next_id
            next_id += 1
    return bt.to(device), next_id, nulls


def _empty_cache(hkv: int, num_blocks: int, device) -> torch.Tensor:
    """A zeroed packed cache as vLLM allocates it.

    Logical ``[NB, Hkv, BS, 2 * (D + 4)]`` with NHD pages (vLLM's LBNHC).
    """
    content = 2 * (D + SCALE_BYTES)
    kv = torch.zeros(num_blocks, BLOCK, hkv, content, dtype=torch.int8, device=device)
    return kv.transpose(1, 2)


def _write(impl, kv, hkv, seq_lens, bt, seed) -> None:
    """Random K/V through ``impl.do_kv_cache_update`` (Triton's writer).

    Tokens on null pages are not written (vLLM evicted them).
    """
    g = torch.Generator().manual_seed(seed)
    keys, values, slots = [], [], []
    for r, L in enumerate(seq_lens):
        t = torch.arange(L)
        blk = bt[r].cpu().long()[t // BLOCK]
        live = blk != NULL_BLOCK
        k = torch.randn(L, hkv, D, generator=g).to(torch.bfloat16)
        v = torch.randn(L, hkv, D, generator=g).to(torch.bfloat16)
        keys.append(k[live])
        values.append(v[live])
        slots.append((blk * BLOCK + t % BLOCK)[live])
    impl.do_kv_cache_update(
        None,
        torch.cat(keys).to(kv.device),
        torch.cat(values).to(kv.device),
        kv,
        torch.cat(slots).to(kv.device),
    )


def _dense(impl, kv, seq_lens, bt) -> list[DenseKV]:
    """Each request's stored ints and scales, read through the Impl's views."""
    k, v, ks, vs = impl._caches(kv)
    out = []
    for r, L in enumerate(seq_lens):
        t = torch.arange(L, device=kv.device)
        blk = bt[r].long()[t // BLOCK]
        sl = t % BLOCK
        out.append(
            DenseKV(
                k[blk, sl].cpu(), ks[blk, sl].cpu(), v[blk, sl].cpu(), vs[blk, sl].cpu()
            )
        )
    return out


def _triton_metadata(cam: CommonAttentionMetadata) -> TritonAttentionMetadata:
    """Triton's metadata for the 2D unified kernel (q > 1 never takes 3D).

    Built from the fields this vLLM's dataclass has (a patched vLLM adds
    some), without its builder, which needs a full ``VllmConfig``.
    """
    values = dict(
        num_actual_tokens=cam.num_actual_tokens,
        max_query_len=cam.max_query_len,
        query_start_loc=cam.query_start_loc,
        max_seq_len=cam.max_seq_len,
        seq_lens=cam.seq_lens,
        block_table=cam.block_table_tensor,
        slot_mapping=cam.slot_mapping,
        causal=cam.causal,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    names = {f.name for f in dataclasses.fields(TritonAttentionMetadata)}
    kwargs = {n: values.get(n) for n in names if n in values or _required(n)}
    return TritonAttentionMetadata(**kwargs)


def _required(name: str) -> bool:
    f = TritonAttentionMetadata.__dataclass_fields__[name]
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


@pytest.mark.parametrize(("hq", "hkv", "seq_lens"), list(_params()))
def test_drafter_layer_matches_triton(device, workspace, record, hq, hkv, seq_lens):
    """CUSTOM and TRITON_ATTN agree on the drafter's layer, within the bound."""
    B = len(seq_lens)
    cfg = make_config(hq, max_num_seqs=B)
    ours = PthFlashAttnImpl(**_impl_args(hq, hkv, W))
    ours._reserve_workspace(cfg)
    tri = TritonAttentionImpl(**_impl_args(hq, hkv, W))
    bt, num_blocks, nulls = _block_table(seq_lens, device)
    kv = _empty_cache(hkv, num_blocks, device)
    # Scale views first: vLLM's helper sets every scale to 1.0 when it builds
    # them, which on a written cache would erase the scales.
    ours._ensure_scale_caches(kv)
    tri._ensure_scale_caches(kv)
    _write(ours, kv, hkv, seq_lens, bt, seed=sum(seq_lens))
    assert torch.equal(tri._k_scale_cache, ours._k_scale_cache)

    q, qsl = make_queries([Q] * B, hq, D, seed=hq + B, dtype=torch.bfloat16)
    cam = CommonAttentionMetadata(
        query_start_loc=qsl.to(device),
        query_start_loc_cpu=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=B,
        num_actual_tokens=Q * B,
        max_query_len=Q,
        max_seq_len=max(seq_lens),
        block_table_tensor=bt,
        slot_mapping=torch.zeros(Q * B, dtype=torch.int64, device=device),
        causal=False,
    )
    spec = SlidingWindowSpec(
        block_size=BLOCK, num_kv_heads=hkv, head_size=D, dtype=torch.int8,
        sliding_window=W,
    )  # fmt: skip
    md = PthFlashAttnMetadataBuilder(spec, ["drafter.attn"], cfg, device).build(0, cam)
    assert md.prefill is None and md.decode is not None and md.causal is False
    plan = md.decode.plan
    qd = q.to(device)
    out = torch.full_like(qd, float("nan"))
    ours.forward(None, qd, None, None, kv, md, out)
    out_tri = torch.full_like(qd, float("nan"))
    tri.forward(None, qd, None, None, kv, _triton_metadata(cam), out_tri)
    torch.cuda.synchronize()

    kv_dense = _dense(ours, kv, seq_lens, bt)
    n_tile, sub, wn = attn_grid.lib().tiling(D, plan.bq, G)
    res = bound(
        q, kv_dense, qsl, D**-0.5,
        tiling=Tiling(n_tile, sub, wn, plan.num_splits, plan.bq),
        out_dtype=torch.bfloat16, window=W, causal=False,
    )  # fmt: skip
    valid = res.valid
    assert valid.all()
    mine = out.double().cpu().numpy()
    theirs = out_tri.double().cpu().numpy()
    vmax = max(float(x.v_deq().abs().max()) for x in kv_dense)
    tol = TRITON_TOL * vmax
    over_bound = float((abs(mine - res.ref) - res.bound).max())
    err_tri = float(abs(theirs - res.ref).max())
    gap = float((abs(mine - theirs) - res.bound - tol).max())
    record(
        test="backend_impl_window_vs_triton",
        heads=f"{hq}/{hkv}",
        seq_lens=list(seq_lens),
        num_splits=plan.num_splits,
        wm=plan.wm,
        null_pages=nulls,
        tol=tol,
        ours_over_bound=over_bound,
        err_triton=err_tri,
        ours_minus_triton_over=gap,
    )
    assert torch.isfinite(out).all() and torch.isfinite(out_tri).all()
    assert over_bound <= 0.0, f"ours exceeds the bound by {over_bound:.3g}"
    assert err_tri <= tol, f"Triton vs fp64 {err_tri:.3g} > {tol:.3g}"
    assert gap <= 0.0, f"ours vs Triton exceeds bound + tol by {gap:.3g}"

    # The null block is never read below the window: poison it, same bytes.
    kv_bytes = kv.transpose(1, 2)  # physical [NB, BS, Hkv, C]
    kv_bytes[NULL_BLOCK].fill_(-1)  # 0xFF: NaN scales
    again = torch.full_like(qd, float("nan"))
    ours.forward(None, qd, None, None, kv, md, again)
    torch.cuda.synchronize()
    assert torch.equal(again.view(torch.int16), out.view(torch.int16))
    if nulls:
        # Control: without a window the same non-causal layer reads block 0.
        # Its output is checked against its own fp64 reference and bound over
        # the clean cache (kv_dense was gathered before the poisoning, block 0
        # still zeros) and must violate it. Finiteness alone cannot be
        # asserted: at S > 1 a split whose tiles all lie in the null block has
        # only NaN scores, writes lse = -inf and the combine skips it as an
        # empty split, so the output is finite but wrong (GPU run
        # 2026-10-17: finite at S = 34 and 17). A non-finite element counts as
        # a violation.
        full = PthFlashAttnImpl(**_impl_args(hq, hkv, None))
        full._k_scale_cache = ours._k_scale_cache  # the views, not a reset
        full._v_scale_cache = ours._v_scale_cache
        full._reserve_workspace(cfg)
        ctl = torch.zeros_like(qd)
        full.forward(None, qd, None, None, kv, md, ctl)
        torch.cuda.synchronize()
        clean = bound(
            q, kv_dense, qsl, D**-0.5,
            tiling=Tiling(n_tile, sub, wn, plan.num_splits, plan.bq),
            out_dtype=torch.bfloat16, window=0, causal=False,
        )  # fmt: skip
        got = ctl.double().cpu().numpy()
        err = abs(got - clean.ref)
        violations = int(((err > clean.bound) | ~np.isfinite(got)).sum())
        record(
            test="backend_impl_window_null_control",
            heads=f"{hq}/{hkv}",
            seq_lens=list(seq_lens),
            num_splits=plan.num_splits,
            null_pages=nulls,
            finite=bool(np.isfinite(got).all()),
            violations=violations,
        )
        assert violations > 0, "control: the poison was not read"
