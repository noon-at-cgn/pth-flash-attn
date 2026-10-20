# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The production op ``pth_flash_attn_fwd`` equals the C-ABI kernel library.

The op (``csrc/pth_attn/launch.cu``) does not contain a second launcher: it
turns tensors into the C ABI's structs and calls ``libpth_attn``'s code. This
test proves that translation, on the production geometry (D = 256, Hkv = 2,
G = 6, block 816, bf16), by running the same inputs through both and demanding
byte-equal outputs (same kernels, same fixed combine order, so equality is the
right gate, not a tolerance):

* decode and verify shapes (q = 1, 2, 4, 10: the swap-AB decode kernel for
  rows <= 32, the 16-row kernel at q = 10) and prefill shapes (q = 37, 200,
  512: wm = 4, bq = 10), each with the planner's S and with S = 1;
* a hand-made call whose ``max_query_len`` is smaller than the device q_len
  (adaptive verification trims drafts on the device, never the other way) and
  one whose bound is larger: any host bound must give the same bytes, because
  the grid is derived from the token count, not from the bound;
* an empty slice returns without launching;
* the op can be captured in a CUDA graph and the replay equals the eager call
  after ``seq_lens`` changed on the device (nothing the op reads on
  the host depends on a length);
* ``torch.library.opcheck`` on real CUDA tensors (schema, fake-tensor
  agreement);
* the windowed path through the op, at the DFlash2
  drafter's shapes (D = 128, G = 4, 16/4 and 32/8 heads, block 864, window
  2048, non-causal, q = 8; plus a causal window): byte-equal to the C ABI,
  ``opcheck`` with the window arguments, the refusals at D = 64 and 256 and
  of window < 0 with the C++ message, and a 15-argument call (the schema's
  defaults) equal to an explicit ``window=0, causal=True`` call.

Run: ``python -m pytest tests/gpu/test_backend_op.py``
inside the dev image (``pth-flash-attn-dev``) with the extension built
(``tools/build_in_image.sh``)
and ``build/attn/libpth_attn.so`` (``tools/build_attn_lib.sh``).
"""

from __future__ import annotations

import types

import attn_lib
import grid_cases as grid
import pytest
import torch
from attn_case import HKV, D, G, build, run
from grid_cuda import cache_to
from pth_flash_attn import ops
from pth_flash_attn.planner import plan_launch, workspace_token_splits
from pth_testing import ref_capi
from pth_testing.reference import make_int8_cache, make_queries

pytestmark = [pytest.mark.attn, pytest.mark.backend]

# (q_len, B, L, forced S or None = the planner's S)
CASES = [
    (1, 1, 4096, None),
    (1, 2, 70000, None),
    (2, 2, 9000, None),
    (4, 2, 40000, None),
    (4, 2, 40000, 1),
    (10, 1, 20000, None),
    (37, 2, 5000, None),
    (200, 1, 3000, None),
    (512, 2, 2000, 1),
]


@pytest.fixture(scope="module")
def lib() -> attn_lib.AttnLib:
    return attn_lib.AttnLib(attn_lib.ATTN_LIB)


def _plan(q_len: int, B: int):
    return plan_launch(
        num_tokens=q_len * B,
        num_reqs=B,
        max_query_len=q_len,
        G=G,
        Hkv=HKV,
        D=D,
        num_sms=grid.NUM_SMS,
        is_prefill=q_len > 10,
        ws_token_splits=grid.production_ws_token_splits(),
    )


def _call_op(c, out, wm: int, max_query_len: int) -> None:
    cache = c.cache
    empty = torch.empty(0, dtype=torch.float32, device=out.device)
    ops.pth_flash_attn_fwd(
        out,
        c.q,
        cache.k_cache,
        cache.v_cache,
        cache.k_scale,
        cache.v_scale,
        cache.block_table,
        cache.seq_lens,
        c.qsl,
        max_query_len,
        wm,
        c.S,
        D**-0.5,
        c.ws_o if c.S > 1 else empty,
        c.ws_lse if c.S > 1 else empty,
        0,
        True,
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    # NaN-safe equality: compare the bits.
    return t.view(torch.int16).cpu()


@pytest.mark.parametrize("q_len,B,L,S", CASES, ids=lambda v: str(v))
def test_op_equals_c_abi_library(lib, q_len, B, L, S) -> None:
    """Same inputs through the C ABI and through the op: equal bytes."""
    c = build(q_len, B, L, num_splits=S)
    stream = torch.cuda.current_stream().cuda_stream
    run(lib, c, stream)
    torch.cuda.synchronize()
    want = _bytes(c.out)
    out = torch.full_like(c.out, float("nan"))  # an unwritten row cannot pass
    _call_op(c, out, _plan(q_len, B).wm, q_len)
    torch.cuda.synchronize()
    assert torch.equal(_bytes(out), want)


@pytest.mark.parametrize("bound", [1, 3, 5, 8], ids=lambda v: f"max_query_len={v}")
def test_any_host_bound_gives_the_same_bytes(lib, bound) -> None:
    """max_query_len is a host bound; the kernel reads the real q_len on the device.

    q = 4 decode (G = 6, planned wm = 2) with bounds 1, 3, 5 and 8 (= T). The op takes
    the balanced tile min(capacity 5, bound) when that keeps wm = 2 (bound 3: a
    3-token tile, so the 4 tokens take two tiles) and the capacity otherwise
    (bound 1 would need wm = 1). Every row's keys, tile order and split
    partition are the same whatever the tile height, so the bytes must be equal.
    """
    c = build(4, 2, 40000)
    stream = torch.cuda.current_stream().cuda_stream
    run(lib, c, stream)
    torch.cuda.synchronize()
    want = _bytes(c.out)
    out = torch.full_like(c.out, float("nan"))
    _call_op(c, out, _plan(4, 2).wm, bound)
    torch.cuda.synchronize()
    assert torch.equal(_bytes(out), want)


def test_wm_outside_the_choices_is_refused() -> None:
    """The kernel family has wm in {1, 2, 4}; anything else is an argument error."""
    c = build(1, 1, 128)
    out = torch.zeros_like(c.out)
    with pytest.raises(RuntimeError, match="wm must be 1, 2 or 4"):
        _call_op(c, out, 3, 1)


def test_empty_slice_returns_without_launching() -> None:
    """T = 0 (no decode or no prefill requests): nothing to launch, no error."""
    c = build(1, 1, 128)
    out = c.out[:0]
    empty = torch.empty(0, dtype=torch.float32, device=out.device)
    ops.pth_flash_attn_fwd(
        out,
        c.q[:0],
        c.cache.k_cache,
        c.cache.v_cache,
        c.cache.k_scale,
        c.cache.v_scale,
        c.cache.block_table,
        c.cache.seq_lens[:0],
        c.qsl[:1],
        0,
        1,
        1,
        D**-0.5,
        empty,
        empty,
        0,
        True,
    )


@pytest.mark.graph
@pytest.mark.parametrize("q_len,S", [(1, None), (4, None), (10, None), (4, 1)])
def test_op_graph_replay_equals_eager(lib, q_len, S) -> None:
    """Capture the op, change seq_lens on the device, replay: equals eager."""
    B = 2
    c = build(q_len, B, 30000, num_splits=S)
    wm = _plan(q_len, B).wm
    out_eager = torch.zeros_like(c.out)
    out_graph = torch.zeros_like(c.out)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _call_op(c, out_graph, wm, q_len)  # warm-up (kernel attributes)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _call_op(c, out_graph, wm, q_len)
    base = c.cache.seq_lens.clone()
    for new in (base - 7 * q_len, base // 2, base):
        c.cache.seq_lens.copy_(new)
        out_graph.zero_()
        graph.replay()
        out_eager.zero_()
        _call_op(c, out_eager, wm, q_len)
        torch.cuda.synchronize()
        assert torch.equal(_bytes(out_graph), _bytes(out_eager))
        assert not torch.isnan(out_graph).any()


def test_opcheck_on_cuda_tensors() -> None:
    """Schema and fake-tensor agreement with real CUDA tensors."""
    c = build(4, 2, 5000)
    out = torch.zeros_like(c.out)
    empty = torch.empty(0, dtype=torch.float32, device=out.device)
    cache = c.cache
    args = (
        out,
        c.q,
        cache.k_cache,
        cache.v_cache,
        cache.k_scale,
        cache.v_scale,
        cache.block_table,
        cache.seq_lens,
        c.qsl,
        4,
        _plan(4, 2).wm,
        c.S,
        D**-0.5,
        c.ws_o if c.S > 1 else empty,
        c.ws_lse if c.S > 1 else empty,
        0,
        True,
    )
    torch.library.opcheck(
        torch.ops.pth_flash_attn.pth_flash_attn_fwd.default,
        args,
        test_utils=("test_schema", "test_faketensor"),
    )


# ------------------------------------------------- windowed path

W_D, W_G, W_BLOCK, W_DRAFTER = 128, 4, grid.DRAFTER_BLOCK, 2048
NEEDS_D128 = "sliding-window and non-causal attention need head size D = 128"

# (Hkv, q_len, seq_lens, forced S or None, window, causal). Hkv 4 and 8 are
# the drafter's TP=2 (16/4) and TP=1 (32/8) ranks.
WINDOW_CASES = [
    (4, 8, (2056,), None, W_DRAFTER, False),
    (4, 8, (8, 5000, 33000), None, W_DRAFTER, False),
    (8, 8, (2049, 33000), None, W_DRAFTER, False),
    (4, 8, (5000,), 1, W_DRAFTER, False),
    (4, 16, (3000, 2048), None, W_DRAFTER, False),
    (4, 8, (5000,), None, 64, True),
    (4, 8, (5000,), None, 0, False),
]


def _window_case(
    hkv: int,
    q_len: int,
    seq_lens: tuple[int, ...],
    num_splits: int | None,
    window: int,
    causal: bool,
) -> types.SimpleNamespace:
    """A windowed launch at D = 128, G = 4, block 864; ``attn_case.build`` style.

    The planner gets the window (its S cap); the row tile is the balanced
    ``min(planner bq, q_len)`` as in ``attn_case.build``.
    """
    dev = torch.device("cuda", 0)
    B, T, hq = len(seq_lens), q_len * len(seq_lens), W_G * hkv
    cache = cache_to(
        make_int8_cache(list(seq_lens), Hkv=hkv, D=W_D, block_size=W_BLOCK, seed=11),
        dev,
    )
    q, qsl = make_queries([q_len] * B, hq, W_D, seed=12, dtype=torch.bfloat16)
    q, qsl = q.to(dev), qsl.to(dev)
    ws = workspace_token_splits(
        max_num_seqs=B, q_dec_max=q_len, Hkv=hkv, D=W_D, num_sms=grid.NUM_SMS
    )
    plan = plan_launch(
        num_tokens=T, num_reqs=B, max_query_len=q_len, G=W_G, Hkv=hkv, D=W_D,
        num_sms=grid.NUM_SMS, is_prefill=False, ws_token_splits=ws,
        window=max(window, 0),
    )  # fmt: skip
    S = num_splits or plan.num_splits
    out = torch.zeros(T, hq, W_D, dtype=torch.bfloat16, device=dev)
    ws_o = torch.zeros(S, T, hq, W_D, device=dev) if S > 1 else None
    ws_lse = torch.zeros(S, T, hq, device=dev) if S > 1 else None
    args = ref_capi.attn_args(
        cache, q, qsl, None if S > 1 else out, softmax_scale=W_D**-0.5,
        bq=min(plan.bq, q_len), num_splits=S, ws_o=ws_o, ws_lse=ws_lse,
        window=window, causal=causal,
    )  # fmt: skip
    cargs = ref_capi.combine_args(ws_o, ws_lse, out, qsl, B, S) if S > 1 else None
    return types.SimpleNamespace(
        cache=cache, q=q, qsl=qsl, out=out, ws_o=ws_o, ws_lse=ws_lse, args=args,
        cargs=cargs, S=S, wm=plan.wm, q_len=q_len, window=window, causal=causal,
    )  # fmt: skip


def _window_op_args(c, out) -> tuple:
    """The 17 op arguments of a windowed case, writing ``out``."""
    cache = c.cache
    empty = torch.empty(0, dtype=torch.float32, device=out.device)
    return (
        out,
        c.q,
        cache.k_cache,
        cache.v_cache,
        cache.k_scale,
        cache.v_scale,
        cache.block_table,
        cache.seq_lens,
        c.qsl,
        c.q_len,
        c.wm,
        c.S,
        W_D**-0.5,
        c.ws_o if c.S > 1 else empty,
        c.ws_lse if c.S > 1 else empty,
        c.window,
        c.causal,
    )


@pytest.mark.parametrize(
    "hkv,q_len,seq_lens,S,window,causal", WINDOW_CASES, ids=lambda v: str(v)
)
def test_windowed_op_equals_c_abi_library(
    lib, hkv, q_len, seq_lens, S, window, causal
) -> None:
    """The op's window and causal arguments reach the windowed kernel: the
    same inputs through the C ABI and through the op give equal bytes."""
    c = _window_case(hkv, q_len, seq_lens, S, window, causal)
    stream = torch.cuda.current_stream().cuda_stream
    run(lib, c, stream)
    torch.cuda.synchronize()
    want = _bytes(c.out)
    out = torch.full_like(c.out, float("nan"))  # an unwritten row cannot pass
    ops.pth_flash_attn_fwd(*_window_op_args(c, out))
    torch.cuda.synchronize()
    assert torch.equal(_bytes(out), want)
    assert not torch.isnan(out).any()


def test_windowed_opcheck_on_cuda_tensors() -> None:
    """Schema and fake-tensor agreement for a drafter call (window 2048,
    non-causal, the planner's S)."""
    c = _window_case(4, 8, (8, 5000), None, W_DRAFTER, False)
    assert c.S > 1
    torch.library.opcheck(
        torch.ops.pth_flash_attn.pth_flash_attn_fwd.default,
        _window_op_args(c, torch.zeros_like(c.out)),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("head_size", [64, 256])
def test_windowed_op_outside_d128_is_refused(head_size: int) -> None:
    """No windowed kernel at D = 64 or 256: the op raises the C++ reason."""
    dev = torch.device("cuda", 0)
    cache = cache_to(
        make_int8_cache([100], Hkv=1, D=head_size, block_size=16, seed=13), dev
    )
    q, qsl = make_queries([4], W_G, head_size, seed=14, dtype=torch.bfloat16)
    q, qsl = q.to(dev), qsl.to(dev)
    out = torch.zeros_like(q)
    empty = torch.empty(0, dtype=torch.float32, device=dev)
    for window, causal in ((W_DRAFTER, False), (64, True), (0, False)):
        with pytest.raises(RuntimeError, match=f"{NEEDS_D128}, got {head_size}"):
            ops.pth_flash_attn_fwd(
                out, q, cache.k_cache, cache.v_cache, cache.k_scale,
                cache.v_scale, cache.block_table, cache.seq_lens, qsl, 4, 1, 1,
                head_size**-0.5, empty, empty, window, causal,
            )  # fmt: skip


def test_negative_window_is_refused() -> None:
    """window < 0 is an argument error, before any kernel choice."""
    c = _window_case(4, 8, (100,), 1, -1, False)
    with pytest.raises(RuntimeError, match="window must be >= 0, got -1"):
        ops.pth_flash_attn_fwd(*_window_op_args(c, torch.zeros_like(c.out)))


def test_15_argument_call_is_the_causal_call() -> None:
    """A call without window and causal (the schema's defaults 0 and True, as
    a caller written against the 15-argument schema makes it) equals the
    explicit one."""
    c = build(4, 2, 5000)
    wm = _plan(4, 2).wm
    explicit = torch.full_like(c.out, float("nan"))
    _call_op(c, explicit, wm, 4)
    legacy = torch.full_like(c.out, float("nan"))
    cache = c.cache
    empty = torch.empty(0, dtype=torch.float32, device=legacy.device)
    ops.pth_flash_attn_fwd(
        legacy, c.q, cache.k_cache, cache.v_cache, cache.k_scale, cache.v_scale,
        cache.block_table, cache.seq_lens, c.qsl, 4, wm, c.S, D**-0.5,
        c.ws_o if c.S > 1 else empty, c.ws_lse if c.S > 1 else empty,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert torch.equal(_bytes(legacy), _bytes(explicit))
    assert not torch.isnan(legacy).any()
