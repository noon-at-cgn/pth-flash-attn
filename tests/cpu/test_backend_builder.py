# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The metadata builder, the Impl and the workspace rules, on CPU tensors.

Everything here runs without a GPU: the op is replaced by a recorder, the
vLLM config by a namespace carrying exactly the fields the backend reads, and
the workspace manager is vLLM's own (on the CPU device). What is protected:

* **Graph-level constants**: ``UNIFORM_BATCH`` (value 2), the
  reorder (decode) threshold ``max(floor(64 / G), 1 + k)`` (10 for production:
  G = 6, MTP k = 3), draft-metadata update allowed, ``q_len`` mismatch between
  host and device allowed.
* **Planner integration**: each launch slice is exactly ``plan_launch`` of its
  own descriptor (decode-like part, prefill part), the prefill ``query_start_loc``
  is rebased on the device value, short extends count as decodes, padded
  graph batches are all decode, and S * T never exceeds the reserved
  workspace for any random batch.
* **Workspace**: the Impl reserves ``W_ts`` token-splits
  (3,355,392 B = 3.2 MiB for production) when it is constructed; with the
  workspace locked, forward launches of every planned size still fit and
  allocate nothing; a request beyond the reservation fails loudly.
* **At most two op calls** per forward and the exact tensors they get.
* **Rejections**: what v1 does not serve raises at construction or build, with
  its reason.

The whole module is skipped (``pytest.importorskip``) when vLLM is not
installed.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from pth_flash_attn import planner, vllm_backend  # noqa: E402
from pth_flash_attn.vllm_backend import (  # noqa: E402
    PthAttnMetadata,
    PthFlashAttnBackend,
    PthFlashAttnImpl,
    PthFlashAttnMetadataBuilder,
    reserved_token_splits,
    verify_query_len,
)
from vllm.v1.attention.backend import (  # noqa: E402
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.worker import workspace as vllm_workspace  # noqa: E402

# Production geometry per rank: Hq = 12, Hkv = 2 (G = 6), D = 256, 68 SMs,
# max_num_seqs = 2, MTP with 3 draft tokens.
HQ, HKV, D, SMS = 12, 2, 256, 68
G = HQ // HKV
WS_BYTES_PROD = 3_355_392  # production workspace reservation, bytes
# The speculative-decoding drafter's attention group: D = 128, window 2048,
# k = 7 (q = 8 per request), heads per rank at TP=2 (16/4) and TP=1 (32/8).
DRAFTER_TP2 = {
    "heads": 16,
    "num_kv_heads": 4,
    "head_size": 128,
    "k": 7,
    "sliding_window": 2048,
}
DRAFTER_TP1 = {**DRAFTER_TP2, "heads": 32, "num_kv_heads": 8}


def make_config(
    *, k: int | None = 3, max_num_seqs: int = 2, heads: int = HQ, rswa=None, dbo=False
) -> SimpleNamespace:
    """The fields of ``VllmConfig`` the backend reads."""
    spec = (
        None
        if k is None
        else SimpleNamespace(num_speculative_tokens=k, parallel_drafting=False)
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            rswa_window=rswa, get_num_attention_heads=lambda pc: heads
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=dbo,
            num_ubatches=2 if dbo else 1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=spec,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def make_spec(
    num_kv_heads: int = HKV, head_size: int = D, sliding_window: int | None = None
) -> SimpleNamespace:
    """The fields of a group's ``AttentionSpec`` the builder reads."""
    spec = SimpleNamespace(num_kv_heads=num_kv_heads, head_size=head_size)
    if sliding_window is not None:
        spec.sliding_window = sliding_window
    return spec


@pytest.fixture(autouse=True)
def fixed_sm_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SM count is a planner input; no GPU is asked for it."""
    monkeypatch.setattr(vllm_backend, "_num_sms", lambda device=None: SMS)


def make_builder(**cfg) -> PthFlashAttnMetadataBuilder:
    spec = make_spec(
        cfg.pop("num_kv_heads", HKV),
        cfg.pop("head_size", D),
        cfg.pop("sliding_window", None),
    )
    return PthFlashAttnMetadataBuilder(
        spec, ["layers.3.attn"], make_config(**cfg), torch.device("cpu")
    )


def make_cam(
    q_lens: list[int],
    seq_lens: list[int] | None = None,
    *,
    device_qsl: list[int] | None = None,
    max_reqs: int = 8,
    **kw,
) -> CommonAttentionMetadata:
    """A batch of requests with the given query lengths on CPU tensors."""
    qsl = [0]
    for q in q_lens:
        qsl.append(qsl[-1] + q)
    seqs = seq_lens if seq_lens is not None else [q + 500 for q in q_lens]
    qsl_t = torch.tensor(qsl, dtype=torch.int32)
    dev_qsl = torch.tensor(device_qsl, dtype=torch.int32) if device_qsl else qsl_t
    seq_buf = torch.zeros(max_reqs, dtype=torch.int32)
    seq_buf[: len(seqs)] = torch.tensor(seqs, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=dev_qsl,
        query_start_loc_cpu=qsl_t,
        seq_lens=seq_buf[: len(q_lens)],
        num_reqs=len(q_lens),
        num_actual_tokens=qsl[-1],
        max_query_len=max(q_lens),
        max_seq_len=max(seqs),
        block_table_tensor=torch.zeros(max_reqs, 8, dtype=torch.int32),
        slot_mapping=torch.zeros(qsl[-1], dtype=torch.int64),
        **kw,
    )


def expected_plan(tokens: int, reqs: int, q_max: int, is_prefill: bool, **cfg):
    b = make_builder(**cfg)
    return planner.plan_launch(
        num_tokens=tokens,
        num_reqs=reqs,
        max_query_len=q_max,
        G=G,
        Hkv=HKV,
        D=D,
        num_sms=SMS,
        is_prefill=is_prefill,
        ws_token_splits=b.ws_token_splits,
    )


# ----------------------------------------------------------- graph constants


def test_graph_level_is_uniform_batch() -> None:
    """FULL graphs for uniform decode batches only."""
    builder = PthFlashAttnBackend.get_builder_cls()
    assert builder._cudagraph_support is AttentionCGSupport.UNIFORM_BATCH
    assert builder._cudagraph_support.value == 2
    got = builder.get_cudagraph_support(make_config(), make_spec())
    assert got is AttentionCGSupport.UNIFORM_BATCH


def test_draft_metadata_update_is_declared_and_a_noop() -> None:
    assert PthFlashAttnMetadataBuilder.supports_draft_decode_metadata_update is True
    b = make_builder()
    md = b.build(0, make_cam([1, 1]))
    assert b.update_draft_decode_metadata(md) is None


@pytest.mark.parametrize(
    "heads, kv, k, want",
    [
        (12, 2, 3, 10),  # production: G = 6, MTP k = 3 -> floor(64/6) = 10
        (12, 2, None, 10),  # no speculation: still one row tile of tokens
        (12, 2, 15, 16),  # speculative verify of 16 outgrows the row tile
        (32, 8, 3, 16),  # Llama-8B: G = 4
        (8, 1, 3, 8),  # G = 8
        (16, 1, 3, 4),  # G = 16: 64 // 16 = 4 = 1 + k
        (64, 1, 3, 4),  # G = 64: floor(64/64) = 1, spec-as-decode raises it
    ],
)
def test_decode_threshold(heads, kv, k, want) -> None:
    """reorder_batch_threshold = max(floor(64 / G), 1 + k) (spec-as-decode)."""
    b = make_builder(heads=heads, num_kv_heads=kv, k=k, head_size=128)
    assert b.reorder_batch_threshold == want


def test_varlen_graph_hook_matches_the_threshold() -> None:
    """The 0.31 hook returns the same bound as the live builder (harmless on 0.30)."""
    cfg = make_config()
    assert (
        PthFlashAttnMetadataBuilder.get_varlen_cudagraph_max_query_len(cfg, make_spec())
        == make_builder().reorder_batch_threshold
    )


def test_host_device_query_len_mismatch_is_allowed() -> None:
    """The kernel reads q_len on the device, so the CPU copy may differ."""
    assert PthFlashAttnBackend.supports_device_cpu_query_lens_mismatch() is True


# ------------------------------------------------------------ planner glue


def test_all_decode_batch_is_one_decode_slice() -> None:
    """Two MTP verifies (q = 4): one decode launch, no prefill launch."""
    b = make_builder()
    cam = make_cam([4, 4])
    md = b.build(0, cam)
    assert isinstance(md, PthAttnMetadata)
    assert md.prefill is None and md.decode is not None
    d = md.decode
    assert (d.token_start, d.token_end, d.max_query_len) == (0, 8, 4)
    assert d.plan == expected_plan(8, 2, 4, False)
    assert d.plan.wm == 2 and d.plan.num_splits > 1
    # Persistent tensors are used as they are: no copy, no per-step allocation.
    assert d.query_start_loc is cam.query_start_loc
    assert d.seq_lens.data_ptr() == cam.seq_lens.data_ptr()
    assert md.num_actual_tokens == 8 and md.causal is True


def test_mixed_batch_splits_into_decode_and_prefill() -> None:
    """Verifies and a 1-token decode come first; the long extends are the prefill."""
    b = make_builder()
    cam = make_cam([4, 4, 1, 300, 20])
    md = b.build(0, cam)
    d, p = md.decode, md.prefill
    assert d is not None and p is not None
    assert (d.token_start, d.token_end, d.max_query_len) == (0, 9, 4)
    assert d.plan == expected_plan(9, 3, 4, False)
    assert (p.token_start, p.token_end, p.max_query_len) == (9, 329, 300)
    assert p.plan == expected_plan(320, 2, 300, True)
    assert p.plan.wm == 4 and p.plan.bq == 10
    assert p.seq_lens.shape == (2,) and p.block_table.shape[0] == 2
    assert p.query_start_loc.tolist() == [0, 300, 320]


def test_prefill_query_start_loc_is_rebased_on_the_device_value() -> None:
    """The device qsl is authoritative (adaptive verification trims drafts on it)."""
    b = make_builder()
    cam = make_cam([2, 2, 300], device_qsl=[0, 3, 5, 305])
    md = b.build(0, cam)
    assert md.prefill is not None
    assert md.prefill.query_start_loc.tolist() == [0, 300]
    # The decode slice uses the device qsl as is; its bound is the host's.
    assert md.decode is not None
    assert md.decode.query_start_loc.tolist() == [0, 3, 5]
    assert md.decode.max_query_len == 2


def test_prefill_rebase_buffer_is_persistent() -> None:
    """One builder reuses one buffer: the step allocates nothing."""
    b = make_builder()
    p1 = b.build(0, make_cam([1, 300])).prefill
    p2 = b.build(0, make_cam([1, 40, 40])).prefill
    assert p1 is not None and p2 is not None
    assert p1.query_start_loc.data_ptr() == p2.query_start_loc.data_ptr()
    assert p2.query_start_loc.tolist() == [0, 40, 80]


def test_prefill_only_batch() -> None:
    b = make_builder()
    md = b.build(0, make_cam([300]))
    assert md.decode is None and md.prefill is not None
    assert md.prefill.plan == expected_plan(300, 1, 300, True)
    assert md.prefill.query_start_loc.tolist() == [0, 300]


def test_short_extend_counts_as_decode() -> None:
    """A chunk tail of 8 tokens (<= threshold 10) is decode-like (spec-as-decode)."""
    b = make_builder()
    md = b.build(0, make_cam([8, 300]))
    assert md.decode is not None and md.decode.token_end == 8
    assert md.decode.max_query_len == 8
    assert md.decode.plan.wm == 4  # 8 tokens x 6 heads = 48 rows


def test_padded_graph_batch_is_all_decode() -> None:
    """A captured descriptor: padding requests have q = 0 and stay in the slice."""
    b = make_builder()
    cam = make_cam([4, 4, 0, 0], seq_lens=[10, 10, 0, 0])
    md = b.build(0, cam)
    assert md.prefill is None and md.decode is not None
    assert md.decode.plan == expected_plan(8, 4, 4, False)
    assert md.decode.seq_lens.shape == (4,)


def test_capture_build_sets_seq_lens_to_one() -> None:
    """As Triton's builder: a capture run must not walk max_model_len keys."""
    b = make_builder()
    cam = make_cam([4, 4], seq_lens=[9000, 9000])
    md = b.build_for_cudagraph_capture(cam)
    assert md.seq_lens.tolist() == [1, 1]
    assert cam.seq_lens.tolist() == [1, 1]


def test_plan_is_independent_of_sequence_length() -> None:
    """Graph rule: the grid cannot depend on L, only on the descriptor."""
    b = make_builder()
    plans = {
        b.build(0, make_cam([4, 4], seq_lens=[s, s])).decode.plan
        for s in (8, 1000, 131072, 262144)
    }
    assert len(plans) == 1


def test_splits_never_exceed_the_reserved_workspace() -> None:
    """For random batches every slice satisfies S * T <= W_ts (or S == 1)."""
    rng = random.Random(0)
    for _ in range(300):
        k = rng.choice([None, 3, 7, 15])
        max_seqs = rng.choice([1, 2, 8, 64])
        b = make_builder(k=k, max_num_seqs=max_seqs)
        # A batch never has more requests than max_num_seqs (the runner's cap).
        n_reqs = rng.randint(1, min(max_seqs, 9))
        n_dec = rng.randint(0, n_reqs)
        q = [rng.randint(1, b.reorder_batch_threshold) for _ in range(n_dec)]
        q += [rng.choice([11, 20, 64, 512, 8192]) for _ in range(n_reqs - n_dec)]
        md = b.build(0, make_cam(q, max_reqs=16))
        for part in (md.decode, md.prefill):
            if part is not None and part.plan.num_splits > 1:
                plan = part.plan
                assert plan.num_splits * plan.num_tokens <= b.ws_token_splits


# ---------------------------------------------------------------- rejections


def test_per_request_causal_tensors_are_refused() -> None:
    """vLLM types ``causal`` as bool | Tensor; only one bool per group is served."""
    for b in (make_builder(), make_builder(**DRAFTER_TP2)):
        with pytest.raises(NotImplementedError, match="per-request causal"):
            b.build(0, make_cam([1], causal=torch.tensor([True])))


@pytest.mark.parametrize("head_size", [64, 256])
def test_non_causal_and_window_off_d128_are_refused(head_size: int) -> None:
    """Windowed kernels exist at D = 128 only: a non-causal batch or
    a windowed group at D = 64 or at the D = 256 MTP target fails with its
    reason instead of reaching the op's own refusal; causal batches without
    window still build at both sizes."""
    b = make_builder(head_size=head_size)
    assert b.build(0, make_cam([1])).decode is not None
    with pytest.raises(
        NotImplementedError, match=f"non-causal attention at head size {head_size}"
    ):
        b.build(0, make_cam([1], causal=False))
    with pytest.raises(
        NotImplementedError, match=f"sliding window at head size {head_size}"
    ):
        make_builder(head_size=head_size, sliding_window=2048)


def test_cascade_prefix_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="cascade"):
        make_builder().build(16, make_cam([1]))


def test_multimodal_prefix_ranges_are_refused_but_empty_ranges_pass() -> None:
    b = make_builder()
    with pytest.raises(NotImplementedError, match="multimodal prefix"):
        b.build(0, make_cam([1, 1], mm_req_doc_ranges={0: [(2, 6)], 1: []}))
    # A runner that passes empty ranges for a text-only batch is fine.
    assert b.build(0, make_cam([1, 1], mm_req_doc_ranges={0: [], 1: []})).decode


def test_rswa_is_refused() -> None:
    b = make_builder()
    with pytest.raises(NotImplementedError, match="R-SWA"):
        b.build(0, make_cam([1], rswa_prefix_lens=torch.zeros(1, dtype=torch.int32)))
    with pytest.raises(NotImplementedError, match="R-SWA"):
        make_builder(rswa=1024)


def test_dbo_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="DBO"):
        make_builder(dbo=True)


def test_unsupported_geometry_is_refused() -> None:
    with pytest.raises(NotImplementedError, match="head size"):
        make_builder(head_size=80)
    with pytest.raises(ValueError, match="exceeds"):
        make_builder(heads=130, num_kv_heads=1, head_size=128)
    with pytest.raises(ValueError, match="multiple"):
        make_builder(heads=12, num_kv_heads=5, head_size=128)


# --------------------------------------------------------------------- Impl


def make_impl(**kw) -> PthFlashAttnImpl:
    args = {
        "num_heads": HQ,
        "head_size": D,
        "scale": D**-0.5,
        "num_kv_heads": HKV,
        "alibi_slopes": None,
        "sliding_window": None,
        "kv_cache_dtype": "int8_per_token_head",
    }
    args.update(kw)
    return PthFlashAttnImpl(**args)


@pytest.fixture
def workspace():
    """vLLM's own workspace manager on the CPU device, reset around the test."""
    vllm_workspace.reset_workspace_manager()
    vllm_workspace.init_workspace_manager(torch.device("cpu"))
    yield vllm_workspace.current_workspace_manager()
    vllm_workspace.reset_workspace_manager()


def test_impl_constructs_for_the_production_layer() -> None:
    impl = make_impl()
    assert impl.num_queries_per_kv == G
    assert impl.supports_quant_query_input is False


@pytest.mark.parametrize(
    "kwargs, what",
    [
        ({"alibi_slopes": [0.5] * HQ}, "ALiBi"),
        ({"use_alibi_sqrt": True}, "ALiBi"),
        ({"sliding_window": 4096}, "sliding window at head size 256"),
        ({"sliding_window": 2048, "head_size": 64}, "sliding window at head size 64"),
        ({"sliding_window": 0}, "sliding window 0"),
        ({"logits_soft_cap": 30.0}, "soft cap"),
        ({"sinks": torch.zeros(HQ)}, "sinks"),
        ({"chunk_lookback": 8}, "chunk lookback"),
        ({"attn_type": AttentionType.ENCODER}, "attention type"),
        ({"kv_cache_dtype": "auto"}, "kv_cache_dtype"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
        ({"head_size": 80}, "head size"),
    ],
)
def test_impl_rejects_what_the_kernel_does_not_serve(kwargs, what) -> None:
    """An explicitly selected backend with an unsupported layer errors at load."""
    with pytest.raises(NotImplementedError, match=what):
        make_impl(**kwargs)


def test_impl_rejects_a_group_larger_than_one_row_tile() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        make_impl(num_heads=130, num_kv_heads=1)


def test_forward_without_metadata_zero_fills() -> None:
    """The memory-profile run executes no attention."""
    impl = make_impl()
    out = torch.ones(3, HQ, D)
    assert impl.forward(None, out, None, None, None, None, out) is out
    assert not out.any()


def test_fused_output_quantisation_is_refused() -> None:
    impl = make_impl()
    out = torch.zeros(1, HQ, D)
    with pytest.raises(NotImplementedError, match="output quantisation"):
        impl.forward(None, out, None, None, None, object(), out, output_scale=out)


# ---------------------------------------------------------------- workspace


def test_reservation_is_the_production_size(workspace) -> None:
    """Impl construction reserves W_ts = 272 token-splits = 3,355,392 B."""
    cfg = make_config()
    assert verify_query_len(cfg) == 4
    assert reserved_token_splits(cfg, num_kv_heads=HKV, head_size=D, num_sms=SMS) == 272
    impl = make_impl()
    impl._reserve_workspace(cfg)
    buf = workspace._current_workspaces[0]
    assert buf is not None and buf.numel() == WS_BYTES_PROD
    assert impl._ws_token_splits == 272
    assert planner.workspace_bytes(ws_token_splits=272, Hq=HQ, D=D) == WS_BYTES_PROD


def test_reservation_without_a_config_reserves_nothing(workspace) -> None:
    """Unit-test construction (no current vLLM config) leaves the manager empty."""
    make_impl()
    assert workspace._current_workspaces[0] is None


def test_second_impl_shares_the_reservation(workspace) -> None:
    """The MTP drafter layer has the same geometry: no second buffer."""
    cfg = make_config()
    a, b = make_impl(), make_impl()
    a._reserve_workspace(cfg)
    ptr = workspace._current_workspaces[0].data_ptr()
    b._reserve_workspace(cfg)
    assert workspace._current_workspaces[0].data_ptr() == ptr


class Recorder:
    """Stands in for ``ops.pth_flash_attn_fwd`` and keeps what it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        self.calls.append(args)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(vllm_backend.ops, "pth_flash_attn_fwd", rec)
    return rec


def run_forward(impl, b, q_lens, *, hq=HQ, hkv=HKV, d=D, **cam_kw):
    cam = make_cam(q_lens, **cam_kw)
    md = b.build(0, cam)
    nb, bs = 4, 16
    kv = torch.zeros(nb, hkv, bs, 2 * (d + 4), dtype=torch.int8)
    tokens = sum(q_lens)
    query = torch.zeros(tokens, hq, d, dtype=torch.bfloat16)
    out = torch.zeros_like(query)
    impl.forward(None, query, None, None, kv, md, out)
    return md, query, out


def test_at_most_two_op_calls_and_their_arguments(workspace, recorder) -> None:
    cfg = make_config()
    impl = make_impl()
    impl._reserve_workspace(cfg)
    b = make_builder()
    md, query, out = run_forward(impl, b, [4, 4, 1, 300, 20])
    assert len(recorder.calls) == 2
    first, second = recorder.calls
    (o1, q1, k, v, ks, vs, bt, sl, qsl, mq, wm, s, scale, ws_o, ws_lse) = first[:15]
    # The MTP target's layers: no window, causal, i.e. the causal kernel as
    # before the window and non-causal variants existed; the 15 arguments
    # before them are the ones it always got.
    assert first[15:] == second[15:] == (0, True)
    d = md.decode
    assert (o1.shape[0], q1.shape[0]) == (9, 9)
    assert o1.data_ptr() == out.data_ptr() and q1.data_ptr() == query.data_ptr()
    assert (wm, s, mq) == (d.plan.wm, d.plan.num_splits, 4)
    assert scale == pytest.approx(D**-0.5)
    # K/V are cut to the head size (the 4 scale bytes of padding are not data).
    assert k.shape == (4, 16, HKV, D) and v.shape == (4, 16, HKV, D)
    assert k.stride(-1) == 1 and v.stride(-1) == 1
    assert ks.shape == vs.shape == (4, 16, HKV) and ks.dtype == torch.float32
    # Workspace views have this launch's [S, T, Hq, D] / [S, T, Hq] shapes.
    assert ws_o.shape == (s, 9, HQ, D) and ws_lse.shape == (s, 9, HQ)
    o2, q2 = second[0], second[1]
    assert (o2.shape[0], q2.shape[0]) == (320, 320)
    assert o2.data_ptr() == out[9:].data_ptr()
    assert second[8].tolist() == [0, 300, 320]
    assert second[10] == 4  # prefill wm
    assert second[11] == md.prefill.plan.num_splits == 1
    assert second[13].numel() == 0  # S == 1: an empty workspace tensor


# ------------------------------------------------- speculative-decoding drafter


def test_impl_keeps_tritons_window_tuple_and_the_ops_window() -> None:
    """The drafter layer (D = 128, window 2048) constructs; vLLM's helpers still
    read Triton's (left, right) tuple, the op gets the int."""
    impl = make_impl(num_heads=16, num_kv_heads=4, head_size=128, sliding_window=2048)
    assert impl.sliding_window == (2047, 0)
    assert impl.window == 2048
    assert make_impl().window == 0


@pytest.mark.parametrize("geom", [DRAFTER_TP2, DRAFTER_TP1], ids=["tp2", "tp1"])
def test_drafter_group_builds_a_windowed_non_causal_decode_slice(geom) -> None:
    """The drafter's step: 2 requests x 8 query tokens, causal=False (bool).

    The q = 8 block is a decode (threshold max(64 // 4, 8) = 16), planned with
    the group's window; the metadata carries causal=False to the Impl.
    """
    b = make_builder(**geom)
    assert b.window == 2048 and b.reorder_batch_threshold == 16
    cam = make_cam([8, 8], seq_lens=[3000, 9000], causal=False)
    md = b.build(0, cam)
    assert md.prefill is None and md.decode is not None and md.causal is False
    hkv = geom["num_kv_heads"]
    want = planner.plan_launch(
        num_tokens=16,
        num_reqs=2,
        max_query_len=8,
        G=4,
        Hkv=hkv,
        D=128,
        num_sms=SMS,
        is_prefill=False,
        ws_token_splits=b.ws_token_splits,
        window=2048,
    )
    assert md.decode.plan == want
    # S = ceil(68 x 2 / Hkv): 34 at TP=2 (the cap, 34, equals it), 17 at TP=1.
    assert (want.wm, want.num_splits) == (2, 136 // hkv)
    # A capture build (seq_lens = 1) gives the same plan: graph-safe.
    assert b.build_for_cudagraph_capture(cam).decode.plan == want


def test_builder_plans_with_the_group_window() -> None:
    """The group's window reaches the planner: with W = 64 the cap binds.

    Why: at W = 2048 on 68 SMs the cap (34) equals the GPU-filling S (34), so
    the drafter test above cannot tell a builder that drops the window from
    one that passes it; W = 64 caps S at ceil((64 + 8 - 2) / 64) + 1 = 3.
    """
    windowed = make_builder(**{**DRAFTER_TP2, "sliding_window": 64})
    plain = make_builder(**{**DRAFTER_TP2, "sliding_window": None})
    cam = make_cam([8, 8], causal=False)
    assert windowed.build(0, cam).decode.plan.num_splits == 3
    assert plain.window == 0
    assert plain.build(0, cam).decode.plan.num_splits == 34


def test_group_window_reads_vllms_specs() -> None:
    """The window comes from the group's spec as vLLM builds it: a
    ``SlidingWindowSpec`` (hybrid allocator, the drafter's group) or a
    ``FullAttentionSpec`` that keeps a window (hybrid allocator off); none
    otherwise."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    geom = {"block_size": 864, "num_kv_heads": 4, "head_size": 128}
    swa = SlidingWindowSpec(**geom, dtype=torch.int8, sliding_window=2048)
    full_swa = FullAttentionSpec(**geom, dtype=torch.int8, sliding_window=2048)
    full = FullAttentionSpec(**geom, dtype=torch.int8)
    assert vllm_backend.group_window(swa) == 2048
    assert vllm_backend.group_window(full_swa) == 2048
    assert vllm_backend.group_window(full) == 0
    assert vllm_backend.group_window(make_spec()) == 0


def test_drafter_graph_query_bound_is_unchanged() -> None:
    """The FULL-graph q bound stays the row-tile/verify rule: 16 for the drafter
    (q = 8 fits), 10 for the MTP target, as without the window variants."""
    hook = PthFlashAttnMetadataBuilder.get_varlen_cudagraph_max_query_len
    assert hook(make_config(k=7, heads=16), make_spec(4, 128, 2048)) == 16
    assert hook(make_config(), make_spec()) == 10


@pytest.mark.parametrize("geom", [DRAFTER_TP2, DRAFTER_TP1], ids=["tp2", "tp1"])
def test_drafter_forward_passes_window_and_causal(geom, workspace, recorder) -> None:
    """One op call with window=2048, causal=False and the group's plan."""
    hq, hkv = geom["heads"], geom["num_kv_heads"]
    impl = make_impl(
        num_heads=hq,
        num_kv_heads=hkv,
        head_size=128,
        scale=128**-0.5,
        sliding_window=2048,
    )
    impl._reserve_workspace(make_config(k=7, heads=hq))
    b = make_builder(**geom)
    md, query, out = run_forward(impl, b, [8, 8], hq=hq, hkv=hkv, d=128, causal=False)
    (args,) = recorder.calls
    assert args[0] is out and args[1] is query
    assert (args[10], args[11]) == (md.decode.plan.wm, md.decode.plan.num_splits)
    assert args[15:] == (2048, False)


def test_causal_windowed_layer_passes_causal_true(workspace, recorder) -> None:
    """A causal sliding-window layer at D = 128 (window without non-causal)."""
    geom = {**DRAFTER_TP2, "k": 3, "sliding_window": 4096}
    impl = make_impl(num_heads=16, num_kv_heads=4, head_size=128, sliding_window=4096)
    impl._reserve_workspace(make_config(heads=16))
    run_forward(impl, make_builder(**geom), [1, 1], hq=16, hkv=4, d=128)
    (args,) = recorder.calls
    assert args[15:] == (4096, True)


def test_decode_only_forward_is_one_call_without_slicing(workspace, recorder) -> None:
    impl = make_impl()
    impl._reserve_workspace(make_config())
    b = make_builder()
    _, query, out = run_forward(impl, b, [4, 4])
    assert len(recorder.calls) == 1
    args = recorder.calls[0]
    assert args[0] is out and args[1] is query  # the whole tensors, no view op


def test_cache_views_are_built_once(workspace, recorder) -> None:
    impl = make_impl()
    impl._reserve_workspace(make_config())
    b = make_builder()
    cam = make_cam([4, 4])
    md = b.build(0, cam)
    kv = torch.zeros(4, HKV, 16, 2 * (D + 4), dtype=torch.int8)
    q = torch.zeros(8, HQ, D, dtype=torch.bfloat16)
    out = torch.zeros_like(q)
    impl.forward(None, q, None, None, kv, md, out)
    impl.forward(None, q, None, None, kv, md, out)
    first, second = recorder.calls
    assert first[2] is second[2] and first[4] is second[4]


def test_locked_workspace_fits_every_planned_launch(workspace, recorder) -> None:
    """After lock_workspace() every launch the builder plans still gets its views."""
    impl = make_impl()
    impl._reserve_workspace(make_config())
    ptr = workspace._current_workspaces[0].data_ptr()
    size = workspace._current_workspaces[0].numel()
    vllm_workspace.lock_workspace()
    b = make_builder()
    for q in ([1, 1], [4, 4], [10, 10], [1], [4], [300], [300, 20], [64, 3, 3]):
        run_forward(impl, b, q, max_reqs=8)
    buf = workspace._current_workspaces[0]
    assert (buf.data_ptr(), buf.numel()) == (ptr, size), "forward grew the workspace"


def test_launch_beyond_the_reservation_fails_loudly(workspace, recorder) -> None:
    """A plan that outgrew the reservation is a clear error, not a silent overrun."""
    impl = make_impl()
    impl._reserve_workspace(make_config())
    vllm_workspace.lock_workspace()
    b = make_builder()
    cam = make_cam([4, 4])
    md = b.build(0, cam)
    plan = md.decode.plan
    # Pretend the planner asked for twice the splits that fit.
    bad = PthAttnMetadata(
        **{
            **md.__dict__,
            "decode": md.decode.__class__(
                **{
                    **md.decode.__dict__,
                    "plan": plan.__class__(
                        **{
                            **plan.__dict__,
                            "num_splits": 2 * impl._ws_token_splits // plan.num_tokens
                            + 2,
                        }
                    ),
                }
            ),
        }
    )
    kv = torch.zeros(4, HKV, 16, 2 * (D + 4), dtype=torch.int8)
    q = torch.zeros(8, HQ, D, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="token-splits"):
        impl.forward(None, q, None, None, kv, bad, torch.zeros_like(q))


def test_vllm_backend_makes_no_cuda_call() -> None:
    """vLLM-facing code asks ``current_platform``, never ``torch.cuda``."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(vllm_backend.__file__).read_text())
    cuda_attrs = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and n.attr == "cuda"
        and isinstance(n.value, ast.Name)
        and n.value.id == "torch"
    ]
    assert not cuda_attrs, f"torch.cuda used at lines {cuda_attrs}"


def test_forward_has_no_host_syncs_or_allocations() -> None:
    """No ``.item()``/``.cpu()``/allocation inside the Impl's forward."""
    import ast
    import inspect
    import textwrap

    banned = {"item", "cpu", "tolist", "numpy", "empty", "zeros", "ones", "clone"}
    for fn in (
        PthFlashAttnImpl.forward,
        PthFlashAttnImpl._run_slice,
        PthFlashAttnImpl._caches,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and n.attr in banned:
                raise AssertionError(
                    f"{fn.__qualname__} uses .{n.attr} (line {n.lineno})"
                )
