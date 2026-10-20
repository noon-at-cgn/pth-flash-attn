# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the split-KV workspace size.

The workspace holds the fp32 partial outputs and log2-LSEs of every split. It
is reserved once, while the attention Impl is constructed at model load, so
vLLM's memory profile counts it inside the KV budget. After
``lock_workspace()`` any request for more bytes raises, and before the lock it
would allocate outside the budget. So the planner must never produce a launch
whose views are larger than the reservation, and the reservation must be
small enough to be worth its KV-cache cost (3.2-12.8 MiB in production).
"""

from __future__ import annotations

import inspect
import itertools
import math

import pytest
from pth_flash_attn.planner import (
    PARTIAL_BYTES_PER_ELEMENT,
    LaunchPlan,
    c_target,
    plan_launch,
    s_alone,
    workspace_bytes,
    workspace_shapes,
    workspace_token_splits,
)

MIB = 1 << 20
# Production per TP rank: Hq/Hkv = 12/2 (G=6), D=256, RTX 3080 (68 SMs),
# max_num_seqs=2, MTP k=3 so q=4; speculative verify q=16.
PROD = {"Hkv": 2, "D": 256, "num_sms": 68}
PROD_HQ = 12
PROD_MAX_NUM_SEQS = 2
# vLLM's WorkspaceManager.get_simultaneous rounds every sub-buffer up to 256 B
# (vllm/v1/worker/workspace.py: aligned_bytes = round_up(actual, 256)).
VLLM_WORKSPACE_ALIGN = 256


def _simultaneous_bytes(*shapes: tuple[int, ...]) -> int:
    """Bytes get_simultaneous needs for these fp32 shapes (vLLM's layout rule)."""
    total = 0
    for shape in shapes:
        actual = math.prod(shape) * PARTIAL_BYTES_PER_ELEMENT
        total += -(-actual // VLLM_WORKSPACE_ALIGN) * VLLM_WORKSPACE_ALIGN
    return total


@pytest.mark.parametrize(
    ("q_dec_max", "w_ts", "nbytes", "mib"),
    [
        (4, 272, 3_355_392, 3.2),  # MTP, k=3 draft tokens: verify q = 4
        (16, 1088, 13_421_568, 12.8),  # DFlash verify, q = 16
    ],
)
def test_production_workspace_reproduces_documented_numbers(
    q_dec_max: int, w_ts: int, nbytes: int, mib: float
) -> None:
    """The production reservation is 3.2 MiB (q=4) and 12.8 MiB (q=16).

    Why: the reservation is taken from the KV cache, so its size matters;
    the arithmetic is fixed here so a formula change cannot drift silently:

        C_target = 68 SMs x 1 CTA/SM (D=256)          = 68
        S_alone  = ceil(68 / Hkv=2)                    = 34
        W_ts     = min(max_num_seqs=2, 34) x q x 34    = 272 (q=4) | 1088 (q=16)
        bytes    = W_ts x Hq=12 x (D=256 + 1) x 4 B    = 3,355,392 | 13,421,568
                 = 3.19995 MiB | 12.79980 MiB, i.e. the documented 3.2 | 12.8
                   (12,336 B per token-split)
    """
    assert c_target(D=256, num_sms=68) == 68
    assert s_alone(**PROD) == 34
    got = workspace_token_splits(
        max_num_seqs=PROD_MAX_NUM_SEQS, q_dec_max=q_dec_max, **PROD
    )
    assert got == w_ts == min(2, 34) * q_dec_max * 34
    assert workspace_bytes(ws_token_splits=got, Hq=PROD_HQ, D=256) == nbytes
    assert nbytes == w_ts * PROD_HQ * (256 + 1) * 4
    assert round(nbytes / MIB, 1) == mib
    # vLLM's 256-B alignment adds less than 2 x 256 B to the reservation.
    o_shape, lse_shape = workspace_shapes(ws_token_splits=got, Hq=PROD_HQ, D=256)
    assert 0 <= _simultaneous_bytes(o_shape, lse_shape) - nbytes < 2 * 256


def test_public_example_llama8b() -> None:
    """Llama-3.1-8B at 256 sequences reserves 18.2 MiB.

    Why: a public model with many sequences is where a per-sequence workspace
    would explode; the min(max_num_seqs, S_alone) cap keeps it flat:
    C_target = 68 x 2 (D=128) = 136; S_alone = ceil(136/8) = 17;
    W_ts = min(256, 17) x 4 x 17 = 1156; bytes = 1156 x 32 x 129 x 4.
    """
    w = workspace_token_splits(max_num_seqs=256, q_dec_max=4, Hkv=8, D=128, num_sms=68)
    assert w == 1156
    nbytes = workspace_bytes(ws_token_splits=w, Hq=32, D=128)
    assert nbytes == 19_087_872
    assert round(nbytes / MIB, 1) == 18.2


def test_q_dec_max_is_the_verify_length_not_the_reorder_threshold() -> None:
    """Sizing by q_spec (4) instead of the threshold (10) is safe, only slower.

    Why: q_dec_max reads like "the decode threshold", but the 3.2 MiB is
    computed with q = 4, while the reorder threshold at G=6 is
    max(64 // 6, 4) = 10 (8.0 MiB). Both are safe: a decode slice of 10-token
    requests on the 4-sized workspace gets S = 272 // 20 = 13 instead of 34,
    never more than the reservation.
    """
    w10 = workspace_token_splits(max_num_seqs=2, q_dec_max=10, **PROD)
    assert workspace_bytes(ws_token_splits=w10, Hq=PROD_HQ, D=256) == 8_388_480
    w4 = workspace_token_splits(max_num_seqs=2, q_dec_max=4, **PROD)
    plan = plan_launch(
        num_tokens=20,
        num_reqs=2,
        max_query_len=10,
        G=6,
        is_prefill=False,
        ws_token_splits=w4,
        **PROD,
    )
    assert plan.num_splits == 13
    assert plan.ws_token_splits_used <= w4


def _all_plans(w_ts: int, geom: dict[str, int], max_reqs: int) -> list[LaunchPlan]:
    plans = []
    for reqs, q, prefill in itertools.product(
        range(1, max_reqs + 1), (1, 2, 4, 6, 10, 16, 64, 512), (False, True)
    ):
        plans.append(
            plan_launch(
                num_tokens=q * reqs,
                num_reqs=reqs,
                max_query_len=q,
                is_prefill=prefill,
                ws_token_splits=w_ts,
                **geom,
            )
        )
    return plans


def _overflowing(plans: list[LaunchPlan], reserved: int) -> list[LaunchPlan]:
    return [
        p
        for p in plans
        if p.num_splits > 1
        and _simultaneous_bytes(p.ws_o_shape, p.ws_lse_shape) > reserved
    ]


@pytest.mark.parametrize("q_dec_max", [1, 4, 16])
@pytest.mark.parametrize(
    ("G", "Hkv", "D"), [(6, 2, 256), (4, 8, 128), (4, 8, 64), (1, 4, 256), (16, 1, 128)]
)
def test_launch_views_never_grow_the_reserved_workspace(
    q_dec_max: int, G: int, Hkv: int, D: int
) -> None:
    """Every launch's workspace views fit in what the Impl reserved at load.

    Why: forward re-requests views of the *current* size each call. If a launch
    ever asked for more bytes than the load-time reservation, vLLM would either
    grow the buffer outside the KV budget (before lock_workspace) or raise
    (after it): an OOM risk or a crash mid-serving. Checked with vLLM's own
    256-B-aligned layout, for decode and prefill launches of any size.
    """
    geom = {"G": G, "Hkv": Hkv, "D": D, "num_sms": 68}
    w = workspace_token_splits(
        max_num_seqs=8, q_dec_max=q_dec_max, Hkv=Hkv, D=D, num_sms=68
    )
    o_shape, lse_shape = workspace_shapes(ws_token_splits=w, Hq=G * Hkv, D=D)
    reserved = _simultaneous_bytes(o_shape, lse_shape)
    assert not _overflowing(_all_plans(w, geom, max_reqs=12), reserved)


def test_negative_control_unclamped_splits_overflow_the_workspace() -> None:
    """Negative control: dropping the workspace clamp is caught by the check.

    Why: without S <= W_ts // T, a decode slice of two 10-token requests would
    take S = 34 (680 token-splits) against a 272-token-split reservation. The
    overflow check above must flag that, or it protects nothing.
    """
    w = workspace_token_splits(max_num_seqs=2, q_dec_max=4, **PROD)
    geom = {"G": 6, **PROD}
    o_shape, lse_shape = workspace_shapes(ws_token_splits=w, Hq=PROD_HQ, D=256)
    reserved = _simultaneous_bytes(o_shape, lse_shape)
    unclamped = [
        plan_launch(
            **{
                **dict(
                    num_tokens=p.num_tokens,
                    num_reqs=1,
                    max_query_len=p.num_tokens,
                    is_prefill=p.is_prefill,
                ),
                **geom,
            },
            ws_token_splits=1 << 30,
        )
        for p in _all_plans(w, geom, max_reqs=2)
    ]
    assert _overflowing(unclamped, reserved)


def test_sized_regime_gives_every_request_its_full_split() -> None:
    """Up to min(max_num_seqs, S_alone) requests of q <= q_dec_max, the
    workspace never lowers S below the per-request fill target.

    Why: this is what W_ts is sized for: each request in such a
    batch, in particular a lone long one, keeps ceil(C_target / (Hkv*rt))
    splits, so the clamp only engages outside the sized regime.
    """
    for max_seqs, q_dec in itertools.product((1, 2, 4, 34, 256), (1, 4, 16)):
        w = workspace_token_splits(max_num_seqs=max_seqs, q_dec_max=q_dec, **PROD)
        for reqs in range(1, min(max_seqs, s_alone(**PROD)) + 1):
            for q in range(1, q_dec + 1):
                plan = plan_launch(
                    num_tokens=q * reqs,
                    num_reqs=reqs,
                    max_query_len=q,
                    G=6,
                    is_prefill=False,
                    ws_token_splits=w,
                    **PROD,
                )
                rt = -(-q // plan.bq)
                assert plan.num_splits == -(-68 // (2 * rt)), (max_seqs, q, reqs)


def test_beyond_s_alone_requests_still_fill_the_gpu() -> None:
    """With more requests than S_alone, the clamp lowers S but the wave stays full.

    Why: past S_alone requests the workspace stops growing (that is the cap),
    so S shrinks; the GPU must then be filled by the requests themselves:
    Hkv * B * rt * S >= C_target for every B up to max_num_seqs.
    """
    for q_dec in (1, 4, 16):
        w = workspace_token_splits(max_num_seqs=256, q_dec_max=q_dec, **PROD)
        for reqs in range(1, 257):
            for q in sorted({1, q_dec}):
                plan = plan_launch(
                    num_tokens=q * reqs,
                    num_reqs=reqs,
                    max_query_len=q,
                    G=6,
                    is_prefill=False,
                    ws_token_splits=w,
                    **PROD,
                )
                rt = -(-q // plan.bq)
                assert 2 * reqs * rt * plan.num_splits >= 68, (q_dec, reqs, q)


def test_workspace_size_is_monotone_and_saturates() -> None:
    """W_ts grows with max_num_seqs, q_dec_max and SMs, and stops at S_alone seqs.

    Why: the cap min(max_num_seqs, S_alone) is what keeps a 256-sequence
    server from reserving a per-sequence workspace; a larger q or a bigger
    GPU must never shrink the reservation below what the planner will use.
    """
    sizes = [
        workspace_token_splits(max_num_seqs=n, q_dec_max=4, **PROD)
        for n in range(1, 300)
    ]
    assert sizes == sorted(sizes)
    assert len(set(sizes[33:])) == 1  # saturated from max_num_seqs = 34
    sizes = [
        workspace_token_splits(max_num_seqs=2, q_dec_max=q, **PROD)
        for q in range(1, 65)
    ]
    assert sizes == sorted(sizes)
    sizes = [
        workspace_token_splits(max_num_seqs=2, q_dec_max=4, Hkv=2, D=256, num_sms=n)
        for n in range(1, 200)
    ]
    assert sizes == sorted(sizes)


def test_workspace_signature_cannot_take_a_length() -> None:
    """workspace_token_splits takes exactly its five keyword-only host inputs.

    Why: the reservation is made once at load; any dependence on a sequence
    length would make it either too small later or a hidden sync now.
    """
    params = inspect.signature(workspace_token_splits).parameters
    assert tuple(params) == ("max_num_seqs", "q_dec_max", "Hkv", "D", "num_sms")
    assert {p.kind for p in params.values()} == {inspect.Parameter.KEYWORD_ONLY}
