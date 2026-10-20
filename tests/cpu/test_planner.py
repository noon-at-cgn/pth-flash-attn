# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the launch planner.

The planner turns a padded batch descriptor into (wm, bq, grid, num_splits).
Its one safety-critical property is that nothing it returns depends on a
sequence length, because CUDA graphs replay a captured grid while the lengths
grow. The tests below check that property structurally (the signature cannot
carry a length) and behaviourally (fuzzed batches with equal descriptors and
different lengths get identical plans), and they prove the check has teeth by
running it against two length-dependent mutant planners that must be caught.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import math
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pth_flash_attn import planner
from pth_flash_attn.planner import (
    DESCRIPTOR_FIELDS,
    LaunchPlan,
    c_target,
    max_tokens_per_tile,
    plan_launch,
    s_alone,
    split_tile_range,
    workspace_token_splits,
)

PLANNER_PATH = Path(planner.__file__)

# Production geometry per tensor-parallel rank (the gate tier): Qwen,
# Hq/Hkv = 12/2 so G = 6, D = 256, RTX 3080 with 68 SMs (the hardware ceilings
# re-measure the SM count), max_num_seqs = 2 and MTP k = 3 so decode q = 4.
PROD = {"G": 6, "Hkv": 2, "D": 256, "num_sms": 68}
PROD_MAX_NUM_SEQS = 2
GS = (1, 4, 6, 8, 16)
DS = (64, 128, 256)
QS = (1, 4, 10, 16)


def _ws(q_dec_max: int, max_num_seqs: int = PROD_MAX_NUM_SEQS, **geom: int) -> int:
    g = {**PROD, **geom}
    return workspace_token_splits(
        max_num_seqs=max_num_seqs,
        q_dec_max=q_dec_max,
        Hkv=g["Hkv"],
        D=g["D"],
        num_sms=g["num_sms"],
    )


def _random_descriptor(rng: random.Random) -> dict[str, Any]:
    """Draw a valid descriptor over the whole supported space."""
    is_prefill = rng.random() < 0.3
    num_reqs = rng.choice((1, 2, 3, 4, 7, 8, 16, 33, 64, 256))
    max_q = rng.randint(1, 8192) if is_prefill else rng.choice((1, 2, 4, 6, 10, 16))
    max_q = max(1, min(max_q, 8192 // num_reqs if is_prefill else max_q))
    # Any T with max_q <= T <= num_reqs * max_q is a consistent descriptor.
    num_tokens = rng.randint(max_q, num_reqs * max_q)
    return {
        "num_tokens": num_tokens,
        "num_reqs": num_reqs,
        "max_query_len": max_q,
        "G": rng.choice((1, 2, 4, 6, 8, 12, 16, 32, 40, 64)),
        "Hkv": rng.choice((1, 2, 4, 8)),
        "D": rng.choice(DS),
        "num_sms": rng.choice((1, 28, 46, 68, 82, 108, 132)),
        "is_prefill": is_prefill,
        "ws_token_splits": rng.choice((0, 1, 272, 1088, 4096, 1 << 20)),
        "window": rng.choice((0, 0, 1, 64, 65, 2048, 4096)),
    }


def _random_batch(rng: random.Random, desc: dict[str, Any]) -> dict[str, list[int]]:
    """Draw per-request query and KV lengths that the descriptor could describe.

    Query lengths may be shorter than the padded descriptor says (padding, and
    adaptive verification trimming drafts on the device), and KV lengths range
    from 1 token to 256k: everything a graph replay may see.
    """
    q_lens = [rng.randint(0, desc["max_query_len"]) for _ in range(desc["num_reqs"])]
    # Log-uniform from 1 to 2**18 = 262,144 tokens, so short contexts (where a
    # "do not split short sequences" heuristic would bite) are as common as long.
    seq_lens = [q + int(2 ** rng.uniform(0, 18)) for q in q_lens]
    return {"query_lens": q_lens, "seq_lens": [max(1, s) for s in seq_lens]}


class LengthOracle:
    """A side channel a careless planner might read lengths from.

    Real code could reach lengths through a global forward context or a cached
    attribute; the harness publishes each fuzzed batch here before planning so
    that a planner which peeks at it is exposed.
    """

    current: dict[str, list[int]] = {"query_lens": [], "seq_lens": []}


def assert_plans_independent_of_lengths(
    plan_fn: Callable[..., LaunchPlan], *, trials: int = 400, seed: int = 0
) -> None:
    """Raise AssertionError unless plan_fn is a function of the descriptor only.

    Two independent checks, because a length can leak in two ways:
    1. Structurally: the parameters must be exactly DESCRIPTOR_FIELDS, all
       keyword-only, with no *args/**kwargs through which a length could pass.
    2. Behaviourally: for random descriptors, several random batches (whose
       lengths are published to LengthOracle) must all get the identical plan.
    """
    params = inspect.signature(plan_fn).parameters
    kinds = {p.kind for p in params.values()}
    assert kinds <= {inspect.Parameter.KEYWORD_ONLY}, (
        f"planner parameters must be keyword-only, got kinds {kinds}"
    )
    assert tuple(params) == DESCRIPTOR_FIELDS, (
        f"planner parameters {tuple(params)} differ from the descriptor "
        f"{DESCRIPTOR_FIELDS}: a sequence-length input would break graph replay"
    )
    rng = random.Random(seed)
    for _ in range(trials):
        desc = _random_descriptor(rng)
        plans = set()
        for _ in range(4):
            LengthOracle.current = _random_batch(rng, desc)
            plans.add(plan_fn(**desc))
        assert len(plans) == 1, f"descriptor {desc} produced {len(plans)} plans"


def test_split_count_never_reads_seq_len() -> None:
    """The plan, and so the split count, cannot depend on any sequence length.

    Why: a FULL CUDA graph bakes the grid (incl. num_splits) of the capture step
    into every replay while seq_lens grow on the device. A length-dependent S
    would replay a grid planned for another length: wrong partition, and with
    S*T above the reservation, out-of-bounds partial writes. Enforced in the
    signature (no parameter can carry a length) and checked by fuzzing.
    """
    assert_plans_independent_of_lengths(plan_launch)
    good = {"num_tokens": 8, "num_reqs": 2, "max_query_len": 4, "is_prefill": False}
    with pytest.raises(TypeError):
        plan_launch(**good, **PROD, ws_token_splits=272, seq_lens=[100, 200])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        plan_launch(8, 2, 4, 6, 2, 256, 68, False, 272)  # type: ignore[misc]


def _mutant_with_length_param(*, max_seq_len: int = 0, **desc: Any) -> LaunchPlan:
    # Mutant A: a "helpful" extra input, e.g. to avoid splitting short contexts.
    plan = plan_launch(**desc)
    tiles = max(1, math.ceil(max_seq_len / 64))
    return dataclasses.replace(plan, num_splits=min(plan.num_splits, tiles))


def _mutant_peeking_at_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    G: int,
    Hkv: int,
    D: int,
    num_sms: int,
    is_prefill: bool,
    ws_token_splits: int,
    window: int = 0,
) -> LaunchPlan:
    # Mutant B: the same heuristic with a clean signature, reading the lengths
    # from a side channel. The structural check passes; only fuzzing catches it.
    plan = plan_launch(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        max_query_len=max_query_len,
        G=G,
        Hkv=Hkv,
        D=D,
        num_sms=num_sms,
        is_prefill=is_prefill,
        ws_token_splits=ws_token_splits,
        window=window,
    )
    longest = max(LengthOracle.current["seq_lens"], default=1)
    tiles = max(1, math.ceil(longest / 64))
    return dataclasses.replace(plan, num_splits=min(plan.num_splits, tiles))


@pytest.mark.parametrize(
    "mutant", [_mutant_with_length_param, _mutant_peeking_at_lengths]
)
def test_negative_control_length_dependent_planner_is_caught(
    mutant: Callable[..., LaunchPlan],
) -> None:
    """Negative control: the independence check FAILS on length-aware planners.

    Why: a property test that passes on every planner proves nothing. Both
    mutants implement the tempting "do not split short contexts" heuristic
    (S <= ceil(L/64)); one through an extra parameter, one through a side
    channel. If either slipped through, test_split_count_never_reads_seq_len
    would not be protecting graph replay.
    """
    with pytest.raises(AssertionError):
        assert_plans_independent_of_lengths(mutant)


def _decode(q: int, reqs: int, ws: int, **geom: int) -> LaunchPlan:
    g = {**PROD, **geom}
    return plan_launch(
        num_tokens=q * reqs,
        num_reqs=reqs,
        max_query_len=q,
        is_prefill=False,
        ws_token_splits=ws,
        **g,
    )


@pytest.mark.parametrize(
    ("q", "ws_q", "wm", "bq", "rows", "splits"),
    [
        # wm holds q*G rows in one tile: 6 -> 1; 24 -> 2; 60 -> 4. The kernel
        # then packs bq = floor(16*wm/6) tokens: 2, 5, 10 (12, 30, 60 rows).
        (1, 4, 1, 2, 12, 34),
        (4, 4, 2, 5, 30, 34),
        (10, 4, 4, 10, 60, 13),  # S limited: 272 // 20 tokens = 13
        (10, 16, 4, 10, 60, 34),
        # q=16: 96 rows -> 2 row tiles, filled 10 + 6 by the kernel rule (a
        # balanced 8 + 8 split is the alternative the row-tile sweep measures).
        # Either way rt = 2, so S = ceil(68 / (2 * 2)) = 17 and the
        # per-request CTA count stays 68.
        (16, 16, 4, 10, 60, 17),
        (16, 4, 4, 10, 60, 8),  # with the q=4-sized workspace: 272 // 32 = 8
    ],
)
def test_production_decode_plans(
    q: int, ws_q: int, wm: int, bq: int, rows: int, splits: int
) -> None:
    """Production shapes (G=6, Hkv=2, D=256, 68 SMs, 2 requests) plan as derived.

    Why: these are the documented production numbers; if a refactor changes one,
    the docs and the GPU test expectations must change with it.
    """
    plan = _decode(q, PROD_MAX_NUM_SEQS, _ws(ws_q))
    assert (plan.wm, plan.bq, plan.rows_per_cta, plan.num_splits) == (
        wm,
        bq,
        rows,
        splits,
    )
    assert plan.grid == (plan.num_tokens // bq + 2, 2, splits)


def test_rt_formula_keeps_rows_within_one_tile() -> None:
    """bq * G <= 64 for every decode shape, which the naive rule can violate.

    Why: the naive rule rt = ceil(q*G/64), bq = ceil(q/rt) fails: for G=6, q=21 that
    gives rt=2, bq=11, 66 rows: more than the 64 rows a CTA owns, i.e. a C++
    check failure or rows silently dropped. The planner instead takes
    rt = ceil(q / floor(64/G)), identical whenever G divides 64.
    """
    naive_rt = math.ceil(21 * 6 / 64)
    assert math.ceil(21 / naive_rt) * 6 == 66  # the counterexample is real
    plan = _decode(21, 1, 0)
    assert (plan.wm, plan.bq, plan.rows_per_cta) == (4, 10, 60)  # 10 + 10 + 1
    for g in range(1, 65):
        for q in range(1, 200):
            p = plan_launch(
                num_tokens=q,
                num_reqs=1,
                max_query_len=q,
                G=g,
                Hkv=1,
                D=128,
                num_sms=68,
                is_prefill=False,
                ws_token_splits=0,
            )
            rt = math.ceil(q / p.bq)
            assert p.rows_per_cta <= 16 * p.wm <= 64
            assert rt == math.ceil(q / max_tokens_per_tile(g)), (g, q)
            if 64 % g == 0:
                assert rt == math.ceil(q * g / 64)


def test_bq_is_the_kernel_rule_and_keeps_the_tile_count() -> None:
    """LaunchPlan.bq is the kernel's floor(16*wm/G), and that changes neither
    the per-request tile count nor S (25,472 cases: G 1-64, q 1-199, both
    launches).

    Why: the op schema has no bq argument, so the kernel (cache_geometry.cuh)
    derives its tile height from wm. If the plan reported another bq, grid.x
    and every number derived from bq would describe a launch that never runs.
    The checks: (1) bq equals the kernel rule; (2) a request of q tokens takes
    the fewest possible tiles, ceil(q / floor(64/G)), the count S was planned
    for; (3) bq >= the balanced ceil(q/rt) alternative, so that
    alternative would take the same tiles and the same S; (4) wm is minimal: one
    step fewer warps would need more tiles; (5) grid.x covers the tiles.
    """
    cases = 0
    for g, q, prefill in itertools.product(range(1, 65), range(1, 200), (False, True)):
        p = plan_launch(
            num_tokens=q,
            num_reqs=1,
            max_query_len=q,
            G=g,
            Hkv=2,
            D=128,
            num_sms=68,
            is_prefill=prefill,
            ws_token_splits=1 << 30,
        )
        cases += 1
        b_max = max_tokens_per_tile(g)
        rt_min = math.ceil(q / b_max)
        assert p.bq == 16 * p.wm // g, (g, q, prefill)  # (1)
        assert math.ceil(q / p.bq) == rt_min, (g, q, prefill)  # (2)
        if prefill:
            assert (p.wm, p.bq) == (4, b_max)
            continue
        assert p.bq >= math.ceil(q / rt_min)  # (3)
        if p.wm > 1:  # (4)
            assert 16 * (p.wm // 2) // g * rt_min < q, (g, q)
        assert p.q_tiles_upper >= rt_min  # (5)
        assert p.num_splits == math.ceil(136 / (2 * rt_min))  # C_target = 68 x 2
    assert cases == 25_472


def _grid_of_descriptors() -> list[dict[str, Any]]:
    out = []
    for g, d, q, reqs, prefill, ws in itertools.product(
        GS, DS, QS + (64, 512, 8192), (1, 2, 3, 8, 64), (False, True), (0, 272, 1088)
    ):
        if q * reqs > 8192 * 4:
            continue
        out.append(
            {
                "num_tokens": q * reqs,
                "num_reqs": reqs,
                "max_query_len": q,
                "G": g,
                "Hkv": 2,
                "D": d,
                "num_sms": 68,
                "is_prefill": prefill,
                "ws_token_splits": ws,
            }
        )
    return out


def test_split_bounds_and_workspace_fit() -> None:
    """1 <= S <= S_cap = min(S_alone, max(1, W_ts // T)); S > 1 => S*T <= W_ts.

    Why: S < 1 is no launch at all; S above S_alone only adds combine work
    (one row tile already fills the GPU at S_alone); S*T above the reserved
    token-splits would make the kernel write partials past the workspace.
    """
    for desc in _grid_of_descriptors():
        plan = plan_launch(**desc)
        cap = min(
            s_alone(Hkv=desc["Hkv"], D=desc["D"], num_sms=desc["num_sms"]),
            max(1, desc["ws_token_splits"] // desc["num_tokens"]),
        )
        assert 1 <= plan.num_splits <= cap, desc
        assert plan.ws_token_splits_used <= desc["ws_token_splits"], desc
        assert plan.rows_per_cta <= 16 * plan.wm, desc
        assert desc["G"] <= 16 * plan.wm, "C++ check G <= 16*wm must hold"


def test_one_request_fills_the_gpu_with_fewest_splits() -> None:
    """With room in the workspace, one decode request alone fills one wave.

    Why: the planner cannot see which request is long, so each must be able to
    occupy every SM by itself (per-request split): Hkv*rt*S >=
    C_target. S is also minimal (Hkv*rt*(S-1) < C_target), since every extra
    split costs a combine input and a pipeline fill.
    """
    for g, d, q, sms, hkv in itertools.product(
        GS, DS, QS, (28, 68, 82, 132), (1, 2, 4, 8)
    ):
        plan = plan_launch(
            num_tokens=q,
            num_reqs=1,
            max_query_len=q,
            G=g,
            Hkv=hkv,
            D=d,
            num_sms=sms,
            is_prefill=False,
            ws_token_splits=1 << 30,
        )
        rt = math.ceil(q / plan.bq)
        target = c_target(D=d, num_sms=sms)
        assert hkv * rt * plan.num_splits >= target
        assert hkv * rt * (plan.num_splits - 1) < target


def test_split_count_monotonicity() -> None:
    """S never grows with load and never shrinks with resources.

    Why: S is non-increasing in num_tokens (workspace clamp, prefill tiles),
    in max_query_len (more row tiles) and in Hkv (more heads fill the GPU),
    and non-decreasing in num_sms and ws_token_splits. A violation means a
    formula bug that would show up as a cliff in a sweep, not as an error.
    """
    base = {
        "num_tokens": 8,
        "num_reqs": 2,
        "max_query_len": 4,
        "G": 6,
        "Hkv": 2,
        "D": 256,
        "num_sms": 68,
        "is_prefill": False,
        "ws_token_splits": 272,
    }

    def s(**over: Any) -> int:
        return plan_launch(**{**base, **over}).num_splits

    for prefill in (False, True):
        seq = [
            s(num_tokens=t, num_reqs=t, max_query_len=1, is_prefill=prefill)
            for t in range(1, 300)
        ]
        assert seq == sorted(seq, reverse=True)
    seq = [
        s(max_query_len=q, num_tokens=q, num_reqs=1, ws_token_splits=1 << 20)
        for q in range(1, 65)
    ]
    assert seq == sorted(seq, reverse=True)
    seq = [s(Hkv=h) for h in range(1, 65)]
    assert seq == sorted(seq, reverse=True)
    seq = [s(num_sms=n) for n in range(1, 200)]
    assert seq == sorted(seq)
    seq = [s(ws_token_splits=w) for w in range(0, 400)]
    assert seq == sorted(seq)


def test_q_tiles_upper_covers_every_request_split() -> None:
    """grid.x = T // bq + num_reqs is >= sum_i ceil(q_i / bq) for any q_i.

    Why: CTAs past the real tile count exit after a binary search, which is
    cheap; a grid smaller than the real tile count would silently skip query
    tokens. The bound holds because ceil(q/bq) <= floor(q/bq) + 1.
    """
    rng = random.Random(1)
    for _ in range(3000):
        desc = _random_descriptor(rng)
        plan = plan_launch(**desc)
        # Split exactly num_tokens tokens across the requests, each <= max_q.
        left = desc["num_tokens"]
        q_lens = []
        for i in range(desc["num_reqs"]):
            rest = desc["num_reqs"] - i - 1
            lo = max(0, left - rest * desc["max_query_len"])
            q = rng.randint(lo, min(left, desc["max_query_len"]))
            q_lens.append(q)
            left -= q
        assert left == 0
        assert sum(math.ceil(q / plan.bq) for q in q_lens) <= plan.q_tiles_upper


@pytest.mark.parametrize("g", GS)
@pytest.mark.parametrize("d", DS)
@pytest.mark.parametrize("reqs", [1, 64])
@pytest.mark.parametrize("q", QS)
def test_edge_case_grid(g: int, d: int, reqs: int, q: int) -> None:
    """Every (G, D, batch, q) edge case gives a launchable plan.

    Why: the gate tier only covers G=6, D=256; the sweep and public tiers
    (G 1-16, D 64/128, batch 1 vs many) must not hit an unplanned corner
    (wm outside {1,2,4}, rows over 64, S < 1) on first contact with a GPU.
    """
    ws = workspace_token_splits(max_num_seqs=reqs, q_dec_max=q, Hkv=2, D=d, num_sms=68)
    plan = _decode(q, reqs, ws, G=g, D=d)
    assert plan.wm in (1, 2, 4)
    assert 1 <= plan.bq <= max_tokens_per_tile(g)
    assert plan.rows_per_cta <= 16 * plan.wm
    assert 1 <= plan.num_splits <= s_alone(Hkv=2, D=d, num_sms=68)
    assert plan.num_splits * plan.num_tokens <= max(ws, plan.num_tokens)
    if reqs == 1:
        # Batch 1 is the case the per-request split exists for: it fills a wave.
        rt = math.ceil(q / plan.bq)
        assert 2 * rt * plan.num_splits >= c_target(D=d, num_sms=68)


def test_prefill_examples() -> None:
    """Prefill splits only when the chunk is too small to fill the GPU.

    Why: an 8192-token chunk already has 820 row tiles per KV head,
    so S_p = 1 (no combine); a 64-token chunk on a long prefix has 7 tiles x 2
    heads = 14 CTAs and wants ceil(68/14) = 5 splits, limited by the q=4
    production workspace to 272 // 64 = 4.
    """
    ws = _ws(4)
    big = plan_launch(
        num_tokens=8192,
        num_reqs=1,
        max_query_len=8192,
        is_prefill=True,
        ws_token_splits=ws,
        **PROD,
    )
    assert (big.wm, big.bq, big.num_splits) == (4, 10, 1)
    assert big.ws_o_shape[0] == 0 and big.ws_token_splits_used == 0
    small = plan_launch(
        num_tokens=64,
        num_reqs=1,
        max_query_len=64,
        is_prefill=True,
        ws_token_splits=ws,
        **PROD,
    )
    assert (small.wm, small.bq, small.num_splits) == (4, 10, 4)
    roomy = plan_launch(
        num_tokens=64,
        num_reqs=1,
        max_query_len=64,
        is_prefill=True,
        ws_token_splits=1 << 20,
        **PROD,
    )
    assert roomy.num_splits == 5


def test_mixed_batch_uses_two_launches_that_share_the_workspace() -> None:
    """A mixed batch is planned as a decode launch plus a prefill launch.

    Why: one prefill-policy launch with S=1 would leave a 128k decode
    request as a straggler. Requests with q <= threshold = max(64 // G, q_spec)
    go to the split decode launch; the rest to the prefill launch with its own
    S_p. The two launches run back to back on one stream, so each may use the
    whole reservation, and each must fit it on its own.
    """
    ws = _ws(4)
    threshold = max(max_tokens_per_tile(PROD["G"]), 4)
    assert threshold == 10
    # Production mixed shape: one 8188-token prefill + 4-token decodes.
    q_lens = [4, 4, 8188]
    decode = [q for q in q_lens if q <= threshold]
    prefill = [q for q in q_lens if q > threshold]
    plans = [
        plan_launch(
            num_tokens=sum(part),
            num_reqs=len(part),
            max_query_len=max(part),
            is_prefill=is_pf,
            ws_token_splits=ws,
            **PROD,
        )
        for part, is_pf in ((decode, False), (prefill, True))
    ]
    assert [p.is_prefill for p in plans] == [False, True]
    assert plans[0].num_splits == 34  # the decode slice still fills the GPU
    assert plans[1].num_splits == 1  # the big chunk needs no split
    assert all(p.ws_token_splits_used <= ws for p in plans)


def test_graph_shape_stable_across_replays() -> None:
    """Replaying one captured descriptor yields one grid and one workspace view.

    Why: during FULL-graph replay only device tensors change: seq_lens grow
    every step and adaptive verification may shrink the device q_len below the
    host's. The plan (grid, wm, bq, S, ws view shapes) must be bit-identical
    for all those steps, and a frozen, hashable plan cannot be mutated by a
    caller between capture and replay.
    """
    desc = {
        "num_tokens": 8,
        "num_reqs": 2,
        "max_query_len": 4,
        "is_prefill": False,
        "ws_token_splits": _ws(4),
        **PROD,
    }
    captured = plan_launch(**desc)
    rng = random.Random(2)
    seq_lens = [1000, 5]
    for _ in range(500):
        seq_lens = [s + rng.randint(0, 4) for s in seq_lens]  # decode steps
        LengthOracle.current = {
            "query_lens": [rng.randint(0, 4) for _ in seq_lens],
            "seq_lens": seq_lens,
        }
        replay = plan_launch(**desc)
        assert replay == captured
        assert (replay.grid, replay.ws_o_shape, replay.ws_lse_shape) == (
            captured.grid,
            captured.ws_o_shape,
            captured.ws_lse_shape,
        )
    assert hash(captured) == hash(plan_launch(**desc))
    with pytest.raises(dataclasses.FrozenInstanceError):
        captured.num_splits = 1  # type: ignore[misc]


def test_rejects_non_host_int_inputs() -> None:
    """Only exact host ints/bools are accepted; device scalars are rejected.

    Why: a 0-d torch tensor converts to int via __index__, which would turn a
    planner call inside the forward path into a device-to-host sync
    and, under graph capture, an error. bool-as-int and floats hide caller bugs.
    """
    import torch  # only the test needs torch; the planner never imports it

    good: dict[str, Any] = {
        "num_tokens": 8,
        "num_reqs": 2,
        "max_query_len": 4,
        "is_prefill": False,
        "ws_token_splits": 272,
        **PROD,
    }
    plan_launch(**good)
    for key, bad in [
        ("num_tokens", torch.tensor(8)),
        ("num_tokens", 8.0),
        ("G", True),
        ("num_sms", None),
        ("is_prefill", 0),
        ("window", True),
        ("window", torch.tensor(2048)),
        ("window", None),
    ]:
        with pytest.raises(TypeError):
            plan_launch(**{**good, key: bad})
    for key, bad in [
        ("G", 65),
        ("D", 96),
        ("num_tokens", 9),
        ("max_query_len", 9),
        ("num_reqs", 0),
        ("ws_token_splits", -1),
        ("window", -1),
    ]:
        with pytest.raises(ValueError):
            plan_launch(**{**good, key: bad})


# ------------------------------------------------------------ sliding window

# Windows around the tile size and the speculative-decoding drafter's 2048.
WINDOWS = (1, 2, 63, 64, 65, 127, 1000, 2047, 2048, 2049, 4096)


def _split_cap(window: int, q_max: int) -> int:
    """The S cap: the most KV tiles W + q_max - 1 consecutive keys can touch."""
    return math.ceil((window + q_max - 2) / 64) + 1


def assert_window_zero_is_the_causal_plan(plan_fn: Callable[..., LaunchPlan]) -> None:
    """Raise AssertionError unless window=0 plans as the causal planner did.

    The default and an explicit 0 agree on every grid descriptor, and the
    production MTP plans (G=6, Hkv=2, D=256, 2 requests; the documented numbers, as
    in test_production_decode_plans) hold with window=0 passed explicitly.
    """
    for desc in _grid_of_descriptors():
        assert plan_fn(**desc) == plan_fn(**desc, window=0), desc
    for q, ws_q, wm, splits in ((1, 4, 1, 34), (4, 4, 2, 34), (10, 4, 4, 13)):
        plan = plan_fn(
            num_tokens=2 * q,
            num_reqs=2,
            max_query_len=q,
            is_prefill=False,
            ws_token_splits=_ws(ws_q),
            window=0,
            **PROD,
        )
        assert (plan.wm, plan.num_splits) == (wm, splits), q


def test_window_zero_is_the_causal_plan() -> None:
    """window=0 (no window, the MTP target's layers) plans exactly as before.

    Why: the causal MTP path must not change: the backend
    now always passes a window, 0 for these layers.
    """
    assert_window_zero_is_the_causal_plan(plan_launch)


def test_window_caps_only_the_split_count() -> None:
    """W > 0: S = min(S without window, ceil((W + q_max - 2) / 64) + 1).

    Why: the cap is the planner's formula and nothing else moves:
    wm, bq, the grid's x and y and the descriptor fields are the unwindowed
    plan's, so the kernel's tiling is untouched; only empty splits are cut.
    """
    for desc in _grid_of_descriptors():
        base = plan_launch(**desc)
        for w in WINDOWS:
            cap = _split_cap(w, desc["max_query_len"])
            got = plan_launch(**desc, window=w)
            want = dataclasses.replace(base, num_splits=min(base.num_splits, cap))
            assert got == want, (desc, w)
            assert 1 <= got.num_splits <= cap


def _max_visible_tiles(window: int, q_max: int, causal: bool) -> int:
    """Brute force: most 64-token KV tiles one row tile of <= q_max tokens sees.

    Semantics: the q queries of a request are its last q
    positions; the query at p sees [max(0, p - W + 1), hi] with hi = p if
    causal, else min(seq_len - 1, p + W - 1). A row tile's range runs from the
    first token's lo to the last token's hi, so the whole query block (q =
    q_max) is the widest row tile. Every alignment mod 64 and the clamp at 0
    are covered by the two seq_len ranges.
    """
    best = 0
    for seq_len in itertools.chain(
        range(q_max, q_max + 200), range(window, window + 200)
    ):
        first, last = seq_len - q_max, seq_len - 1
        lo = max(0, first - window + 1)
        hi = last if causal else min(seq_len - 1, last + window - 1)
        best = max(best, hi // 64 - lo // 64 + 1)
    return best


def assert_cap_is_the_exact_tile_count(cap_fn: Callable[[int, int], int]) -> None:
    """Raise AssertionError unless cap_fn(W, q_max) equals the most KV tiles a
    row tile can see, causal or not, for every W in WINDOWS and q_max in 1..17."""
    for w, q_max, causal in itertools.product(WINDOWS, range(1, 18), (True, False)):
        tiles = _max_visible_tiles(w, q_max, causal)
        assert cap_fn(w, q_max) == tiles, (w, q_max, causal, tiles)


def test_window_cap_is_the_exact_tile_count() -> None:
    """The cap is the most KV tiles one row tile can see, no more, no fewer.

    Why: a cap below the visible tile count would make every split walk more
    tiles than needed (slower, never wrong); a cap above it leaves at least
    one empty split whenever it binds (an extra CTA writing lse = -inf and an
    extra combine input). At exactly the count, the cap removes only splits
    that would have been empty. Checked by brute force over windows around the
    tile size and 2048, query blocks up to 17 tokens (the drafter's is 8),
    causal and non-causal.
    """
    assert_cap_is_the_exact_tile_count(_split_cap)
    # The planner applies the same number (the cap binds: 4096 SMs, room).
    for w, q_max in itertools.product(WINDOWS, range(1, 18)):
        plan = plan_launch(
            num_tokens=q_max,
            num_reqs=1,
            max_query_len=q_max,
            G=1,
            Hkv=1,
            D=128,
            num_sms=1 << 12,
            is_prefill=False,
            ws_token_splits=1 << 30,
            window=w,
        )
        assert plan.num_splits == _split_cap(w, q_max), (w, q_max)


@pytest.mark.parametrize(
    "mutant",
    [
        pytest.param(lambda w, q: math.ceil(w / 64) + 1, id="forgets_q_max"),
        pytest.param(
            lambda w, q: math.ceil((w + q - 1) / 64), id="forgets_misalignment"
        ),
        # The first formula considered: the count plus a spare (one empty split).
        pytest.param(
            lambda w, q: math.ceil((w + q + 63) / 64) + 1, id="count_plus_spare"
        ),
        # One too many when W + q - 1 = 1 mod 64 (e.g. W = 64, q = 2).
        pytest.param(
            lambda w, q: math.ceil((w + q - 1) / 64) + 1, id="off_by_one_at_1_mod_64"
        ),
    ],
)
def test_negative_control_inexact_window_cap_is_caught(
    mutant: Callable[[int, int], int],
) -> None:
    """Negative control: caps that are too small or too large fail the check.

    Why: shows the brute-force check tells the exact count from the tempting
    shortcuts (only the window; tile-aligned keys) and from the over-counts
    that leave an empty split (the spare; ceil(n / 64) + 1).
    """
    with pytest.raises(AssertionError):
        assert_cap_is_the_exact_tile_count(mutant)


def test_negative_control_cap_applied_without_a_window_is_caught() -> None:
    """Negative control: a planner that caps S at window 0 changes the MTP plan.

    Why: shows test_window_zero_is_the_causal_plan has teeth: the window-0
    formula would give ceil((1 - 2) / 64) + 1 = 1 split for MTP q = 1, not 34.
    """

    def cap_applied_at_zero(**desc: Any) -> LaunchPlan:
        plan = plan_launch(**desc)
        cap = _split_cap(desc.get("window", 0), desc["max_query_len"])
        return dataclasses.replace(plan, num_splits=min(plan.num_splits, cap))

    with pytest.raises(AssertionError):
        assert_window_zero_is_the_causal_plan(cap_applied_at_zero)


@pytest.mark.parametrize(
    ("Hkv", "num_sms", "splits"),
    [
        # TP=2 rank: 16/4 heads. S_want = ceil(2 x SMs / 4) = 34, 35, 41; the
        # window cap (34) binds at 70 and 82 SMs.
        (4, 68, 34),
        (4, 70, 34),
        (4, 82, 34),
        # TP=1: 32/8 heads. S_want = ceil(2 x SMs / 8) = 17, 18, 21: below 34.
        (8, 68, 17),
        (8, 70, 18),
        (8, 82, 21),
    ],
)
def test_drafter_plans(Hkv: int, num_sms: int, splits: int) -> None:
    """The speculative-decoding drafter's layers (D = 128, G = 4, window 2048,
    q = 1 + 7 = 8) on 68 SMs (RTX 3080), 70 and 82 SMs (other sm_86 boards).

    Why: these are the split counts the GPU tests must replay. One
    row tile of 8 tokens x 4 heads = 32 rows (wm = 2, bq = 8). Its keys
    [p - 2047, seq_len) are 2055 tokens, at most ceil(2054 / 64) + 1 = 34 KV
    tiles, so the cap is 34: where it binds no split is empty at the worst
    alignment (at a better one, 33 tiles leave one split empty). A small
    window caps harder: W = 64 gives ceil(70 / 64) + 1 = 3.
    """
    geom = {"G": 4, "Hkv": Hkv, "D": 128, "num_sms": num_sms}
    ws = workspace_token_splits(
        max_num_seqs=2, q_dec_max=8, Hkv=Hkv, D=128, num_sms=num_sms
    )
    assert _split_cap(2048, 8) == 34
    for reqs in (1, 2):
        desc = {
            "num_tokens": 8 * reqs,
            "num_reqs": reqs,
            "max_query_len": 8,
            "is_prefill": False,
            "ws_token_splits": ws,
            **geom,
        }
        plan = plan_launch(**desc, window=2048)
        assert (plan.wm, plan.bq, plan.rows_per_cta) == (2, 8, 32)
        assert plan.num_splits == splits, (Hkv, num_sms, reqs)
        unwindowed = plan_launch(**desc)
        assert plan == dataclasses.replace(unwindowed, num_splits=splits)
        assert plan_launch(**desc, window=64).num_splits == 3


def test_split_tile_ranges_partition_and_balance() -> None:
    """Splits partition [0, T) into disjoint ranges differing by <= 1 tile.

    Why: balanced splits keep the slowest CTA within one tile of
    the mean; disjoint covering ranges mean each KV token is counted exactly
    once in the combine. When T < S the extra splits are empty and must be
    representable (they write lse = -inf).
    """
    for num_splits in range(1, 70):
        for num_tiles in range(0, 300, 7):
            ranges = [
                split_tile_range(split=s, num_splits=num_splits, num_tiles=num_tiles)
                for s in range(num_splits)
            ]
            assert ranges[0][0] == 0 and ranges[-1][1] == num_tiles
            assert all(a[1] == b[0] for a, b in itertools.pairwise(ranges))
            sizes = [e - b for b, e in ranges]
            assert min(sizes) >= 0 and max(sizes) - min(sizes) <= 1
            assert sizes.count(0) == max(0, num_splits - num_tiles)


def test_planner_is_host_only() -> None:
    """planner.py imports no torch and touches no device.

    Why: the planner runs in the metadata builder on every
    step and in CPU-only CI; a device query (SM count, torch.cuda) there would
    tie the plan to the build host and add a sync. Checked statically (no
    torch import, no `cuda` attribute anywhere) and dynamically (importing the
    file alone does not load torch).
    """
    tree = ast.parse(PLANNER_PATH.read_text())
    imported = {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for a in n.names
    }
    imported |= {
        (n.module or "").split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
    }
    assert imported <= {"__future__", "dataclasses"}, imported
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "cuda" for n in ast.walk(tree)
    )
    # Load planner.py by path (not through the package __init__, which may
    # import torch for ops.py) and check that it alone never pulls torch in.
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(PLANNER_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['p'] = m\n"
        "spec.loader.exec_module(m)\n"
        "m.plan_launch(num_tokens=8, num_reqs=2, max_query_len=4, G=6, Hkv=2, "
        "D=256, num_sms=68, is_prefill=False, ws_token_splits=272)\n"
        "assert 'torch' not in sys.modules, 'planner pulled in torch'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
