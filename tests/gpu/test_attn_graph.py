# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""CUDA-graph replay of the tensor-core kernels equals eager, byte for byte.

vLLM captures the attention call once per padded batch descriptor and replays
it while ``seq_lens`` / ``query_start_loc`` (device tensors) change. The launch
parameters (grid, splits, row tile) are fixed at capture, so the kernels must
read every length from the device. This is test_graph_replay.py (the
reference library's test, same steps and gate) run with ``libpth_attn.so`` on
the production geometry (D = 256, Hkv = 2, G = 6, block 816, bf16), for the
decode kernel (q = 1, 4: rows 6, 24) and the 16-row kernel (q = 10: rows 60),
each with S = 1 (direct output) and S = 34 (workspace + combine; most splits
are empty at the short steps). The windowed kernel runs the same
steps at the DFlash2 drafter's geometry (D = 128, G = 4, q = 8, window 2048,
non-causal) with S = 1 and 3.

After every replay
* out and the workspace (canary bytes included) equal an eager launch with the
  same device contents, byte for byte,
* 16 further back-to-back replays (no host synchronisation: consecutive graph
  launches overlap CTA tails and heads on the SMs, which an eager launch with
  a synchronise does not) still give the same bytes, and
* the output is within the bound of the kernel's own tiling.

Negative control: a replay with stale device lengths must differ from the eager
result for the new lengths (otherwise a baked-in length would go unseen).

History: this test failed at its first version for every kernel because the
test's own case builder freed the query tensors whose raw pointers the launch
arguments held (attn_case.build now owns them); it was not a kernel fault.
"""

from __future__ import annotations

import attn_grid
import attn_lib
import grid_cases as grid
import numpy as np
import pytest
import torch
from grid_cuda import copy_back, prepare
from test_graph_replay import _fill_canary, _gate, _qsl, _snapshot

pytestmark = [pytest.mark.attn, pytest.mark.graph]

# Capture-time descriptor: 2 requests of up to CAPACITY tokens (block 816), 2
# padding rows; the cache holds the longest step of each request.
CAPACITY = (20000, 1700)
BLOCK = 816
# (seq_lens, q_len of request 0 and 1): page edges (815/816/817, 1632), tile
# edges (64 multiples), the longest length, a shrink, and an empty second
# request (graph padding; q = 0 and seq = 0 is the vLLM padding descriptor).
STEPS = (
    ((815, 1700), (1, 1)),
    ((816, 1699), (1, 1)),
    ((817, 64), (1, 1)),
    ((20000, 1633), (1, 1)),
    ((1632, 1632), (1, 1)),
    ((65, 16), (1, 1)),
    ((19999, 817), (1, 1)),
    ((1000, 0), (1, 0)),
    ((64 * 3, 12), (1, 1)),
)
BACK_TO_BACK = 16


def _case(q_len: int, S: int) -> grid.GridCase:
    return grid.GridCase(
        f"agraph-q{q_len}-S{S}",
        "graph",
        (q_len, q_len),
        CAPACITY,
        block_size=BLOCK,
        num_splits=S,
        pad_rows=2,
        sentinel=True,
    )


def _window_case(S: int) -> grid.GridCase:
    return grid.GridCase(
        f"agraph-win-q8-S{S}",
        "graph",
        (8, 8),
        CAPACITY,
        G=4,
        D=128,
        block_size=BLOCK,
        num_splits=S,
        pad_rows=2,
        sentinel=True,
        window=2048,
        causal=False,
    )


def _set_lengths(dl, seq, q_lens, q_len: int) -> None:
    dl.cache.seq_lens.copy_(torch.tensor(seq, dtype=torch.int32))
    dl.qsl.copy_(_qsl(tuple(q_len * x for x in q_lens)))


@pytest.fixture(scope="module")
def lib(device) -> attn_lib.AttnLib:
    """build/attn, or the library of ``-p attn_grid --attn-lib`` (A/B builds)."""
    return attn_grid.lib()


@pytest.mark.parametrize("S", [1, 34], ids=["S1", "S34"])
@pytest.mark.parametrize("q_len", [1, 4, 10], ids=["q1", "q4", "q10"])
def test_graph_replay_equals_eager(lib, device, record, q_len, S):
    """Replay == eager byte for byte, repeatable, within the bound (module doc)."""
    _replay_equals_eager(lib, device, record, _case(q_len, S), q_len)


@pytest.mark.parametrize("S", [1, 3], ids=["S1", "S3"])
def test_graph_replay_equals_eager_windowed(lib, device, record, S):
    """The windowed kernel: the same steps and checks."""
    _replay_equals_eager(lib, device, record, _window_case(S), 8)


def _replay_equals_eager(lib, device, record, grid_case, q_len) -> None:
    case = attn_lib.attn_case(grid_case, lib)
    inp = grid.build_inputs(case)
    host_bufs = grid.make_buffers(case)
    dl = prepare(lib, inp, host_bufs, device)
    stream = lambda: torch.cuda.current_stream().cuda_stream  # noqa: E731
    # Warm-up: the kernels' dynamic-shared-memory opt-in (cudaFuncSetAttribute)
    # is illegal inside a capture and happens at the first launch.
    dl.launch(stream())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dl.launch(stream())
    torch.cuda.synchronize()
    for seq, q_lens in STEPS:
        _set_lengths(dl, seq, q_lens, q_len)
        _fill_canary(dl)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _snapshot(dl)
        _fill_canary(dl)
        dl.launch(stream())
        torch.cuda.synchronize()
        eager = _snapshot(dl)
        for r, e in zip(replayed, eager):
            assert np.array_equal(r, e), f"replay != eager at L={seq} q={q_lens}"
        _fill_canary(dl)
        for _ in range(BACK_TO_BACK):
            graph.replay()
        torch.cuda.synchronize()
        for r, e in zip(_snapshot(dl), eager):
            assert np.array_equal(r, e), (
                f"back-to-back replays != eager at L={seq} q={q_lens}"
            )
        copy_back(dl.bufs, host_bufs)
        n = q_len * sum(1 for x in q_lens if x)
        assert (host_bufs.out_buf[n:].view(torch.uint8) == grid.CANARY_BYTE).all(), (
            "padding rows were written"
        )
        _gate(case, inp, host_bufs, seq, tuple(q_len * x for x in q_lens))
        record(
            test="attn_graph",
            case=f"{case.name}-L{seq[0]}-{seq[1]}-q{q_lens[1]}",
            passed=True,
        )
    # Negative control: stale device lengths must change the result.
    _set_lengths(dl, STEPS[0][0], STEPS[0][1], q_len)
    _fill_canary(dl)
    graph.replay()
    torch.cuda.synchronize()
    stale = _snapshot(dl)[0]
    _set_lengths(dl, STEPS[3][0], STEPS[3][1], q_len)
    _fill_canary(dl)
    dl.launch(stream())
    torch.cuda.synchronize()
    assert not np.array_equal(stale, _snapshot(dl)[0])
