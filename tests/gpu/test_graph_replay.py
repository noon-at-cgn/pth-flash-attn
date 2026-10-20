# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""CUDA-graph capture/replay of the reference launch equals eager.

vLLM captures one graph per padded batch descriptor and replays it while the
sequence lengths grow on the device (planner.py module doc). The
launch therefore must read ``seq_lens`` and ``query_start_loc`` from device
memory on every replay and must not bake any length into the grid or the
arguments. This test captures the reference launch (attention plus combine)
once, then replays it after rewriting ``seq_lens`` and ``query_start_loc`` in
place: lengths that cross page and tile edges, shrink, and make the second
request empty (graph padding). After every replay

* the whole out and workspace buffers (canary bytes included) equal an eager
  launch with the same device contents, byte for byte (the kernel is
  deterministic: fixed-order combine), and
* the output is within the bound for those lengths.

Negative control: replaying WITHOUT updating ``seq_lens`` must differ from
the eager result for new lengths (otherwise the comparison could not see a
baked-in length).
"""

from __future__ import annotations

import grid_cases as grid
import numpy as np
import pytest
import torch
from grid_cuda import copy_back, prepare
from pth_testing.bound import bound
from pth_testing.reference import DenseKV

pytestmark = pytest.mark.graph

# Capture-time descriptor: 2 requests, 1 query token each, 2 padding rows
# (T = 4), production heads; the cache holds the longest step of each request.
CAPACITY = (4096, 817)
# (seq_lens, q_lens) per replay: page edges at block 16, tile edges at 64,
# a shrink, and an empty second request (its row becomes padding).
STEPS = (
    ((815, 1), (1, 1)),
    ((816, 64), (1, 1)),
    ((817, 65), (1, 1)),
    ((4096, 817), (1, 1)),
    ((65, 16), (1, 1)),
    ((1000, 0), (1, 0)),
)


def _case(S: int) -> grid.GridCase:
    return grid.GridCase(
        f"graph-S{S}",
        "graph",
        (1, 1),
        CAPACITY,
        block_size=16,
        num_splits=S,
        pad_rows=2,
        sentinel=True,
    )


def _qsl(q_lens):
    return torch.tensor([0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32)


def _fill_canary(dl) -> None:
    for t in (dl.bufs.out_buf, dl.bufs.ws_o_buf, dl.bufs.ws_lse_buf):
        if t is not None:
            t.view(torch.uint8).fill_(grid.CANARY_BYTE)


def _snapshot(dl) -> list[np.ndarray]:
    return [
        t.view(torch.uint8).cpu().numpy().copy()
        for t in (dl.bufs.out_buf, dl.bufs.ws_o_buf, dl.bufs.ws_lse_buf)
        if t is not None
    ]


def _gate(case, inp, host_bufs, seq, q_lens):
    """Bound check of the out rows for the lengths of one step."""
    kv = [
        DenseKV(x.k_int[:L], x.k_scale[:L], x.v_int[:L], x.v_scale[:L])
        for x, L in zip(inp.kv, seq)
    ]
    qsl = _qsl(q_lens)
    n = int(qsl[-1])
    b = bound(
        inp.q[:n],
        kv,
        qsl,
        inp.softmax_scale,
        tiling=case.tiling,
        out_dtype=case.torch_out_dtype,
        window=case.window,
        causal=case.causal,
    )
    o = host_bufs.out[:n].double().numpy()
    err = np.abs(o - b.ref)
    ok = (err <= b.bound)[b.valid]
    assert ok.all(), (
        f"graph step L={seq} q={q_lens}: {int((~ok).sum())} "
        f"elements over the bound, max err {err.max():.3g}"
    )


@pytest.mark.parametrize("S", [1, 34], ids=["S1", "S34"])
def test_graph_replay_equals_eager_with_changing_lengths(ref_lib, device, record, S):
    """Replay == eager byte for byte, and within the bound, at every step."""
    case = _case(S)
    inp = grid.build_inputs(case)
    host_bufs = grid.make_buffers(case)
    dl = prepare(ref_lib, inp, host_bufs, device)
    seq_d, qsl_d = dl.cache.seq_lens, dl.qsl

    # Warm-up (lazy module loading must not happen inside the capture).
    dl.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dl.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    for seq, q_lens in STEPS:
        seq_d.copy_(torch.tensor(seq, dtype=torch.int32))
        qsl_d.copy_(_qsl(q_lens))
        _fill_canary(dl)
        graph.replay()
        torch.cuda.synchronize()
        replayed = _snapshot(dl)
        _fill_canary(dl)
        dl.launch(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        eager = _snapshot(dl)
        for r, e in zip(replayed, eager):
            assert np.array_equal(r, e), f"replay != eager at L={seq} q={q_lens}"
        copy_back(dl.bufs, host_bufs)
        n = sum(q_lens)
        assert (host_bufs.out_buf[n:].view(torch.uint8) == grid.CANARY_BYTE).all(), (
            "padding rows were written"
        )
        _gate(case, inp, host_bufs, seq, q_lens)
        record(
            test="graph",
            case=f"{case.name}-L{seq[0]}-{seq[1]}-q{q_lens[1]}",
            passed=True,
        )

    # Negative control: stale device lengths must change the result.
    seq_d.copy_(torch.tensor(STEPS[0][0], dtype=torch.int32))
    qsl_d.copy_(_qsl(STEPS[0][1]))
    _fill_canary(dl)
    graph.replay()
    torch.cuda.synchronize()
    stale = _snapshot(dl)[0]
    seq_d.copy_(torch.tensor(STEPS[3][0], dtype=torch.int32))
    _fill_canary(dl)
    dl.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert not np.array_equal(stale, _snapshot(dl)[0])
