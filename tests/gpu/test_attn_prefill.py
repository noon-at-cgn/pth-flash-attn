# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Prefill (q = 2048, 8192 with prefix) and mixed launches within the bound.

The kernel runs the whole request on the GPU; the bound is checked on windows
of whole row tiles (attn_prefill.py explains why that is exact) because the
fp64 reference of a whole q = 8192 request would take hours. Also checked on
the whole output: every element finite and rows past the request tokens
(graph padding) untouched.
"""

from __future__ import annotations

import attn_grid
import attn_prefill
import grid_cases as grid
import numpy as np
import pytest
import torch
from attn_lib import attn_case
from grid_cuda import copy_back, prepare

pytestmark = pytest.mark.attn


@pytest.mark.parametrize(
    "name", [c.name for c in attn_prefill.cases()], ids=lambda n: n
)
def test_prefill_within_bound(name, device, record):
    """Every window of every request within the bound; canaries intact."""
    case = attn_case(attn_prefill.case_by_name(name), attn_grid.lib())
    inp = grid.build_inputs(case)
    bufs = grid.make_buffers(case)
    dl = prepare(attn_grid.lib(), inp, bufs, device)
    dl.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    copy_back(dl.bufs, bufs)
    n = int(inp.qsl[-1])
    out = bufs.out[:n].double().numpy()
    assert np.isfinite(out).all(), "non-finite output"
    assert (bufs.out_buf[n:].view(torch.uint8) == grid.CANARY_BYTE).all(), (
        "padding rows were written"
    )
    worst = 0.0
    for w in attn_prefill.windows(case):
        ref, bnd, valid, _ = attn_prefill.expected(case, w, inp)
        start = int(inp.qsl[w.req])
        got = out[start + w.t_start : start + w.t_end]
        err = np.abs(got - ref)
        bad = int((err > bnd)[valid].sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(err == 0, 0.0, err / bnd)
        worst = max(worst, float(np.max(ratio[valid])))
        assert bad == 0, f"{name} {w}: {bad} elements over the bound"
    record(test="attn_prefill", case=name, passed=True, tightness=worst)
