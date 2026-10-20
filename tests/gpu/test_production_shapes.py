# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Production shapes: split-count independence and workspace size.

Production runs D=256, Hkv=2, G=6 with max_num_seqs=2 and verify lengths up
to 16. The planner picks S from the
SM count and reserves W_ts token-splits of workspace at load. The grid's
``production`` family runs each production descriptor at the planner's S and
at S in {1, 2, 34} inside buffers of exactly the reserved size (canaries
beyond S*T); this module adds the checks that need the device:

* the device has the SM count the planner was given (68 on an RTX 3080), so
  the S and W_ts used in the grid are the ones production would use;
* the outputs for different S agree with each other to within the sum of
  their bounds (split-count independence, element by element).
"""

from __future__ import annotations

import grid_cases as grid
import numpy as np
import pytest
from grid_cuda import ref_kernel
from pth_flash_attn.planner import workspace_bytes


def test_device_matches_planner_inputs(device_info, record):
    """The planner input NUM_SMS equals the device's SM count.

    Why: S, the grid and W_ts are pure functions of the SM count (planner.py);
    if the device differed, the production rows would test a different
    launch than production runs.
    """
    record(
        test="production",
        case="device",
        passed=device_info["sm_count"] == grid.NUM_SMS,
        **{"sm_count": device_info["sm_count"]},
    )
    assert device_info["sm_count"] == grid.NUM_SMS


def test_workspace_size_is_the_planner_formula():
    """W_ts = 1088 token-splits, 13,421,568 B; every planner S fits.

    Why: the workspace is reserved once at load; a launch whose
    S*T exceeded it would write past the buffer.
    """
    w = grid.production_ws_token_splits()
    assert w == 1088
    assert workspace_bytes(ws_token_splits=w, Hq=12, D=256) == 1088 * 12 * 257 * 4
    for _, q, _ in grid.production_descriptors():
        assert grid.production_splits(q) * sum(q) <= w


@pytest.mark.parametrize("desc", grid.production_descriptors(), ids=lambda d: d[0])
def test_split_count_independence(ref_lib, device, record, desc):
    """Outputs at the planner's S and at S in {1, 2, 34} pass the gate, and
    pairwise |o_S - o_S'| <= bound_S + bound_S' (the triangle inequality
    through the fp64 reference: a split-dependent bug breaks it)."""
    name = desc[0]
    cases = [c for c in grid.cases("full") if c.name.startswith(f"prod-{name}-")]
    assert len(cases) >= 3
    outs = {}
    for c in cases:
        inp = grid.build_inputs(c)
        exp = grid.load_expected(c, inp)
        run = ref_kernel(ref_lib, device)(inp, grid.make_buffers(c))
        r = grid.check(c, inp, run, exp)
        assert r.passed, r.message()
        outs[c.num_splits] = (run.bufs.out.double().numpy(), exp.bound, exp.valid)
    worst = 0.0
    for s1, (o1, bd1, v) in outs.items():
        for s2, (o2, bd2, _) in outs.items():
            if s1 < s2:
                diff = np.abs(o1 - o2)[v]
                worst = max(worst, float((diff / (bd1 + bd2)[v]).max()))
    record(
        test="production",
        case=f"independence-{name}",
        passed=worst <= 1.0,
        ratio=worst,
        splits=sorted(outs),
    )
    assert worst <= 1.0
