# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The scalar reference kernel and the combine against fp64 + bound.

Every case of the grid (tests/gpu/grid_cases.py) runs
through ``libpth_ref.so`` on the GPU. A case passes when every valid output
element is within the data-dependent bound (``bound()`` with the
reference's tiling, Tiling(64, 64, 1, S, bq)), no valid element is NaN/inf,
and the canary bytes show that nothing outside the request rows of the views
was written (out and workspace). Tightness ``max(err/bound)`` is recorded per
case; for the fp32 scalar reference it is expected around 1e-3..1e-2
(a property of the scalar arithmetic), which is not a defect: the bound is
written for the coarser
tensor-core kernel.

Why it matters: the reference kernel is the executable specification the
tensor-core kernel is later compared with, and the combine is shared with
it. A wrong page, slot, head, causal limit, split range or combine weight
here would propagate into every later comparison.
"""

from __future__ import annotations

import grid_cases as grid
import pytest
from grid_cuda import ref_kernel


def _param(c: grid.GridCase) -> object:
    marks = [pytest.mark.quick] if c.tier == "quick" else [pytest.mark.full]
    if c.name in grid.SANITIZER_CASES:
        marks.append(pytest.mark.sanitizer)
    return pytest.param(c, id=c.name, marks=marks)


@pytest.mark.parametrize("case", [_param(c) for c in grid.cases("full")])
def test_reference_kernel_within_bound(case, ref_lib, device, record):
    """Every valid element within the bound; canaries intact (see module doc)."""
    r = grid.run_case(case, ref_kernel(ref_lib, device))
    record(test="reference_grid", **r.record())
    assert r.passed, r.message()
