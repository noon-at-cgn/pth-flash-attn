# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The tensor-core kernel on the full correctness grid.

Every case of tests/gpu/grid_cases.py (paging edges x block sizes x layouts,
row tiling x GQA x head sizes, mixed batches with padding rows, splits up to
S_max, L = 256k, scale extremes, dtypes, stride fuzz, production
descriptors) runs through ``libpth_attn.so``. A case passes when every valid
element is within ``bound()`` for the kernel's own tiling
(Tiling(64, SUB, WN, S, bq), reported by the library), nothing is NaN/inf,
and the canary bytes show that only request rows of the views were written,
in ``out`` and in the workspace: the same gate the scalar reference passed in
the GPU correctness grid.

Why it matters: this is the correctness gate ("0 bound violations"); the
grid's sentinel keys make an off-by-one tile, page, split or causal edge
visible even at L = 256k.
"""

from __future__ import annotations

import attn_grid
import grid_cases as grid
import pytest
from grid_cuda import ref_kernel

pytestmark = pytest.mark.attn


def _param(c: grid.GridCase) -> object:
    marks = [pytest.mark.quick] if c.tier == "quick" else [pytest.mark.full]
    return pytest.param(c.name, id=c.name, marks=marks)


@pytest.mark.parametrize("name", [_param(c) for c in grid.cases("full")])
def test_attn_kernel_within_bound(name, device, record):
    """Every valid element within the bound; canaries intact (module doc)."""
    case = attn_grid.case_by_name(name)
    r = grid.run_case(case, ref_kernel(attn_grid.lib(), device))
    record(test="attn_grid", tiling=str(case.tiling), **r.record())
    assert r.passed, r.message()
