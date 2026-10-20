# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/reg_need.py (register need vs register use).

Tile choices are compared by register NEED (smallest spill-free cap),
because ptxas' reported count under a 255 budget includes registers spent on
scheduling. These tests check the need finder on crafted reports and on a
real kernel whose need is bracketed by construction.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import ptxas_gate  # noqa: E402
import reg_need  # noqa: E402

needs_nvcc = pytest.mark.skipif(
    not Path("/usr/local/cuda/bin/nvcc").is_file() and shutil.which("nvcc") is None,
    reason="nvcc not available",
)


def _res(regs: int, spill: int) -> ptxas_gate.KernelResources:
    return ptxas_gate.KernelResources(
        name="k", arch="sm_86", registers=regs, spill_stores=spill, spill_loads=spill
    )


def test_need_is_smallest_cap_above_the_first_spill():
    """Need = the last spill-free cap before the first spill, going down."""
    per_cap = {255: _res(200, 0), 248: _res(200, 0), 240: _res(240, 0),
               232: _res(232, 8), 224: _res(224, 0)}  # fmt: skip
    # 224 does not spill again, but 232 already did: the need is 240, not 224
    # (spill-freeness must hold for every cap above the need).
    assert reg_need.need_of(per_cap) == (240, True)


def test_never_spilling_and_always_spilling():
    """No spill at any cap: need <= smallest cap; spill at the top: unknown."""
    assert reg_need.need_of({255: _res(90, 0), 128: _res(90, 0)}) == (128, False)
    assert reg_need.need_of({255: _res(255, 4)}) == (None, False)


KERNEL = r"""
// 96 independent accumulators that must all be live across the loop: the
// need cannot be much below ~96 + addressing, and is far below 255.
extern "C" __global__ void __maxnreg__(PTH_PROBE_MAXNREG)
live96(const float* x, float* y, int n) {
  float a[96];
#pragma unroll
  for (int i = 0; i < 96; ++i) a[i] = x[threadIdx.x + i * 128];
  for (int k = 0; k < n; ++k) {
#pragma unroll
    for (int i = 0; i < 96; ++i) a[i] = a[i] * a[(i + 1) % 96] + 1.0f;
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 96; ++i) s += a[i];
  y[threadIdx.x] = s;
}
"""


@needs_nvcc
def test_real_kernel_need_is_bracketed(tmp_path, capsys):
    """A kernel with 96 live floats needs between 96 and 255 registers.

    The negative control is built in: one step below the reported need,
    ptxas must spill (that is what makes the need tight), which the tool
    records in ``spill_below_need``.
    """
    src = tmp_path / "live96.cu"
    src.write_text(KERNEL)
    out = tmp_path / "need.json"
    rc = reg_need.main([str(src), "--arch", "86", "--start", "160", "--stop", "64",
                        "--step", "16", "--json", str(out)])  # fmt: skip
    assert rc == 0
    import json

    (row,) = json.loads(out.read_text())["rows"]
    assert row["need_exact"] and 96 <= row["need"] <= 160
    st, ld = row["spill_below_need"]
    assert st > 0 and ld > 0
