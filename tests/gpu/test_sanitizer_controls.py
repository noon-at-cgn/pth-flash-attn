# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Negative controls: compute-sanitizer must see planted errors.

A clean compute-sanitizer report only means something if the tool actually
instruments the kernels of ``libpth_ref.so`` (a ctypes-loaded library with a
statically linked CUDA runtime, inside a Python process). These tests plant
one error each and are run by ``tests/gpu/run_grid.sh`` under the matching tool,
which must then FAIL (exit code 86 from ``--error-exitcode``):

* ``oob``: block ids far beyond the cache, so the kernel reads unmapped
  global memory (memcheck: invalid __global__ read);
* ``uninit``: queries in a fresh, never-written device allocation
  (initcheck: uninitialized __global__ memory read).

They are deselected unless ``-m sanitizer_control`` is given, because without
a sanitizer the out-of-bounds read can fault the device context.
"""

from __future__ import annotations

import grid_cases as grid
import pytest
import torch
from grid_cuda import prepare

pytestmark = pytest.mark.sanitizer_control

# Block id for the out-of-bounds control: 2^30 pages past the pool, far
# beyond any allocation (int64 address math, keeps it unwrapped).
OOB_BLOCK_ID = 2**30


def _launch(ref_lib, device, mutate) -> None:
    case = grid.case_by_name("page-L65-bs16-NHD-D128-gqa1")
    inp = grid.build_inputs(case)
    dl = prepare(ref_lib, inp, grid.make_buffers(case), device)
    mutate(dl)
    dl.launch(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()


def test_oob(ref_lib, device):
    """Planted invalid global read (run under memcheck; must be reported)."""
    _launch(ref_lib, device, lambda dl: dl.cache.block_table.fill_(OOB_BLOCK_ID))


def test_uninit(ref_lib, device):
    """Planted read of never-written memory (run under initcheck)."""

    def mutate(dl):
        dl.q = torch.empty_like(dl.q)

    _launch(ref_lib, device, mutate)
