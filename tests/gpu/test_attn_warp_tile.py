# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Build ladder step 12: one warp's QK and PV fragments on the GPU.

The tensor-core kernel computes S = Q K^T and O = P'' V through three
unproved-on-hardware mappings (proved only in the simulator):
``ldmatrix.b16`` on int8 K with Q pre-permuted by pi, the magic int8 -> fp16
conversion, and the V byte pairing with the d-permutation. This test runs the
kernel's own ``qk_subtile`` / ``pv_subtile`` (csrc/pth_attn/tiling.cuh) for
one warp, 16 rows and 64 tokens, through ``pth_attn_test_warp_tile``, and
compares with fp64.

Products are exact (fp16 x int8 needs <= 19 bits), so the only error is the
tensor cores' fp32 accumulation: at most ``n * 2^-23 * sum |terms|`` for n
terms (truncating alignment, Fasi et al. 2021; the same model as bound.py).
A wrong permutation, swizzle or byte pairing moves results by about the
size of the terms themselves, millions of times that tolerance; the negative
controls prove the comparison sees it.
"""

from __future__ import annotations

import pytest
import torch
from attn_lib import ATTN_LIB, AttnLib

pytestmark = pytest.mark.attn

ROWS, TOKENS = 16, 64  # one warp's M; one load tile
FP32_ACC_REL = 2.0**-23  # truncating fp32 accumulation per term (bound.py)


@pytest.fixture(scope="module")
def lib(device) -> AttnLib:
    if not ATTN_LIB.exists():
        pytest.fail(f"{ATTN_LIB} missing: run tools/build_attn_lib.sh")
    return AttnLib(ATTN_LIB)


def _inputs(D: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(ROWS, D, generator=g) * 4).half()
    k = torch.randint(-128, 128, (TOKENS, D), generator=g, dtype=torch.int8)
    v = torch.randint(-128, 128, (TOKENS, D), generator=g, dtype=torch.int8)
    p = torch.rand(ROWS, TOKENS, generator=g).half()
    return q, k, v, p


def _run(lib: AttnLib, device, q, k, v, p):
    D = q.shape[1]
    s = torch.full((ROWS, TOKENS), float("nan"), device=device)
    o = torch.full((ROWS, D), float("nan"), device=device)
    rc = lib.test_warp_tile(
        q.to(device), k.to(device), v.to(device), p.to(device), s, o
    )
    assert rc == 0, lib.last_error()
    torch.cuda.synchronize(device)
    return s.cpu().double(), o.cpu().double()


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("seed", [0, 1])
def test_qk_and_pv_fragments_match_fp64(lib, device, D, seed):
    """S and O of the kernel's fragment loops equal fp64 within accumulation.

    Why: every later step (and every grid case) relies on these loops; a
    fragment bug here would look like a numerics problem everywhere else.
    """
    q, k, v, p = _inputs(D, seed)
    s, o = _run(lib, device, q, k, v, p)
    qd, kd, vd, pd = q.double(), k.double(), v.double(), p.double()
    s_ref = qd @ kd.T
    o_ref = pd @ vd
    s_tol = D * FP32_ACC_REL * (qd.abs() @ kd.abs().T)
    o_tol = TOKENS * FP32_ACC_REL * (pd.abs() @ vd.abs())
    assert torch.isfinite(s).all() and torch.isfinite(o).all()
    assert ((s - s_ref).abs() <= s_tol).all(), (s - s_ref).abs().max()
    assert ((o - o_ref).abs() <= o_tol).all(), (o - o_ref).abs().max()


PI = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]  # K permutation pi


@pytest.mark.parametrize("D", [64, 256])
def test_negative_controls_are_far_outside_tolerance(lib, device, D):
    """The comparison discriminates: plausible wrong mappings fail by far.

    Controls: K with its d permuted by pi inside each 16-chunk (what a kernel
    that forgot to permute Q would compute), V with d in natural n-tile order
    (a kernel without the d-permutation), and the transposed token order of a
    wrong byte pairing (tokens 2q+1, 2q swapped). Each must differ from the
    kernel's output by more than 1000x the tolerance somewhere.
    """
    q, k, v, p = _inputs(D, 7)
    s, o = _run(lib, device, q, k, v, p)
    qd, kd, vd, pd = q.double(), k.double(), v.double(), p.double()
    perm = torch.tensor([16 * (d // 16) + PI[d % 16] for d in range(D)])
    s_tol = D * FP32_ACC_REL * (qd.abs() @ kd.abs().T)
    o_tol = TOKENS * FP32_ACC_REL * (pd.abs() @ vd.abs())
    s_wrong = qd @ kd[:, perm].T
    assert ((s - s_wrong).abs() / s_tol).max() > 1e3
    # Natural order: n-tile j column n holds d = 8 j + n instead of
    # 32 (j // 4) + 4 n + j % 4.
    dperm = torch.tensor(
        [32 * (j // 4) + 4 * n + j % 4 for j in range(D // 8) for n in range(8)]
    )
    o_wrong = (pd @ vd)[:, dperm]
    assert ((o - o_wrong).abs() / o_tol).max() > 1e3
    swap = torch.arange(TOKENS).view(-1, 2).flip(1).reshape(-1)
    o_swapped = pd @ vd[swap]
    assert ((o - o_swapped).abs() / o_tol).max() > 1e3
