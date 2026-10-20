# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""One CTA of the decode kernel's QK and PV fragments on the GPU (D = 256).

The decode kernel (attn_decode.cuh) computes S^T = K Q^T and
O^T = V^T P'' with the roles of the shipped kernel's operands exchanged: K is
the MMA's A operand (ldmatrix.x4 of int8 + cvt4, K permuted by pi), Q^T and
P''^T are B operands (ldmatrix of 8-row groups; P'' through a shared buffer
that the four warps share), and V^T is an A operand built from LDS.32 + PRMT
pairs with the d-permutation. test_attn_warp_tile.py covers the
16-row kernel's mappings; this test runs the decode kernel's own ``dec_qk`` /
``dec_pv`` (through ``pth_attn_test_dec_tile``: four warps, 64 tokens, QK split
by tokens and PV by d exactly as in the kernel) for 1..32 rows and compares
with fp64.

Products are exact (fp16 x int8), so the only error is the tensor cores'
fp32 accumulation, at most ``n * 2^-23 * sum |terms|`` (the model of
bound.py). A wrong mapping moves results by the size of the terms
themselves, millions of times the tolerance; the negative controls prove the
comparison sees each mapping (pi, d-permutation, token pairing, the rows of a
column pair, the token halves of a fragment).
"""

from __future__ import annotations

import pytest
import torch
from attn_lib import ATTN_LIB, AttnLib

pytestmark = pytest.mark.attn

D = 256
TOKENS = 64  # one load tile
FP32_ACC_REL = 2.0**-23  # truncating fp32 accumulation per term (bound.py)
# n-tile edges (8, 16, 24, 32), partial n-tiles (1, 6, 13, 30), production
# decode (q = 1: 6 rows, q = 4: 24 rows).
ROWS = [1, 6, 8, 13, 16, 24, 30, 32]
PI = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]  # K permutation pi


@pytest.fixture(scope="module")
def lib(device) -> AttnLib:
    if not ATTN_LIB.exists():
        pytest.fail(f"{ATTN_LIB} missing: run tools/build_attn_lib.sh")
    return AttnLib(ATTN_LIB)


def _inputs(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(rows, D, generator=g) * 4).half()
    k = torch.randint(-128, 128, (TOKENS, D), generator=g, dtype=torch.int8)
    v = torch.randint(-128, 128, (TOKENS, D), generator=g, dtype=torch.int8)
    p = torch.rand(rows, TOKENS, generator=g).half()
    return q, k, v, p


def _run(lib: AttnLib, device, q, k, v, p):
    rows = q.shape[0]
    s = torch.full((rows, TOKENS), float("nan"), device=device)
    o = torch.full((rows, D), float("nan"), device=device)
    rc = lib.test_dec_tile(
        rows, q.to(device), k.to(device), v.to(device), p.to(device), s, o
    )
    assert rc == 0, lib.last_error()
    torch.cuda.synchronize(device)
    return s.cpu().double(), o.cpu().double()


def _tolerances(q, k, v, p):
    qd, kd, vd, pd = q.double(), k.double(), v.double(), p.double()
    s_tol = D * FP32_ACC_REL * (qd.abs() @ kd.abs().T)
    o_tol = TOKENS * FP32_ACC_REL * (pd.abs() @ vd.abs())
    return qd, kd, vd, pd, s_tol, o_tol


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("seed", [0, 1])
def test_decode_fragments_match_fp64(lib, device, rows, seed):
    """S and O of the decode kernel's fragment loops equal fp64 within accumulation.

    Why: the decode kernel serves every q <= 5 launch at D = 256; a mapping
    bug in one n-tile count (a ldmatrix row group, the P'' pitch, the
    odd-NT clamp of dec_pv) would look like a numerics problem in the grid.
    """
    q, k, v, p = _inputs(rows, seed)
    s, o = _run(lib, device, q, k, v, p)
    qd, kd, vd, pd, s_tol, o_tol = _tolerances(q, k, v, p)
    assert torch.isfinite(s).all() and torch.isfinite(o).all()
    assert ((s - qd @ kd.T).abs() <= s_tol).all(), (s - qd @ kd.T).abs().max()
    assert ((o - pd @ vd).abs() <= o_tol).all(), (o - pd @ vd).abs().max()


@pytest.mark.parametrize("rows", [6, 24, 32])
def test_negative_controls_are_far_outside_tolerance(lib, device, rows):
    """The comparison discriminates: plausible wrong mappings fail by far.

    Controls (each must differ from the kernel's output by > 1000x the
    tolerance somewhere): K with d permuted by pi in each 16-chunk (a kernel
    that forgot to permute Q), V with d in natural n-tile order (no
    d-permutation), V tokens 2q / 2q + 1 swapped (a wrong byte pairing), the
    two rows of a column pair swapped in S and in O (a wrong C-fragment
    reading), and the token halves g / g + 8 of a fragment exchanged in S.
    """
    q, k, v, p = _inputs(rows, 7)
    s, o = _run(lib, device, q, k, v, p)
    qd, kd, vd, pd, s_tol, o_tol = _tolerances(q, k, v, p)
    s_ref, o_ref = qd @ kd.T, pd @ vd

    def far(got, wrong, tol):
        return ((got - wrong).abs() / tol).max() > 1e3

    perm = torch.tensor([16 * (d // 16) + PI[d % 16] for d in range(D)])
    assert far(s, qd @ kd[:, perm].T, s_tol)
    # n-tile j column n holds d = 8 j + n instead of 32 (j // 4) + 4 n + j % 4.
    dperm = torch.tensor(
        [32 * (j // 4) + 4 * n + j % 4 for j in range(D // 8) for n in range(8)]
    )
    assert far(o, o_ref[:, dperm], o_tol)
    swap_tok = torch.arange(TOKENS).view(-1, 2).flip(1).reshape(-1)
    assert far(o, pd @ vd[swap_tok], o_tol)
    swap_row = torch.arange(rows) ^ 1
    swap_row = torch.where(swap_row < rows, swap_row, torch.arange(rows))
    assert far(s, s_ref[swap_row], s_tol)
    assert far(o, o_ref[swap_row], o_tol)
    halves = torch.arange(TOKENS).view(-1, 16)
    halves = torch.cat([halves[:, 8:], halves[:, :8]], dim=1).reshape(-1)
    assert far(s, s_ref[:, halves], s_tol)
