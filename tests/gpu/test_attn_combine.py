# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""pth_attn_combine is bit-identical to the reference combine.

The tensor-core library has its own combine kernel (csrc/pth_attn/
attn_combine.cu) because the reference's walked the splits one load latency
at a time (14.6 us at S = 34). It keeps the reference's arithmetic in the
same order, so its output must equal pth_ref_combine's bit for bit, for
every split count including empty (-inf) splits and S not a multiple of the
kernel's load batch. A wrong order or a read of an empty split's o (which
is canary NaN here) would break the equality.
"""

from __future__ import annotations

import pytest
import torch
from attn_lib import ATTN_LIB, AttnLib
from pth_testing import ref_capi

pytestmark = pytest.mark.attn


@pytest.fixture(scope="module")
def libs(device):
    if not ATTN_LIB.exists() or not ref_capi.CUDA_LIB.exists():
        pytest.fail("build/attn/libpth_attn.so and build/ref/libpth_ref.so needed")
    return AttnLib(ATTN_LIB), ref_capi.RefLib(ref_capi.CUDA_LIB)


@pytest.mark.parametrize("S", [2, 7, 8, 9, 34, 204])
@pytest.mark.parametrize("D", [64, 256])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_combine_bit_identical_to_reference(libs, device, S, D, out_dtype):
    """Same bits as pth_ref_combine, empty splits never read (NaN canary)."""
    attn, ref = libs
    T, Hq = 5, 12
    g = torch.Generator().manual_seed(S * 1000 + D)
    ws_o = torch.randn(S, T, Hq, D, generator=g).to(device)
    ws_lse = (torch.randn(S, T, Hq, generator=g) * 8).to(device)
    empty = torch.rand(S, T, Hq, generator=g) < 0.3
    empty[0] = False  # every row keeps at least one live split
    ws_lse[empty.to(device)] = float("-inf")
    ws_o[empty.to(device)] = float("nan")
    qsl = torch.tensor([0, 2, T], dtype=torch.int32, device=device)
    outs = []
    for lib in (attn, ref):
        out = torch.full((T, Hq, D), float("nan"), dtype=out_dtype, device=device)
        args = ref_capi.combine_args(ws_o, ws_lse, out, qsl, 2, S)
        assert lib.combine(args, host=False) == 0, lib.last_error()
        torch.cuda.synchronize(device)
        outs.append(out)
    assert torch.isfinite(outs[0].float()).all()
    assert torch.equal(outs[0].view(torch.uint8), outs[1].view(torch.uint8))
