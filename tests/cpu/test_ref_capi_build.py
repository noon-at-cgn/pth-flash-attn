# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The scalar reference kernel and the combine, built and checked on the CPU.

``tools/build_ref_lib.sh`` builds two libraries from ``csrc/pth_attn/ref_capi.cu``:
``libpth_ref.so`` (nvcc, sm_80/86/89) and ``libpth_ref_host.so`` (g++). The
host library runs the SAME kernel source (``reference_kernel.cu``) with each
warp emulated by 32 lock-step threads, so logic bugs in the executable
specification are found here, before any GPU run. The checks:

- the reference output on real vLLM-layout caches is within the error bound of
  ``pth_testing.bound`` against the fp64 reference, for every layout, block
  size, head size, dtype and split count in the matrix below;
- split partials are the per-split softmax (lse in log2 units) and the combine
  reproduces the unsplit result;
- the combine follows the planner's log2 rule (skip empty splits before
  reading their O) and the 16-bit stores round like torch;
- deliberate bugs (causal off by one, V read 4 bytes early, int32 offsets)
  fail the bound gate;
- the CUDA library builds without local memory and fails cleanly without a GPU.

The ctypes mirror of ``ref_capi.h`` is ``pth_testing.ref_capi`` (shared with
the GPU correctness grid, ``tests/gpu/grid*``).
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pth_testing.bound import Tiling, bound, tightness
from pth_testing.ref_capi import (
    ABI_VERSION,
    ECUDA,
    EINVAL,
    ENOCUDA,
    OK,
    RefLib,
    attn_args,
    combine_args,
)
from pth_testing.reference import make_int8_cache, make_queries

REPO = Path(__file__).resolve().parents[2]
TILE = 64  # KV tokens per tile: the split partition unit (planner)


def build_libs(repo: Path, host_only: bool) -> subprocess.CompletedProcess:
    """Run tools/build_ref_lib.sh of ``repo``."""
    cmd = ["bash", str(repo / "tools" / "build_ref_lib.sh")]
    if host_only:
        cmd.append("--host-only")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def host_lib() -> RefLib:
    """The g++ host-emulator library (skip if g++ is missing)."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    r = build_libs(REPO, host_only=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return RefLib(REPO / "build" / "ref" / "libpth_ref_host.so")


# --------------------------------------------------------------------------
# One case: run the reference (host or CUDA) and gate it with the bound
# --------------------------------------------------------------------------
SENTINEL = 1234.5  # never a valid attention output (|o| <= max |v| ~ 4)


def run_case(
    lib: RefLib,
    *,
    lens,
    q_lens,
    Hkv=2,
    G=6,
    D=128,
    block_size=16,
    layout="NHD",
    pad=0,
    high=False,
    q_dtype=torch.bfloat16,
    out_dtype=torch.float32,
    num_splits=1,
    bq=None,
    pad_tokens=3,
    seed=0,
    host=True,
    window=0,
    causal=True,
):
    """Run one case; returns (gate, out, ws_o, ws_lse, cache, q, qsl, scale, bq)."""
    c_cpu = make_int8_cache(
        lens,
        Hkv=Hkv,
        D=D,
        block_size=block_size,
        layout=layout,
        pad_page_bytes=pad,
        high_block_ids=high,
        seed=seed,
    )
    q_real, qsl = make_queries(q_lens, Hkv * G, D, seed=seed + 1, dtype=q_dtype)
    # Padding rows after the last request: must never be written.
    q = torch.cat([q_real, torch.randn(pad_tokens, Hkv * G, D).to(q_dtype)])
    T = q.shape[0]
    bq = bq or max(1, 64 // G)
    scale = D**-0.5
    dev = "cpu" if host else "cuda"
    out = torch.full((T, Hkv * G, D), SENTINEL, dtype=out_dtype)
    ws_o = ws_lse = None
    if num_splits > 1:
        ws_o = torch.full((num_splits, T, Hkv * G, D), float("nan"))
        ws_lse = torch.full((num_splits, T, Hkv * G), float("nan"))
    tensors = dict(q=q, out=out, ws_o=ws_o, ws_lse=ws_lse)
    if host:
        c, qsl_dev = c_cpu, qsl
    else:
        c = _cache_to(c_cpu, dev)
        tensors = {
            k: (v.to(dev) if v is not None else None) for k, v in tensors.items()
        }
        qsl_dev = qsl.to(dev)
    a = attn_args(
        c,
        tensors["q"],
        qsl_dev,
        tensors["out"],
        softmax_scale=scale,
        bq=bq,
        num_splits=num_splits,
        ws_o=tensors["ws_o"],
        ws_lse=tensors["ws_lse"],
        window=window,
        causal=causal,
    )
    rc = lib.attention(a, host=host)
    assert rc == OK, lib.last_error()
    if num_splits > 1:
        ca = combine_args(
            tensors["ws_o"],
            tensors["ws_lse"],
            tensors["out"],
            qsl_dev,
            len(lens),
            num_splits,
        )
        rc = lib.combine(ca, host=host)
        assert rc == OK, lib.last_error()
    if not host:
        torch.cuda.synchronize()
        tensors = {k: (v.cpu() if v is not None else None) for k, v in tensors.items()}
    c = c_cpu
    out = tensors["out"]
    kv = c.dense_all()
    b = bound(
        q,
        kv,
        qsl,
        scale,
        tiling=Tiling(TILE, TILE, 1, num_splits, bq),
        out_dtype=out_dtype,
        window=window,
        causal=causal,
    )
    err = np.abs(out.double().numpy() - b.ref)
    valid = b.valid
    gate = dict(
        violations=int((err > b.bound)[valid].sum()),
        frac=float((err > b.bound)[valid].mean()),
        tightness=tightness(err, b.bound, valid),
        rel_err=float(err[valid].max() / max(np.abs(b.ref[valid]).max(), 1e-30)),
        untouched=bool((out[~torch.from_numpy(valid)] == SENTINEL).all()),
    )
    return gate, out, tensors["ws_o"], tensors["ws_lse"], c, q, qsl, scale, bq


def _cache_to(c, dev):
    """A copy of an Int8Cache whose storage lives on ``dev``.

    The four views are rebuilt on the moved storage with the same element
    offsets and strides, so every byte keeps its address relative to the
    storage (the kernel sees the same layout on the device).
    """
    import copy

    d = copy.copy(c)
    d.storage = c.storage.to(dev)
    base = c.storage.untyped_storage().data_ptr()

    def view(t):
        off = (t.data_ptr() - base) // t.element_size()
        return torch.empty(0, dtype=t.dtype, device=dev).set_(
            d.storage.untyped_storage(), off, t.shape, t.stride()
        )

    for name in ("k_cache", "v_cache", "k_scale", "v_scale"):
        setattr(d, name, view(getattr(c, name)))
    d.block_table = c.block_table.to(dev)
    d.seq_lens = c.seq_lens.to(dev)
    return d


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
_SLOW = pytest.mark.slow  # host emulation of long rows: 20 s+ each

MATRIX = [
    pytest.param(
        dict(lens=[900, 0, 90], q_lens=[1, 0, 1], D=256, block_size=816),
        id="decode-D256-gqa6-bs816",
    ),
    pytest.param(
        dict(lens=[130, 75], q_lens=[4, 4], D=256, layout="HND", pad=72),
        id="verify-q4-D256-HND-pad",
    ),
    pytest.param(
        dict(lens=[40, 33], q_lens=[40, 9], G=4, block_size=1, bq=16),
        id="prefill-D128-gqa4-bs1",
        marks=_SLOW,
    ),
    pytest.param(
        dict(
            lens=[600],
            q_lens=[3],
            D=64,
            G=1,
            block_size=528,
            q_dtype=torch.float16,
            out_dtype=torch.float16,
        ),
        id="D64-gqa1-bs528-fp16",
    ),
    pytest.param(
        dict(lens=[200, 17], q_lens=[5, 17], G=8, out_dtype=torch.bfloat16),
        id="bf16-out-gqa8",
        marks=_SLOW,
    ),
    pytest.param(
        dict(lens=[150, 40], q_lens=[2, 1], Hkv=1, D=64, high=True),
        id="high-block-ids",
    ),
    pytest.param(
        dict(lens=[1100], q_lens=[2], D=128, block_size=1056, layout="HND"),
        id="bs1056-HND",
        marks=_SLOW,
    ),
    # Sliding window and non-causal. Ranges that start inside a
    # tile, windows wider than the sequence, the drafter's window 2048
    # non-causal at q = 8 (slow: long host rows).
    pytest.param(
        dict(lens=[150, 90], q_lens=[4, 8], D=128, G=4, window=40),
        id="window40-causal-D128-gqa4",
    ),
    pytest.param(
        dict(
            lens=[130, 60],
            q_lens=[8, 5],
            D=128,
            G=6,
            layout="HND",
            pad=72,
            window=33,
            causal=False,
        ),
        id="window33-noncausal-D128-HND-pad",
    ),
    pytest.param(
        dict(lens=[70], q_lens=[8], D=128, G=2, window=0, causal=False),
        id="noncausal-nowindow-D128",
    ),
    pytest.param(
        dict(lens=[40], q_lens=[3], D=128, G=4, block_size=1, window=500),
        id="window-wider-than-L-bs1",
    ),
    pytest.param(
        dict(
            lens=[2056, 300],
            q_lens=[8, 8],
            Hkv=1,
            D=128,
            G=4,
            block_size=864,
            window=2048,
            causal=False,
        ),
        id="drafter-w2048-noncausal-bs864",
        marks=_SLOW,
    ),
]


@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("kw", MATRIX)
def test_reference_within_bound(host_lib, kw, num_splits):
    """The executable spec is within the error bound of the fp64 reference.

    Why: the scalar kernel is the oracle the optimised kernel is compared with,
    so it must itself be right on every layout (NHD/HND, padded pages, block
    sizes 1..1056, block ids past 2^31/stride), head size, G, dtype and split
    count. Its arithmetic is plain fp32, so every rounding is no larger than
    the bound's corresponding term with Tiling(64, 64, 1, S, bq);
    tightness is therefore small (~1e-3) by design. Also checks that padding
    rows (after the last request) and empty requests are never written.
    """
    gate = run_case(host_lib, num_splits=num_splits, **kw)[0]
    assert gate["violations"] == 0, gate
    assert gate["untouched"], "a row outside every request was written"
    if kw.get("out_dtype", torch.float32) == torch.float32:
        assert gate["rel_err"] < 1e-5, gate  # fp32 arithmetic, no fp16 rounding


def test_split_partials_are_per_split_softmax(host_lib):
    """With S > 1 every partial is its split's softmax and lse is log2-sum-exp.

    Why: the combine (and the optimised kernel, compared split by split) relies on
    ws_lse = log2(sum over the split's visible tokens of 2^score) with
    score = scale * log2(e) * q . k_hat, and ws_o = that split's normalised
    output; empty splits must have lse = -inf. Checked in fp64 per row.
    """
    S = 4
    L, ql, G = 100, 3, 2  # 2 KV tiles < 4 splits: some splits are empty
    _, _, ws_o, ws_lse, c, q, qsl, scale, bq = run_case(
        host_lib, lens=[L], q_lens=[ql], G=G, D=64, num_splits=S, bq=2, pad_tokens=0
    )
    x = c.dense(0)
    kd, vd = x.k_deq(), x.v_deq()
    n_empty = 0
    for i in range(ql):
        tile_first = (i // bq) * bq
        end = L - ql + min(tile_first + bq, ql)  # row tile's KV range end
        T = -(-end // TILE)
        lim = L - ql + i
        for h in range(q.shape[1]):
            s = (kd[: lim + 1, h // G] @ q[i, h].double()) * scale / math.log(2.0)
            for sp in range(S):
                a, b = (sp * T) // S * TILE, ((sp + 1) * T) // S * TILE
                b = min(b, lim + 1)
                lse = float(ws_lse[sp, i, h])
                if a >= b:
                    assert lse == -math.inf
                    n_empty += 1
                    continue
                ss = s[a:b]
                m = ss.max()
                p = torch.exp2(ss - m)
                want_lse = float(m + torch.log2(p.sum()))
                want_o = (p @ vd[a:b, h // G]) / p.sum()
                assert abs(lse - want_lse) < 1e-5 * max(1.0, abs(want_lse))
                torch.testing.assert_close(
                    ws_o[sp, i, h].double(), want_o, rtol=1e-5, atol=1e-6
                )
    assert n_empty > 0  # T = 2 tiles, S = 4: splits 0 and 2 are empty


def test_split_equals_unsplit(host_lib):
    """Splitting and combining gives the unsplit result up to fp32 rounding.

    Why: split-KV is only an execution strategy; S must not change
    the answer beyond rounding. Both runs are within the bound (previous
    tests); here their difference is also small against max |o|.
    """
    kw = dict(lens=[300, 120], q_lens=[2, 3], D=128, seed=3)
    o1 = run_case(host_lib, num_splits=1, **kw)[1]
    for S in (2, 5):
        oS = run_case(host_lib, num_splits=S, **kw)[1]
        valid = o1 != SENTINEL
        diff = (oS[valid] - o1[valid]).abs().max() / o1[valid].abs().max()
        assert diff < 1e-5, (S, float(diff))


def _np_combine(o, lse):
    """The planner's rule (test_planner_combine._combine_log2), fp64."""
    live = np.isfinite(lse)
    w = np.exp2(lse[live] - lse[live].max())
    return (w[:, None] * o[live]).sum(axis=0) / w.sum()


def test_combine_matches_planner_rule(host_lib):
    """The combine kernel is the planner's log2 merge, and empty splits are inert.

    Why: the rule is fixed (max-lse subtract, skip lse = -inf BEFORE
    reading O, fixed order). Empty splits carry NaN garbage in O here (as an
    uninitialised workspace may); a combine that multiplied them would return
    NaN. Rows past qsl[num_reqs] stay untouched.
    """
    rng = np.random.default_rng(0)
    S, T, H, D = 7, 5, 3, 64
    o = torch.from_numpy(rng.standard_normal((S, T, H, D)).astype(np.float32))
    lse = torch.from_numpy((rng.standard_normal((S, T, H)) * 20).astype(np.float32))
    lse[2] = -math.inf
    o[2] = float("nan")
    lse[5, 1] = -math.inf
    o[5, 1] = float("nan")
    out = torch.full((T, H, D), SENTINEL)
    qsl = torch.tensor([0, 1, 4], dtype=torch.int32)  # rows 0..3 valid
    rc = host_lib.combine(combine_args(o, lse, out, qsl, 2, S), host=True)
    assert rc == OK, host_lib.last_error()
    for t in range(4):
        for h in range(H):
            want = _np_combine(
                o[:, t, h].double().numpy(), lse[:, t, h].double().numpy()
            )
            np.testing.assert_allclose(out[t, h].numpy(), want, rtol=2e-6, atol=2e-6)
    assert (out[4] == SENTINEL).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_store_rounding_matches_torch(host_lib, dtype):
    """The kernel's bit-level fp32 -> fp16/bf16 store rounds exactly like torch.

    Why: the conversions are hand-written (identical host and device code), so
    they are checked bit for bit: random values over the whole range,
    exact halfway cases (ties to even), fp16 subnormals and the 65504/65520
    overflow edge. A combine with one live split of lse 0 stores o unchanged.
    """
    rng = np.random.default_rng(1)
    vals = [rng.standard_normal(200000) * 10.0 ** rng.uniform(-9, 5, 200000)]
    bits = np.arange(0, 1 << 16, dtype=np.uint32)
    if dtype == torch.float16:
        h = (
            torch.from_numpy(bits.astype(np.uint16).view(np.int16))
            .view(torch.float16)
            .float()
        )
        mid = (h[:-1] + h[1:]) / 2  # exact halfway points (in fp32)
        vals += [
            h.numpy(),
            mid.numpy(),
            np.array(
                [
                    65504,
                    65519.996,
                    65520,
                    1e-8,
                    2.0**-25,
                    2.0**-25 * 1.0000001,
                    -(2.0**-24),
                ]
            ),
        ]
    else:
        h = torch.from_numpy((bits << 16).view(np.float32))
        h = h[torch.isfinite(h)]
        mid = h + torch.from_numpy(
            np.ldexp(np.ones(len(h)), np.frexp(h.numpy())[1] - 9)
        )
        vals += [h.numpy(), mid.numpy()]
    x = torch.from_numpy(np.concatenate(vals).astype(np.float32))
    # The combine computes fma(1, o, +0), which maps -0 to +0; zeros say
    # nothing about rounding, so they are left out.
    x = x[torch.isfinite(x) & (x != 0)]
    n = x.numel() // 64 * 64
    x = x[:n].reshape(-1, 1, 64)
    o = x.unsqueeze(0).contiguous()
    lse = torch.zeros(1, x.shape[0], 1)
    out = torch.empty(x.shape[0], 1, 64, dtype=dtype)
    qsl = torch.tensor([0, x.shape[0]], dtype=torch.int32)
    rc = host_lib.combine(combine_args(o, lse, out, qsl, 1, 1), host=True)
    assert rc == OK, host_lib.last_error()
    want = x.to(dtype)
    same = out.view(torch.int16) == want.view(torch.int16)
    assert bool(same.all()), x.flatten()[~same.flatten()][:10]


def test_abi_checks(host_lib):
    """A stale mirror (wrong ABI version or struct size) is rejected by name."""
    gate_args = _small_args()
    a = gate_args[0]
    a.abi_version = ABI_VERSION + 1
    assert host_lib.attention(a, host=True) == EINVAL
    assert "abi_version" in host_lib.last_error()
    a.abi_version = ABI_VERSION
    a.struct_bytes -= 8
    assert host_lib.attention(a, host=True) == EINVAL
    assert "struct_bytes" in host_lib.last_error()
    a.struct_bytes += 8
    a.head_size = 96
    assert host_lib.attention(a, host=True) == EINVAL
    assert "head_size" in host_lib.last_error()
    a.head_size = 64
    assert host_lib.attention(a, host=True) == OK
    assert host_lib.attention(a, host=False) == ENOCUDA


def _small_args():
    c = make_int8_cache([20], Hkv=1, D=64, block_size=16, seed=9)
    q, qsl = make_queries([2], 2, 64, seed=10)
    out = torch.empty(2, 2, 64)
    a = attn_args(c, q, qsl, out, softmax_scale=0.125, bq=4, num_splits=1)
    return a, (c, q, qsl, out)  # keep the tensors alive with the args


# --------------------------------------------------------------------------
# Mutations: deliberate bugs in the kernel source must fail the bound gate
# --------------------------------------------------------------------------
MUTATIONS = {
    "causal-off-by-one": (
        "reference_kernel.cu",
        "const int32_t hi = kv_hi < vis_hi + 1 ? kv_hi : vis_hi + 1;",
        "const int32_t hi = kv_hi < vis_hi ? kv_hi : vis_hi;",
    ),
    # The window's oldest key is one position too old; a
    # non-causal row loses the query block's later tokens; the windowed CTA's
    # first tile is rounded up past the range begin.
    "window-off-by-one": (
        "cache_geometry.cuh",
        "return window > 0 && p >= window ? p - window + 1 : 0;",
        "return window > 0 && p >= window ? p - window : 0;",
    ),
    "noncausal-masks-block-future": (
        "cache_geometry.cuh",
        "if (causal) return p;",
        "if (causal || seq_len > 0) return p;",
    ),
    "tile-lo-rounds-up": (
        "cache_geometry.cuh",
        "return range_begin / tile_tokens;",
        "return (range_begin + tile_tokens - 1) / tile_tokens;",
    ),
    "v-offset-minus-4": (
        "reference_kernel.cu",
        "const char* vrow = p.stream[kStreamV].address(page, slot, kv_head);",
        "const char* vrow = p.stream[kStreamV].address(page, slot, kv_head) - 4;",
    ),
    "int32-offsets": (
        "reference_kernel.cu",
        "const char* krow = p.stream[kStreamK].address(page, slot, kv_head);",
        (
            "const char* krow = p.stream[kStreamK].base + static_cast<int32_t>("
            "p.stream[kStreamK].offset(page, slot, kv_head));"
        ),
    ),
}

_GATE_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import test_ref_capi_build as t
from pathlib import Path
lib = t.RefLib(Path({lib!r}))
worst = 0.0
for kw in ({cases}):
    g = t.run_case(lib, **kw)[0]
    print(kw.get("name", ""), g, flush=True)
    worst = max(worst, g["frac"])
sys.exit(1 if worst > 0 else 0)
"""

MUTATION_CASES = (
    "dict(lens=[64, 40], q_lens=[4, 3], D=128, G=6, layout='HND', pad=72),"
    "dict(lens=[100, 37], q_lens=[2, 1], Hkv=1, D=64, G=4, high=True, num_splits=2),"
    "dict(lens=[150, 100], q_lens=[8, 4], D=128, G=4, window=40, causal=False,"
    " num_splits=2),"
    "dict(lens=[150], q_lens=[4], D=128, G=6, window=40),"
)


def _gate_subprocess(lib_path: Path) -> subprocess.CompletedProcess:
    script = _GATE_SCRIPT.format(
        tests=str(Path(__file__).parent), lib=str(lib_path), cases=MUTATION_CASES
    )
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )


def test_gate_passes_unmutated(host_lib):
    """Control for the mutation tests: the same gate passes the real source."""
    r = _gate_subprocess(REPO / "build" / "ref" / "libpth_ref_host.so")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mutation_fails_gate(tmp_path, name):
    """A deliberate bug in the kernel source fails the gate (or crashes).

    Mutations: the causal limit excludes the diagonal key; V is read 4 bytes
    early (at byte D of the cell, the vLLM layout's K scale); K offsets are
    truncated to int32 (the case with block ids past 2^31/stride reads the
    wrong page, or faults); the window keeps one key too many, a non-causal
    row masks the query block's later tokens, the windowed CTA's first tile
    is rounded up. The gate is the error bound with the cases of
    MUTATION_CASES. A crash (SIGSEGV from the int32 offset) also counts as
    detected: the test process sees a non-zero exit status.
    """
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    fname, old, new = MUTATIONS[name]
    root = tmp_path / "repo"
    shutil.copytree(REPO / "csrc", root / "csrc")
    (root / "tools").mkdir()
    shutil.copy(REPO / "tools" / "build_ref_lib.sh", root / "tools")
    src = root / "csrc" / "pth_attn" / fname
    text = src.read_text()
    assert text.count(old) == 1, f"mutation anchor not found: {old}"
    src.write_text(text.replace(old, new))
    b = build_libs(root, host_only=True)
    assert b.returncode == 0, b.stdout + b.stderr
    r = _gate_subprocess(root / "build" / "ref" / "libpth_ref_host.so")
    assert r.returncode != 0, f"mutant {name} passed the gate:\n{r.stdout}"


# --------------------------------------------------------------------------
# The CUDA library: builds without local memory; fails cleanly with no GPU
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cuda_lib() -> RefLib:
    """Build libpth_ref.so with nvcc (skip if nvcc is missing)."""
    if not Path("/usr/local/cuda/bin/nvcc").exists():
        pytest.skip("nvcc not available")
    r = build_libs(REPO, host_only=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "build_ref_lib: PASS" in r.stdout
    return RefLib(REPO / "build" / "ref" / "libpth_ref.so")


def test_cuda_library_builds_without_local_memory(cuda_lib):
    """nvcc builds sm_80/86/89 kernels with 0 spill and no stack frame.

    Why: the per-lane q and accumulator arrays must stay in registers
    (D / 32 <= 8 floats each); local memory would make the oracle slow and
    hide indexing mistakes behind memory. The build script fails on any
    spill or stack frame (ptxas -v), per arch.
    """
    for arch in (80, 86, 89):
        log = (REPO / "build" / "ref" / f"ptxas_sm{arch}.log").read_text()
        assert log.count("0 bytes stack frame, 0 bytes spill stores") == 6, log
    assert cuda_lib.has_cuda()


def test_cuda_entry_fails_cleanly_without_gpu(cuda_lib):
    """With no visible GPU the CUDA entry point returns ECUDA with a message."""
    if torch.cuda.is_available():
        pytest.skip("a GPU is visible; covered by the gpu test")
    a, keep = _small_args()
    assert cuda_lib.attention(a, host=False) == ECUDA
    assert "launch" in cuda_lib.last_error()
    assert keep


@pytest.mark.gpu
@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("kw", MATRIX)
def test_gpu_reference_within_bound(cuda_lib, kw, num_splits):
    """On a GPU: the CUDA build is within the bound and equals the host build.

    The host emulator runs the same source; the two may differ only by the
    libm (exp2f/log2f) and the device's fp32 rounding of the same operations,
    i.e. a few ulp. Needs a visible CUDA device.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    g_dev, o_dev = run_case(cuda_lib, num_splits=num_splits, host=False, **kw)[:2]
    g_host, o_host = run_case(cuda_lib, num_splits=num_splits, host=True, **kw)[:2]
    assert g_dev["violations"] == 0 and g_dev["untouched"], g_dev
    # The two builds compute the same fp32 operations; the libm (exp2f/log2f)
    # differs by a few ulp, so a 16-bit output may round to the neighbouring
    # value: allow 1e-5 of max |o| plus one ulp of the output type.
    valid = o_host != SENTINEL
    h, d = o_host[valid].double(), o_dev[valid].double()
    ulp = {torch.float32: 0.0, torch.float16: 2.0**-10, torch.bfloat16: 2.0**-7}
    tol = 1e-5 * h.abs().max() + ulp[kw.get("out_dtype", torch.float32)] * h.abs()
    assert bool(((d - h).abs() <= tol).all()), float(((d - h).abs() - tol).max())
