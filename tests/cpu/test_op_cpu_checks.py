# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the argument checks of the op (csrc/pth_attn/launch.cu).

The op validates every argument of ``pth_flash_attn_fwd`` before it launches a
kernel; on CPU tensors (a registration that exists to test the checks) a valid
call then raises "computes on CUDA tensors only". These tests pin down three
things:

1. The registered schema is the one documented in ``docs/api.md`` verbatim
   (the frozen interface).
2. A valid call shaped like production (D=256, Hq/Hkv=12/2, the real
   ``[K D | ks f32 | V D | vs f32]`` cell with V at byte offset D+4) passes
   every check and reaches the CPU-only error. This is the positive control: a
   check that wrongly rejects the real layout fails here.
3. Each invalid argument is rejected with its own specific message. Matching
   the message (not just "some error") is what makes these negative controls:
   if the wrong check fired, or none did and the CPU-only error appeared instead,
   the test fails.

The tests need the built library (``tools/build_in_image.sh``); without it they
skip, and ``tools/run_ci_local.sh`` builds it before running them.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "python" / "pth_flash_attn" / "_C.abi3.so"

# The schema of docs/api.md, verbatim (comments removed).
API_SCHEMA = (
    "pth_flash_attn::pth_flash_attn_fwd(Tensor! out, Tensor q, Tensor k_cache, "
    "Tensor v_cache, Tensor k_scale, Tensor v_scale, Tensor block_table, "
    "Tensor seq_lens, Tensor query_start_loc, int max_query_len, int wm, "
    "int num_splits, float softmax_scale, Tensor! ws_o, Tensor! ws_lse, "
    "int window=0, bool causal=True) -> ()"
)
NOT_IMPLEMENTED = "computes on CUDA tensors only"  # the end of a valid CPU call

# Production geometry (the gate tier), small block counts.
T, HQ, HKV, D, NB, BS = 3, 12, 2, 256, 4, 16
SCALE_BYTES = 4  # one float32 scale per (token, head) and per K/V half


@pytest.fixture(scope="module")
def op():
    """The registered op; skips the module when the library is not built."""
    if not LIB.is_file():
        pytest.skip(f"{LIB} not built; run tools/build_in_image.sh")
    torch.ops.load_library(str(LIB))
    return torch.ops.pth_flash_attn.pth_flash_attn_fwd.default


@dataclass
class Args:
    """A valid argument set; tests mutate one field at a time."""

    out: torch.Tensor
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int = 2
    wm: int = 1
    num_splits: int = 1
    softmax_scale: float = 1.0 / math.sqrt(D)
    ws_o: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    ws_lse: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    window: int = 0
    causal: bool = True

    def call(self, op):
        """Invokes the op with these arguments in schema order."""
        return op(
            self.out, self.q, self.k_cache, self.v_cache, self.k_scale,
            self.v_scale, self.block_table, self.seq_lens, self.query_start_loc,
            self.max_query_len, self.wm, self.num_splits, self.softmax_scale,
            self.ws_o, self.ws_lse, self.window, self.causal,
        )  # fmt: skip


def production_cache(d: int = D, cell_pad: int = 0):
    """K/V int8 views and f32 scale views over one packed int8 cache.

    Mirrors vLLM's int8_per_token_head cell ``[K d | ks | V d | vs]`` (2d+8
    bytes, plus ``cell_pad`` extra bytes to test stride rules). V starts at
    byte d+4, which is 4- but not 8-aligned for d=256: the reason the cache
    streams only require 4-byte alignment.
    """
    cell = 2 * d + 2 * SCALE_BYTES + cell_pad
    raw = torch.zeros(NB, BS, HKV, cell, dtype=torch.int8)
    k = raw[..., :d]
    v = raw[..., d + SCALE_BYTES : 2 * d + SCALE_BYTES]
    f32 = torch.tensor([], dtype=torch.float32).set_(raw.untyped_storage())
    unit = SCALE_BYTES  # bytes per float32 element
    strides = tuple(s // unit for s in raw.stride()[:3]) if cell % unit == 0 else None
    if strides is None:
        ks = torch.ones(NB, BS, HKV)
        vs = torch.ones(NB, BS, HKV)
    else:
        ks = torch.as_strided(f32, (NB, BS, HKV), strides, d // unit)
        vs = torch.as_strided(f32, (NB, BS, HKV), strides, (2 * d + unit) // unit)
    return k, v, ks, vs


def valid_args(**overrides) -> Args:
    """Builds a valid production-shaped argument set, then applies overrides."""
    k, v, ks, vs = production_cache()
    q = torch.zeros(T, HQ, D, dtype=torch.bfloat16)
    args = Args(
        out=torch.empty_like(q),
        q=q,
        k_cache=k,
        v_cache=v,
        k_scale=ks,
        v_scale=vs,
        block_table=torch.zeros(2, 4, dtype=torch.int32),
        seq_lens=torch.tensor([5, 7], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_schema_matches_api_doc(op):
    """The registered schema equals the one in docs/api.md exactly.

    The schema is a frozen interface: ops.py's fake and the vLLM backend are
    written against it. Comparing parsed schemas (not
    strings) makes the test independent of how torch prints alias sets.
    """
    assert str(op._schema) == str(torch._C.parse_schema(API_SCHEMA))
    mutable = [a.name for a in op._schema.arguments if a.is_write]
    assert mutable == ["out", "ws_o", "ws_lse"]


def test_valid_production_call_reaches_cpu_only_error(op):
    """Positive control: the real cache layout passes every check.

    D=256, G=6, V at byte 260 (4-aligned only), scale views with element
    strides 130/... exactly as vLLM's _ensure_scale_caches builds them. The
    only error allowed is the deliberate CPU-tensor one; anything else means a
    check rejects production.
    """
    with pytest.raises(RuntimeError, match=NOT_IMPLEMENTED):
        valid_args().call(op)


def test_valid_split_call_with_workspace_reaches_cpu_only_error(op):
    """Positive control for S > 1 with a workspace larger than needed.

    The workspace is reserved at maximum size at model load and sliced per
    call, so leading sizes above S and T are valid.
    """
    s = 2
    args = valid_args(
        num_splits=s,
        wm=1,
        ws_o=torch.zeros(s + 3, T + 5, HQ, D),
        ws_lse=torch.zeros(s + 3, T + 5, HQ),
    )
    with pytest.raises(RuntimeError, match=NOT_IMPLEMENTED):
        args.call(op)


def test_fp16_and_d64_d128_are_valid(op):
    """Every supported (dtype, D) pair is accepted."""
    for d in (64, 128, 256):
        for dtype in (torch.float16, torch.bfloat16):
            k, v, ks, vs = production_cache(d)
            q = torch.zeros(T, HQ, d, dtype=dtype)
            args = valid_args(
                q=q, out=torch.empty_like(q), k_cache=k, v_cache=v,
                k_scale=ks, v_scale=vs,
            )  # fmt: skip
            with pytest.raises(RuntimeError, match=NOT_IMPLEMENTED):
                args.call(op)


def test_window_and_noncausal_are_valid_at_d128(op):
    """Sliding window and non-causal calls pass every check at D = 128,
    including a window wider than any sequence."""
    k, v, ks, vs = production_cache(128)
    q = torch.zeros(T, HQ, 128, dtype=torch.bfloat16)
    for window, causal in ((2048, False), (64, True), (0, False), (2**40, True)):
        args = valid_args(
            q=q, out=torch.empty_like(q), k_cache=k, v_cache=v,
            k_scale=ks, v_scale=vs, window=window, causal=causal,
        )  # fmt: skip
        with pytest.raises(RuntimeError, match=NOT_IMPLEMENTED):
            args.call(op)


def test_window_and_noncausal_are_refused_at_d64_and_d256(op):
    """No windowed kernel at D = 64 or 256: the op says so."""
    for d in (64, 256):
        k, v, ks, vs = production_cache(d)
        q = torch.zeros(T, HQ, d, dtype=torch.bfloat16)
        for window, causal in ((2048, False), (64, True), (0, False)):
            args = valid_args(
                q=q, out=torch.empty_like(q), k_cache=k, v_cache=v,
                k_scale=ks, v_scale=vs, window=window, causal=causal,
            )  # fmt: skip
            with pytest.raises(RuntimeError) as info:
                args.call(op)
            text = str(info.value)
            assert NOT_IMPLEMENTED not in text
            assert f"need head size D = 128, got {d}" in text


def test_old_fifteen_argument_call_means_window_0_causal(op):
    """Callers written for ABI 1 (the 15-argument form of the op schema)
    still work: the schema defaults are window = 0
    and causal = True. At D = 64, where any window or non-causal call is
    refused, the 15-argument call passes every check, so the defaults are the
    causal no-window attention (negative control: causal=False is refused)."""
    defaults = {a.name: a.default_value for a in op._schema.arguments[-2:]}
    assert defaults == {"window": 0, "causal": True}
    k, v, ks, vs = production_cache(64)
    q = torch.zeros(T, HQ, 64, dtype=torch.bfloat16)
    a = valid_args(
        q=q, out=torch.empty_like(q), k_cache=k, v_cache=v, k_scale=ks, v_scale=vs
    )
    fifteen = (
        a.out, a.q, a.k_cache, a.v_cache, a.k_scale, a.v_scale, a.block_table,
        a.seq_lens, a.query_start_loc, a.max_query_len, a.wm, a.num_splits,
        a.softmax_scale, a.ws_o, a.ws_lse,
    )  # fmt: skip
    with pytest.raises(RuntimeError, match=NOT_IMPLEMENTED):
        op(*fifteen)
    with pytest.raises(RuntimeError, match="need head size D = 128"):
        op(*fifteen, 0, False)


def _offset_view(base_numel: int, offset: int, shape, dtype):
    """A contiguous-strided view starting ``offset`` elements into a buffer."""
    buf = torch.zeros(base_numel, dtype=dtype)
    strides = []
    acc = 1
    for s in reversed(shape):
        strides.insert(0, acc)
        acc *= s
    return torch.as_strided(buf, shape, strides, offset)


def _bad_cases():
    k, v, ks, vs = production_cache()
    q = torch.zeros(T, HQ, D, dtype=torch.bfloat16)
    raw = k.as_strided((NB, BS, HKV, 2 * D + 8), k.stride())  # whole cells
    padded_k, padded_v, _, _ = production_cache(cell_pad=2)  # 522-byte cells
    q40 = torch.zeros(T, 40, D, dtype=torch.bfloat16)  # G = 20
    q130 = torch.zeros(T, 130, D, dtype=torch.bfloat16)  # G = 65
    return {
        "q float32": (dict(q=q.float()), "q must be bfloat16 or float16"),
        "q rank 2": (dict(q=q[0]), "q must be 3-D"),
        "D = 96": (
            dict(q=q[..., :96], out=torch.empty(T, HQ, 96, dtype=torch.bfloat16)),
            "head size D = q.size(2) must be 64, 128 or 256",
        ),
        "q inner stride 2": (
            dict(q=torch.zeros(T, HQ, 2 * D, dtype=torch.bfloat16)[..., ::2]),
            "q must have stride 1 in its last dimension",
        ),
        "out dtype": (dict(out=torch.empty_like(q, dtype=torch.float16)),
                      "out (same dtype as q) must be bfloat16"),
        "out shape": (dict(out=torch.empty(T + 1, HQ, D, dtype=torch.bfloat16)),
                      "out.size(0) must equal T"),
        "k_cache uint8": (dict(k_cache=k.view(torch.uint8)), "k_cache must be int8"),
        "k_cache rank 3": (dict(k_cache=k[0]), "k_cache must be 4-D"),
        "k_cache wrong D": (dict(k_cache=raw[..., : D + 4]),
                            "k_cache.size(3) must equal D"),
        "v_cache block size": (dict(v_cache=v[:, : BS - 1]),
                               "v_cache.size(1) must equal BS"),
        "k_cache inner stride 2": (
            dict(k_cache=raw[..., : 2 * D : 2]),
            "k_cache must have stride 1 in its last dimension",
        ),
        "k_scale float16": (dict(k_scale=ks.half()), "k_scale must be float32"),
        "v_scale shape": (dict(v_scale=vs[:, :, :1]), "v_scale.size(2) must equal Hkv"),
        "Hq not multiple of Hkv": (
            dict(q=q[:, :11], out=torch.empty(T, 11, D, dtype=torch.bfloat16)),
            "must be a positive multiple of Hkv",
        ),
        "wm = 3": (dict(wm=3), "wm must be 1, 2 or 4"),
        "G = 20 > 16*wm": (
            dict(q=q40, out=torch.empty_like(q40), wm=1),
            "must be <= 16*wm = 16",
        ),
        "G = 65 > 64": (
            dict(q=q130, out=torch.empty_like(q130), wm=4),
            "exceeds the supported maximum 64",
        ),
        "block_table int64": (
            dict(block_table=torch.zeros(2, 4, dtype=torch.int64)),
            "block_table must be int32",
        ),
        "seq_lens rank 2": (
            dict(seq_lens=torch.zeros(2, 1, dtype=torch.int32)),
            "seq_lens must be 1-D",
        ),
        "query_start_loc length": (
            dict(query_start_loc=torch.tensor([0, 3], dtype=torch.int32)),
            "query_start_loc.size(0) must equal num_reqs + 1",
        ),
        "block_table rows < num_reqs": (
            dict(block_table=torch.zeros(1, 4, dtype=torch.int32)),
            "fewer than num_reqs",
        ),
        "max_query_len > T": (dict(max_query_len=T + 1), "max_query_len must be in"),
        "num_splits = 0": (dict(num_splits=0), "num_splits must be >= 1"),
        "softmax_scale nan": (dict(softmax_scale=float("nan")),
                              "softmax_scale must be finite and > 0"),
        "softmax_scale 0": (dict(softmax_scale=0.0),
                            "softmax_scale must be finite and > 0"),
        "window < 0": (dict(window=-1), "window must be >= 0"),
        "window at D = 256": (
            dict(window=2048),
            "sliding-window and non-causal attention need head size D = 128",
        ),
        "non-causal at D = 256": (
            dict(causal=False),
            "sliding-window and non-causal attention need head size D = 128",
        ),
        "S = 2 without workspace": (dict(num_splits=2), "ws_o must be 4-D"),
        "workspace too few splits": (
            dict(num_splits=3, ws_o=torch.zeros(2, T, HQ, D),
                 ws_lse=torch.zeros(2, T, HQ)),
            "ws_o must have size(0) >= num_splits",
        ),
        "workspace too few tokens": (
            dict(num_splits=2, ws_o=torch.zeros(2, T, HQ, D),
                 ws_lse=torch.zeros(2, T - 1, HQ)),
            "ws_lse must have size(0) >= num_splits",
        ),
        "workspace wrong Hq": (
            dict(num_splits=2, ws_o=torch.zeros(2, T, HQ, D),
                 ws_lse=torch.zeros(2, T, HQ + 1)),
            "ws_lse.size(2) must equal Hq",
        ),
        "workspace float16": (
            dict(num_splits=2, ws_o=torch.zeros(2, T, HQ, D, dtype=torch.float16),
                 ws_lse=torch.zeros(2, T, HQ)),
            "ws_o must be float32",
        ),
        # Address checks (not mirrored by the fake: they need data pointers).
        "k_cache at odd byte": (
            dict(k_cache=raw[..., 1 : D + 1]),
            "k_cache data pointer must be 4-byte aligned",
        ),
        "k_cache 522-byte cell stride": (
            dict(k_cache=padded_k, v_cache=padded_v),
            "k_cache byte stride of dim 2 (522) must be a multiple of 4",
        ),
        "q at 8-byte offset": (
            dict(q=_offset_view(T * HQ * D + 4, 4, (T, HQ, D), torch.bfloat16)),
            "q data pointer must be 16-byte aligned",
        ),
    }  # fmt: skip


@pytest.mark.parametrize("case", list(_bad_cases()))
def test_invalid_argument_rejected_with_specific_message(op, case):
    """Each invalid argument fails with its own message, never the CPU-only error.

    Why it matters: a check that is missing lets the (future) kernel read out
    of bounds or misinterpret bytes silently; a check with a vague message
    costs the user hours. The message match is the negative control: if the
    check were absent, the call would reach "computes on CUDA tensors only"
    and this test would fail.
    """
    overrides, message = _bad_cases()[case]
    with pytest.raises(RuntimeError) as info:
        valid_args(**overrides).call(op)
    text = str(info.value)
    assert NOT_IMPLEMENTED not in text, f"{case}: check missing, reached the end"
    assert message in text, f"{case}: wrong check fired: {text}"
    assert "pth_flash_attn_fwd: " in text


EXTENT_TEST_CPP = r"""
#include "ops.h"
#include <cstdint>
#include <cstdio>
using pth_attn::detail::view_extent_bytes;
int main() {
  int failures = 0;
  auto expect = [&](bool got, bool want, const char* what) {
    if (got != want) { std::printf("FAIL %s\n", what); ++failures; }
  };
  int64_t e = 0;
  // A real cache view: [4096 blocks, 816 tokens, 2 heads, 256 B], cell 520 B.
  const int64_t sizes[4] = {4096, 816, 2, 256};
  const int64_t strides[4] = {816 * 2 * 520, 2 * 520, 520, 1};
  expect(view_extent_bytes(sizes, strides, 4, 1, &e), true, "real cache");
  expect(e == 4095LL * 816 * 2 * 520 + 815LL * 2 * 520 + 520 + 255 + 1, true,
         "real cache extent value");
  // (size-1)*stride overflows int64: 2^32 * 2^32 = 2^64.
  const int64_t big_s[2] = {(1LL << 32) + 1, 1};
  const int64_t big_t[2] = {1LL << 32, 1};
  expect(view_extent_bytes(big_s, big_t, 2, 1, &e), false, "mul overflow");
  // Product fits, times element size 4 does not: 2^62 * 4.
  const int64_t es_s[1] = {(1LL << 31) + 1};
  const int64_t es_t[1] = {1LL << 31};
  expect(view_extent_bytes(es_s, es_t, 1, 4, &e), false, "elsize overflow");
  // Each term fits, the sum does not: 2 * (2^62) + 1.
  const int64_t sum_s[2] = {2, 2};
  const int64_t sum_t[2] = {1LL << 62, 1LL << 62};
  expect(view_extent_bytes(sum_s, sum_t, 2, 1, &e), false, "add overflow");
  // Negative stride: not produced by torch views, rejected by contract.
  const int64_t neg_s[1] = {4};
  const int64_t neg_t[1] = {-1};
  expect(view_extent_bytes(neg_s, neg_t, 1, 1, &e), false, "negative stride");
  // Empty view touches nothing, whatever its strides.
  const int64_t emp_s[2] = {0, 1LL << 40};
  const int64_t emp_t[2] = {1LL << 40, 1LL << 40};
  expect(view_extent_bytes(emp_s, emp_t, 2, 8, &e) && e == 0, true, "empty");
  // Negative control: the same inputs DO wrap with unchecked arithmetic, so
  // the overflow cases above are real overflows, not values that fit.
  uint64_t naive = (uint64_t)(big_s[0] - 1) * (uint64_t)big_t[0];
  expect(naive == 0, true, "naive product wraps to 0 (control)");
  std::printf("%d failures\n", failures);
  return failures != 0;
}
"""


def test_int64_extent_overflow_is_detected(tmp_path):
    """view_extent_bytes refuses views whose byte extent overflows int64.

    The kernel computes addresses as int64 sums of index*stride. If the extent
    fits, every partial sum fits. No tensor a CPU test can allocate is large
    enough to overflow, so this compiles the header's helper with g++ and drives
    it with crafted sizes/strides. The
    last assertion is the negative control: the same inputs wrap to 0 with
    unchecked arithmetic, proving the cases really overflow.
    """
    sys.path.insert(0, str(REPO / "tools"))
    import build_flags

    src = tmp_path / "extent_test.cpp"
    exe = tmp_path / "extent_test"
    src.write_text(EXTENT_TEST_CPP)
    includes = [
        f"-I{d}" for d in [str(build_flags.CSRC_DIR), *build_flags.torch_include_dirs()]
    ]
    build = subprocess.run(
        ["g++", "-std=c++17", "-O1", *build_flags.define_flags(), *includes,
         str(src), "-o", str(exe)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout
    assert "0 failures" in run.stdout
