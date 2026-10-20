# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""ctypes mirror of ``csrc/pth_attn/ref_capi.h`` (test builds only).

The scalar reference kernel and the split combine are driven from Python
through two libraries that ``tools/build_ref_lib.sh`` builds into
``build/ref/``: ``libpth_ref.so`` (nvcc; CUDA entry points on device pointers,
plus the host emulator) and ``libpth_ref_host.so`` (g++; host emulator only).
This module is the single Python view of that C ABI, shared by the CPU tests
(``tests/cpu/test_ref_capi_build.py``) and the GPU grid (``tests/gpu``).

It is not part of the production op: it never loads a library at import time,
reads no environment variable and does not touch ``torch.cuda``. Tensors may
live on the CPU (host entry points) or on a CUDA device (CUDA entry points);
only their data pointers and byte strides are read here.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import torch

ABI_VERSION = 2  # PTH_ATTN_ABI_VERSION
F32, F16, BF16 = 0, 1, 2  # PthAttnDtype
OK, EINVAL, ECUDA, ENOCUDA = 0, -1, -2, -3  # PthAttnStatus
DTYPE_CODE = {torch.float32: F32, torch.float16: F16, torch.bfloat16: BF16}
# Logical dimensions of a cache stream view with free strides: [NB, BS, Hkv].
STREAM_DIMS = 3

LIB_DIR = Path(__file__).resolve().parents[2] / "build" / "ref"
CUDA_LIB = LIB_DIR / "libpth_ref.so"
HOST_LIB = LIB_DIR / "libpth_ref_host.so"


class PthAttnStream(ctypes.Structure):
    """One cache stream: data pointer and byte strides of [NB, BS, Hkv]."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("stride_block", ctypes.c_int64),
        ("stride_slot", ctypes.c_int64),
        ("stride_head", ctypes.c_int64),
    ]


class PthAttnArgs(ctypes.Structure):
    """Mirror of ``PthAttnArgs``; the field order is the ABI."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("struct_bytes", ctypes.c_int32),
        ("q", ctypes.c_void_p),
        ("q_dtype", ctypes.c_int32),
        ("q_stride_token", ctypes.c_int64),
        ("q_stride_head", ctypes.c_int64),
        ("out", ctypes.c_void_p),
        ("out_dtype", ctypes.c_int32),
        ("out_stride_token", ctypes.c_int64),
        ("out_stride_head", ctypes.c_int64),
        ("k_cache", PthAttnStream),
        ("v_cache", PthAttnStream),
        ("k_scale", PthAttnStream),
        ("v_scale", PthAttnStream),
        ("num_blocks", ctypes.c_int32),
        ("block_size", ctypes.c_int32),
        ("num_kv_heads", ctypes.c_int32),
        ("head_size", ctypes.c_int32),
        ("block_table", ctypes.c_void_p),
        ("block_table_row_stride", ctypes.c_int64),
        ("seq_lens", ctypes.c_void_p),
        ("query_start_loc", ctypes.c_void_p),
        ("num_reqs", ctypes.c_int32),
        ("num_tokens", ctypes.c_int32),
        ("num_q_heads", ctypes.c_int32),
        ("bq", ctypes.c_int32),
        ("num_q_tiles", ctypes.c_int32),
        ("num_splits", ctypes.c_int32),
        ("softmax_scale", ctypes.c_float),
        ("ws_o", ctypes.c_void_p),
        ("ws_o_stride_split", ctypes.c_int64),
        ("ws_o_stride_token", ctypes.c_int64),
        ("ws_o_stride_head", ctypes.c_int64),
        ("ws_lse", ctypes.c_void_p),
        ("ws_lse_stride_split", ctypes.c_int64),
        ("ws_lse_stride_token", ctypes.c_int64),
        ("ws_lse_stride_head", ctypes.c_int64),
        ("window", ctypes.c_int32),
        ("causal", ctypes.c_int32),
    ]


class PthAttnCombineArgs(ctypes.Structure):
    """Mirror of ``PthAttnCombineArgs``; the field order is the ABI."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("struct_bytes", ctypes.c_int32),
        ("ws_o", ctypes.c_void_p),
        ("ws_o_stride_split", ctypes.c_int64),
        ("ws_o_stride_token", ctypes.c_int64),
        ("ws_o_stride_head", ctypes.c_int64),
        ("ws_lse", ctypes.c_void_p),
        ("ws_lse_stride_split", ctypes.c_int64),
        ("ws_lse_stride_token", ctypes.c_int64),
        ("ws_lse_stride_head", ctypes.c_int64),
        ("out", ctypes.c_void_p),
        ("out_dtype", ctypes.c_int32),
        ("out_stride_token", ctypes.c_int64),
        ("out_stride_head", ctypes.c_int64),
        ("query_start_loc", ctypes.c_void_p),
        ("num_reqs", ctypes.c_int32),
        ("num_tokens", ctypes.c_int32),
        ("num_q_heads", ctypes.c_int32),
        ("head_size", ctypes.c_int32),
        ("num_splits", ctypes.c_int32),
    ]


def byte_strides(t: torch.Tensor, n: int) -> list[int]:
    """The first ``n`` strides of a view in bytes (strides from views only)."""
    return [s * t.element_size() for s in t.stride()[:n]]


def stream_of(view: torch.Tensor) -> PthAttnStream:
    """PthAttnStream of a cache view [NB, BS, Hkv, row] (row contiguous)."""
    sb, ss, sh = byte_strides(view, STREAM_DIMS)
    return PthAttnStream(view.data_ptr(), sb, ss, sh)


class RefLib:
    """ctypes wrapper of one built library (CUDA or host-only).

    Loading checks the library's ABI version and both struct sizes against
    this mirror, so a stale mirror fails at load time, not with corrupt
    arguments.
    """

    def __init__(self, path: Path) -> None:
        """Load ``path`` and check the ABI."""
        self.lib = ctypes.CDLL(str(path))
        lib = self.lib
        lib.pth_ref_abi_version.restype = ctypes.c_int32
        lib.pth_ref_attn_args_bytes.restype = ctypes.c_int32
        lib.pth_ref_combine_args_bytes.restype = ctypes.c_int32
        lib.pth_ref_has_cuda.restype = ctypes.c_int32
        lib.pth_ref_last_error.restype = ctypes.c_char_p
        attn_p = ctypes.POINTER(PthAttnArgs)
        comb_p = ctypes.POINTER(PthAttnCombineArgs)
        lib.pth_ref_attention.restype = ctypes.c_int32
        lib.pth_ref_attention.argtypes = [attn_p, ctypes.c_void_p]
        lib.pth_ref_combine.restype = ctypes.c_int32
        lib.pth_ref_combine.argtypes = [comb_p, ctypes.c_void_p]
        lib.pth_ref_attention_host.restype = ctypes.c_int32
        lib.pth_ref_attention_host.argtypes = [attn_p]
        lib.pth_ref_combine_host.restype = ctypes.c_int32
        lib.pth_ref_combine_host.argtypes = [comb_p]
        if lib.pth_ref_abi_version() != ABI_VERSION:
            raise RuntimeError(f"{path}: ABI {lib.pth_ref_abi_version()} != mirror")
        if lib.pth_ref_attn_args_bytes() != ctypes.sizeof(PthAttnArgs):
            raise RuntimeError(f"{path}: PthAttnArgs size differs from mirror")
        if lib.pth_ref_combine_args_bytes() != ctypes.sizeof(PthAttnCombineArgs):
            raise RuntimeError(f"{path}: PthAttnCombineArgs size differs from mirror")

    def has_cuda(self) -> bool:
        """True for the nvcc build (CUDA entry points available)."""
        return bool(self.lib.pth_ref_has_cuda())

    def last_error(self) -> str:
        """Message of the last failure on this thread ("" if none)."""
        return self.lib.pth_ref_last_error().decode()

    def attention(self, args: PthAttnArgs, *, host: bool, stream: int = 0) -> int:
        """Call pth_ref_attention_host (host) or pth_ref_attention (CUDA)."""
        if host:
            return int(self.lib.pth_ref_attention_host(ctypes.byref(args)))
        return int(self.lib.pth_ref_attention(ctypes.byref(args), stream or None))

    def combine(self, args: PthAttnCombineArgs, *, host: bool, stream: int = 0) -> int:
        """Call pth_ref_combine_host (host) or pth_ref_combine (CUDA)."""
        if host:
            return int(self.lib.pth_ref_combine_host(ctypes.byref(args)))
        return int(self.lib.pth_ref_combine(ctypes.byref(args), stream or None))


def attn_args(
    cache: Any,
    q: torch.Tensor,
    qsl: torch.Tensor,
    out: torch.Tensor | None,
    *,
    softmax_scale: float,
    bq: int,
    num_splits: int,
    ws_o: torch.Tensor | None = None,
    ws_lse: torch.Tensor | None = None,
    window: int = 0,
    causal: bool = True,
) -> PthAttnArgs:
    """Fill PthAttnArgs.

    Args:
        cache: A ``reference.Int8Cache`` (or any object with ``k_cache``,
            ``v_cache``, ``k_scale``, ``v_scale``, ``block_table``,
            ``seq_lens``, ``block_size``, ``Hkv``, ``D``), on the same device
            as ``q``.
        q: ``[T, Hq, D]`` fp16/bf16 queries.
        qsl: ``[R + 1]`` int32 query_start_loc.
        out: ``[T, Hq, D]`` output (fp32/fp16/bf16) or None when S > 1.
        softmax_scale: Scale of q . k.
        bq: Query tokens per row tile (``16 * wm // G`` mirrors the kernel).
        num_splits: S.
        ws_o: ``[S, T, Hq, D]`` fp32 partials (S > 1).
        ws_lse: ``[S, T, Hq]`` fp32 log2-LSE partials (S > 1).
        window: Sliding window in tokens, 0 = none.
        causal: Bottom-right causal (True) or non-causal (False).
    """
    a = PthAttnArgs()
    a.abi_version = ABI_VERSION
    a.struct_bytes = ctypes.sizeof(PthAttnArgs)
    a.q = q.data_ptr()
    a.q_dtype = DTYPE_CODE[q.dtype]
    a.q_stride_token, a.q_stride_head = byte_strides(q, 2)
    a.out_dtype = F32
    if out is not None:
        a.out = out.data_ptr()
        a.out_dtype = DTYPE_CODE[out.dtype]
        a.out_stride_token, a.out_stride_head = byte_strides(out, 2)
    a.k_cache = stream_of(cache.k_cache)
    a.v_cache = stream_of(cache.v_cache)
    a.k_scale = stream_of(cache.k_scale)
    a.v_scale = stream_of(cache.v_scale)
    a.num_blocks = cache.k_cache.shape[0]
    a.block_size = cache.block_size
    a.num_kv_heads = cache.Hkv
    a.head_size = cache.D
    a.block_table = cache.block_table.data_ptr()
    a.block_table_row_stride = cache.block_table.stride(0)
    a.seq_lens = cache.seq_lens.data_ptr()
    a.query_start_loc = qsl.data_ptr()
    a.num_reqs = cache.seq_lens.shape[0]
    a.num_tokens = q.shape[0]
    a.num_q_heads = q.shape[1]
    a.bq = bq
    a.num_q_tiles = 0
    a.num_splits = num_splits
    a.softmax_scale = softmax_scale
    a.window = window
    a.causal = int(causal)
    if ws_o is not None and ws_lse is not None:
        a.ws_o = ws_o.data_ptr()
        a.ws_o_stride_split, a.ws_o_stride_token, a.ws_o_stride_head = byte_strides(
            ws_o, 3
        )
        a.ws_lse = ws_lse.data_ptr()
        a.ws_lse_stride_split, a.ws_lse_stride_token, a.ws_lse_stride_head = (
            byte_strides(ws_lse, 3)
        )
    return a


def combine_args(
    ws_o: torch.Tensor,
    ws_lse: torch.Tensor,
    out: torch.Tensor,
    qsl: torch.Tensor,
    num_reqs: int,
    num_splits: int,
) -> PthAttnCombineArgs:
    """Fill PthAttnCombineArgs (shapes taken from ``out``: [T, Hq, D])."""
    a = PthAttnCombineArgs()
    a.abi_version = ABI_VERSION
    a.struct_bytes = ctypes.sizeof(PthAttnCombineArgs)
    a.ws_o = ws_o.data_ptr()
    a.ws_o_stride_split, a.ws_o_stride_token, a.ws_o_stride_head = byte_strides(ws_o, 3)
    a.ws_lse = ws_lse.data_ptr()
    a.ws_lse_stride_split, a.ws_lse_stride_token, a.ws_lse_stride_head = byte_strides(
        ws_lse, 3
    )
    a.out = out.data_ptr()
    a.out_dtype = DTYPE_CODE[out.dtype]
    a.out_stride_token, a.out_stride_head = byte_strides(out, 2)
    a.query_start_loc = qsl.data_ptr()
    a.num_reqs = num_reqs
    a.num_tokens = out.shape[0]
    a.num_q_heads = out.shape[1]
    a.head_size = out.shape[2]
    a.num_splits = num_splits
    return a
