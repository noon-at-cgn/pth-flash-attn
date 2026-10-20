# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Kernel adapters for the correctness grid: the reference library and the loader.

``ref_kernel(lib, device)`` turns ``pth_testing.ref_capi.RefLib`` into a
:data:`grid_cases.Kernel`: it copies a case's host tensors to ``device`` with
their exact strides and offsets, launches ``pth_ref_attention`` (plus
``pth_ref_combine`` when S > 1), and copies every buffer back, so the gate in
``grid_cases.check`` sees the kernel's output and every canary byte.
``device="cpu"`` with the host library (``libpth_ref_host.so``) runs the same
kernel source under its warp emulator and needs no GPU.

Timings here are dev timings (CUDA events around one launch), recorded for
orientation only; they are never gate evidence.
"""

from __future__ import annotations

import ctypes
import dataclasses
import time
from collections.abc import Sequence

import torch
from grid_cases import Buffers, CaseInputs, Kernel, KernelRun, rebase_view
from pth_testing import ref_capi
from pth_testing.reference import Int8Cache

CACHE_VIEWS = (
    "kv_packed",
    "key_cache",
    "value_cache",
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
)


def move_view(view: torch.Tensor, storage: torch.Tensor) -> torch.Tensor:
    """``view`` rebuilt on ``storage`` (a copy of its backing storage), with
    the same dtype, element offset, shape and strides (the kernel
    must see the strides of the real views, padding included)."""
    return torch.empty(0, dtype=view.dtype, device=storage.device).set_(
        storage.untyped_storage(), view.storage_offset(), view.shape, view.stride()
    )


def cache_to(cache: Int8Cache, device: torch.device | str) -> Int8Cache:
    """The cache with its whole backing storage and every view on ``device``.

    Copying the storage (not the views) keeps poison bytes, page padding and
    high block ids exactly where the host cache has them.
    """
    if torch.device(device).type == "cpu":
        return cache
    base = cache.storage.untyped_storage().data_ptr()
    for name in CACHE_VIEWS:
        if getattr(cache, name).untyped_storage().data_ptr() != base:
            raise ValueError(f"{name} is not a view of the cache storage")
    storage = cache.storage.to(device)
    moved = {n: move_view(getattr(cache, n), storage) for n in CACHE_VIEWS}
    return dataclasses.replace(
        cache,
        storage=storage,
        block_table=cache.block_table.to(device),
        seq_lens=cache.seq_lens.to(device),
        **moved,
    )


def _pair(
    view: torch.Tensor | None, buf: torch.Tensor | None, device: torch.device | str
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if view is None or buf is None:
        return None, None
    nb = buf.to(device)
    return rebase_view(view, buf, nb), nb


def _check_rc(lib: ref_capi.RefLib, rc: int, what: str) -> None:
    if rc != ref_capi.OK:
        raise RuntimeError(f"{what} returned {rc}: {lib.last_error()}")


@dataclasses.dataclass
class DeviceLaunch:
    """Everything one reference launch needs, resident on one device.

    Kept as an object so the graph-replay test can capture ``launch()`` once
    and replay it after changing ``seq_lens``/``qsl`` in place.
    """

    lib: ref_capi.RefLib
    host: bool
    cache: Int8Cache
    q: torch.Tensor
    qsl: torch.Tensor
    bufs: Buffers  # device buffers and views
    softmax_scale: float
    bq: int
    num_splits: int
    window: int = 0
    causal: bool = True

    def launch(self, stream: int = 0) -> None:
        """Enqueue the reference kernel (and the combine if S > 1)."""
        b = self.bufs
        S = self.num_splits
        args = ref_capi.attn_args(
            self.cache,
            self.q,
            self.qsl,
            b.out,
            softmax_scale=self.softmax_scale,
            bq=self.bq,
            num_splits=S,
            ws_o=b.ws_o,
            ws_lse=b.ws_lse,
            window=self.window,
            causal=self.causal,
        )
        _check_rc(
            self.lib,
            self.lib.attention(args, host=self.host, stream=stream),
            "pth_ref_attention",
        )
        if S > 1:
            assert b.ws_o is not None and b.ws_lse is not None
            cargs = ref_capi.combine_args(
                b.ws_o, b.ws_lse, b.out, self.qsl, self.cache.num_reqs, S
            )
            _check_rc(
                self.lib,
                self.lib.combine(cargs, host=self.host, stream=stream),
                "pth_ref_combine",
            )


def prepare(
    lib: ref_capi.RefLib, inp: CaseInputs, bufs: Buffers, device: torch.device | str
) -> DeviceLaunch:
    """Copy a case's inputs and canary buffers to ``device``."""
    c = inp.case
    q, _ = _pair(inp.q, inp.q_buf, device)
    out, out_buf = _pair(bufs.out, bufs.out_buf, device)
    ws_o, ws_o_buf = _pair(bufs.ws_o, bufs.ws_o_buf, device)
    ws_lse, ws_lse_buf = _pair(bufs.ws_lse, bufs.ws_lse_buf, device)
    assert q is not None and out is not None and out_buf is not None
    dbufs = Buffers(out_buf, out, ws_o_buf, ws_o, ws_lse_buf, ws_lse)
    return DeviceLaunch(
        lib,
        torch.device(device).type == "cpu",
        cache_to(inp.cache, device),
        q,
        inp.qsl.to(device),
        dbufs,
        inp.softmax_scale,
        c.bq,
        c.num_splits,
        c.window,
        c.causal,
    )


def copy_back(dev: Buffers, host: Buffers) -> None:
    """Host buffers := device buffers (all bytes, canaries included)."""
    pairs: Sequence[tuple[torch.Tensor | None, torch.Tensor | None]] = (
        (dev.out_buf, host.out_buf),
        (dev.ws_o_buf, host.ws_o_buf),
        (dev.ws_lse_buf, host.ws_lse_buf),
    )
    for d, h in pairs:
        if d is not None and h is not None and d.data_ptr() != h.data_ptr():
            h.copy_(d.cpu())


def ref_kernel(lib: ref_capi.RefLib, device: torch.device | str) -> Kernel:
    """The reference library as a grid kernel on ``device`` ("cuda" or "cpu")."""

    def run(inp: CaseInputs, bufs: Buffers) -> KernelRun:
        dl = prepare(lib, inp, bufs, device)
        if dl.host:
            t0 = time.perf_counter()
            dl.launch()
            secs = time.perf_counter() - t0
        else:
            stream = torch.cuda.current_stream()
            start, stop = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record(stream)
            dl.launch(stream.cuda_stream)
            stop.record(stream)
            stop.synchronize()
            secs = start.elapsed_time(stop) / 1e3
        copy_back(dl.bufs, bufs)
        return KernelRun(bufs, secs, checks_workspace=True)

    return run


# ---------------------------------------------------------------- loader
VARIANT_SEPARATE = 0  # loader_capi.h PTH_LOADER_VARIANT_SEPARATE
VARIANT_CELL = 1  # loader_capi.h PTH_LOADER_VARIANT_CELL
LOADER_ABI_VERSION = 2  # loader_capi.h PTH_LOADER_ABI_VERSION
LOADER_ERR_NOT_CELL = -3  # PTH_LOADER_ERR_NOT_CELL
LOADER_ERR_WIDTH = -2  # PTH_LOADER_ERR_WIDTH


def _i64(vals: Sequence[int]) -> ctypes.Array[ctypes.c_int64]:
    return (ctypes.c_int64 * len(vals))(*[int(x) for x in vals])


class LoaderLib:
    """ctypes wrapper of ``pth_loader_run`` / ``pth_loader_stage_bytes``."""

    def __init__(self, path: str) -> None:
        self.lib = ctypes.CDLL(path)
        if self.lib.pth_loader_abi_version() != LOADER_ABI_VERSION:
            raise RuntimeError(f"{path}: loader ABI differs from this wrapper")
        self.lib.pth_loader_run.restype = ctypes.c_int32

    def stage_bytes(self, variant: int, D: int) -> int:
        return int(self.lib.pth_loader_stage_bytes(variant, D))

    def run(
        self,
        variant: int,
        D: int,
        views: Sequence[torch.Tensor],
        block_table: torch.Tensor,
        req: int,
        kv_head: int,
        tile_begin: int,
        tile_end: int,
        tok_end: int,
        out: torch.Tensor,
        override: Sequence[int] | None = None,
        stream: int = 0,
        tok_begin: int = 0,
    ) -> int:
        """Run the GPU loader; ``views`` = (k_cache, v_cache, k_scale, v_scale)
        on the device, ``out`` a device uint8 buffer of the stage images.
        ``tok_begin > 0`` also zero-fills below it (the windowed kernels'
        loader; separate streams only)."""
        k = views[0]
        ov = (ctypes.c_int32 * 4)(*override) if override is not None else None
        return int(
            self.lib.pth_loader_run(
                ctypes.c_int32(variant),
                ctypes.c_int32(D),
                *(ctypes.c_void_p(t.data_ptr()) for t in views),
                _i64(k.shape[:3]),
                *(_i64(t.stride()[:3]) for t in views),
                ov,
                ctypes.c_void_p(block_table.data_ptr()),
                ctypes.c_int64(block_table.stride(0)),
                ctypes.c_int32(k.shape[1]),
                ctypes.c_int32(req),
                ctypes.c_int32(kv_head),
                ctypes.c_int32(tile_begin),
                ctypes.c_int32(tile_end),
                ctypes.c_int32(tok_begin),
                ctypes.c_int32(tok_end),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int64(out.numel()),
                ctypes.c_void_p(stream or None),
            )
        )
