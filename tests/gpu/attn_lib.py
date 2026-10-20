# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""ctypes wrapper of ``build/attn/libpth_attn.so`` (``csrc/pth_attn/attn_capi.h``).

The library takes the reference library's argument structs unchanged
(``pth_testing.ref_capi.PthAttnArgs`` / ``PthAttnCombineArgs``), so
:class:`AttnLib` has the same ``attention`` / ``combine`` / ``last_error`` /
``has_cuda`` methods as ``ref_capi.RefLib`` and plugs into
``grid_cuda.ref_kernel`` unchanged. On top it reports the bound's tiling of the
kernel (:meth:`AttnLib.tiling`) and runs the one-warp fragment unit test
(:meth:`AttnLib.test_warp_tile`).

:func:`attn_case` turns a grid ``GridCase`` into an :class:`AttnCase` whose
``tiling`` is the kernel's, so ``grid_cases.run_case`` computes (and caches,
under a different key) the bound for this kernel instead of the reference's.
"""

from __future__ import annotations

import ctypes
import dataclasses
from pathlib import Path

import grid_cases as grid
import torch
from pth_testing import ref_capi
from pth_testing.bound import Tiling

REPO = Path(__file__).resolve().parents[2]
ATTN_LIB = REPO / "build" / "attn" / "libpth_attn.so"


class PthAttnOptions(ctypes.Structure):
    """Mirror of ``PthAttnOptions``."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("struct_bytes", ctypes.c_int32),
        ("loader", ctypes.c_int32),
    ]


class PthAttnTileTestArgs(ctypes.Structure):
    """Mirror of ``PthAttnTileTestArgs``."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("struct_bytes", ctypes.c_int32),
        ("head_size", ctypes.c_int32),
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("p", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p),
        ("o_out", ctypes.c_void_p),
    ]


class PthAttnDecTileTestArgs(ctypes.Structure):
    """Mirror of ``PthAttnDecTileTestArgs``."""

    _fields_ = [
        ("abi_version", ctypes.c_int32),
        ("struct_bytes", ctypes.c_int32),
        ("head_size", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("p", ctypes.c_void_p),
        ("s_out", ctypes.c_void_p),
        ("o_out", ctypes.c_void_p),
    ]


class AttnLib:
    """The tensor-core library, duck-typed like ``ref_capi.RefLib``."""

    def __init__(self, path: Path = ATTN_LIB, loader: int = 0) -> None:
        """Load ``path`` and check the ABI against the mirrors.

        ``loader``: 0 shipped default, 1 separate streams, 2 whole cell.
        """
        self.path = path
        self.loader = loader
        lib = self.lib = ctypes.CDLL(str(path))
        i32 = ctypes.c_int32
        for name in (
            "pth_attn_abi_version",
            "pth_attn_attn_args_bytes",
            "pth_attn_combine_args_bytes",
            "pth_attn_options_bytes",
            "pth_attn_tile_test_args_bytes",
            "pth_attn_dec_tile_test_args_bytes",
            "pth_attn_has_cuda",
        ):
            getattr(lib, name).restype = i32
        lib.pth_attn_last_error.restype = ctypes.c_char_p
        lib.pth_attn_attention_ex.restype = i32
        lib.pth_attn_attention_ex.argtypes = [
            ctypes.POINTER(ref_capi.PthAttnArgs),
            ctypes.POINTER(PthAttnOptions),
            ctypes.c_void_p,
        ]
        lib.pth_attn_combine.restype = i32
        lib.pth_attn_combine.argtypes = [
            ctypes.POINTER(ref_capi.PthAttnCombineArgs),
            ctypes.c_void_p,
        ]
        lib.pth_attn_tiling.restype = i32
        lib.pth_attn_tiling.argtypes = [i32, i32, i32] + [ctypes.POINTER(i32)] * 3
        lib.pth_attn_test_warp_tile.restype = i32
        lib.pth_attn_test_warp_tile.argtypes = [
            ctypes.POINTER(PthAttnTileTestArgs),
            ctypes.c_void_p,
        ]
        lib.pth_attn_test_dec_tile.restype = i32
        lib.pth_attn_test_dec_tile.argtypes = [
            ctypes.POINTER(PthAttnDecTileTestArgs),
            ctypes.c_void_p,
        ]
        checks = {
            "pth_attn_abi_version": ref_capi.ABI_VERSION,
            "pth_attn_attn_args_bytes": ctypes.sizeof(ref_capi.PthAttnArgs),
            "pth_attn_combine_args_bytes": ctypes.sizeof(ref_capi.PthAttnCombineArgs),
            "pth_attn_options_bytes": ctypes.sizeof(PthAttnOptions),
            "pth_attn_tile_test_args_bytes": ctypes.sizeof(PthAttnTileTestArgs),
            "pth_attn_dec_tile_test_args_bytes": ctypes.sizeof(PthAttnDecTileTestArgs),
        }
        for fn, want in checks.items():
            got = getattr(lib, fn)()
            if got != want:
                raise RuntimeError(
                    f"{path}: {fn}() = {got}, mirror has {want}. The library and "
                    "this attn_lib.py come from different source trees: run with "
                    "the tree the library was built from (build/k/src for the "
                    "shipped kernel), or pass that tree's libpth_attn.so with "
                    "--kernel-lib to bench/kernel_bench.py"
                )

    def has_cuda(self) -> bool:
        """Always True (GPU-only library)."""
        return bool(self.lib.pth_attn_has_cuda())

    def last_error(self) -> str:
        """Message of the last failure on this thread."""
        return self.lib.pth_attn_last_error().decode()

    def attention(
        self, args: ref_capi.PthAttnArgs, *, host: bool, stream: int = 0
    ) -> int:
        """pth_attn_attention_ex with this object's options; host is refused."""
        if host:
            raise RuntimeError("libpth_attn has no host entry points")
        opt = PthAttnOptions(
            ref_capi.ABI_VERSION, ctypes.sizeof(PthAttnOptions), self.loader
        )
        return int(
            self.lib.pth_attn_attention_ex(
                ctypes.byref(args), ctypes.byref(opt), stream or None
            )
        )

    def combine(
        self, args: ref_capi.PthAttnCombineArgs, *, host: bool, stream: int = 0
    ) -> int:
        """pth_attn_combine (same arithmetic as pth_ref_combine)."""
        if host:
            raise RuntimeError("libpth_attn has no host entry points")
        return int(self.lib.pth_attn_combine(ctypes.byref(args), stream or None))

    def tiling(self, head_size: int, bq: int, group: int) -> tuple[int, int, int]:
        """(n_tile, sub_tile, wn) the kernel uses, from the library itself."""
        n, sub, wn = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        rc = self.lib.pth_attn_tiling(
            head_size, bq, group, ctypes.byref(n), ctypes.byref(sub), ctypes.byref(wn)
        )
        if rc != ref_capi.OK:
            raise RuntimeError(f"pth_attn_tiling: {self.last_error()}")
        return n.value, sub.value, wn.value

    def test_warp_tile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        s_out: torch.Tensor,
        o_out: torch.Tensor,
    ) -> int:
        """pth_attn_test_warp_tile on device tensors (dense row-major)."""
        a = PthAttnTileTestArgs(
            ref_capi.ABI_VERSION,
            ctypes.sizeof(PthAttnTileTestArgs),
            q.shape[-1],
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            p.data_ptr(),
            s_out.data_ptr(),
            o_out.data_ptr(),
        )
        stream = torch.cuda.current_stream().cuda_stream
        return int(self.lib.pth_attn_test_warp_tile(ctypes.byref(a), stream or None))

    def test_dec_tile(
        self,
        rows: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        s_out: torch.Tensor,
        o_out: torch.Tensor,
    ) -> int:
        """pth_attn_test_dec_tile on device tensors (dense row-major)."""
        a = PthAttnDecTileTestArgs(
            ref_capi.ABI_VERSION,
            ctypes.sizeof(PthAttnDecTileTestArgs),
            q.shape[-1],
            rows,
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            p.data_ptr(),
            s_out.data_ptr(),
            o_out.data_ptr(),
        )
        stream = torch.cuda.current_stream().cuda_stream
        return int(self.lib.pth_attn_test_dec_tile(ctypes.byref(a), stream or None))


@dataclasses.dataclass(frozen=True)
class AttnCase(grid.GridCase):
    """A grid case run by the tensor-core kernel: its tiling is the kernel's.

    ``kernel_sub`` / ``kernel_wn`` are part of the dataclass, so
    ``describe()`` (the expected-value cache key) differs from the
    reference's case of the same name; inputs (seeded by name) are identical.
    """

    kernel_sub: int = 0
    kernel_wn: int = 0

    @property
    def tiling(self) -> Tiling:
        return Tiling(
            grid.REF_N_TILE, self.kernel_sub, self.kernel_wn, self.num_splits, self.bq
        )


def attn_case(c: grid.GridCase, lib: AttnLib) -> AttnCase:
    """``c`` with the kernel's tiling (from ``lib.tiling``).

    ``c`` may itself be an :class:`AttnCase` (e.g. ``dataclasses.replace`` of
    one with another window): only the ``GridCase`` fields are carried over
    and the tiling is asked from the library again for the new case.
    """
    n, sub, wn = lib.tiling(c.D, c.bq, c.G)
    if n != grid.REF_N_TILE:
        raise RuntimeError(f"kernel load tile {n} != {grid.REF_N_TILE}")
    fields = {f.name: getattr(c, f.name) for f in dataclasses.fields(grid.GridCase)}
    return AttnCase(**fields, kernel_sub=sub, kernel_wn=wn)
