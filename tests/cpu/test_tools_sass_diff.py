# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/sass_diff.py (SASS identity of two builds).

A kernel refactor is claimed free of performance and numerics changes when
the machine code is identical. The tool must therefore (1) call two builds of
the same source identical, (2) catch a one-instruction difference, and (3)
report a kernel that disappeared, unless the caller allowed that removal.
Each case compiles a tiny kernel for sm_86 on the CPU.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import sass_check  # noqa: E402
import sass_diff  # noqa: E402

NVCC = "/usr/local/cuda/bin/nvcc"


def _have_tools() -> bool:
    try:
        sass_check.find_cuobjdump()
    except FileNotFoundError:
        return False
    return Path(NVCC).is_file() or shutil.which("nvcc") is not None


needs_tools = pytest.mark.skipif(not _have_tools(), reason="nvcc/cuobjdump absent")

SOURCE = """
__global__ void scale(float* x, float s) {{ x[threadIdx.x] *= {factor}; }}
{extra}
"""
EXTRA_KERNEL = "__global__ void extra_kernel(int* y) { y[threadIdx.x] = 7; }"


def _cubin(tmp: Path, name: str, factor: str, extra: str = "") -> Path:
    src = tmp / f"{name}.cu"
    src.write_text(SOURCE.format(factor=factor, extra=extra))
    out = tmp / f"{name}.cubin"
    nvcc = NVCC if Path(NVCC).is_file() else "nvcc"
    subprocess.run(
        [nvcc, "-O3", "-arch=sm_86", "-cubin", "-o", str(out), str(src)],
        check=True,
    )
    return out


def test_anonymous_namespace_is_keyed_independently_of_its_file():
    """A kernel in an anonymous namespace carries a per-file identifier in its
    mangled name; moving it to another .cu file must not make it 'new'."""
    old = "_ZN45_GLOBAL__N__3db8e440_16_attn_capi_cu_644f431f6kernelILi64EEEvi"
    new = "_ZN53_GLOBAL__N__1a2b3c4d_20_attn_test_kernels_cu_9e8d7c6b6kernelILi64EEEvi"
    assert sass_diff.normalise_name(old) == sass_diff.normalise_name(new)
    assert sass_diff.normalise_name(old) == "_ZN(anonymous)6kernelILi64EEEvi"
    assert sass_diff.normalise_name("_Z5scalePff") == "_Z5scalePff"


@needs_tools
def test_same_source_is_identical(tmp_path):
    """Two compiles of one source: identical, exit 0."""
    a = _cubin(tmp_path, "a", "s", EXTRA_KERNEL)
    b = _cubin(tmp_path, "b", "s", EXTRA_KERNEL)
    assert sass_diff.main([str(a), str(b)]) == 0


@needs_tools
def test_one_changed_instruction_is_caught(tmp_path):
    """Negative control: `x * s` vs `x * (s + 1)` differs, exit 1."""
    a = _cubin(tmp_path, "a", "s")
    b = _cubin(tmp_path, "b", "(s + 1.0f)")
    assert sass_diff.main([str(a), str(b)]) == 1


@needs_tools
def test_removed_kernel_fails_unless_allowed(tmp_path):
    """A kernel missing from the second build is a difference, unless its
    name matches --allow-removed (e.g. test kernels moved to a test TU)."""
    a = _cubin(tmp_path, "a", "s", EXTRA_KERNEL)
    b = _cubin(tmp_path, "b", "s")
    assert sass_diff.main([str(a), str(b)]) == 1
    assert sass_diff.main([str(a), str(b), "--allow-removed", "extra_kernel"]) == 0
