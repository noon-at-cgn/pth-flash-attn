# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/ptxas_gate.py (registers, spills, static smem).

The gate is what stands between a kernel edit and a silent spill in the main
loop (the D=256 probe has 7 registers of headroom). A gate that
parses nothing passes everything, so each rule has a negative control: a
crafted ptxas report or a real compile that MUST fail.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import ptxas_gate  # noqa: E402

NVCC = "/usr/local/cuda/bin/nvcc"
needs_nvcc = pytest.mark.skipif(
    not Path(NVCC).is_file() and shutil.which("nvcc") is None,
    reason="nvcc not available",
)

# Real nvcc 13.0 output for sm_86 (captured in the dev image), one kernel.
CLEAN_LOG = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z5smallPf' for 'sm_86'
ptxas info    : Function properties for _Z5smallPf
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 1 barriers, 4096 bytes smem, 360 bytes cmem[0]
"""


def _log(regs: int, smem: int, st: int, ld: int, name: str = "_Z1kv") -> str:
    return (
        f"ptxas info    : Compiling entry function '{name}' for 'sm_86'\n"
        f"ptxas info    : Function properties for {name}\n"
        f"    0 bytes stack frame, {st} bytes spill stores, {ld} bytes spill loads\n"
        f"ptxas info    : Used {regs} registers, used 1 barriers, {smem} bytes smem, "
        "360 bytes cmem[0]\n"
    )


def test_parse_real_log():
    """The parser reads registers, smem and spills from real ptxas output."""
    (k,) = ptxas_gate.parse_ptxas_log(CLEAN_LOG)
    assert (k.name, k.arch, k.registers, k.smem_bytes) == (
        "_Z5smallPf",
        "sm_86",
        10,
        4096,
    )
    assert ptxas_gate.evaluate([k])


@pytest.mark.parametrize(
    "regs,smem,st,ld,expected",
    [
        (255, 101376, 0, 0, None),  # exactly at both limits: passes
        (256, 0, 0, 0, "registers 256 > 255"),
        (200, 101377, 0, 0, "static smem 101377 B > 101376 B"),
        (248, 0, 36, 0, "spills: 36 B stores"),
        (248, 0, 0, 36, "spills: 0 B stores, 36 B loads"),
    ],
)
def test_limits_on_crafted_reports(regs, smem, st, ld, expected):
    """Each limit fires exactly at its boundary (negative controls).

    The limits are: 255 registers, 0 spill bytes,
    101376 B smem per block on sm_86. The first case is the positive control
    at the boundary; the others must each produce their own violation.
    """
    kernels = ptxas_gate.parse_ptxas_log(_log(regs, smem, st, ld))
    ok = ptxas_gate.evaluate(kernels)
    if expected is None:
        assert ok and not kernels[0].violations
    else:
        assert not ok
        assert any(expected in v for v in kernels[0].violations), kernels[0].violations


def test_missing_register_line_is_a_failure_not_a_pass():
    """If ptxas' format drifts and "Used N registers" vanishes, the gate fails.

    A parser that silently finds nothing would report every kernel clean.
    """
    log = "\n".join(CLEAN_LOG.splitlines()[:4]) + "\n"
    kernels = ptxas_gate.parse_ptxas_log(log)
    assert not ptxas_gate.evaluate(kernels)
    assert "register count not found" in kernels[0].violations[0]


def test_callee_spills_count_against_the_kernel():
    """Spills inside a non-inlined callee execute in the kernel: they count."""
    log = (
        "ptxas info    : Compiling entry function '_Z1kv' for 'sm_86'\n"
        "ptxas info    : Function properties for _Z1kv\n"
        "    8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info    : Function properties for _Z6helperv\n"
        "    16 bytes stack frame, 12 bytes spill stores, 12 bytes spill loads\n"
        "ptxas info    : Used 40 registers, 360 bytes cmem[0]\n"
    )
    (k,) = ptxas_gate.parse_ptxas_log(log)
    assert (k.stack_bytes, k.spill_stores, k.spill_loads) == (8, 12, 12)
    assert not ptxas_gate.evaluate([k])


PRESSURE_CU = r"""
// 64 values live at once: fine with the default register budget, but forced
// to spill when ptxas may use only 32 registers.
__global__ void pressure(const float* x, float* y, int n) {
  float a[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) a[i] = x[threadIdx.x + i * n];
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += a[i] * a[63 - i] + a[(i * 5) & 63];
  y[threadIdx.x] = s;
}
"""


@needs_nvcc
def test_real_compile_passes_then_spills_under_maxrregcount(tmp_path, capsys):
    """End to end: the same kernel passes, then fails when forced to spill.

    This exercises nvcc, ptxas -v and the parser together with the extension's
    flags. The negative control (-maxrregcount=32) proves the gate reacts to a
    real spill, not only to crafted text.
    """
    src = tmp_path / "pressure.cu"
    src.write_text(PRESSURE_CU)
    out = tmp_path / "out"
    assert ptxas_gate.main([str(src), "--arch", "86", "--out-dir", str(out)]) == 0
    rc = ptxas_gate.main(
        [
            str(src),
            "--arch",
            "86",
            "--out-dir",
            str(out),
            "--nvcc-flag=-maxrregcount=32",
        ]
    )
    assert rc == 1
    assert "spills:" in capsys.readouterr().out


@needs_nvcc
def test_zero_kernels_passes_unless_min_kernels(tmp_path):
    """A file without kernels passes vacuously by default.

    ``--min-kernels`` turns "nothing was checked" into a failure, so a
    gate run on the real kernel files can never pass by finding nothing.
    """
    src = tmp_path / "host_only.cu"
    src.write_text("int host_only() { return 0; }\n")
    out = tmp_path / "out"
    assert ptxas_gate.main([str(src), "--arch", "86", "--out-dir", str(out)]) == 0
    rc = ptxas_gate.main(
        [str(src), "--arch", "86", "--out-dir", str(out), "--min-kernels", "1"]
    )
    assert rc == 1


@needs_nvcc
def test_compile_error_is_exit_2(tmp_path):
    """A source that does not compile is a tool failure (2), not a pass."""
    src = tmp_path / "broken.cu"
    src.write_text("__global__ void k( { }\n")
    assert ptxas_gate.main([str(src), "--arch", "86", "--out-dir", str(tmp_path)]) == 2
