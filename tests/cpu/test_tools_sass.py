# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/sass_check.py (static SASS lint).

The SASS lint turns three design claims into checks on the compiled code:
the int8 path never converts with I2F in a loop (it converts with magic numbers),
nothing touches local memory (no LDL/STL), and the intended hardware paths
are present (LDGSTS for cp.async, HMMA for mma.sync, LDSM for ldmatrix).
Each rule is tested on a kernel that satisfies it and on one that violates
it; the violating one is the negative control that proves the rule can fail.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import sass_check  # noqa: E402

NVCC = "/usr/local/cuda/bin/nvcc"


def _have_tools() -> bool:
    try:
        sass_check.find_cuobjdump()
    except FileNotFoundError:
        return False
    return Path(NVCC).is_file() or shutil.which("nvcc") is not None


needs_tools = pytest.mark.skipif(not _have_tools(), reason="nvcc/cuobjdump absent")

# Crafted in the format of real cuobjdump -sass output (sm_86; the encoding
# comments after ";" are dropped, the parser ignores them): a forward branch,
# a loop body with I2FP, a backward branch, then the self-branch trap ptxas
# puts after EXIT (which is not a loop).
CRAFTED = """\
\tcode for sm_86
\t\tFunction : _Z4loopPfi
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0010*/              @!P0 BRA 0x80 ;
        /*0020*/                   I2FP.F32.S32 R2, R9 ;
        /*0030*/                   FADD R7, R2, R7 ;
        /*0040*/               @P1 BRA 0x20 ;
        /*0050*/                   I2F R3, R4 ;
        /*0060*/                   STL [R1], R3 ;
        /*0070*/                   HMMA.16816.F32 R4, R8, R12, R4 ;
        /*0080*/                   EXIT ;
        /*0090*/                   BRA 0x90;
"""


def test_parse_and_loop_detection_on_crafted_sass():
    """A backward BRA closes a loop; forward branches do not."""
    (instrs,) = sass_check.parse_sass(CRAFTED).values()
    assert [i.opcode for i in instrs][:3] == ["IMAD.MOV.U32", "BRA", "I2FP.F32.S32"]
    assert sass_check.find_loops(instrs) == [(0x20, 0x40)]


def test_opcode_matching_is_dotted_prefix():
    """HMMA matches HMMA.16816.F32; I2F does not match I2FP (both listed)."""
    assert sass_check.opcode_matches("HMMA.16816.F32", "HMMA")
    assert sass_check.opcode_matches("HMMA.16816.F32", "HMMA.16816.F32")
    assert not sass_check.opcode_matches("I2FP.F32.S32", "I2F")
    assert not sass_check.opcode_matches("LDLX", "LDL")


def test_rules_on_crafted_sass():
    """In-loop I2FP is caught, out-of-loop I2F is not; STL is caught.

    The I2F at 0x50 lies after the loop, so forbidding I2F in loops must not
    flag it: this distinguishes "in a loop" from "anywhere".
    """
    (instrs,) = sass_check.parse_sass(CRAFTED).values()
    rep = sass_check.check_kernel(
        "k", instrs, ["LDL", "STL"], ["I2F", "I2FP"], ["HMMA", "LDGSTS"]
    )
    text = "\n".join(rep.violations)
    assert "I2FP inside loop 0x0020-0x0040 at 0x0020" in text
    assert not any("0x0050" in v for v in rep.violations)  # I2F after the loop
    assert "forbidden STL at 0x0060" in text
    assert "required LDGSTS not present" in text
    assert "required HMMA not present" not in text


FEATURES_CU = r"""
#include <cstdint>
// Uses the three paths the kernel needs: cp.async (LDGSTS), ldmatrix (LDSM)
// and mma.sync m16n8k16 f16 -> f32 (HMMA.16816.F32).
__global__ void features(const int4* g, float* out) {
  __shared__ __align__(16) int4 s[128];
  unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(&s[threadIdx.x]));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(saddr),
               "l"(g + threadIdx.x));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  unsigned a0, a1, a2, a3;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(saddr));
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a0), "r"(a1));
  out[threadIdx.x] = d0 + d1 + d2 + d3;
}
"""

I2F_LOOP_CU = r"""
__global__ void convert_in_loop(const int* x, float* y, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i)
    acc += static_cast<float>(x[i * blockDim.x + threadIdx.x]);
  y[threadIdx.x] = acc;
}
"""

NO_CONVERT_LOOP_CU = r"""
__global__ void no_convert_in_loop(const float* x, float* y, int n) {
  float acc = 0.f;
  for (int i = 0; i < n; ++i) acc += x[i * blockDim.x + threadIdx.x];
  y[threadIdx.x] = acc;
}
"""

LOCAL_ARRAY_CU = r"""
__global__ void local_array(const float* x, float* y, int n) {
  float a[256];  // dynamically indexed: must live in local memory
  for (int i = 0; i < 256; ++i) a[i] = x[i * n + threadIdx.x];
  y[threadIdx.x] = a[(threadIdx.x * 7 + n) & 255];
}
"""


def _run(tmp_path: Path, name: str, src: str, *extra: str) -> int:
    cu = tmp_path / f"{name}.cu"
    cu.write_text(src)
    argv = ["--cu", str(cu), "--arch", "86", "--out-dir", str(tmp_path), *extra]
    return sass_check.main(argv)


@needs_tools
def test_required_paths_present_and_absent(tmp_path):
    """LDGSTS/LDSM/HMMA are found where used and reported missing where not.

    The second kernel has none of them: requiring them must fail (negative
    control), otherwise a kernel that lost its tensor-core path would pass.
    """
    req = ["--require", "LDGSTS,LDSM,HMMA.16816.F32"]
    assert _run(tmp_path, "features", FEATURES_CU, *req) == 0
    assert _run(tmp_path, "plain", NO_CONVERT_LOOP_CU, *req) == 1


@needs_tools
def test_int_to_float_in_loop_is_caught(tmp_path):
    """A static_cast<float>(int) in a loop compiles to I2F/I2FP and fails.

    The same loop without the conversion passes: the rule reacts to the
    conversion, not to the loop.
    """
    assert _run(tmp_path, "i2f", I2F_LOOP_CU) == 1
    assert _run(tmp_path, "noconv", NO_CONVERT_LOOP_CU) == 0


@needs_tools
def test_local_memory_is_caught(tmp_path, capsys):
    """A dynamically indexed local array produces LDL/STL and fails.

    ptxas reports such arrays as "stack frame", not as spills, so
    ptxas_gate.py alone would pass this kernel; this rule closes that gap.
    """
    assert _run(tmp_path, "local", LOCAL_ARRAY_CU) == 1
    assert "forbidden STL" in capsys.readouterr().out


@needs_tools
def test_unmatched_function_filter_fails(tmp_path):
    """A --function regex that matches no kernel is a failure, not a pass."""
    rc = _run(tmp_path, "noconv", NO_CONVERT_LOOP_CU, "--function", "^does_not_exist$")
    assert rc == 1


@needs_tools
def test_loopless_kernel_reports_vacuous_loop_rule(tmp_path, capsys):
    """Without a backward branch the loop rule is reported as unchecked."""
    src = "__global__ void k(float* y) { y[threadIdx.x] = 2.f; }\n"
    assert _run(tmp_path, "flat", src) == 0
    assert "holds vacuously" in capsys.readouterr().out
