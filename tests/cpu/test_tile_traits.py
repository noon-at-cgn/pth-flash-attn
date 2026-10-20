# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for csrc/pth_attn/tile_traits.cuh.

The tile geometry is a handful of byte and register counts that decide
whether the kernel fits on the SM at all. These tests recompute every number
from first principles in Python, with the arithmetic written out, and compare
it with what the C++ header computes (tests/cpp/test_tile_traits.cpp, built
with g++). The documented table is pinned row by row, and the rejected
geometries (N = 128, 3 stages, 2 CTAs/SM at D = 256, a SUB that
does not divide the warp's slice) are negative controls that must fail to
compile with their specific message.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CSRC = REPO / "csrc" / "pth_attn"
CPP_TEST = REPO / "tests" / "cpp" / "test_tile_traits.cpp"

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")

# Hardware (sm_86/sm_89) and the three choices.
SMEM_PER_BLOCK, SMEM_PER_SM, RESERVED = 101376, 102400, 1024
REGS_PER_SM, MAX_REGS, ALLOC_UNIT = 65536, 255, 8
N, STAGES, WARPS, THREADS, ROWS_PER_WARP = 64, 2, 4, 128, 16

# Tile table: D, WM -> (Q smem, stage, 2 stages, total, CTAs/SM, SUB)
GEOMETRY_TABLE = {
    (256, 4): (32768, 33280, 66560, 99328, 1, 32),
    (256, 2): (16384, 33280, 66560, 82944, 1, 32),
    (256, 1): (8192, 33280, 66560, 74752, 1, 16),
    (128, 4): (16384, 16896, 33792, 50176, 2, 64),
    (64, 4): (8192, 8704, 17408, 25600, 3, 64),
}


def expected(d: int, wm: int) -> dict[str, int]:
    """Every traits value, derived step by step.

    rows   = 16 * WM                    (m16 MMA: 16 rows per warp)
    WN     = 4 / WM                     (4 warps as WM x WN)
    slice  = N / WN                     (tokens per warp per tile)
    sub    = min(slice, 32 if D == 256 else 64)
    Q      = rows * D * 2               (fp16 Q)
    stage  = N * (D + D + 4 + 4)        (K int8, V int8, ks f32, vs f32)
    smem   = Q + 2 * stage
    ctas   = 1, 2, 3 for D = 256, 128, 64
    budget = min(255, floor(65536 / (ctas * 128) / 8) * 8)
    merge  = (WN - 1) * WM * 16 * (D * 4 + 2 * 4)
    """
    rows = ROWS_PER_WARP * wm
    wn = WARPS // wm
    slice_ = N // wn
    sub = min(slice_, 32 if d == 256 else 64)
    q = rows * d * 2
    stage = N * (d + d + 4 + 4)
    ctas = {256: 1, 128: 2, 64: 3}[d]
    budget = min(MAX_REGS, REGS_PER_SM // (ctas * THREADS) // ALLOC_UNIT * ALLOC_UNIT)
    return {
        "D": d,
        "WM": wm,
        "WN": wn,
        "rows": rows,
        "N": N,
        "stages": STAGES,
        "tokens_per_warp": slice_,
        "sub": sub,
        "sub_steps": slice_ // sub,
        "q_bytes": q,
        "stage_bytes": stage,
        "smem_bytes": q + STAGES * stage,
        "merge_bytes": (wn - 1) * wm * 16 * (d * 4 + 2 * 4),
        "ctas_per_sm": ctas,
        "reg_budget": budget,
        "o_regs": 16 * d // 32,
        "s_regs": 16 * sub // 32,
        "out_bytes": 2,
    }


def _compile(src: Path, exe: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Werror",
            f"-I{CSRC}",
            str(src),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def cpp_rows(tmp_path_factory) -> dict[tuple[int, int], dict[str, int]]:
    """Builds and runs the C++ test; returns its printed rows by (D, WM)."""
    exe = tmp_path_factory.mktemp("traits") / "test_tile_traits"
    build = _compile(CPP_TEST, exe)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    rows = {}
    for line in run.stdout.splitlines():
        kv = dict(item.split("=") for item in line.split())
        row = {k: int(v) for k, v in kv.items()}
        rows[(row["D"], row["WM"])] = row
    return rows


def test_all_nine_configurations_match_first_principles(cpp_rows):
    """Every (D, WM) the kernel ships has exactly the derived geometry.

    The header and this test compute the numbers independently (C++ templates
    vs the formulas in ``expected``); any drift in either is caught.
    """
    assert sorted(cpp_rows) == sorted(
        (d, wm) for d in (64, 128, 256) for wm in (1, 2, 4)
    )
    for key, row in cpp_rows.items():
        assert row == expected(*key), key


@pytest.mark.parametrize("key", list(GEOMETRY_TABLE))
def test_reproduces_geometry_table(cpp_rows, key):
    """The tile table, reproduced row by row from the header.

    Worked example for the first row (D = 256, WM = 4):
    Q = 64 rows * 256 * 2 B = 32768; stage = 64 * (256 + 256 + 4 + 4) = 33280;
    total = 32768 + 2 * 33280 = 99328 <= 101376 (2048 B spare).
    """
    q, stage, two, total, ctas, sub = GEOMETRY_TABLE[key]
    row = cpp_rows[key]
    assert (row["q_bytes"], row["stage_bytes"], 2 * row["stage_bytes"]) == (
        q,
        stage,
        two,
    )
    assert (row["smem_bytes"], row["ctas_per_sm"], row["sub"]) == (total, ctas, sub)


def test_occupancy_claims(cpp_rows):
    """CTAs/SM fit both shared memory and the register file.

    D = 128: 2 * (50176 + 1024) = 102400, the per-SM limit exactly.
    D = 64:  3 * (25600 + 1024) = 79872 <= 102400 (smem would allow 3, not 4:
             4 * 26624 = 106496), and the register budget is 168:
             168 * 128 * 3 = 64512 <= 65536.
    """
    for (d, wm), row in cpp_rows.items():
        assert row["ctas_per_sm"] * (row["smem_bytes"] + RESERVED) <= SMEM_PER_SM
        assert row["ctas_per_sm"] * THREADS * row["reg_budget"] <= REGS_PER_SM
    assert 2 * (cpp_rows[(128, 4)]["smem_bytes"] + RESERVED) == SMEM_PER_SM
    assert cpp_rows[(64, 4)]["reg_budget"] == 168


def test_n_is_the_largest_load_tile_that_fits():
    """N = 64 is the largest multiple of 16 * WN with Q + 2 N (2D + 8) <= limit.

    At D = 256, WM = 4: 32768 + 2 * N * 520 <= 101376 gives N <= 65.96, so
    N = 64; the next multiple of 16 (80) needs 32768 + 83200 = 115968 B.
    """
    q = 64 * 256 * 2
    n_max = (SMEM_PER_BLOCK - q) // (2 * (2 * 256 + 8))
    assert n_max == 65
    assert max(n for n in range(16, 129, 16) if q + 2 * n * 520 <= SMEM_PER_BLOCK) == 64


NEGATIVE = {
    # (D, WM, N, STAGES, SUB, CTAS): expected static_assert text
    "N = 128 at D = 256 does not fit": (
        "256, 4, 128, 2, 32, 1",
        "shared memory exceeds the sm_86 per-block limit (101376 B)",
    ),
    "3 stages at D = 256 do not fit": (
        "256, 4, 64, 3, 32, 1",
        "shared memory exceeds the sm_86 per-block limit (101376 B)",
    ),
    "2 CTAs/SM at D = 256 do not fit": (
        "256, 4, 64, 2, 32, 2",
        "CTAS blocks do not fit in 102400 B",
    ),
    "4 CTAs/SM at D = 64 do not fit": (
        "64, 4, 64, 2, 64, 4",
        "CTAS blocks do not fit in 102400 B",
    ),
    "SUB = 48 does not divide a 64-token slice": (
        "128, 4, 64, 2, 48, 2",
        "SUB must be a multiple of 16 that divides the warp's slice",
    ),
    "N = 48 is not whole k-steps per warp at WN = 4": (
        "64, 1, 48, 2, 16, 3",
        "each warp's token slice must be whole PV k-steps of 16",
    ),
}


def test_ctas_per_sm_agrees_with_the_planner(cpp_rows):
    """The C++ CTAs/SM equal the Python planner's CTAS_PER_SM.

    The planner sizes splits from C_target = SMs x CTAs/SM(D);
    the kernel's __launch_bounds__ use tile_traits' value. Two sources of one
    number must agree, or the split count fills the GPU only on paper.
    """
    from pth_flash_attn.planner import CTAS_PER_SM

    for (d, _wm), row in cpp_rows.items():
        assert row["ctas_per_sm"] == CTAS_PER_SM[d], d


@pytest.mark.parametrize("case", list(NEGATIVE))
def test_rejected_geometries_do_not_compile(tmp_path, case):
    """The rejected options fail at compile time with their reason.

    Negative control for the static_asserts: if a check were missing, the
    snippet would compile and the test would fail.
    """
    args, message = NEGATIVE[case]
    src = tmp_path / "neg.cpp"
    src.write_text(
        '#include "tile_traits.cuh"\n'
        f"constexpr int kBytes = pth_attn::TileGeometry<{args}>::kSmemBytes;\n"
        "int main() { return kBytes > 0 ? 0 : 1; }\n"
    )
    build = _compile(src, tmp_path / "neg")
    assert build.returncode != 0, f"{case}: compiled, a static_assert is missing"
    assert message in build.stderr, build.stderr


def test_positive_control_snippet_compiles(tmp_path):
    """The snippet harness itself works: the shipped geometry compiles."""
    src = tmp_path / "pos.cpp"
    src.write_text(
        '#include "tile_traits.cuh"\n'
        "constexpr int kBytes = pth_attn::TileGeometry<256, 4, 64, 2, 32, 1>"
        "::kSmemBytes;\n"
        "int main() { return kBytes == 99328 ? 0 : 1; }\n"
    )
    exe = tmp_path / "pos"
    build = _compile(src, exe)
    assert build.returncode == 0, build.stderr
    assert subprocess.run([str(exe)], check=False).returncode == 0
