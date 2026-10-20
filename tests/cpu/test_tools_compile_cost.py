# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/compile_cost.py (nvcc time and binary size).

Compile cost is an acceptance criterion for kernel changes, so the
report must be real: measured with the extension's flags, per arch, with a
failure recorded as a failure rather than as a zero-cost row.

The ``--extension`` mode (full ``setup.py`` builds per arch, ~20 s each) is
exercised by ``tools/run_ci_local.sh``, not here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import compile_cost  # noqa: E402

needs_nvcc = pytest.mark.skipif(
    not Path("/usr/local/cuda/bin/nvcc").is_file() and shutil.which("nvcc") is None,
    reason="nvcc not available",
)

KERNEL_CU = "__global__ void k(float* y) { y[threadIdx.x] *= 2.f; }\n"


@needs_nvcc
def test_per_arch_rows_with_sizes(tmp_path):
    """One row per arch, each with a positive time and the real cubin size."""
    cu = tmp_path / "k.cu"
    cu.write_text(KERNEL_CU)
    out = tmp_path / "cost.json"
    rc = compile_cost.main(
        ["--tu", str(cu), "--arch", "80", "86", "--out-dir", str(tmp_path),
         "--json", str(out)]
    )  # fmt: skip
    assert rc == 0
    rows = json.loads(out.read_text())["tu"]
    assert [r["arch"] for r in rows] == ["sm_80", "sm_86"]
    for r in rows:
        cubin = tmp_path / f"k.{r['arch']}.cubin"
        assert r["ok"] and r["seconds"] > 0
        assert r["bytes"] == cubin.stat().st_size > 0


@needs_nvcc
def test_host_translation_unit_is_measured_once(tmp_path):
    """A .cpp TU is compiled once by the host compiler, not per arch."""
    cpp = tmp_path / "h.cpp"
    cpp.write_text("int f(int x) { return x + 2; }\n")
    out = tmp_path / "cost.json"
    assert compile_cost.main(["--tu", str(cpp), "--out-dir", str(tmp_path),
                              "--json", str(out)]) == 0  # fmt: skip
    (row,) = json.loads(out.read_text())["tu"]
    assert row["arch"] == "host" and row["bytes"] > 0


@needs_nvcc
def test_failed_compile_is_reported_not_hidden(tmp_path):
    """Negative control: a broken TU gives exit 2 and a row with ok=False.

    A cost table that silently dropped failing units would under-report.
    """
    cu = tmp_path / "broken.cu"
    cu.write_text("__global__ void k( {}\n")
    out = tmp_path / "cost.json"
    rc = compile_cost.main(
        [
            "--tu",
            str(cu),
            "--arch",
            "86",
            "--out-dir",
            str(tmp_path),
            "--json",
            str(out),
        ]
    )
    assert rc == 2
    (row,) = json.loads(out.read_text())["tu"]
    assert row["ok"] is False and row["bytes"] is None and "error" in row["error"]


def test_nothing_to_measure_is_a_usage_error():
    """Calling without --tu or --extension is an error, not an empty report."""
    with pytest.raises(SystemExit):
        compile_cost.main([])
