# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/collect_env.sh (read-only environment capture).

collect_env.sh runs on machines whose GPU settings belong to someone else, where a
stray ``nvidia-smi -pl`` or ``--lock-gpu-clocks`` would change them.
So besides "it runs and reports", the tests scan the script for any GPU
state-changing command, with a negative control proving the scanner flags
one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "collect_env.sh"

# nvidia-smi options that change GPU state (nvidia-smi --help): power limit,
# clock locks/resets, application clocks, persistence, compute mode, ECC,
# GPU reset, MIG, and driver-model changes.
MUTATING_NVIDIA_SMI = re.compile(
    r"nvidia-smi\b[^\n]*\s(-pl\b|--power-limit|-lgc\b|--lock-gpu-clocks|-rgc\b|"
    r"--reset-gpu-clocks|-lmc\b|--lock-memory-clocks|-rmc\b|-ac\b|"
    r"--applications-clocks|-rac\b|-pm\b|--persistence-mode|-c\b|"
    r"--compute-mode|-e\b|--ecc-config|-r\b|--gpu-reset|-mig\b|-dm\b)"
)
# Python imports that can initialise CUDA; versions must come from metadata.
CUDA_INITIALISING = re.compile(r"\bimport (torch|vllm)\b")


def mutating_commands(text: str) -> list[str]:
    """Lines of a shell script that would change GPU state or init CUDA."""
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return [
        ln
        for ln in code
        if MUTATING_NVIDIA_SMI.search(ln) or CUDA_INITIALISING.search(ln)
    ]


def test_script_is_read_only():
    """collect_env.sh contains no GPU-state-changing command."""
    assert mutating_commands(SCRIPT.read_text()) == []


@pytest.mark.parametrize(
    "line",
    [
        "nvidia-smi -pl 200",
        "nvidia-smi --lock-gpu-clocks=1500,1500",
        "sudo nvidia-smi -i 0 -r",
        'python3 -c "import torch; print(torch.cuda.is_available())"',
    ],
)
def test_scanner_flags_mutating_commands(line):
    """Negative control: the scanner catches each kind of forbidden command."""
    assert mutating_commands(f"#!/bin/sh\n{line}\n") == [line]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_runs_and_reports_every_section(tmp_path):
    """The script exits 0 and prints every section, tools present or not.

    In the CPU-only dev container nvidia-smi is absent; the script must say
    so instead of failing, because a missing probe is information too.
    """
    out = tmp_path / "env.txt"
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(out)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    text = out.read_text()
    for section in (
        "## time (UTC)",
        "## host",
        "## git",
        "## GPU (nvidia-smi, read-only queries)",
        "## CUDA toolkit",
        "## Python packages (metadata only, no import)",
    ):
        assert section in text
    assert re.search(r"^torch: \S+", text, re.MULTILINE)
