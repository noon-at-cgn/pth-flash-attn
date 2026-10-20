# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The CPU limit of the dev containers is ``min(4, nproc)``, overridable.

Why it matters: Docker rejects ``--cpus N`` when N exceeds the number of CPUs of
the machine, so a hard-coded 4 made every step of ``tools/run_ci_local.sh`` fail
at once (exit 125) on a 2-CPU CI runner. ``tools/ci_cpus.sh`` holds the rule once;
every script that starts a container takes its limit from it. ``nproc`` is shimmed
so that the tests do not depend on the machine they run on.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "tools" / "ci_cpus.sh"


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _cpus(tmp_path: Path, nproc: int, override: str | None = None):
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    _exe(shim / "nproc", f"echo {nproc}\n")
    env = {k: v for k, v in os.environ.items() if k != "PTH_CI_CPUS"}
    env["PATH"] = f"{shim}:{env['PATH']}"
    if override is not None:
        env["PTH_CI_CPUS"] = override
    return subprocess.run(
        ["bash", str(HELPER)], capture_output=True, text=True, env=env, timeout=30
    )


@pytest.mark.parametrize(
    ("nproc", "want"),
    [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (6, "4"), (24, "4"), (128, "4")],
)
def test_the_limit_is_min_4_nproc(tmp_path, nproc, want):
    """No override: min(4, nproc), at least 1 (a 2-CPU runner gets 2, not 4)."""
    r = _cpus(tmp_path, nproc)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == want


@pytest.mark.parametrize(("nproc", "override"), [(2, "8"), (64, "2"), (4, "1")])
def test_the_environment_variable_overrides(tmp_path, nproc, override):
    """PTH_CI_CPUS wins over nproc in both directions."""
    r = _cpus(tmp_path, nproc, override)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == override


@pytest.mark.parametrize("bad", ["0", "-1", "two", "1.5", "08x"])
def test_a_bad_override_is_refused(tmp_path, bad):
    """A malformed PTH_CI_CPUS fails loudly instead of reaching docker."""
    r = _cpus(tmp_path, 8, bad)
    assert r.returncode == 2
    assert "PTH_CI_CPUS must be a positive integer" in r.stderr
    assert r.stdout == ""


def test_no_script_hard_codes_a_cpu_limit_of_four():
    """Every container started by tools/ or bench/ takes its limit from the
    helper: no ``--cpus 4`` literal is left (run_grid.sh has its own --jobs
    rule)."""
    scripts = [
        p
        for d in ("tools", "bench")
        for p in (ROOT / d).rglob("*.sh")
        if p.name != "ci_cpus.sh"
    ]
    assert len(scripts) > 5, "the scan would be vacuous"
    bad = [
        p.name
        for p in scripts
        if re.search(r"--cpus\s+[0-9]", p.read_text(encoding="utf-8"))
    ]
    assert bad == []
    users = [p for p in scripts if "--cpus" in p.read_text(encoding="utf-8")]
    for p in users:
        text = p.read_text(encoding="utf-8")
        assert "ci_cpus.sh" in text, p.name


def test_build_in_image_uses_the_limit_for_docker_and_make_jobs(tmp_path):
    """tools/build_in_image.sh on a 2-CPU machine: ``--cpus 2`` and MAX_JOBS=2
    (the same number), and PTH_CI_CPUS changes both."""
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "build_in_image.sh", repo / "tools")
    shutil.copy(HELPER, repo / "tools")
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "calls.log"
    _exe(shim / "nproc", "echo 2\n")
    _exe(shim / "docker", f'printf "%s\\n" "$*" >>"{log}"\n')
    for override, want in ((None, "2"), ("3", "3")):
        log.write_text("")
        env = {k: v for k, v in os.environ.items() if k != "PTH_CI_CPUS"}
        env["PATH"] = f"{shim}:{env['PATH']}"
        if override:
            env["PTH_CI_CPUS"] = override
        r = subprocess.run(
            ["bash", str(repo / "tools" / "build_in_image.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert r.returncode == 0, r.stderr
        line = log.read_text()
        assert f"--cpus {want} " in line and f"MAX_JOBS={want} " in line
