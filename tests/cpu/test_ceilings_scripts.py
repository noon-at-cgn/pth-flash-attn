# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the hardware-ceiling scripts (bench/ceilings_build.sh,
ceilings_run.sh).

These scripts run for hours on a GPU that must be otherwise idle. A script that
does nothing and exits 0 is the worst kind of failure: the operator believes
the ceilings were built or measured. The shipped failure mode: `ceilings_build.sh`
piped its in-container build script into `docker run ... bash -s` without `-i`,
docker never attached stdin, bash read EOF and exited 0, and nothing was built.
These tests pin down the rules that prevent the whole class:

* every `docker run` that feeds the container on stdin passes `-i` (static
  scan of all shell scripts under bench/ and tools/, with negative controls);
* `ceilings_build.sh` fails loudly (exit 2) if the container produced no fresh
  SHA256SUMS, and propagates a failing build's exit code;
* `ceilings_run.sh` refuses (exit 65) before touching anything when a compute
  process exists on the GPU, the lock file given with `--lock` is held,
  nvidia-smi fails, the binaries are missing or modified, or the result
  directory exists; bad arguments exit 64;
* the result directory name carries the power limit in watts.

No GPU and no docker: `docker` and `nvidia-smi` are replaced by shims on PATH
that record their calls.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRS = ("bench", "tools")

# ---------------------------------------------------------------- static rule
_CONT = re.compile(r"\\\n")
# Input on stdin: a heredoc or herestring (<<, <<-, <<<) or a `< file`
# redirection, or a shell told to read its script from stdin (`bash -s`).
_STDIN = re.compile(r"(?<![<>&0-9])<(?![(&])|\s-s(\s|$)")
_INTERACTIVE = re.compile(r"\s(-i|--interactive|-[a-zA-Z]*i[a-zA-Z]*)(\s|=|$)")


def docker_run_commands(text: str) -> list[str]:
    """Logical `docker run` commands (backslash continuations joined)."""
    joined = _CONT.sub(" ", text)
    return [
        line
        for line in joined.splitlines()
        if re.search(r"\bdocker\s+run\b", line) and not line.lstrip().startswith("#")
    ]


def stdin_without_interactive(text: str) -> list[str]:
    """`docker run` commands that feed stdin but do not pass -i."""
    bad = []
    for cmd in docker_run_commands(text):
        run_part = cmd[cmd.index("docker") :]
        # Only the part up to the image matters for docker's own flags; the
        # whole command is checked for stdin feeding.
        if _STDIN.search(run_part) and not _INTERACTIVE.search(
            run_part.split(" pth-flash-attn-dev")[0].split('"${image}"')[0]
        ):
            bad.append(cmd.strip())
    return bad


def test_stdin_rule_negative_controls():
    """The scanner must flag the exact shipped bug and accept its fix, or a
    green result would mean nothing."""
    shipped_bug = (
        "docker run --rm --network none \\\n"
        '  --entrypoint /bin/bash "${image}" -s -- "${out}" <<\'IN\'\necho hi\nIN\n'
    )
    fixed = shipped_bug.replace("docker run --rm", "docker run -i --rm")
    herestring = 'docker run --rm img python - <<<"print(1)"\n'
    no_stdin = 'docker run --rm --entrypoint /bin/bash img -c "echo $x" 2>&1\n'
    assert stdin_without_interactive(shipped_bug)
    assert not stdin_without_interactive(fixed)
    assert stdin_without_interactive(herestring)
    assert not stdin_without_interactive(no_stdin)
    assert not stdin_without_interactive("docker run -it --rm img bash -s <<X\nX\n")


def test_every_docker_run_that_feeds_stdin_passes_i():
    """Repository rule: no `docker run` in bench/ or tools/ may feed a script
    on stdin without -i (it would be silently dropped)."""
    scripts = sorted(p for d in SCRIPT_DIRS for p in (ROOT / d).rglob("*.sh"))
    assert scripts, "no shell scripts found: the scan would be vacuous"
    offenders = {
        str(p.relative_to(ROOT)): stdin_without_interactive(p.read_text())
        for p in scripts
    }
    assert {k: v for k, v in offenders.items() if v} == {}


# ---------------------------------------------------------------- shims
def _write_exe(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_repo(tmp_path):
    """A throwaway repo root holding copies of the scripts, plus a shim
    directory whose tools log every call to calls.log."""
    repo = tmp_path / "repo"
    (repo / "bench").mkdir(parents=True)
    for name in ("ceilings_build.sh", "ceilings_run.sh", "lib.sh"):
        shutil.copy(ROOT / "bench" / name, repo / "bench" / name)
    (repo / "tools").mkdir()
    shutil.copy(ROOT / "tools" / "ci_cpus.sh", repo / "tools" / "ci_cpus.sh")
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "calls.log"
    rec = f'printf "%s\\n" "$(basename "$0") $*" >>"{log}"\n'
    # docker: FAKE_DOCKER_MODE=noop (exit 0, do nothing), fail (exit 7) or
    # build (write the SHA256SUMS a real build writes). Records whether -i was
    # passed and whether anything arrived on stdin.
    _write_exe(
        shim / "docker",
        rec
        + 'case " $* " in *" -i "*) cat >/dev/null; echo "stdin-attached" >>"'
        + str(log)
        + '";; esac\n'
        + 'case "${FAKE_DOCKER_MODE:-noop}" in\n'
        + "  fail) exit 7 ;;\n"
        + '  build) mkdir -p "$FAKE_REPO/build/ceilings"'
        + ' && echo x >"$FAKE_REPO/build/ceilings/SHA256SUMS" ;;\n'
        + "esac\nexit 0\n",
    )
    _write_exe(
        shim / "nvidia-smi",
        rec + '[[ -n "${FAKE_SMI_FAIL:-}" ]] && exit 9\n'
        # 220 W: part of the result name
        'case "$*" in\n'
        '  *power.limit*) echo "${FAKE_POWER-220.00}" ;;\n'
        '  *) printf "%s" "${FAKE_APPS:-}" ;;\n'
        "esac\n",
    )
    _write_exe(shim / "systemctl", rec + '[[ -n "${FAKE_UNIT_ACTIVE:-}" ]]\n')
    _write_exe(shim / "curl", rec + '[[ -n "${FAKE_HEALTH:-}" ]]\n')
    return repo, shim, log


def run_script(repo, shim, script, args=(), **env):
    e = dict(os.environ, PATH=f"{shim}:{os.environ['PATH']}", FAKE_REPO=str(repo))
    e.update(env)
    return subprocess.run(
        ["bash", str(repo / "bench" / script), *args],
        capture_output=True,
        text=True,
        env=e,
        timeout=60,
    )


# ---------------------------------------------------------------- ceilings_build.sh
def test_ceilings_build_attaches_stdin(fake_repo):
    """The build container must receive its script: `-i` is passed and the
    heredoc arrives on the shim's stdin."""
    repo, shim, log = fake_repo
    run_script(repo, shim, "ceilings_build.sh", FAKE_DOCKER_MODE="build")
    calls = log.read_text()
    assert "docker run -i " in calls
    assert "stdin-attached" in calls


def test_ceilings_build_noop_container_fails_loudly(fake_repo):
    """A container that exits 0 without building (the shipped bug's symptom)
    must make ceilings_build.sh exit 2 with a message, never 0."""
    repo, shim, _ = fake_repo
    r = run_script(repo, shim, "ceilings_build.sh", FAKE_DOCKER_MODE="noop")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no fresh build/ceilings/SHA256SUMS" in r.stderr


def test_ceilings_build_stale_sums_do_not_count(fake_repo):
    """SHA256SUMS left by an earlier build must not make a no-op look like a
    success: only a file newer than this run's start counts."""
    repo, shim, _ = fake_repo
    (repo / "build" / "ceilings").mkdir(parents=True)
    old = repo / "build" / "ceilings" / "SHA256SUMS"
    old.write_text("old\n")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    r = run_script(repo, shim, "ceilings_build.sh", FAKE_DOCKER_MODE="noop")
    assert r.returncode == 2


def test_ceilings_build_success_and_failure_propagate(fake_repo):
    """Negative controls for the guard: a container that writes fresh sums
    passes (exit 0); a failing container's exit code is propagated."""
    repo, shim, _ = fake_repo
    assert (
        run_script(repo, shim, "ceilings_build.sh", FAKE_DOCKER_MODE="build").returncode
        == 0
    )
    r = run_script(repo, shim, "ceilings_build.sh", FAKE_DOCKER_MODE="fail")
    assert r.returncode == 7 and "failed (exit 7)" in r.stderr


# ------------------------------------------------------------ ceilings_run.sh checks
@pytest.fixture
def run_ready(fake_repo):
    """A fake repo where every ceilings_run.sh check would pass: binaries present
    and matching their sums."""
    repo, shim, log = fake_repo
    build = repo / "build" / "ceilings"
    build.mkdir(parents=True)
    for b in ("sol_copy", "mma_ceiling"):
        (build / b).write_text(b)
    subprocess.run(
        "sha256sum sol_copy mma_ceiling > SHA256SUMS", shell=True, cwd=build, check=True
    )
    return repo, shim, log


GUARD_ARGS = ("--gpu", "0", "--date", "2026-10-20")
# The result name carries the GPU's power limit in watts; the fake nvidia-smi
# answers 220 W for power.limit.
RES_NAME = "2026-10-20-ceilings-220w-gpu0"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "--gpu N is required"),
        (("--gpu", "0", "--date", "20261020"), "--date must be YYYY-MM-DD"),
        (("--gpu", "0", "--timer", "wallclock"), "--timer must be cupti or events"),
        (("--gpu", "0", "--bogus"), "unknown argument"),
    ],
)
def test_ceilings_run_usage_errors(run_ready, args, message):
    """Bad arguments stop with exit 64 before any query is made."""
    repo, shim, log = run_ready
    r = run_script(repo, shim, "ceilings_run.sh", args)
    assert r.returncode == 64 and message in r.stderr
    assert not log.exists()


@pytest.mark.parametrize(
    ("env", "prepare", "message"),
    [
        (
            {"FAKE_APPS": "00000000:01:00.0, 99, python, 900 MiB"},
            None,
            "compute processes",
        ),
        ({"FAKE_SMI_FAIL": "1"}, None, "cannot read the power limit"),
        ({"FAKE_POWER": ""}, None, "cannot read the power limit"),
        ({}, "lock", "another run holds the lock"),
        ({}, "no_sums", "no build/ceilings/SHA256SUMS"),
        ({}, "tamper", "differ from build/ceilings/SHA256SUMS"),
        ({}, "result_exists", "results are never overwritten"),
    ],
)
def test_ceilings_run_refuses_before_touching_anything(
    run_ready, tmp_path, env, prepare, message
):
    """Every safety check fires with exit 65 and its own message, and when it
    does, no container was started and no result directory was created."""
    repo, shim, log = run_ready
    build = repo / "build" / "ceilings"
    res = repo / "results" / RES_NAME
    args = GUARD_ARGS
    holder = None
    if prepare == "no_sums":
        (build / "SHA256SUMS").unlink()
    elif prepare == "tamper":
        (build / "sol_copy").write_text("modified")
    elif prepare == "result_exists":
        res.mkdir(parents=True)
    elif prepare == "lock":
        lock = tmp_path / "gpu0.lock"
        holder = open(lock, "w")  # noqa: SIM115 (closed below)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = (*GUARD_ARGS, "--lock", str(lock))
    try:
        r = run_script(repo, shim, "ceilings_run.sh", args, **env)
    finally:
        if holder is not None:
            holder.close()
    assert r.returncode == 65, r.stdout + r.stderr
    assert "REFUSED" in r.stderr and message in r.stderr
    calls = log.read_text() if log.exists() else ""
    assert "docker run" not in calls
    assert res.exists() == (prepare == "result_exists")
    if prepare == "result_exists":
        assert not any(res.iterdir())


def test_ceilings_run_passes_checks_when_all_clear(run_ready):
    """Negative control for the checks: with no compute process and good
    binaries, ceilings_run.sh gets past them, creates the result directory
    (named with the power limit) and starts capturing (it then fails on the
    inert shims, which is expected). Nothing is locked without --lock."""
    repo, shim, _ = run_ready
    r = run_script(repo, shim, "ceilings_run.sh", GUARD_ARGS)
    assert "REFUSED" not in r.stderr
    assert (repo / "results" / RES_NAME / "env").is_dir()


def test_ceilings_run_help_explains_inputs_outputs_and_the_gpu_requirement(run_ready):
    repo, shim, log = run_ready
    r = run_script(repo, shim, "ceilings_run.sh", ("--help",))
    assert r.returncode == 0 and not log.exists()
    assert "Needs: one NVIDIA GPU" in r.stdout and "--lock FILE" in r.stdout
    assert "Output: results/<date>-ceilings-<W>w-gpu<N>/" in r.stdout
