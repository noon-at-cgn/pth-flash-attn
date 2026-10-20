# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the GPU grid run script and its report.

``tests/gpu/run_grid.sh`` runs once per result directory on an otherwise idle
GPU; a script that "passes" without testing anything is the failure to
prevent (the ceilings script once did: see
tests/cpu/test_ceilings_scripts.py). These tests pin down that it

* refuses (exit 65) before starting any container when the power limit
  cannot be read, a compute process exists on the GPU, nvidia-smi fails, the
  lock is held, an image is missing or the result directory exists; bad
  arguments exit 64;
* runs every step on the GPU under test only (``--gpus device=N``, offline),
  the sanitizer steps in the sanitizer image with torch's caching allocator
  off, and passes when every step passes (positive control);
* FAILS when a pytest step runs no tests, when a sanitizer misses its planted
  error, when the build produces nothing, or when the libraries change
  during the run;
* feeds no container on stdin without ``-i`` (the bug class of a silently
  empty container run);

and that ``tests/gpu/grid_report.py`` turns a results directory into a
schema-valid report whose verdict is ``fail`` for a missing case, a bound
violation, a canary hit or an empty step.

No GPU or docker is needed: ``docker``, ``nvidia-smi`` and ``git`` are shims
on PATH that log their calls.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
grid_report = importlib.import_module("grid_report")
ceilings_scripts = importlib.import_module("test_ceilings_scripts")

ARGS = ("--gpu", "1", "--date", "2026-10-21", "--jobs", "2")
RES = Path("results") / "2026-10-21-grid-220w-gpu1"
LIBS = (
    "build/ref/libpth_ref.so",
    "build/ref/libpth_ref_host.so",
    "build/loader/libpth_loader.so",
    "build/loader/libpth_loader_host.so",
)
SAN_EXIT = 86  # sanitizer --error-exitcode


def _write_exe(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# The docker shim plays every container run_grid.sh starts:
#   image inspect     fails if FAKE_NO_IMAGE
#   nvml logger (-d)  writes one line to its --out file
#   pytest            writes a JUnit file with FAKE_TESTS tests (default 1),
#                     exits FAKE_PYTEST_RC (default 0); sanitizer controls exit
#                     86 unless FAKE_CTL_BLIND; FAKE_NO_JUNIT writes none
#   grid_cases expected  modifies a library if FAKE_TAMPER (mid-run rebuild)
#   grid_report.py      writes a passing report and SUMMARY.txt
# The nvidia-smi shim answers the power-limit query with 220.00 (nothing if
# FAKE_NO_POWER), the compute-apps query with FAKE_APPS (exit 9 if
# FAKE_APPS_FAIL) and every other query with a line of text.
DOCKER_SHIM = r"""
printf '%s\n' "docker $*" >>"$FAKE_LOG"
args=" $* "
case "$1" in
  image) [[ -n "${FAKE_NO_IMAGE:-}" ]] && exit 1; echo sha256:fake; exit 0 ;;
  stop) exit 0 ;;
esac
out=""; junit=""; prev=""
for a in "$@"; do
  [[ "$prev" == "--out" ]] && out="$a"
  case "$a" in --junitxml=*) junit="${a#--junitxml=}" ;; esac
  prev="$a"
done
if [[ "$args" == *" -d "* && -n "$out" ]]; then
  echo '{"t": 0}' >"$FAKE_REPO/$out"; exit 0
fi
if [[ "$args" == *"grid_cases.py expected"* ]]; then
  [[ -n "${FAKE_TAMPER:-}" ]] && echo changed >>"$FAKE_REPO/build/ref/libpth_ref.so"
  exit 0
fi
if [[ "$args" == *"grid_report.py"* ]]; then
  res="${@: -1}"
  echo '{"verdict": "pass"}' >"$FAKE_REPO/$res/grid_report.json"
  echo "grid fake: PASS" >"$FAKE_REPO/$res/SUMMARY.txt"
  exit 0
fi
if [[ "$args" == *" pytest "* ]]; then
  n="${FAKE_TESTS:-1}"
  if [[ -z "${FAKE_NO_JUNIT:-}" && -n "$junit" ]]; then
    x="<testsuite name=\"pytest\" failures=\"0\" skipped=\"${FAKE_SKIPPED:-0}\""
    printf '<testsuites>%s tests="%s"/></testsuites>\n' "$x" "$n" >"$FAKE_REPO/$junit"
  fi
  if [[ "$args" == *"sanitizer_control"* ]]; then
    [[ -n "${FAKE_CTL_BLIND:-}" ]] && exit 0
    exit 86
  fi
  exit "${FAKE_PYTEST_RC:-0}"
fi
exit 0
"""

BUILD_SHIM = r"""
[[ -n "${FAKE_BUILD_FAIL:-}" ]] && exit 2
mkdir -p "$FAKE_REPO/build/ref" "$FAKE_REPO/build/loader"
for f in build/ref/libpth_ref.so build/ref/libpth_ref_host.so \
         build/loader/libpth_loader.so build/loader/libpth_loader_host.so; do
  echo "$f" >"$FAKE_REPO/$f"
done
"""


@pytest.fixture
def grid_repo(tmp_path):
    """A throwaway repo with run_grid.sh, fake build scripts and shims."""
    repo = tmp_path / "repo"
    (repo / "bench").mkdir(parents=True)
    (repo / "tests" / "gpu").mkdir(parents=True)
    (repo / "tools").mkdir()
    shutil.copy(
        ROOT / "tests" / "gpu" / "run_grid.sh", repo / "tests" / "gpu" / "run_grid.sh"
    )
    _write_exe(repo / "tools" / "build_ref_lib.sh", BUILD_SHIM)
    _write_exe(repo / "tools" / "build_loader_lib.sh", "exit 0\n")
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "calls.log"
    rec = f'printf "%s\\n" "$(basename "$0") $*" >>"{log}"\n'
    _write_exe(shim / "docker", DOCKER_SHIM)
    _write_exe(
        shim / "nvidia-smi",
        rec
        + 'case "$*" in\n'
        + '  *power.limit*) [[ -n "${FAKE_NO_POWER:-}" ]] && exit 9; echo 220.00 ;;\n'
        + '  *query-compute-apps*) [[ -n "${FAKE_APPS_FAIL:-}" ]] && exit 9\n'
        + '    printf "%s" "${FAKE_APPS:-}" ;;\n'
        + "  *) echo nvidia-smi fake ;;\n"
        + "esac\n",
    )
    _write_exe(shim / "git", rec + 'case "$*" in *rev-parse*) echo 0123abcd ;; esac\n')
    return repo, shim, log


def run_grid(grid_repo, args=ARGS, **env):
    repo, shim, log = grid_repo
    e = dict(
        os.environ,
        PATH=f"{shim}:{os.environ['PATH']}",
        FAKE_REPO=str(repo),
        FAKE_LOG=str(log),
    )
    e.update(env)
    return subprocess.run(
        ["bash", str(repo / "tests" / "gpu" / "run_grid.sh"), *args],
        capture_output=True,
        text=True,
        env=e,
        timeout=120,
    )


def steps(repo):
    rows = (repo / RES / "steps.tsv").read_text().splitlines()[1:]
    return {r.split("\t")[0]: r.split("\t")[1:] for r in rows}


# ---------------------------------------------------------------- static rule
def test_run_grid_feeds_no_container_on_stdin_without_i():
    """The empty-run bug class (``docker run`` reading a script from stdin without
    ``-i`` silently does nothing) cannot occur in run_grid.sh; the scan is not
    vacuous (it sees the script's docker runs)."""
    text = (ROOT / "tests" / "gpu" / "run_grid.sh").read_text()
    assert len(ceilings_scripts.docker_run_commands(text)) >= 4
    assert ceilings_scripts.stdin_without_interactive(text) == []


# ---------------------------------------------------------------- guards
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "--gpu N is required"),
        (("--gpu", "1", "--date", "21-10-2026"), "--date must be YYYY-MM-DD"),
        (("--gpu", "1", "--jobs", "0"), "--jobs must be a positive integer"),
        (("--gpu", "1", "--bogus"), "unknown argument"),
    ],
)
def test_run_grid_usage_errors(grid_repo, args, message):
    """Bad arguments stop with exit 64 before any query is made."""
    r = run_grid(grid_repo, args)
    assert r.returncode == 64 and message in r.stderr
    assert not grid_repo[2].exists()


@pytest.mark.parametrize(
    ("env", "result_exists", "message"),
    [
        ({"FAKE_NO_POWER": "1"}, False, "cannot read the power limit"),
        (
            {"FAKE_APPS": "00000000:04:00.0, 99, python, 900 MiB"},
            False,
            "compute processes",
        ),
        ({"FAKE_APPS_FAIL": "1"}, False, "nvidia-smi failed"),
        ({"FAKE_NO_IMAGE": "1"}, False, "not found"),
        ({}, True, "results are never overwritten"),
    ],
)
def test_run_grid_refuses_before_starting_anything(
    grid_repo, env, result_exists, message
):
    """Every guard fires with exit 65 and its own message; no container ran
    and no result directory was created (or the existing one is untouched)."""
    repo, _, log = grid_repo
    if result_exists:
        (repo / RES).mkdir(parents=True)
    r = run_grid(grid_repo, **env)
    assert r.returncode == 65, r.stdout + r.stderr
    assert "REFUSED" in r.stderr and message in r.stderr
    assert "docker run" not in (log.read_text() if log.exists() else "")
    assert (repo / RES).exists() == result_exists
    if result_exists:
        assert not any((repo / RES).iterdir())


def test_run_grid_refuses_when_the_lock_is_held(grid_repo, tmp_path):
    """--lock FILE: a second run on the same lock file stops (exit 65) before
    starting anything; without the option nothing is locked."""
    lock = tmp_path / "gpu.lock"
    with open(lock, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        r = run_grid(grid_repo, (*ARGS, "--lock", str(lock)))
    assert r.returncode == 65, r.stdout + r.stderr
    assert "another run holds the lock" in r.stderr
    log = grid_repo[2]
    assert "docker run" not in (log.read_text() if log.exists() else "")


# ---------------------------------------------------------------- pipeline
def test_run_grid_passes_when_every_step_passes(grid_repo):
    """Positive control: all steps ok -> exit 0, every step in steps.tsv with
    its expected exit, library sums recorded, manifest written; GPU
    containers see only the GPU under test, offline; sanitizer steps use the
    sanitizer image with torch's caching allocator off."""
    repo, _, log = grid_repo
    r = run_grid(grid_repo)
    assert r.returncode == 0, r.stdout + r.stderr
    st = steps(repo)
    assert list(st) == [
        "build",
        "expected",
        "quick",
        "full",
        "san-memcheck",
        "san-racecheck",
        "san-synccheck",
        "san-initcheck",
        "ctl-memcheck",
        "ctl-initcheck",
        "graph",
        "report",
    ]
    assert st["ctl-memcheck"][:2] == [str(SAN_EXIT), str(SAN_EXIT)]
    assert all(v[0] == v[1] for v in st.values())
    sums = (repo / RES / "build" / "SHA256SUMS").read_text()
    assert all(lib in sums for lib in LIBS)
    assert (repo / RES / "MANIFEST.sha256").exists()
    runs = [c for c in log.read_text().splitlines() if c.startswith("docker run")]
    gpu_runs = [c for c in runs if "--gpus" in c and " -d " not in c]
    assert gpu_runs and all(
        "--gpus device=1 " in c and "--network none" in c for c in gpu_runs
    )
    san = [c for c in runs if "compute-sanitizer" in c]
    assert len(san) == 6
    assert all(
        "pth-flash-attn-sanitizer" in c and "PYTORCH_NO_CUDA_MEMORY_CACHING=1" in c
        for c in san
    )
    assert all("--gpus all" not in c for c in runs if " -d " not in c)


def _docker_runs(log):
    return [c for c in log.read_text().splitlines() if c.startswith("docker run")]


@pytest.mark.parametrize(
    ("cores", "jobs"), [(64, 4), (24, 4), (16, 4), (8, 2), (3, 1), (1, 1)]
)
def test_run_grid_default_jobs_leave_the_host_usable(grid_repo, cores, jobs):
    """Without --jobs the CPU workers and the CPU quota are min(4, nproc / 4),
    at least 1.

    Why it matters: library threads multiply with workers (every worker of a
    16-worker pool starting one thread per core overloads a 24-core host), so the
    default must scale with the host and stay small.
    """
    repo, shim, log = grid_repo
    _write_exe(shim / "nproc", f"echo {cores}\n")
    r = run_grid(grid_repo, ("--gpu", "1", "--date", "2026-10-21"))
    assert r.returncode == 0, r.stdout + r.stderr
    runs = _docker_runs(log)
    cpu = [c for c in runs if "--cpus" in c]
    assert cpu and all(f"--cpus {jobs} " in c for c in cpu)
    expected = [c for c in runs if "grid_cases.py expected" in c]
    assert len(expected) == 1 and expected[0].endswith(f"--jobs {jobs}")


@pytest.mark.parametrize(
    ("env", "omp", "mkl", "blas"),
    [
        ({}, "1", "1", "1"),
        ({"OMP_NUM_THREADS": "3"}, "3", "1", "1"),
        ({"MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "5"}, "1", "2", "5"),
    ],
)
def test_run_grid_pins_library_threads_in_every_container(
    grid_repo, env, omp, mkl, blas
):
    """Every container the script starts (but the detached NVML logger) gets one
    thread per numerical library unless the caller set the variable.

    Why it matters: one thread per core in each of several workers multiplies the
    load by the core count.
    """
    base = {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": "", "OPENBLAS_NUM_THREADS": ""}
    r = run_grid(grid_repo, **{**base, **env})
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [c for c in _docker_runs(grid_repo[2]) if " -d " not in c]
    assert len(runs) >= 10
    want = (
        f"-e OMP_NUM_THREADS={omp} -e MKL_NUM_THREADS={mkl} "
        f"-e OPENBLAS_NUM_THREADS={blas} "
    )
    assert all(want in c for c in runs), [c for c in runs if want not in c][:1]


@pytest.mark.parametrize(
    ("env", "bad_step", "message"),
    [
        ({"FAKE_TESTS": "0", "FAKE_PYTEST_RC": "5"}, "quick", "quick exited 5"),
        ({"FAKE_TESTS": "0"}, "quick", "quick ran no tests"),
        ({"FAKE_NO_JUNIT": "1", "FAKE_PYTEST_RC": "5"}, "full", "full exited 5"),
        ({"FAKE_CTL_BLIND": "1"}, "ctl-memcheck", "exited 0 (expected 86)"),
        ({"FAKE_TAMPER": "1"}, "quick", "quick exited 3"),
        ({"FAKE_TESTS": "4", "FAKE_SKIPPED": "4"}, "quick", "quick ran no tests"),
    ],
)
def test_run_grid_fails_when_nothing_was_tested(grid_repo, env, bad_step, message):
    """No tests, a blind sanitizer or libraries changed mid-run -> exit 1
    with the step named, never a pass."""
    repo, _, _ = grid_repo
    r = run_grid(grid_repo, **env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "GRID FAILED" in r.stdout and bad_step in r.stdout
    assert message in r.stdout


def test_run_grid_stops_when_the_build_fails(grid_repo):
    """A failing build stops the run before any GPU step (exit 1)."""
    repo, _, log = grid_repo
    r = run_grid(grid_repo, FAKE_BUILD_FAIL="1")
    assert r.returncode == 1 and "produced no libraries" in r.stderr
    assert " -m pytest " not in log.read_text()  # (tmp paths contain "pytest")


# ---------------------------------------------------------------- report
GRID = [("a", "paging"), ("b", "paging"), ("c", "splits")]


def _grid_rec(case, *, passed=True, violations=0, tightness=0.01, canary=True):
    return {
        "test": "reference_grid",
        "case": case,
        "passed": passed,
        "violations": violations,
        "nonfinite": 0,
        "tightness": tightness,
        "canary_out_ok": canary,
        "canary_ws_ok": None,
        "worst": {"index": [0, 0, 0]},
    }


def _results(tmp_path, recs, steps_rows=None):
    res = tmp_path / "2026-10-21-grid-220w-gpu1"
    (res / "records").mkdir(parents=True)
    rows = steps_rows or [
        ("build", 0, 0, "-"),
        ("quick", 0, 0, 3),
        ("ctl-memcheck", 86, 86, 1),
    ]
    (res / "steps.tsv").write_text(
        "step\texit\texpect\tseconds\ttests\n"
        + "".join(f"{n}\t{rc}\t{ex}\t5\t{t}\n" for n, rc, ex, t in rows)
    )
    (res / "records" / "quick.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs)
    )
    return res


def test_report_passes_a_complete_clean_run(tmp_path):
    """Positive control: all grid cases recorded and passed, all steps as
    expected (a sanitizer control's 86 counts as expected) -> pass, and the
    JSON is valid against the schema and strict JSON (no NaN/Infinity)."""
    res = _results(
        tmp_path,
        [_grid_rec(n) for n, _ in GRID]
        + [{"test": "loader", "case": "x", "passed": True}],
    )
    rep = grid_report.build_report(res, GRID)
    assert rep["verdict"] == "pass", rep["reasons"]
    assert rep["grid"]["cases_passed"] == 3
    assert rep["other_tests"]["loader"] == {"recorded": 1, "failed": []}
    json.dumps(rep, allow_nan=False)


@pytest.mark.parametrize(
    ("recs", "rows", "reason"),
    [
        ([_grid_rec("a"), _grid_rec("b")], None, "1 of 3 grid cases have no record"),
        (
            [
                _grid_rec("a"),
                _grid_rec("b", passed=False, violations=4, tightness=7.0),
                _grid_rec("c"),
            ],
            None,
            "1 grid cases failed",
        ),
        (
            [
                _grid_rec("a"),
                _grid_rec("b", passed=False, canary=False),
                _grid_rec("c"),
            ],
            None,
            "1 grid cases failed",
        ),
        ([_grid_rec(n) for n, _ in GRID], [("quick", 5, 0, 0)], "quick ran no tests"),
        (
            [_grid_rec(n) for n, _ in GRID],
            [("ctl-memcheck", 0, 86, 1)],
            "ctl-memcheck exited 0",
        ),
        (
            [_grid_rec(n) for n, _ in GRID]
            + [{"test": "graph", "case": "g", "passed": False}],
            None,
            "1 graph records failed",
        ),
    ],
    ids=["missing", "violation", "canary", "no-tests", "blind-sanitizer", "graph"],
)
def test_report_fails_for_every_kind_of_problem(tmp_path, recs, rows, reason):
    """Each failure kind turns the verdict to fail with its own reason."""
    rep = grid_report.build_report(_results(tmp_path, recs, rows), GRID)
    assert rep["verdict"] == "fail"
    assert any(reason in r for r in rep["reasons"]), rep["reasons"]


def test_report_keeps_inf_tightness_as_text(tmp_path, monkeypatch):
    """A NaN output gives tightness inf; the report and SUMMARY.txt keep it
    visible ("inf") instead of writing invalid JSON, and main() exits 1."""
    recs = [
        _grid_rec("a"),
        _grid_rec("b", passed=False, violations=1, tightness=float("inf")),
        _grid_rec("c"),
    ]
    res = _results(tmp_path, recs)
    monkeypatch.setattr(grid_report, "expected_grid_cases", lambda: GRID)
    assert grid_report.main([str(res)]) == 1
    rep = json.loads((res / "grid_report.json").read_text())
    assert rep["grid"]["tightness"]["max"] == "inf"
    assert "FAILED b" in (res / "SUMMARY.txt").read_text()


def test_report_refuses_a_directory_without_steps(tmp_path, monkeypatch):
    """No steps.tsv (the run never got going) is exit 2, not a report."""
    monkeypatch.setattr(grid_report, "expected_grid_cases", lambda: GRID)
    assert grid_report.main([str(tmp_path)]) == 2


def test_report_counts_sanitizer_errors_once(tmp_path):
    """compute-sanitizer prints "ERROR SUMMARY: 66 errors" and, when it hides
    some, "ERROR SUMMARY: 63 errors were not printed"; only the first is the
    count. Racecheck's summary line has its own format."""
    res = _results(
        tmp_path,
        [_grid_rec(n) for n, _ in GRID],
        [("ctl-memcheck", 86, 86, 1), ("san-racecheck", 0, 0, 10)],
    )
    (res / "logs").mkdir()
    (res / "logs" / "ctl-memcheck.log").write_text(
        "========= ERROR SUMMARY: 66 errors\n"
        "========= ERROR SUMMARY: 63 errors were not printed. Use --print-limit\n"
    )
    (res / "logs" / "san-racecheck.log").write_text(
        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
    )
    st = {s["name"]: s for s in grid_report.build_report(res, GRID)["steps"]}
    assert st["ctl-memcheck"]["sanitizer_errors"] == 66
    assert st["san-racecheck"]["sanitizer_errors"] == 0
