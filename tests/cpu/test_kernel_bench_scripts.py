# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the kernel benchmark scripts (bench/k_run.sh,
bench/k_build.sh, bench/k_sweep.sh) and the shared bench/lib.sh.

k_run.sh produces the numbers the kernel gates are decided on. What must never
happen, and is tested here with shims in place of docker, nvidia-smi, git and
python3 (no GPU, no containers):

* starting while any process uses the GPU under test, while another run holds
  the lock file given with ``--lock``, with unbuilt or modified binaries, or
  over an existing results directory (exit 65, no container started, nothing
  written); bad arguments exit 64; without ``--lock`` nothing is locked;
* touching the other GPUs: every container gets ``--gpus device=N`` only,
  including the NVML logger;
* naming a result directory without the GPU's power limit in watts;
* keeping timings taken while a foreign process used the GPU: the result
  directory is moved to ``results/aborted-*`` and the run fails;
* passing silently: a failing kernel process, a gate refusal or a gate that
  did not pass make the exit code non-zero; reruns of short processes go to
  fresh processes with ``--only`` and a new attempt number;
* feeding a container a script on stdin without ``-i`` (a silent no-op);

and that k_build.sh fails loudly (exit 2) when a step fails or the kernel
library's build script does not exist, never leaving a SHA256SUMS that
looks like a fresh build.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
ceilings_scripts = importlib.import_module("test_ceilings_scripts")

ARGS = ("--gpu", "1", "--date", "2026-10-22", "--processes", "2")
# The result name carries the GPU's power limit in watts: the fake nvidia-smi
# below answers 220 W.
POWER_W = 220
RES = Path("results") / f"2026-10-22-kernel-{POWER_W}w-gpu1"


def _exe(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# Plays every container the scripts start (dispatch on the arguments).
DOCKER = r"""
printf '%s\n' "docker $*" >>"$FAKE_LOG"
args=" $* "
case "$1" in
  image)
    [[ -n "${FAKE_NO_IMAGE:-}" ]] && exit 1
    echo "sha256:$(printf '0%.0s' {1..64})"; exit 0 ;;
  stop) exit 0 ;;
esac
val() {  # value after the argument named $K
  local prev="" a
  for a in "$@"; do
    [[ "$prev" == "$K" ]] && { echo "$a"; return; }
    prev="$a"
  done
}
case "$args" in
  *" -d "*)
    K=--out; out="$(val "$@")"; echo '{"kind":"header"}' >"$FAKE_REPO/$out"; exit 0 ;;
  *"--device-info"*) echo '{"device": {"sm_count": 68}}'; exit 0 ;;
  *"nvcc"*"--version"*) echo "Cuda compilation tools, release 13.0, V13.0.88"; exit 0 ;;
  *"collect_env.sh"*) echo env; exit 0 ;;
  *"kernel_bench.py list"*"--copy-ceilings"*) echo "1 65536"; exit 0 ;;
  *"kernel_bench.py list"*)
    [[ -n "${FAKE_LIST_RC:-}" ]] && exit "$FAKE_LIST_RC"
    printf 'dec-a decode quick 0\ndec-b decode quick 0\n'; exit 0 ;;
  *"k_sweep.py list"*)
    printf 'swp-S-a S planner,planner#aa,S9\nswp-BQ-b BQ planner,planner#aa,wm2\n'
    exit 0 ;;
  *"sweep_table.py"*) echo "table"; exit 0 ;;
  *"stats.py schedule"*) echo 0; exit 0 ;;
  *" -c "*) echo 34; exit 0 ;;
  *"kernel_bench.py run"* | *"k_sweep.py run"*)
    # someone rebuilding a hashed file while the run is in progress
    t="$FAKE_REPO/build/k/libpth_cupti_timer.so"
    [[ -n "${FAKE_CHANGE_BUILD:-}" ]] && echo x >>"$t"
    exit "${FAKE_KB_RC:-0}" ;;
  *"/w/build/k/sol_copy"* | *"/w/build/k/mma_ceiling"*) exit 0 ;;
  *"bench_report.py domain"*) echo mem; exit 0 ;;
  *"bench_report.py status"*) exit "${FAKE_STATUS:-0}" ;;
  *"bench_report.py report"*)
    K=--experiment; exp="$(val "$@")"; K=--out-dir; out="$(val "$@")"
    f=0; [[ "$exp" == "${FAKE_FOREIGN:-none}" ]] && f=2
    # a tiny foreign context: the blocks are rejected, the record is not void
    fv=""; [[ "$exp" == "${FAKE_FOREIGN_SMALL:-none}" ]] && {
      f=1
      fv=', "foreign": {"blocks": 1, "large_blocks": 0, "worst_fraction": 0.03'
      fv="$fv"', "void": false}'; }
    rej='"rejections": {"by_reason": {"foreign_process": %s}%s}'
    blk='"samples": {"blocks": [{"nvml": {"max_compute_procs": 1}}]}'
    printf "{$rej, $blk}\n" "$f" "$fv" >"$FAKE_REPO/$out/${exp}__kernel.json"
    echo "report $exp"; exit 0 ;;
  *"kernel_gate.py"*)
    res="${@: -1}"; echo '{}' >"$FAKE_REPO/$res/k_gates.json"; echo "gates"
    exit "${FAKE_GATE_RC:-0}" ;;
esac
exit 0
"""


@pytest.fixture
def krepo(tmp_path):
    """A fake repo ready for k_run.sh: built files with matching sums."""
    repo = tmp_path / "repo"
    (repo / "bench").mkdir(parents=True)
    scripts = ("k_run.sh", "k_build.sh", "lib.sh", "k_sweep.sh")
    for script in scripts:
        shutil.copy(ROOT / "bench" / script, repo / "bench" / script)
    (repo / "tools").mkdir()
    shutil.copy(ROOT / "tools" / "ci_cpus.sh", repo / "tools" / "ci_cpus.sh")
    k = repo / "build" / "k"
    k.mkdir(parents=True)
    (k / "libpth_cupti_timer.so").write_text("timer")
    subprocess.run(
        "sha256sum build/k/libpth_cupti_timer.so > build/k/SHA256SUMS",
        shell=True,
        cwd=repo,
        check=True,
    )
    (k / "KERNEL").write_text(f"kernel attn\nsource {'0' * 40}\n")
    shim = tmp_path / "shim"
    shim.mkdir()
    log = tmp_path / "calls.log"
    rec = f'printf "%s\\n" "$(basename "$0") $*" >>"{log}"\n'
    _exe(shim / "docker", DOCKER)
    _exe(
        shim / "nvidia-smi",
        rec + '[[ -n "${FAKE_SMI_FAIL:-}" ]] && exit 9\n'
        'case "$*" in *power.limit*) echo "${FAKE_POWER-220.00}" ;;\n'
        "  *driver_version*) echo 580 ;;\n"
        '  *compute-apps*) printf "%s" "${FAKE_APPS:-}" ;; esac\n',
    )
    # git: `rev-parse` gives a fixed sha (FAKE_BAD_COMMIT: unknown revision);
    # `archive` packs tmp_path/src, the fake tree of the kernel commit.
    src = tmp_path / "src"
    src.mkdir()
    _exe(
        shim / "git",
        rec + 'case "$*" in\n'
        '  *rev-parse*) [[ -n "${FAKE_BAD_COMMIT:-}" ]] && exit 128\n'
        '    printf "%040d\\n" 0 ;;\n'
        f'  *archive*) tar -c -C "{src}" . ;;\n'
        "esac\n",
    )
    _exe(shim / "python3", f'exec "{sys.executable}" "$@"\n')
    return repo, shim, log


def run(krepo, args=ARGS, script="k_run.sh", **env):
    repo, shim, log = krepo
    e = dict(
        os.environ,
        PATH=f"{shim}:{os.environ['PATH']}",
        FAKE_REPO=str(repo),
        FAKE_LOG=str(log),
    )
    e.update(env)
    return subprocess.run(
        ["bash", str(repo / "bench" / script), *args],
        capture_output=True,
        text=True,
        env=e,
        timeout=120,
    )


def calls(krepo) -> list[str]:
    log = krepo[2]
    return log.read_text().splitlines() if log.exists() else []


# ---------------------------------------------------------------- static rule
@pytest.mark.parametrize("script", ["lib.sh", "k_build.sh"])
def test_no_container_is_fed_stdin_without_i(script):
    """The silent no-op (docker run reading a script from stdin without -i)
    cannot occur; the scan sees the script's docker runs. The run scripts
    start their containers through lib.sh."""
    text = (ROOT / "bench" / script).read_text()
    assert ceilings_scripts.docker_run_commands(text)
    assert ceilings_scripts.stdin_without_interactive(text) == []


@pytest.mark.parametrize("script", ["k_run.sh", "k_sweep.sh"])
def test_run_scripts_start_no_container_themselves(script):
    """Every container goes through in_gpu / in_cpu, which check the build
    hashes around it and pin the GPU: a bare `docker run` in a run script
    would bypass both."""
    text = (ROOT / "bench" / script).read_text()
    assert ceilings_scripts.docker_run_commands(text) == []


# ---------------------------------------------------------------- usage / guards
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "--gpu N is required"),
        (("--gpu", "1", "--date", "22-10-2026"), "--date must be YYYY-MM-DD"),
        (("--gpu", "1", "--tier", "all"), "--tier must be quick or full"),
        (("--gpu", "1", "--family", "x"), "--family must be"),
        (("--gpu", "1", "--processes", "1"), "--processes must be 2..29"),
        (("--gpu", "1", "--tag", "a/b"), "--tag may only use"),
        (("--gpu", "1", "--session", "my-run"), "unknown argument"),
        (("--gpu", "1", "--bogus"), "unknown argument"),
        (("--gpu", "1", "--only", "a b"), "--only takes comma-separated"),
    ],
)
def test_usage_errors(krepo, args, message):
    """Bad arguments exit 64 before any query."""
    r = run(krepo, args)
    assert r.returncode == 64 and message in r.stderr
    assert calls(krepo) == []


@pytest.mark.parametrize("script", ["k_run.sh", "k_sweep.sh", "k_build.sh"])
def test_help_explains_inputs_outputs_and_the_gpu_requirement(krepo, script):
    """--help prints the header: what the script needs (GPU, docker, image),
    what it writes and its exit codes, without starting anything."""
    r = run(krepo, ("--help",), script)
    assert r.returncode == 0 and calls(krepo) == []
    assert "Needs:" in r.stdout and "Exit" in r.stdout
    if script != "k_build.sh":
        assert "NVIDIA GPU" in r.stdout and "--lock FILE" in r.stdout
    else:
        assert "No GPU is needed" in r.stdout


@pytest.mark.parametrize(
    ("env", "prepare", "message"),
    [
        ({"FAKE_APPS": "77, python, 900 MiB"}, None, "compute processes on GPU 1"),
        ({"FAKE_SMI_FAIL": "1"}, None, "cannot read the power limit of GPU 1"),
        ({"FAKE_POWER": ""}, None, "cannot read the power limit of GPU 1"),
        ({}, "lock", "another run holds the lock"),
        ({}, "no_sums", "no build/k/SHA256SUMS"),
        ({}, "tamper", "differ from build/k/SHA256SUMS"),
        ({}, "no_kernel", "which kernel was built"),
        ({"FAKE_NO_IMAGE": "1"}, None, "not found"),
        ({}, "result_exists", "results are never overwritten"),
    ],
)
def test_refuses_before_starting_anything(krepo, tmp_path, env, prepare, message):
    """Every check exits 65 with its own message; no container ran and no
    results directory was created (an existing one stays untouched)."""
    repo = krepo[0]
    holder = None
    args = ARGS
    if prepare == "lock":
        lock = tmp_path / "gpu1.lock"
        holder = open(lock, "w")  # noqa: SIM115 (closed below)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = (*ARGS, "--lock", str(lock))
    elif prepare == "no_sums":
        (repo / "build/k/SHA256SUMS").unlink()
    elif prepare == "tamper":
        (repo / "build/k/libpth_cupti_timer.so").write_text("changed")
    elif prepare == "no_kernel":
        (repo / "build/k/KERNEL").unlink()
    elif prepare == "result_exists":
        (repo / RES).mkdir(parents=True)
    try:
        r = run(krepo, args, **env)
    finally:
        if holder is not None:
            holder.close()
    assert r.returncode == 65, r.stdout + r.stderr
    assert "REFUSED" in r.stderr and message in r.stderr
    assert not any(c.startswith("docker run") for c in calls(krepo))
    assert (repo / RES).exists() == (prepare == "result_exists")
    if prepare == "result_exists":
        assert not any((repo / RES).iterdir())


def test_a_held_lock_file_is_ignored_without_the_lock_option(krepo, tmp_path):
    """Nothing is locked by default: a file that another process holds
    exclusively blocks the run only when it is named with --lock."""
    lock = tmp_path / "gpu1.lock"
    with open(lock, "w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run(krepo, ARGS)
        assert r.returncode == 0, r.stdout + r.stderr
        shutil.rmtree(krepo[0] / RES)
        r = run(krepo, (*ARGS, "--lock", str(lock)))
        assert r.returncode == 65 and "another run holds the lock" in r.stderr


def test_a_free_lock_file_lets_the_run_through(krepo, tmp_path):
    r = run(krepo, (*ARGS, "--lock", str(tmp_path / "free.lock")))
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- flow
def test_full_flow_on_one_gpu(krepo):
    """Positive control: ceilings (copy for each decode (B, L), then mma),
    P kernel processes, one report per experiment with cluster-robust CIs
    and the A/A pair, the gates, a manifest; every container on GPU 1 only,
    the run name (with the power limit) as session id in env.json; exit 0
    when the gates pass."""
    repo = krepo[0]
    r = run(krepo)
    assert r.returncode == 0, r.stdout + r.stderr
    cs = calls(krepo)
    runs = [c for c in cs if c.startswith("docker run")]
    gpu = [c for c in runs if "--gpus" in c]
    assert gpu and all("--gpus device=1 " in c and "--network none" in c for c in gpu)
    assert not any("--gpus all" in c for c in runs)
    kb_runs = [c for c in runs if "kernel_bench.py run" in c]
    assert len(kb_runs) == 2 and all("--kernel-kind attn" in c for c in kb_runs)
    # the commit-built copy, never the working-tree build/attn
    assert all("--kernel-lib /w/build/k/libpth_attn.so" in c for c in kb_runs)
    reports = [c for c in runs if "bench_report.py report" in c]
    exps = [c.split("--experiment ")[1].split()[0] for c in reports]
    assert exps == ["ceil-copy-b1-L64k", "ceil-mma", "dec-a", "dec-b"]
    assert all("--cluster-robust" in c for c in reports)
    assert all("--aa kernel kernel#aa" in c for c in reports[2:])
    assert "--clock-domain mem" in reports[0] and "--clock-domain both" in reports[1]
    env = json.loads((repo / RES / "env.json").read_text())
    assert env["session"] == RES.name == "2026-10-22-kernel-220w-gpu1"
    assert env["kernel"] == "attn" and env["power_limit_w"] == POWER_W
    assert set(env) == {
        "git_sha",
        "git_dirty",
        "image_digest",
        "nvcc_version",
        "driver_version",
        "power_limit_w",
        "captures",
        "session",
        "kernel",
        "kernel_source",
    }
    assert len(env["captures"]) == 3
    # env.json feeds every record's env.captures: each item must already be
    # schema-valid.
    schema = json.loads((ROOT / "bench" / "results.schema.json").read_text())
    item = schema["properties"]["env"]["properties"]["captures"]["items"]
    for c in env["captures"]:
        jsonschema.validate(c, {"$defs": schema["$defs"], **item})
    session = schema["properties"]["session"]
    assert re.fullmatch(session["pattern"], env["session"])
    assert (repo / RES / "MANIFEST.sha256").exists()
    assert (repo / RES / "k_gates.json").exists()
    assert (repo / RES / "nvml.jsonl.gz").exists() or not (
        repo / RES / "nvml.jsonl"
    ).exists()
    limits = (repo / RES / "env" / "power_limit.txt").read_text()
    assert "power_limit_w_before=220" in limits and "power_limit_w_after=220" in limits


def test_the_result_name_follows_the_power_limit(krepo):
    """The name carries the limit read from nvidia-smi, rounded down to whole
    watts and nothing else."""
    repo = krepo[0]
    r = run(krepo, FAKE_POWER="320.00")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (repo / "results" / "2026-10-22-kernel-320w-gpu1").is_dir()
    assert not (repo / RES).exists()


def test_only_reruns_a_subset_as_a_run_of_its_own(krepo):
    """--only restricts the listing (ceilings and experiments) and every
    kernel process to the named experiments, so noisy rows can be rerun as
    a new tagged run without touching the first one's records."""
    r = run(krepo, (*ARGS, "--tag", "again", "--only", "dec-a"))
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [c for c in calls(krepo) if c.startswith("docker run")]
    lists = [c for c in runs if "kernel_bench.py list" in c]
    assert lists and all("--only dec-a" in c for c in lists)
    assert all("--only dec-a" in c for c in runs if "kernel_bench.py run" in c)
    assert (krepo[0] / f"{RES}-again").is_dir()


def test_only_with_an_unknown_name_stops_before_the_ceilings(krepo):
    """A name outside the selection must not cost the ceiling runs first."""
    r = run(krepo, (*ARGS, "--only", "dec-zz"), FAKE_LIST_RC="1")
    assert r.returncode == 1
    assert "--only names an experiment outside" in r.stdout + r.stderr
    assert not any("--arms" in c for c in calls(krepo))  # no ceiling program ran


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"FAKE_GATE_RC": "1"}, "gate exit 1"),
        ({"FAKE_GATE_RC": "2"}, "kernel_gate.py refused"),
        ({"FAKE_KB_RC": "2"}, "kernel_bench failed in process 0"),
        ({"FAKE_CHANGE_BUILD": "1"}, "built files changed during the run"),
    ],
)
def test_never_passes_silently(krepo, env, message):
    """A gate that did not pass, a gate refusal or a failed kernel process
    give exit 1 with the reason."""
    r = run(krepo, **env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert message in r.stdout + r.stderr


def test_foreign_process_voids_the_run(krepo):
    """A record with a foreign_process rejection: the directory is moved to
    results/aborted-<name> and the run fails; no gate is computed."""
    repo = krepo[0]
    r = run(krepo, FAKE_FOREIGN="dec-a")
    assert r.returncode == 1 and "foreign process used GPU 1" in r.stderr
    assert not (repo / RES).exists()
    assert (repo / "results" / f"aborted-{RES.name}").is_dir()
    assert not any("kernel_gate.py" in c for c in calls(krepo))


def test_a_tiny_foreign_context_does_not_void_the_run(krepo):
    """The record says the foreign hits are small and
    few (foreign.void false), so the blocks are only rejected from the
    statistics: the run goes on to the gates."""
    repo = krepo[0]
    r = run(krepo, FAKE_FOREIGN_SMALL="dec-a")
    assert "foreign process used GPU" not in r.stderr
    assert (repo / RES).is_dir()
    assert not (repo / "results" / f"aborted-{RES.name}").exists()
    assert any("kernel_gate.py" in c for c in calls(krepo))


def test_short_processes_rerun_in_fresh_processes(krepo):
    """Status 3 (fewer than 30 accepted blocks): each process of each kernel
    experiment is rerun alone (--only, attempt 1..3), then reported."""
    r = run(krepo, FAKE_STATUS="3")
    assert r.returncode == 0, r.stdout + r.stderr
    reruns = [c for c in calls(krepo) if "kernel_bench.py run" in c and "--only" in c]
    # 2 experiments x 2 processes x 3 reruns
    assert len(reruns) == 12
    assert {c.split("--attempt ")[1].split()[0] for c in reruns} == {"1", "2", "3"}


# ---------------------------------------------------------------- k_build.sh
def _build_shims(krepo, attn: bool, fail: str = ""):
    repo = krepo[0]
    (repo / "tools").mkdir(exist_ok=True)
    body = (
        'mkdir -p "$FAKE_REPO/build/ceilings";'
        ' echo s >"$FAKE_REPO/build/ceilings/sol_copy";'
        ' echo m >"$FAKE_REPO/build/ceilings/mma_ceiling"\n'
    )
    _exe(
        repo / "bench" / "ceilings_build.sh",
        ("exit 7\n" if fail == "ceilings" else "") + body,
    )
    _exe(
        repo / "tools" / "build_ref_lib.sh",
        'mkdir -p "$FAKE_REPO/build/ref"\n'
        'echo r >"$FAKE_REPO/build/ref/libpth_ref.so"\n',
    )
    if attn:
        # The kernel commit's own build script, in the archived tree: it
        # builds into the tree it lives in (as tools/build_attn_lib.sh does).
        tools = repo.parent / "src" / "tools"
        tools.mkdir(parents=True, exist_ok=True)
        _exe(
            tools / "build_attn_lib.sh",
            'd="$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p "$d/build/attn"\n'
            'echo commit-build >"$d/build/attn/libpth_attn.so"\n',
        )


def test_build_records_every_file_it_needs(krepo):
    """Positive control: all steps pass -> SHA256SUMS over the ceilings, the
    reference library, the kernel library built from the commit and copied
    to build/k, the timer and KERNEL (kind and source commit)."""
    repo = krepo[0]
    _build_shims(krepo, attn=True)
    r = run(krepo, (), "k_build.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    sums = (repo / "build/k/SHA256SUMS").read_text()
    for f in (
        "build/k/sol_copy",
        "build/k/mma_ceiling",
        "build/k/libpth_ref.so",
        "build/k/libpth_attn.so",
        "build/k/libpth_cupti_timer.so",
        "build/k/KERNEL",
    ):
        assert f in sums
    # the working-tree builds are not what a run hashes or times
    assert "build/ceilings" not in sums and "build/ref" not in sums
    assert (repo / "build/k/KERNEL").read_text() == f"kernel attn\nsource {'0' * 40}\n"


def test_a_rebuild_of_build_ceilings_or_ref_does_not_void_a_run(krepo):
    """The ceilings, the ref library and the kernel come from build/k copies,
    so rebuilding the working-tree build/ceilings or build/ref at any time is
    harmless, and the run times the copies."""
    repo = krepo[0]
    _build_shims(krepo, attn=True)
    assert run(krepo, (), "k_build.sh").returncode == 0
    (repo / "build/ref/libpth_ref.so").write_text("rebuilt by someone else")
    (repo / "build/ceilings/sol_copy").write_text("rebuilt")
    (repo / "build/ceilings/mma_ceiling").write_text("rebuilt")
    r = run(krepo)
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [c for c in calls(krepo) if c.startswith("docker run")]
    assert any("/w/build/k/sol_copy" in c and "--arms" in c for c in runs)
    assert any("/w/build/k/mma_ceiling" in c and "--arms" in c for c in runs)
    assert not any("/w/build/ceilings" in c for c in runs)
    kb = [c for c in runs if "kernel_bench.py run" in c]
    assert kb and all("--ref-lib /w/build/k/libpth_ref.so" in c for c in kb)


@pytest.mark.parametrize(
    ("args", "attn", "fail", "message"),
    [
        ((), False, "", "build_attn_lib.sh does not exist at"),
        ((), True, "ceilings", "ceilings_build failed (exit 7)"),
    ],
)
def test_build_failures_leave_no_sums(krepo, args, attn, fail, message):
    """A missing kernel build script or a failing step: exit 2, and the old
    SHA256SUMS is gone, so k_run.sh refuses instead of timing stale files."""
    repo = krepo[0]
    _build_shims(krepo, attn=attn, fail=fail)
    r = run(krepo, args, "k_build.sh")
    assert r.returncode == 2 and message in r.stderr
    assert not (repo / "build/k/SHA256SUMS").exists()


def test_build_times_the_commit_not_the_working_tree(krepo):
    """The timed library is built from `git archive` of the commit, so a
    working-tree build/attn (a dev build, changing at any time) is neither
    used nor touched; an unknown revision is exit 2."""
    repo = krepo[0]
    _build_shims(krepo, attn=True)
    (repo / "build/attn").mkdir(parents=True)
    (repo / "build/attn/libpth_attn.so").write_text("dev build\n")
    r = run(krepo, ("--attn-commit", "main"), "k_build.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (repo / "build/k/libpth_attn.so").read_text() == "commit-build\n"
    assert (repo / "build/attn/libpth_attn.so").read_text() == "dev build\n"
    r = run(krepo, ("--attn-commit", "nope"), "k_build.sh", FAKE_BAD_COMMIT="1")
    assert r.returncode == 2 and "is not a commit" in r.stderr


def test_build_stand_in_needs_no_kernel_library(krepo):
    """--kernel ref: the reference library is timed as a stand-in kernel;
    the build succeeds without build_attn_lib.sh and records the choice."""
    repo = krepo[0]
    _build_shims(krepo, attn=False)
    r = run(krepo, ("--kernel", "ref"), "k_build.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (repo / "build/k/KERNEL").read_text() == "kernel ref\nsource -\n"
    assert "libpth_attn" not in (repo / "build/k/SHA256SUMS").read_text()


# ---------------------------------------------------------------- k_sweep.sh
SWEEP_RES = Path("results") / f"2026-10-22-sweep-{POWER_W}w-gpu1"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((), "--gpu N is required"),
        (("--gpu", "1", "--date", "22-10-2026"), "--date must be YYYY-MM-DD"),
        (("--gpu", "1", "--processes", "1"), "--processes must be 2..29"),
        (("--gpu", "1", "--tag", "a/b"), "--tag may only use"),
        (("--gpu", "1", "--bogus"), "unknown argument"),
    ],
)
def test_sweep_script_checks_its_arguments(krepo, args, message):
    """The shared argument checks (lib.sh session_setup) and the script's own
    flags exit 64 before anything starts."""
    r = run(krepo, args, "k_sweep.sh")
    assert r.returncode == 64 and message in r.stderr
    assert calls(krepo) == []


def test_sweep_kind_is_checked(krepo):
    r = run(krepo, (*ARGS, "--kind", "X"), "k_sweep.sh")
    assert r.returncode == 64 and "--kind must be" in r.stderr


def test_sweep_flow(krepo):
    """Positive control: the sweep lists its experiments for this card's SM
    count, P processes time them with the commit-built library, every
    experiment is reported with the planner's A/A pair and one claim per
    variant against the planner, then the table and the manifest; every
    container on GPU 1 only."""
    repo = krepo[0]
    r = run(krepo, ARGS, "k_sweep.sh")
    assert r.returncode == 0, r.stdout + r.stderr
    cs = calls(krepo)
    runs = [c for c in cs if c.startswith("docker run")]
    assert all("--gpus device=1 " in c for c in runs if "--gpus" in c)
    lists = [c for c in runs if "k_sweep.py list" in c]
    assert len(lists) == 1 and "--num-sms 68" in lists[0] and "--kind all" in lists[0]
    sweeps = [c for c in runs if "k_sweep.py run" in c]
    assert len(sweeps) == 2
    assert all("--kernel-lib /w/build/k/libpth_attn.so" in c for c in sweeps)
    assert all("--kernel-kind attn" in c for c in sweeps)
    reports = [c for c in runs if "bench_report.py report" in c]
    assert [c.split("--experiment ")[1].split()[0] for c in reports] == [
        "swp-S-a",
        "swp-BQ-b",
    ]
    assert "--aa planner planner#aa --compare S9 planner" in reports[0]
    assert "--aa planner planner#aa --compare wm2 planner" in reports[1]
    assert all("--cluster-robust" in c for c in reports)
    assert any("sweep_table.py" in c for c in runs)
    assert (repo / SWEEP_RES / "SWEEP.txt").read_text().strip() == "table"
    assert (repo / SWEEP_RES / "MANIFEST.sha256").exists()


def test_sweep_refuses_the_reference_kernel(krepo):
    """The reference library has no row tile to sweep: refuse before
    creating anything rather than time it under the sweep's name."""
    repo = krepo[0]
    (repo / "build/k/KERNEL").write_text("kernel ref\nsource -\n")
    r = run(krepo, ARGS, "k_sweep.sh")
    assert r.returncode == 65 and "must say 'kernel attn'" in r.stderr
    assert not (repo / SWEEP_RES).exists()
