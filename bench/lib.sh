# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# lib.sh: helpers shared by the single-GPU benchmark drivers bench/ceilings_run.sh,
# bench/k_run.sh and bench/k_sweep.sh. Sourced, never run.
#
# A timing is only usable as evidence if nobody else used the GPU, the timed
# binaries are the hashed ones, and nothing is overwritten. This file is where
# those rules live, once, so the drivers cannot drift apart. It also captures the
# environment of a run (GPU, driver, power limit, git commit, image) and starts the
# NVML logger that the noise-rejection rules read.
#
# It only reads device state: it never sets clocks, power limits or persistence.
#
# Requirements on the host: bash, docker with the NVIDIA container runtime, nvidia-smi,
# git, python3 (standard library only) and, for the optional lock, flock. Everything
# that needs torch, numpy, jsonschema or nvidia-ml-py (the statistics, the reports,
# the NVML logger) runs inside the dev image (docker/dev.Dockerfile), which must have
# python3 and nvcc on its PATH; PTH_BENCH_PYTHON names another interpreter.
#
# The caller sets, before calling anything here:
#   repo image gpu date_tag tag processes max_reruns settle_s rerun_exit
#   hash_lists   array of "DIR|SUMSFILE": every sha256sum list that must hold before
#                and after each GPU container (DIR relative to the repo; the lists are
#                written by bench/k_build.sh and bench/ceilings_build.sh)
#   ceil_dir     directory (relative to the repo) of sol_copy and mma_ceiling
#   lock_file    optional (empty = none): the --lock FILE option of the drivers. The run
#                then holds an exclusive flock on FILE until it ends, so two runs that
#                name the same file never overlap. Without it nothing is locked.
#   run_argv     optional array: the command line, recorded in env/command.txt
# and then runs: session_setup KIND, guards, env_capture, nvml_start, the experiments
# (run_exp / kernel_processes / kernel_reports), manifest. A caller with more to check
# before anything is created defines `extra_guard` (called by guards, before the result
# directory exists; it calls refuse).
#
# Result directory: results/<date>-<kind>-<W>w-gpu<N>[-<tag>]/ where kind is ceilings,
# kernel or sweep and W is the GPU's power limit in watts, read from nvidia-smi. The
# name is also the `session` field of every record.
#
# Exit codes of the drivers: 64 bad command line, 65 refused (nothing was started),
# 1 a step failed.
# shellcheck shell=bash
# The variables listed above are set by the sourcing driver, not here.
# shellcheck disable=SC2154

log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 65
}
# fd 8 = the script's own stderr: die is also called inside commands whose output
# goes to a log file (in_gpu ... >logs/x.log 2>&1), and the reason must still reach
# the operator.
exec 8>&2
die() {
  printf 'ERROR: %s\n' "$*" >&8
  if [[ -d "${res:-}" ]]; then
    printf 'Partial results stay in %s; move it aside (never delete).\n' "${res_rel}" >&8
  fi
  exit 1
}

# The interpreter and the nvcc used inside the image.
py_bin="${PTH_BENCH_PYTHON:-python3}"
nvcc_bin="${PTH_BENCH_NVCC:-/usr/local/cuda/bin/nvcc}"

# py_json CODE ARGS...: small JSON jobs with the host's python3 (standard library
# only), run from the repo root so relative result paths work.
py_json() {
  local code="$1"
  shift
  (cd "${repo}" && python3 -c "import hashlib, json, sys
${code}" "$@")
}

# hashes_ok: every listed sha256sum list holds.
hashes_ok() {
  local entry
  for entry in "${hash_lists[@]}"; do
    (cd "${repo}/${entry%%|*}" && sha256sum --quiet -c "${entry#*|}" >/dev/null 2>&1) || return 1
  done
}

# gpu_power_limit: the enforced power limit of GPU ${gpu} in whole watts (empty if
# nvidia-smi cannot say).
gpu_power_limit() {
  nvidia-smi -i "${gpu}" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null |
    cut -d. -f1 | tr -d ' ' || true
}

# ---------------------------------------------------------------- guards
# guards: refuse (exit 65, nothing started) unless no process uses GPU N, the lock
# (if one was asked for) is free, every hashed file matches, the image exists and
# the result directory is new.
guards() {
  local apps
  apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader)" || refuse "nvidia-smi failed for GPU ${gpu}"
  [[ -z "${apps}" ]] || refuse "compute processes on GPU ${gpu}: ${apps}"
  if [[ -n "${lock_file:-}" ]]; then
    exec 9>"${lock_file}"
    flock -n 9 || refuse "another run holds the lock ${lock_file}"
  fi
  local entry shown=""
  for entry in "${hash_lists[@]}"; do
    local f="${entry%%|*}/${entry#*|}"
    f="${f#./}"
    shown+="${f} "
    [[ -s "${repo}/${entry%%|*}/${entry#*|}" ]] ||
      refuse "no ${f}: build first (bench/k_build.sh or bench/ceilings_build.sh)"
  done
  hashes_ok || refuse "built files differ from ${shown% }: rebuild"
  if declare -F extra_guard >/dev/null; then extra_guard; fi
  docker image inspect "${image}" >/dev/null 2>&1 || refuse "image ${image} not found"
  [[ ! -e "${res}" ]] || refuse "${res_rel} exists; results are never overwritten"
  mkdir -p "${res}/raw" "${res}/env" "${res}/logs"
  # The first list is the run's build record (build.SHA256SUMS), further lists are
  # numbered.
  local i=0
  for entry in "${hash_lists[@]}"; do
    cp "${repo}/${entry%%|*}/${entry#*|}" \
      "${res}/build$( ((i)) && echo ".${i}").SHA256SUMS"
    i=$((i + 1))
  done
  trap cleanup EXIT # from here on the NVML logger may exist
}

# ---------------------------------------------------------------- docker helpers
in_cpu() {
  # shellcheck source=tools/ci_cpus.sh
  source "${repo}/tools/ci_cpus.sh"
  docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
    -v "${repo}:/w" -w /w -e PYTHONPATH=/w/python -e CUDA_VISIBLE_DEVICES= \
    --entrypoint "${py_bin}" "${image}" "$@"
}
# The binaries and libraries must stay the ones hashed at the start: a rebuild in
# the middle of a run would put a timing of a different library under this run's
# hashes. Checked before and after every GPU container.
same_build() {
  hashes_ok || die "built files changed during the run (SHA256SUMS); results void"
}
# in_gpu ENTRYPOINT ARGS...: the GPU under test only, offline.
in_gpu() {
  local entry="$1" rc=0
  shift
  same_build
  docker run --rm --gpus "device=${gpu}" --network none \
    -v "${repo}:/w" -w /w -e PYTHONPATH=/w/python \
    --entrypoint "${entry}" "${image}" "$@" || rc=$?
  same_build
  return "${rc}"
}
nvml_name="pth-bench-nvml-$$"
# SIGTERM lets the logger finish its tick and write the footer.
cleanup() { docker stop -t 10 "${nvml_name}" >/dev/null 2>&1 || true; }
# (the EXIT trap that calls cleanup is set at the end of `guards`)

# ---------------------------------------------------------------- environment
# snapshot LABEL: nvidia-smi power, clock, temperature and performance state.
snapshot() {
  { date -Is; nvidia-smi -i "${gpu}" -q -d POWER,CLOCK,TEMPERATURE,PERFORMANCE; } \
    >"${res}/env/gpu_${1}.txt" 2>&1 || true
}

# env_capture [KERNEL_KIND KERNEL_SOURCE]: the "before" snapshot, the command line,
# nvidia-smi -q, collect_env, the device info, and env.json (the fields
# bench/bench_report.py reads), plus the kernel kind and the commit the timed kernel
# was built from when the run times a kernel. Sets sm_count.
env_capture() {
  local kernel_kind="${1:-}" kernel_source="${2:-}"
  log "environment capture -> ${res_rel}/env (${session}${kernel_kind:+, kernel ${kernel_kind}})"
  mkdir -p "${res}/env" "${res}/scripts"
  power_limit_before="$(gpu_power_limit)"
  snapshot before
  { printf '%q ' "${run_argv[@]:-$0}"; echo; } >"${res}/env/command.txt"
  # The scripts as they were when the run started.
  cp "${repo}"/bench/*.sh "${repo}"/bench/*.py "${res}/scripts/" 2>/dev/null || true
  nvidia-smi -q -i "${gpu}" >"${res}/env/nvidia-smi-q.txt" || die "nvidia-smi -q failed"
  in_gpu /bin/bash tools/collect_env.sh >"${res}/env/collect_env.txt" 2>&1 ||
    log "WARNING: collect_env.sh failed (kept its output)"
  in_gpu "/w/${ceil_dir}/sol_copy" --device-info >"${res}/env/device.json" ||
    die "sol_copy --device-info failed"
  sm_count="$(py_json 'print(json.load(open(sys.argv[1]))["device"]["sm_count"])' \
    "${res}/env/device.json")"
  [[ "${sm_count}" =~ ^[0-9]+$ ]] || die "could not read the SM count"
  local power_limit driver nvcc_v digest sha dirty=false
  power_limit="$(nvidia-smi -i "${gpu}" --query-gpu=power.limit --format=csv,noheader,nounits |
    tr -d ' ')"
  # "[N/A]" or empty would not be a number: record null instead.
  [[ "${power_limit}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || power_limit=null
  driver="$(nvidia-smi -i "${gpu}" --query-gpu=driver_version --format=csv,noheader | tr -d ' ')"
  # "Cuda compilation tools, release 13.0, V13.0.88" -> 13.0.88
  nvcc_v="$(docker run --rm --network none --entrypoint "${nvcc_bin}" "${image}" \
    --version | sed -n 's/.*, V\([0-9.]*\)$/\1/p')"
  [[ -n "${nvcc_v}" ]] || die "could not read the nvcc version"
  digest="$(docker image inspect --format '{{.Id}}' "${image}")"
  sha="$(git -C "${repo}" rev-parse HEAD)"
  [[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || dirty=true
  py_json '
out, sha, dirty, digest, nvcc, driver, power, kernel, ksrc, sess = sys.argv[1:11]
cap = [{"name": p.rsplit("/", 1)[-1], "path": p,
        "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
       for p in sys.argv[11:]]
env = {"git_sha": sha, "git_dirty": dirty == "true", "image_digest": digest,
       "nvcc_version": nvcc, "driver_version": driver,
       "power_limit_w": None if power == "null" else float(power),
       "captures": cap, "session": sess}
if kernel:
    env["kernel"], env["kernel_source"] = kernel, ksrc
json.dump(env, open(out, "w"))' \
    "${res_rel}/env.json" "${sha}" "${dirty}" "${digest}" "${nvcc_v}" "${driver}" \
    "${power_limit}" "${kernel_kind}" "${kernel_source:--}" "${session}" \
    "${res_rel}/env/nvidia-smi-q.txt" "${res_rel}/env/collect_env.txt" \
    "${res_rel}/env/device.json"
  [[ "${dirty}" == false ]] || log "WARNING: tracked files are modified; records say git.dirty=true"
}

nvml_start() {
  log "NVML logger -> ${res_rel}/nvml.jsonl"
  docker run -d --rm --name "${nvml_name}" --gpus "device=${gpu}" --network none --pid=host \
    -v "${repo}:/w" -w /w --entrypoint "${py_bin}" "${image}" \
    bench/nvml_log.py --out "${res_rel}/nvml.jsonl" >/dev/null
  sleep 2
  [[ -s "${res}/nvml.jsonl" ]] || die "the NVML logger wrote nothing"
}

# ---------------------------------------------------------------- helpers
# no_foreign RECORD...: a voiding foreign-process verdict voids the run (a
# foreign process above stats.FOREIGN_MAX_MIB or of unknown memory, or more than
# stats.FOREIGN_MAX_BLOCK_FRACTION of one process's blocks hit). Blocks hit by a tiny
# foreign context are only rejected from the statistics. A record without "foreign"
# predates that verdict: any foreign_process rejection voids it.
no_foreign() {
  local n
  n="$(py_json 'def hit(f):
    rej = json.load(open(f))["rejections"]
    foreign = rej.get("foreign")
    if foreign is None:
        return rej["by_reason"].get("foreign_process", 0)
    return foreign["blocks"] if foreign["void"] else 0
print(sum(hit(f) for f in sys.argv[1:]))' "$@")"
  if [[ "${n}" != 0 ]]; then
    cleanup
    mv "${res}" "${repo}/results/aborted-${name}"
    die "a foreign process used GPU ${gpu} during ${n} blocks; moved to results/aborted-${name}"
  fi
}

# status RAW DOMAIN: 0 ok, 3 rerun, other = error
status() {
  sleep "${settle_s}"
  in_cpu bench/bench_report.py status --raw "$1" --nvml "${res_rel}/nvml.jsonl" \
    --clock-domain "$2"
}

# report NAME DOMAIN RERUNS RAW... [-- EXTRA...]
report() {
  local exp="$1" dom="$2" reruns="$3" extra=()
  shift 3
  local raws=()
  while (($#)) && [[ "$1" != -- ]]; do raws+=("$1"); shift; done
  (($#)) && { shift; extra=("$@"); }
  in_cpu bench/bench_report.py report --raw "${raws[@]}" --nvml "${res_rel}/nvml.jsonl" \
    --env "${res_rel}/env.json" --session "${session}" --experiment "${exp}" \
    --out-dir "${res_rel}" --reruns "${reruns}" --clock-domain "${dom}" \
    --cluster-robust "${extra[@]}" | tee -a "${res}/SUMMARY.txt" ||
    die "report failed for ${exp}"
  no_foreign "${res}/${exp}"__*.json
}

# run_exp NAME PROGRAM ARMS DOMAIN [ARGS...] [-- REPORT_ARGS...]: P fresh processes of
# a ceiling program (ARMS comma-separated, mirrored rounds from stats.py schedule), a
# status check and up to max_reruns reruns per process, then one record per arm.
# REPORT_ARGS go to bench_report.py (--aa, --compare).
run_exp() {
  local exp="$1" prog="$2" arms="$3" dom="$4" p attempt rc raw sched worst=0 n_arms
  shift 4
  local targs=() rargs=()
  while (($#)) && [[ "$1" != -- ]]; do targs+=("$1"); shift; done
  (($#)) && { shift; rargs=("$@"); }
  local raws=() commas="${arms//[^,]/}"
  n_arms=$((${#commas} + 1))
  log "experiment ${exp}: ${prog} ${arms}"
  for ((p = 0; p < processes; p++)); do
    sched="$(in_cpu bench/stats.py schedule --arms "${n_arms}" --process "${p}")"
    attempt=0
    while :; do
      raw="${res_rel}/raw/${exp}.p${p}.a${attempt}.json"
      in_gpu "/w/${ceil_dir}/${prog}" --arms "${arms}" --schedule "${sched}" \
        --process "${p}" --seed "${p}" --out "${raw}" "${targs[@]}" ||
        die "${prog} failed in ${exp} process ${p}"
      rc=0
      status "${raw}" "${dom}" >/dev/null || rc=$?
      if ((rc == 0)); then break; fi
      ((rc == rerun_exit)) || die "status check failed for ${raw}"
      if ((attempt < max_reruns)); then attempt=$((attempt + 1)); else break; fi
    done
    ((attempt > worst)) && worst="${attempt}"
    raws+=("${raw}")
  done
  report "${exp}" "${dom}" "${worst}" "${raws[@]}" -- "${rargs[@]}"
}

# splits B: the decode split count of the planner for B requests of q = 1 on this
# card (the target geometry: D = 256, Hkv = 2, G = 6).
splits() {
  in_cpu -c "
from pth_flash_attn.planner import plan_launch, workspace_token_splits
B = $1
ws = workspace_token_splits(max_num_seqs=B, q_dec_max=1, Hkv=2, D=256, num_sms=${sm_count})
print(plan_launch(num_tokens=B, num_reqs=B, max_query_len=1, G=6, Hkv=2, D=256,
                  num_sms=${sm_count}, is_prefill=False, ws_token_splits=ws).num_splits)"
}

# ---------------------------------------------------------------- python experiments
# For runs whose experiments are timed by a python program with a `run` subcommand
# (bench/kernel_bench.py, bench/k_sweep.py). The caller sets exps (experiment names,
# in the program's order) and, optionally, a function report_args_for EXP that prints
# the bench_report.py arguments (--aa, --compare), one per line.

# kernel_processes SCRIPT ARGS...: P fresh processes, each timing every experiment
# (mirrored order for odd processes, inside the program).
kernel_processes() {
  local script="$1" p what
  what="$(basename "$1" .py)"
  shift
  for ((p = 0; p < processes; p++)); do
    log "${what} process ${p}"
    in_gpu "${py_bin}" "${script}" run "$@" \
      --process "${p}" --attempt 0 --out-dir "${res_rel}/raw" \
      >"${res}/logs/kernel.p${p}.log" 2>&1 ||
      die "${what} failed in process ${p} (logs/kernel.p${p}.log)"
  done
}

# kernel_reports SCRIPT ARGS...: per experiment the clock domain measured from its own
# blocks (bench_report.py domain), the status of every process with reruns of short
# ones (fresh processes with --only), then one record per arm.
kernel_reports() {
  local script="$1" exp p dom attempt rc raw worst
  shift
  for exp in "${exps[@]}"; do
    local first=() raws=() extra=()
    for ((p = 0; p < processes; p++)); do first+=("${res_rel}/raw/${exp}.p${p}.a0.json"); done
    dom="$(in_cpu bench/bench_report.py domain --raw "${first[@]}" --nvml "${res_rel}/nvml.jsonl" \
      2>>"${res}/logs/domains.log")" || die "domain check failed for ${exp}"
    [[ "${dom}" == mem || "${dom}" == both ]] || die "bad clock domain '${dom}' for ${exp}"
    echo "${exp} ${dom}" >>"${res}/logs/domains.log"
    worst=0
    for ((p = 0; p < processes; p++)); do
      attempt=0
      while :; do
        raw="${res_rel}/raw/${exp}.p${p}.a${attempt}.json"
        rc=0
        status "${raw}" "${dom}" >/dev/null || rc=$?
        if ((rc == 0)); then break; fi
        ((rc == rerun_exit)) || die "status check failed for ${raw}"
        ((attempt < max_reruns)) || break
        attempt=$((attempt + 1))
        log "${exp} process ${p}: rerun ${attempt}/${max_reruns}"
        in_gpu "${py_bin}" "${script}" run "$@" --only "${exp}" \
          --process "${p}" --attempt "${attempt}" --out-dir "${res_rel}/raw" \
          >>"${res}/logs/kernel.p${p}.log" 2>&1 || die "rerun failed (${exp}, ${p})"
      done
      ((attempt > worst)) && worst="${attempt}"
      raws+=("${raw}")
    done
    if declare -F report_args_for >/dev/null; then
      mapfile -t extra < <(report_args_for "${exp}")
    fi
    report "${exp}" "${dom}" "${worst}" "${raws[@]}" -- "${extra[@]}"
  done
}

# nvml_check EXP: stop the logger and check that it saw the benchmark's own process,
# or the foreign-process rule was blind in this run.
nvml_check() {
  cleanup
  trap - EXIT
  local seen
  seen="$(py_json 'print(max(b["nvml"].get("max_compute_procs") or 0
                     for f in sys.argv[1:] for b in json.load(open(f))["samples"]["blocks"]))' \
    "${res}/$1"__*.json)"
  [[ "${seen}" -ge 1 ]] || die "NVML never saw the benchmark process: foreign-process rule blind"
}

# manifest: the closing steps, then the sha256 of every file of the run, written
# last. Closing steps: the "after" snapshot, the power-limit check (a run whose power
# limit changed is no comparison row, and says so in env/power_limit.txt) and
# nvml.jsonl -> nvml.jsonl.gz (the records hash nvml.jsonl; gunzip -k restores the
# exact file they hash).
manifest() {
  snapshot after
  local power_limit_after note=""
  power_limit_after="$(gpu_power_limit)"
  [[ "${power_limit_after}" == "${power_limit_before:-}" ]] ||
    note="POWER-LIMIT-CHANGED(${power_limit_before:-?}->${power_limit_after:-?})"
  printf 'power_limit_w_before=%s\npower_limit_w_after=%s\n%s\n' \
    "${power_limit_before:-}" "${power_limit_after}" "${note}" >"${res}/env/power_limit.txt"
  [[ -z "${note}" ]] || log "WARNING: ${note}: this run is not a comparison row"
  [[ -s "${res}/nvml.jsonl" ]] && gzip -f "${res}/nvml.jsonl"
  # The list is written next to the directory first, so it never hashes itself.
  (cd "${res}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${res}.manifest.tmp"
  mv "${res}.manifest.tmp" "${res}/MANIFEST.sha256"
}

# session_setup KIND: the checks every driver makes of its common arguments (exit 64)
# and the names derived from them: session, name, res_rel, res. The first nvidia-smi
# query happens here, after the command line was accepted. Refuses (65) if the power
# limit cannot be read, because it is part of the result name.
session_setup() {
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "--gpu N is required (nvidia-smi index)" >&2; exit 64; }
  [[ "${date_tag}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "--date must be YYYY-MM-DD" >&2; exit 64; }
  [[ -z "${tag}" || "${tag}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "--tag may only use [A-Za-z0-9_.-]" >&2; exit 64; }
  [[ "${processes}" =~ ^[2-9]$|^[1-2][0-9]$ ]] || { echo "--processes must be 2..29 (a confidence interval needs >= 2 processes)" >&2; exit 64; }
  local power_w
  power_w="$(gpu_power_limit)"
  [[ "${power_w}" =~ ^[0-9]+$ ]] || refuse "cannot read the power limit of GPU ${gpu} (nvidia-smi power.limit)"
  name="${date_tag}-${1}-${power_w}w-gpu${gpu}${tag:+-${tag}}"
  session="${name}"
  res_rel="results/${name}"
  res="${repo}/${res_rel}"
}
