#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# run_grid.sh: the GPU correctness grid on one GPU (tests/gpu/grid_*): the scalar
# reference kernel and combine against the fp64 reference and bound, the paged
# loader byte-exact, compute-sanitizer, CUDA-graph replay.
#
#   tests/gpu/run_grid.sh --gpu N [--date YYYY-MM-DD] [--image IMAGE]
#                   [--sanitizer-image IMAGE] [--jobs J] [--lock FILE]
#
# Needs: one NVIDIA GPU (sm80, sm86 or sm89), docker with the NVIDIA runtime,
# nvidia-smi and git on the host, the dev image (docker/dev.Dockerfile, default tag
# pth-flash-attn-dev, with nvcc 13, g++, python3, torch and pytest on its PATH) and
# the sanitizer image (docker/sanitizer.Dockerfile, default tag
# pth-flash-attn-sanitizer). A full run took about 20 minutes on one RTX 3080 20 GB
# with the CPU reference values computed (about 4 minutes of CPU at four workers).
#
# CPU use: --jobs (CPU workers of the bound pool, and the CPU quota of the CPU
# containers) defaults to min(4, nproc / 4), at least 1. Every container runs with
# OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS set to 1 unless the
# variable is set in the calling environment, so that the load stays near --jobs
# instead of --jobs times the host's core count.
#
# It refuses to start (exit 65, nothing started) unless
#   * no compute process exists on GPU N (the other GPUs may be in use);
#   * with --lock FILE: no other run holds the lock FILE (flock); without the
#     option nothing is locked;
#   * both images exist;
#   * the result directory does not exist yet (results are never overwritten).
# It only reads device state; it never sets clocks, power or persistence.
#
# Steps (each logged with its exit code and duration in steps.tsv):
#   build      tools/build_ref_lib.sh, tools/build_loader_lib.sh (CPU-only
#              containers); the libraries' SHA-256 go to build/SHA256SUMS
#              and are re-checked before every GPU step
#   expected   fp64 reference + bound of every grid case, CPU pool (cached in
#              build/grid_expected/; fast when precomputed beforehand)
#   quick      pytest -m "gpu and quick"  (grid subset + loader subset)
#   full       pytest -m "gpu and not quick and not graph" (rest of the grid,
#              loader, production split/workspace checks)
#   san-<tool> compute-sanitizer memcheck/racecheck/synccheck/initcheck on
#              pytest -m sanitizer (sanitizer image)
#   ctl-<tool> planted errors that memcheck/initcheck must report (exit 86). The
#              planted out-of-bounds kernel makes the driver log one
#              `NVRM: Xid (PCI:<bus id>): 43, pid=<pid>, name=python3, ...` for
#              the GPU under test; that is expected here and nowhere else.
#              The Xid 43 lines appear in the kernel log (dmesg) during the
#              ctl-memcheck step only. An Xid in any other step, or any other
#              Xid number, is a fault to investigate, not a planted-error
#              side effect.
#   graph      pytest -m graph (capture/replay == eager, lengths changing)
#   report     tests/gpu/grid_report.py -> grid_report.json, SUMMARY.txt
# A step that runs no test is a failure (pytest exit 5, or 0 tests in its
# JUnit file). Every step runs even after a failure, so one run yields
# all the evidence; the exit code says whether everything passed.
#
# Output: results/<date>-grid-<W>w-gpu<N>/ (W = the GPU's power limit in watts, read
# from nvidia-smi) with env/, env.json, nvml.jsonl, build/, logs/, junit/,
# records/, steps.tsv, grid_report.json, SUMMARY.txt, MANIFEST.sha256.
# Exit: 0 all passed; 1 a step or the report failed; 64 usage; 65 refused.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="pth-flash-attn-dev"
san_image="pth-flash-attn-sanitizer"
gpu=""
date_tag="$(date +%Y-%m-%d)"
lock_file=""
# CPU workers for the bound pool: min(4, nproc / 4), at least 1
jobs=$(($(nproc 2>/dev/null || echo 4) / 4))
((jobs >= 1)) || jobs=1
((jobs <= 4)) || jobs=4
# Thread counts of the numerical libraries inside every container (one each unless set).
thread_env=(-e "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  -e "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
  -e "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}")
san_exit=86       # --error-exitcode, distinct from pytest's 0-5

usage() { sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; } # the header comment above
while (($#)); do
  case "$1" in
    --gpu)
      gpu="${2:?--gpu needs an index}"
      shift
      ;;
    --date)
      date_tag="${2:?--date needs YYYY-MM-DD}"
      shift
      ;;
    --image)
      image="${2:?--image needs a value}"
      shift
      ;;
    --sanitizer-image)
      san_image="${2:?--sanitizer-image needs a value}"
      shift
      ;;
    --jobs)
      jobs="${2:?--jobs needs a number}"
      shift
      ;;
    --lock)
      lock_file="${2:?--lock needs a file}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 64
      ;;
  esac
  shift
done
[[ "${gpu}" =~ ^[0-9]+$ ]] || {
  echo "--gpu N is required (nvidia-smi index)" >&2
  exit 64
}
[[ "${date_tag}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || {
  echo "--date must be YYYY-MM-DD" >&2
  exit 64
}
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "--jobs must be a positive integer" >&2
  exit 64
}

log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
refuse() {
  printf 'REFUSED: %s\n' "$*" >&2
  exit 65
}
# The result name carries the GPU's power limit in watts (first nvidia-smi query,
# after the command line was accepted).
power_w="$(nvidia-smi -i "${gpu}" --query-gpu=power.limit --format=csv,noheader,nounits \
  2>/dev/null | cut -d. -f1 | tr -d ' ' || true)"
[[ "${power_w}" =~ ^[0-9]+$ ]] || refuse "cannot read the power limit of GPU ${gpu} (nvidia-smi power.limit)"
session="${date_tag}-grid-${power_w}w-gpu${gpu}"
res_rel="results/${session}"
res="${repo}/${res_rel}"
die() {
  printf 'ERROR: %s\n' "$*" >&2
  if [[ -d "${res}" ]]; then
    printf 'Partial results stay in %s; move it aside (never delete) before a rerun.\n' \
      "${res_rel}" >&2
  fi
  exit 1
}

# ---------------------------------------------------------------- 1. checks
apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader)" || refuse "nvidia-smi failed"
[[ -z "${apps}" ]] || refuse "compute processes on GPU ${gpu}: ${apps}"
if [[ -n "${lock_file}" ]]; then
  exec 9>"${lock_file}"
  flock -n 9 || refuse "another run holds the lock ${lock_file}"
fi
for img in "${image}" "${san_image}"; do
  docker image inspect "${img}" >/dev/null 2>&1 || refuse "image ${img} not found"
done
[[ ! -e "${res}" ]] || refuse "${res_rel} exists; results are never overwritten"
mkdir -p "${res}/env" "${res}/build" "${res}/logs" "${res}/junit" "${res}/records"
printf 'step\texit\texpect\tseconds\ttests\n' >"${res}/steps.tsv"

# ---------------------------------------------------------------- docker helpers
# in_cpu ARGS...: python in a CPU-only, offline container.
in_cpu() {
  docker run --rm --network none --cpus "${jobs}" "${thread_env[@]}" \
    -v "${repo}:/w" -w /w -e PYTHONPATH=/w/python -e CUDA_VISIBLE_DEVICES= \
    --entrypoint python3 "${image}" "$@"
}
# in_gpu IMAGE ENTRYPOINT ARGS...: the GPU under test only, offline.
in_gpu() {
  local img="$1" entry="$2"
  shift 2
  docker run --rm --gpus "device=${gpu}" --network none "${thread_env[@]}" \
    -v "${repo}:/w" -w /w -e PYTHONPATH=/w/python \
    --entrypoint "${entry}" "${img}" "$@"
}

nvml_name="pth-grid-nvml-$$"
cleanup() {
  # SIGTERM lets the logger finish its tick and write the footer.
  docker stop -t 10 "${nvml_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

failed=()
# step NAME EXPECT CMD...: run CMD with output to logs/NAME.log, append a
# steps.tsv row, remember a failure (exit != EXPECT) without stopping.
step() {
  local name="$1" expect="$2" t0 rc=0 secs
  shift 2
  log "step ${name}"
  t0="$(date +%s)"
  "$@" >"${res}/logs/${name}.log" 2>&1 || rc=$?
  secs=$(($(date +%s) - t0))
  local tests="-" junit="${res}/junit/${name}.xml" total skipped
  if [[ -f "${junit}" ]]; then
    # Tests that ran = first tests="N" minus first skipped="N" of the JUnit
    # file: a step whose tests were all skipped tested nothing. (`|| true`:
    # a missing attribute must not trip set -e/pipefail.)
    total="$( (grep -o 'tests="[0-9]*"' "${junit}" || true) | head -n 1 | tr -dc '0-9')"
    skipped="$( (grep -o 'skipped="[0-9]*"' "${junit}" || true) | head -n 1 |
      tr -dc '0-9')"
    tests=$((${total:-0} - ${skipped:-0}))
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${rc}" "${expect}" "${secs}" "${tests}" \
    >>"${res}/steps.tsv"
  if [[ "${rc}" -ne "${expect}" ]]; then
    log "  FAILED: ${name} exited ${rc} (expected ${expect}); see logs/${name}.log"
    failed+=("${name}")
  elif [[ "${tests}" == 0 ]]; then
    log "  FAILED: ${name} ran no tests"
    failed+=("${name}")
  else
    log "  ok (${secs} s, tests: ${tests})"
  fi
}

# Libraries must stay the ones built at the start of this run.
verify_libs() {
  (cd "${repo}" && sha256sum --quiet -c "${res_rel}/build/SHA256SUMS")
}

# pytest_gpu NAME MARKEXPR [EXTRA...]: the GPU tier in the dev image.
pytest_gpu() {
  local name="$1" expr="$2"
  shift 2
  verify_libs || return 3 # the libraries changed since the build
  in_gpu "${image}" python3 -m pytest tests/gpu -m "${expr}" \
    -p no:cacheprovider -q -rfE "--junitxml=${res_rel}/junit/${name}.xml" \
    "--grid-record=${res_rel}/records/${name}.jsonl" "$@"
}

# pytest_san NAME TOOL MARKEXPR [EXTRA...]: the same under compute-sanitizer.
# PYTORCH_NO_CUDA_MEMORY_CACHING=1 (torch's switch, read by torch, not by our
# code): every tensor gets its own cudaMalloc, so an access just past a
# buffer is outside any allocation and memcheck/initcheck can see it.
pytest_san() {
  local name="$1" tool="$2" expr="$3"
  shift 3
  verify_libs || return 3 # the libraries changed since the build
  docker run --rm --gpus "device=${gpu}" --network none "${thread_env[@]}" \
    -v "${repo}:/w" -w /w -e PYTHONPATH=/w/python \
    -e PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    --entrypoint /usr/local/cuda/bin/compute-sanitizer "${san_image}" \
    --tool "${tool}" --error-exitcode "${san_exit}" \
    python3 -m pytest tests/gpu -m "${expr}" -p no:cacheprovider -q \
    -rfE "--junitxml=${res_rel}/junit/${name}.xml" "$@"
}

# ---------------------------------------------------------------- 2. environment
log "environment capture -> ${res_rel}/env"
nvidia-smi -q >"${res}/env/nvidia-smi-q.txt" || die "nvidia-smi -q failed"
in_gpu "${image}" /bin/bash tools/collect_env.sh >"${res}/env/collect_env.txt" 2>&1 ||
  log "WARNING: collect_env.sh failed (kept its output)"
sha="$(git -C "${repo}" rev-parse HEAD)"
dirty=false
[[ -z "$(git -C "${repo}" status --porcelain --untracked-files=no)" ]] || dirty=true
printf '{"session": "%s", "gpu": %s, "git_sha": "%s", "git_dirty": %s,
 "image": "%s", "image_digest": "%s",
 "sanitizer_image": "%s", "sanitizer_image_digest": "%s"}\n' \
  "${session}" "${gpu}" "${sha}" "${dirty}" \
  "${image}" "$(docker image inspect --format '{{.Id}}' "${image}")" \
  "${san_image}" "$(docker image inspect --format '{{.Id}}' "${san_image}")" \
  >"${res}/env.json"
if [[ "${dirty}" == true ]]; then
  log "WARNING: tracked files are modified; env.json says git_dirty=true"
fi

# ---------------------------------------------------------------- 3. NVML logger
log "NVML logger -> ${res_rel}/nvml.jsonl"
docker run -d --rm --name "${nvml_name}" --gpus "device=${gpu}" --network none \
  -v "${repo}:/w" -w /w --entrypoint python3 "${image}" \
  bench/nvml_log.py --out "${res_rel}/nvml.jsonl" >/dev/null
sleep 2
[[ -s "${res}/nvml.jsonl" ]] || die "the NVML logger wrote nothing"

# ---------------------------------------------------------------- 4. build
build_libs() {
  "${repo}/tools/build_ref_lib.sh" && "${repo}/tools/build_loader_lib.sh" &&
    (cd "${repo}" && sha256sum build/ref/libpth_ref.so build/ref/libpth_ref_host.so \
      build/loader/libpth_loader.so build/loader/libpth_loader_host.so) \
      >"${res}/build/SHA256SUMS"
}
step build 0 build_libs
[[ -s "${res}/build/SHA256SUMS" ]] || die "the build produced no libraries (logs/build.log)"
cp "${repo}"/build/ref/*.log "${res}/build/" 2>/dev/null || true

# ---------------------------------------------------------------- 5. expected values
step expected 0 in_cpu tests/gpu/grid_cases.py expected --tier full --jobs "${jobs}"

# ---------------------------------------------------------------- 6. GPU steps
step quick 0 pytest_gpu quick "gpu and quick"
step full 0 pytest_gpu full "gpu and not quick and not graph"
for tool in memcheck racecheck synccheck initcheck; do
  step "san-${tool}" 0 pytest_san "san-${tool}" "${tool}" sanitizer
done
# Negative controls: each planted error must make its tool fail.
step ctl-memcheck "${san_exit}" pytest_san ctl-memcheck memcheck sanitizer_control -k oob
step ctl-initcheck "${san_exit}" pytest_san ctl-initcheck initcheck sanitizer_control \
  -k uninit
step graph 0 pytest_gpu graph graph

# ---------------------------------------------------------------- 7. report
cleanup
trap - EXIT
step report 0 in_cpu tests/gpu/grid_report.py "${res_rel}"
[[ -s "${res}/grid_report.json" && -s "${res}/SUMMARY.txt" ]] ||
  failed+=("report-missing")
# The list is written next to the directory first, so it never hashes itself.
(cd "${res}" && find . -type f -print0 | sort -z | xargs -0 sha256sum) >"${res}.manifest.tmp"
mv "${res}.manifest.tmp" "${res}/MANIFEST.sha256"
[[ -s "${res}/SUMMARY.txt" ]] && cat "${res}/SUMMARY.txt"
if ((${#failed[@]})); then
  log "GRID FAILED: ${failed[*]} (results in ${res_rel})"
  exit 1
fi
log "GRID passed: ${res_rel}"
