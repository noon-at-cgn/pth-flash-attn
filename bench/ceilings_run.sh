#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# ceilings_run.sh: measure the hardware ceilings (copy bandwidth, tensor-core
# throughput, CUDA-graph node overhead) of ONE GPU, the denominators of the kernel
# gates.
#
#   bench/ceilings_run.sh --gpu N [--date YYYY-MM-DD] [--quick] [--image IMAGE]
#                   [--build DIR] [--timer cupti|events] [--lock FILE]
#
# Needs: one NVIDIA GPU (sm80, sm86 or sm89) that nothing else is using, docker with
# the NVIDIA runtime, nvidia-smi and git on the host, and the dev image
# (docker/dev.Dockerfile, default tag pth-flash-attn-dev). Run
# bench/ceilings_build.sh first: it builds the programs and writes the hashes this
# script checks. A full run takes a few hours; --quick is a subset for a second card.
#
# --timer events: only if CUPTI cannot attach in the container (the binaries exit 2
# with a CUPTI error). Records then say timer=cuda_event and
# duration_kind=graph_replay_mean; they are not comparable with CUPTI rows.
#
# This script refuses to start (exit 65, nothing started) unless
#   * no compute process exists on GPU N (the other GPUs may be in use);
#   * with --lock FILE: no other run holds the lock FILE (flock); without the
#     option nothing is locked;
#   * the binaries in build/ceilings match build/ceilings/SHA256SUMS
#     (bench/ceilings_build.sh);
#   * the result directory does not exist yet (it never overwrites results).
# It only reads device state (nvidia-smi queries, NVML): it never sets clocks, power
# limits, persistence or application clocks.
#
# What runs (every timed experiment: mirrored arm order, 36 rounds, 20 launches per
# block, 3 fresh processes with the order reversed in process 1, 2 s warm-up; a
# process that keeps < 30 accepted blocks of an arm is rerun, at most 3 times, then
# reported "noisy"):
#   soak-copy, soak-mma   2 min each of one workload: clock/power behaviour under
#                         the GPU's power limit (justifies the 2 s warm-up)
#   copy-*                sol_copy at D=256, Hkv=2: separate streams vs whole
#                         cell vs plain stream; NHD/HND; block 816 and 16;
#                         L 4k..256k; B 1/2/4; A/A copy for the MDE
#   mma-*                 mma_ceiling: f16f32 chain sweep at the kernel's occupancy
#                         (4 warps, 1 CTA/SM), best cases, A/A; other kinds
#                         one experiment per kind (no power carry-over)
#   graph-overhead        empty graph nodes, 65/129/257 per graph
# --quick: the subset for a second GPU (copy at 128k, f16f32, graph).
#
# Output: results/<date>-ceilings-<W>w-gpu<N>/ (W = the GPU's power limit in watts,
# read from nvidia-smi)
#   env/ (nvidia-smi -q, tools/collect_env.sh, device.json, before/after snapshots),
#   env.json, nvml.jsonl.gz (NVML at 100 ms for the whole run),
#   raw/<experiment>.p<P>.a<A>.json (raw per process and attempt),
#   <experiment>__<arm>.json (bench/results.schema.json records), SUMMARY.txt,
#   MANIFEST.sha256.
# bench/ceilings.py turns the records into the JSON that tests/gpu/attn_dev_timing.py
# reads.
# Exit: 0 done; 1 a step failed; 64 usage; 65 refused (safety check).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="pth-flash-attn-dev"
build="build/ceilings"
gpu=""
date_tag="$(date +%Y-%m-%d)"
tag=""
lock_file=""
quick=0
timer="cupti"
# Measurement constants.
rounds=36           # scheduled blocks: 30 accepted + 1/6 rejection slack
processes=3         # fresh processes per experiment
max_reruns=3        # reruns per process before the run is reported noisy
settle_s=0.5        # NVML settle time: > NVML period + write latency
rerun_exit=3        # exit status of bench_report.py that requests a rerun
soak_s=120          # soak time: clock/power behaviour over 2 min

usage() { sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; } # the header comment above
run_argv=("$0" "$@")             # recorded in env/command.txt
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
    --quick) quick=1 ;;
    --timer)
      timer="${2:?--timer needs cupti or events}"
      shift
      ;;
    --image)
      image="${2:?--image needs a value}"
      shift
      ;;
    --build)
      build="${2:?--build needs a directory relative to the repo}"
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
[[ "${timer}" == cupti || "${timer}" == events ]] || {
  echo "--timer must be cupti or events" >&2
  exit 64
}
# shellcheck source=bench/lib.sh
source "${repo}/bench/lib.sh"
session_setup ceilings # the name carries the power limit in watts

# ---------------------------------------------------------------- 1. checks
hash_lists=("${build}|SHA256SUMS")
ceil_dir="${build}"
guards

# ---------------------------------------------------------------- 2. environment
env_capture

# ---------------------------------------------------------------- 3. NVML logger
nvml_start

# ---------------------------------------------------------------- 4. experiments
# run_exp NAME PROGRAM ARMS: uses the arrays tool_args and report_args and the
# scalars exp_procs / exp_rounds / exp_sched ("mirrored" or "single") and
# exp_domain, the NVML clocks the drift rule watches (stats.CLOCK_DOMAINS):
# "mem" for copies at L >= 32k, whose time was measured not to depend on the
# SM clock (docs/benchmark-methodology.md); "both" for everything else.
# (This run needs its own run_exp: unlike lib.sh's, it varies the round count and
# the schedule per experiment.)
tool_args=()
report_args=()
exp_procs="${processes}"
exp_rounds="${rounds}"
exp_sched="mirrored"
exp_domain="both"
run_exp() {
  local name="$1" prog="$2" arms="$3" n_arms p attempt rc raw sched worst=0
  local raws=()
  n_arms="$(awk -F, '{print NF}' <<<"${arms}")"
  log "experiment ${name}: ${arms}"
  for ((p = 0; p < exp_procs; p++)); do
    attempt=0
    while :; do
      if [[ "${exp_sched}" == single ]]; then
        # one arm, every block: "0,0,...,0" (no `yes | head`: SIGPIPE under pipefail)
        sched="$(seq "${exp_rounds}" | sed 's/.*/0/' | paste -sd, -)"
      else
        sched="$(in_cpu bench/stats.py schedule --arms "${n_arms}" \
          --rounds "${exp_rounds}" --process "${p}")"
      fi
      raw="${res_rel}/raw/${name}.p${p}.a${attempt}.json"
      in_gpu "/w/${build}/${prog}" --arms "${arms}" --schedule "${sched}" \
        --process "${p}" --seed "${p}" --out "${raw}" "${tool_args[@]}" ||
        die "${prog} failed in ${name} process ${p}"
      sleep "${settle_s}"
      set +e
      in_cpu bench/bench_report.py status --raw "${raw}" --nvml "${res_rel}/nvml.jsonl" \
        --clock-domain "${exp_domain}"
      rc=$?
      set -e
      if ((rc == 0)); then
        break
      elif ((rc == rerun_exit && attempt < max_reruns)); then
        attempt=$((attempt + 1))
        log "process ${p} kept < 30 blocks of an arm: rerun ${attempt}/${max_reruns}"
      elif ((rc == rerun_exit)); then
        log "process ${p} still short after ${max_reruns} reruns: reported as noisy"
        break
      else
        die "status check failed for ${raw}"
      fi
    done
    ((attempt > worst)) && worst="${attempt}"
    raws+=("${raw}")
  done
  in_cpu bench/bench_report.py report --raw "${raws[@]}" --nvml "${res_rel}/nvml.jsonl" \
    --env "${res_rel}/env.json" --session "${session}" --experiment "${name}" \
    --out-dir "${res_rel}" --reruns "${worst}" --clock-domain "${exp_domain}" \
    "${report_args[@]}" |
    tee -a "${res}/SUMMARY.txt"
}

copy_shape() { # copy_shape B L BLOCK_SIZE: sets tool_args for sol_copy
  # The planner's answer is validated: an empty or non-numeric --splits must
  # stop the run here, not reach sol_copy.
  local s
  s="$(splits "$1")"
  [[ "${s}" =~ ^[0-9]+$ ]] || die "planner returned no split count for B=$1: '${s}'"
  tool_args=(--batch "$1" --seq-len "$2" --block-size "$3" --splits "${s}"
    --timer "${timer}")
  # memory-bound only at long L
  if (($2 >= 32768)); then exp_domain=mem; else exp_domain=both; fi
}

s_b1="$(splits 1)"
[[ "${s_b1}" =~ ^[0-9]+$ ]] || die "planner returned no split count: '${s_b1}'"
log "planner: ${sm_count} SMs -> decode splits ${s_b1} at B=1"

# 4a. soaks: one arm, one process, enough blocks for ~2 minutes.
# B=4, L=256k is ~1.1 GB per launch (~1.5 ms at ~700 GB/s); 50 launches per
# block make ~75 ms blocks, so 1600 blocks ~ 2 min. The mma block is
# 50 x 2 ms = 100 ms, so 1200 blocks ~ 2 min.
# Soaks only in the full run (--quick skips them).
if [[ ${quick} != 1 ]]; then
  exp_procs=1
  exp_sched=single
  copy_shape 4 262144 816
  tool_args+=(--replays 50 --warmup-s 0)
  report_args=(--min-accepted 1)
  exp_rounds=$((soak_s * 1000 / 75))
  run_exp soak-copy sol_copy sep-nhd
  tool_args=(--mode mma --replays 50 --warmup-s 0 --timer "${timer}")
  exp_domain=both
  exp_rounds=$((soak_s * 1000 / 100))
  run_exp soak-mma mma_ceiling f16f32:4:1:8
  exp_procs="${processes}"
  exp_rounds="${rounds}"
  exp_sched=mirrored
fi

# 4b. copy ceilings (benchmark geometry: D=256, Hkv=2, NHD).
copy_shape 1 131072 816
report_args=(--aa sep-nhd "sep-nhd#aa"
  --compare cell-nhd sep-nhd --compare sep-hnd sep-nhd
  --compare cell-hnd sep-hnd --compare plain sep-nhd)
if [[ ${quick} == 1 ]]; then
  report_args=(--aa sep-nhd "sep-nhd#aa" --compare cell-nhd sep-nhd --compare plain sep-nhd)
  run_exp copy-L128k sol_copy "sep-nhd,cell-nhd,plain,sep-nhd#aa"
else
  run_exp copy-L128k sol_copy "sep-nhd,cell-nhd,sep-hnd,cell-hnd,plain,sep-nhd#aa"
  report_args=()
  for L in 4096 32768 65536 262144; do
    copy_shape 1 "${L}" 816
    run_exp "copy-L$((L / 1024))k" sol_copy "sep-nhd,cell-nhd,plain"
  done
  for B in 2 4; do
    copy_shape "${B}" 131072 816
    run_exp "copy-b${B}-L128k" sol_copy "sep-nhd,cell-nhd,plain"
  done
  copy_shape 1 131072 16
  run_exp copy-bs16-L128k sol_copy "sep-nhd,cell-nhd,sep-hnd,plain"
fi

# 4c. tensor-core ceilings. 4:1 = the kernel's occupancy (4 warps, 1 CTA/SM); 8:2 and
# 8:4 = 4 and 8 warps per sub-partition (best cases); chains 1..8 = latency
# sweep at the kernel's occupancy.
tool_args=(--mode mma --timer "${timer}")
exp_domain=both
report_args=(--aa f16f32:4:1:8 "f16f32:4:1:8#aa")
if [[ ${quick} == 1 ]]; then
  run_exp mma-f16f32 mma_ceiling "f16f32:4:1:8,f16f32:8:4:8,f16f32:4:1:8#aa"
else
  run_exp mma-f16f32 mma_ceiling \
    "f16f32:4:1:1,f16f32:4:1:2,f16f32:4:1:4,f16f32:4:1:8,f16f32:8:2:8,f16f32:8:4:8,f16f32:4:1:8#aa"
  report_args=()
  # One experiment per kind: interleaving kinds of different power draw left
  # the governor's response to the previous block inside the next one
  # (bf16 blocks after s8 started 6.8 % slower; docs/benchmark-methodology.md).
  for kind in f16f16 bf16f32 s8s32; do
    run_exp "mma-${kind}" mma_ceiling "${kind}:4:1:8,${kind}:8:4:8"
  done
fi

# 4d. graph-node overhead (T_ovh part 1).
# The slope needs no 1-node arm: a 2-us replay is two steps of the 1024-ns
# event timer and carried no information in the first runs. 257 nodes adds a
# third point.
tool_args=(--mode graph-overhead)
exp_domain=both
report_args=()
run_exp graph-overhead mma_ceiling "nodes:65,nodes:129,nodes:257"

# ---------------------------------------------------------------- 5. close
cleanup
trap - EXIT
# Field check: the logger must have seen the benchmark's own process during
# the measured blocks; if it never did, NVML in the logger container cannot
# see other containers' processes and the foreign-process rule was blind.
seen="$(py_json 'print(max(b.get("nvml", {}).get("max_compute_procs") or 0
    for f in sys.argv[1:] for b in json.load(open(f))["samples"]["blocks"]))' \
  "${res}"/copy-L128k__*.json)"
if [[ "${seen}" -lt 1 ]]; then
  log "WARNING: NVML never saw a compute process during copy-L128k blocks:"
  log "         the foreign-process rejection rule could not fire in this run."
  echo "WARNING foreign-process rule blind (max_compute_procs=${seen})" >>"${res}/SUMMARY.txt"
fi
manifest
log "done: ${res_rel} (SUMMARY.txt, MANIFEST.sha256)"
