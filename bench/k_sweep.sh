#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# k_sweep.sh: the split-count and row-tile sweeps on ONE GPU:
# the planner's launch against variants, at the decode shapes of the benchmark
# grid, for the kernel built by bench/k_build.sh.
#
#   bench/k_sweep.sh --gpu N [--date YYYY-MM-DD] [--tag TAG] [--lock FILE]
#                    [--kind all|S|BQ] [--processes P] [--image IMAGE]
#
# Needs: one NVIDIA GPU (sm80, sm86 or sm89) that nothing else is using, docker with
# the NVIDIA runtime, nvidia-smi and git on the host, the dev image
# (docker/dev.Dockerfile, default tag pth-flash-attn-dev) and a kernel built by
# bench/k_build.sh (--kernel attn).
#
# What each experiment is and why: bench/k_sweep.py. Per experiment the arms are the
# planner's launch (``planner``), its A/A copy (``planner#aa``) and the variants; the
# report claims "variant differs from planner" only beyond 2 x the A/A MDE. A sweep
# finds what to look at; it is not a gate and has no ceilings (the gate numbers come
# from bench/k_run.sh).
#
# Same checks, hashed build, memory-clock rule, process count and results layout as
# k_run.sh (bench/lib.sh): refuses (exit 65) while a compute process uses GPU N, when
# another run holds the --lock FILE (if given), when build/k does not match its
# hashes, or when the result directory exists.
# results/<date>-sweep-<W>w-gpu<N>[-TAG]/ (W = the GPU's power limit in watts) gets one
# record per arm, SUMMARY.txt, SWEEP.txt (bench/sweep_table.py) and MANIFEST.sha256.
# Exit 0 when every experiment reported; 1 a step failed; 64 usage; 65 refused.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="pth-flash-attn-dev"
gpu=""
date_tag="$(date +%Y-%m-%d)"
tag=""
kind="all"
lock_file=""
processes=6     # fresh processes per experiment
max_reruns=3    # reruns per process before the run is reported noisy
settle_s=0.5    # NVML settle time
rerun_exit=3    # exit status of bench_report.py that requests a rerun

usage() { sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; }
run_argv=("$0" "$@") # recorded in env/command.txt
while (($#)); do
  case "$1" in
    --gpu) gpu="${2:?--gpu needs an index}"; shift ;;
    --date) date_tag="${2:?--date needs YYYY-MM-DD}"; shift ;;
    --tag) tag="${2:?--tag needs a value}"; shift ;;
    --lock) lock_file="${2:?--lock needs a file}"; shift ;;
    --kind) kind="${2:?--kind needs all, S or BQ}"; shift ;;
    --processes) processes="${2:?--processes needs a number}"; shift ;;
    --image) image="${2:?--image needs a value}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done
# shellcheck source=bench/lib.sh
source "${repo}/bench/lib.sh"
# own argument first: no query (nvidia-smi) for a bad command line
[[ "${kind}" =~ ^(all|S|BQ)$ ]] || { echo "--kind must be all, S or BQ" >&2; exit 64; }
session_setup sweep

# The sweep times the attention kernel only (the reference has no tiling to sweep);
# the library is k_build.sh's copy built from a commit.
hash_lists=(".|build/k/SHA256SUMS")
ceil_dir=build/k
extra_guard() {
  kernel_kind="$(sed -n 's/^kernel //p' "${repo}/build/k/KERNEL" 2>/dev/null || true)"
  kernel_source="$(sed -n 's/^source //p' "${repo}/build/k/KERNEL" 2>/dev/null || true)"
  [[ "${kernel_kind}" == attn ]] ||
    refuse "build/k/KERNEL must say 'kernel attn' for a sweep (found '${kernel_kind}')"
}
guards
cp "${repo}/build/k/KERNEL" "${res}/build.KERNEL"
env_capture "${kernel_kind}" "${kernel_source}"
nvml_start

# "<experiment> <kind> <arm,arm,...>" per line, planned for this card's SM count.
mapfile -t listing < <(in_cpu bench/k_sweep.py list --kind "${kind}" --num-sms "${sm_count}")
((${#listing[@]})) || die "the sweep selection is empty"
exps=()
declare -A arms_of
for line in "${listing[@]}"; do
  read -r e _ arms <<<"${line}"
  exps+=("${e}")
  arms_of["${e}"]="${arms}"
done
log "sweep: ${#exps[@]} experiments x ${processes} processes"

# Claims: every variant against the planner's launch, MDE from planner#aa.
report_args_for() {
  local a
  printf '%s\n' --aa planner "planner#aa"
  for a in ${arms_of["$1"]//,/ }; do
    [[ "${a}" == planner || "${a}" == "planner#aa" ]] || printf '%s\n' --compare "${a}" planner
  done
}
kb_args=(--kind "${kind}" --kernel-kind attn --kernel-lib /w/build/k/libpth_attn.so)
kernel_processes bench/k_sweep.py "${kb_args[@]}"
kernel_reports bench/k_sweep.py "${kb_args[@]}"

nvml_check "${exps[0]}"
in_cpu bench/sweep_table.py "${res_rel}" | tee "${res}/SWEEP.txt"
manifest
log "done: ${res_rel} (SWEEP.txt, SUMMARY.txt, one record per arm, MANIFEST.sha256)"
