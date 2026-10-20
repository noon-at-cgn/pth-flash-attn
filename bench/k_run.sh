#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# k_run.sh: kernel benchmark on ONE GPU: the attention kernel over the benchmark grid
# (bench/kernel_bench.py), the hardware ceilings it is gated against measured in the
# same run on the same GPU, and the pass or fail results at fixed reference levels (bench/kernel_gate.py).
#
#   bench/k_run.sh --gpu N [--date YYYY-MM-DD] [--tag TAG] [--lock FILE]
#                  [--tier quick|full] [--family all|decode|prefill|mixed|smoke]
#                  [--processes P] [--image IMAGE] [--only EXP[,EXP...]]
#
# Needs: one NVIDIA GPU (sm80, sm86 or sm89) that nothing else is using, docker with
# the NVIDIA runtime, nvidia-smi and git on the host, and the dev image
# (docker/dev.Dockerfile, default tag pth-flash-attn-dev). Run bench/k_build.sh first:
# it builds the kernel, the ceiling programs and the CUPTI timer, and writes the
# hashes this script checks.
#
# Refuses (exit 65, nothing started) unless
#   * no compute process exists on GPU N (other GPUs may be in use);
#   * with --lock FILE: no other run holds the lock FILE (flock); without the
#     option nothing is locked;
#   * build/k/SHA256SUMS exists and every hashed file matches (bench/k_build.sh),
#     and the image exists;
#   * results/<date>-kernel-<W>w-gpu<N>[-TAG]/ does not exist yet (W = the GPU's
#     power limit in watts, read from nvidia-smi).
# The checks, the environment capture, the NVML logger and the per-experiment
# statistics are bench/lib.sh, shared with bench/k_sweep.sh and bench/ceilings_run.sh.
# It only reads device state; it never sets clocks, power or persistence.
#
# Steps (the measurement design: mirrored rounds, 20 replays per block, 2 s warm-up, P
# fresh processes (default 6), NVML at 100 ms, rejection rules, at most 3 reruns per
# process, then "noisy"):
#   ceilings  sol_copy sep-nhd for every decode (B, L) of the grid (copy ceiling
#             for eta_mem) and mma_ceiling f16f32:4:1:8 (eta_mma)
#   kernel    bench/kernel_bench.py: P processes, each running every experiment of
#             the selected grid; then per experiment: the clock domain measured from
#             its own blocks (bench_report.py domain), status per process (reruns of
#             short processes), one record per arm (bench_report.py report,
#             cluster-robust CIs, A/A MDE)
#   gates     bench/kernel_gate.py -> k_gates.json, GATES.txt
# A timing taken while another process used GPU N is never kept: if any record shows
# a foreign_process rejection, the run stops and the directory is renamed
# results/aborted-<name>.
# --only runs the named experiments of the selection (e.g. the rows a run left
# "noisy") as a run of its own, with its own --tag: the earlier records are never
# edited or merged, and the gates of the rows not run read "missing" in GATES.txt.
#
# Output: results/<date>-kernel-<W>w-gpu<N>[-TAG]/ with env/ (GPU, driver, power limit,
# commit, image), nvml.jsonl.gz, raw/, one <experiment>__<arm>.json record per arm
# (bench/results.schema.json), SUMMARY.txt, GATES.txt, k_gates.json and
# MANIFEST.sha256.
#
# Exit: 0 the gate_verdict is pass; 1 a step failed or the gates did not pass;
# 64 usage; 65 refused.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="pth-flash-attn-dev"
gpu=""
date_tag="$(date +%Y-%m-%d)"
tag=""
tier="full"
family="all"
only=""
lock_file=""
processes=6     # 6 fresh processes halve the between-process CI term
                # of 3 (docs/benchmark-methodology.md)
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
    --tier) tier="${2:?--tier needs quick or full}"; shift ;;
    --family) family="${2:?--family needs a value}"; shift ;;
    --processes) processes="${2:?--processes needs a number}"; shift ;;
    --image) image="${2:?--image needs a value}"; shift ;;
    --only) only="${2:?--only needs experiment names}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done
# shellcheck source=bench/lib.sh
source "${repo}/bench/lib.sh"
# The script's own arguments are checked before session_setup: its first query
# (nvidia-smi, for the power limit in the name) must not run for a bad command line
# (exit 64 before any query).
[[ "${tier}" == quick || "${tier}" == full ]] || { echo "--tier must be quick or full" >&2; exit 64; }
[[ "${family}" =~ ^(all|decode|prefill|mixed|smoke)$ ]] || { echo "--family must be all, decode, prefill, mixed or smoke" >&2; exit 64; }
[[ -z "${only}" || "${only}" =~ ^[A-Za-z0-9_.,-]+$ ]] || { echo "--only takes comma-separated experiment names" >&2; exit 64; }
session_setup kernel # --gpu/--date/--tag/--processes checks; name, res

# ---------------------------------------------------------------- 1. checks
# The kernel, the ref library, the ceiling programs and the timer are k_build.sh's
# copies in build/k (build/ceilings and build/ref may be rebuilt at any time, which
# would void a running measurement); build/k/SHA256SUMS hashes them all.
hash_lists=(".|build/k/SHA256SUMS")
ceil_dir=build/k
extra_guard() {
  kernel_kind="$(sed -n 's/^kernel //p' "${repo}/build/k/KERNEL" 2>/dev/null || true)"
  kernel_source="$(sed -n 's/^source //p' "${repo}/build/k/KERNEL" 2>/dev/null || true)"
  [[ "${kernel_kind}" == attn || "${kernel_kind}" == ref ]] ||
    refuse "build/k/KERNEL does not say which kernel was built"
}
guards
cp "${repo}/build/k/KERNEL" "${res}/build.KERNEL"

# ---------------------------------------------------------------- 2. environment
env_capture "${kernel_kind}" "${kernel_source}"

# ---------------------------------------------------------------- 3. NVML logger
nvml_start
# (helpers run_exp, status, report, splits: bench/lib.sh)

kb_args=(--family "${family}" --tier "${tier}" --kernel-kind "${kernel_kind}"
  --ref-lib /w/build/k/libpth_ref.so)
list_args=(--family "${family}" --tier "${tier}")
if [[ -n "${only}" ]]; then
  kb_args+=(--only "${only}")
  list_args+=(--only "${only}")
  # An unknown name would otherwise surface only after the ceilings ran.
  in_cpu bench/kernel_bench.py list "${list_args[@]}" >/dev/null ||
    die "--only names an experiment outside --family ${family} --tier ${tier}"
fi
# The kernel library is k_build.sh's copy built from a commit, never the working-tree
# build/attn (which may change during the run).
[[ "${kernel_kind}" == ref ]] || kb_args+=(--kernel-lib /w/build/k/libpth_attn.so)

# ---------------------------------------------------------------- 4. ceilings
mapfile -t pairs < <(in_cpu bench/kernel_bench.py list "${list_args[@]}" --copy-ceilings)
for pair in "${pairs[@]}"; do
  read -r B L <<<"${pair}"
  s="$(splits "${B}")"
  [[ "${s}" =~ ^[0-9]+$ ]] || die "planner returned no split count for B=${B}: '${s}'"
  dom=both
  ((L >= 32768)) && dom=mem # memory-bound from 32k (measured on the ceilings runs)
  run_exp "ceil-copy-b${B}-L$((L / 1024))k" sol_copy sep-nhd "${dom}" \
    --batch "${B}" --seq-len "${L}" --block-size 816 --splits "${s}" --timer cupti
done
run_exp ceil-mma mma_ceiling f16f32:4:1:8 both --mode mma --timer cupti

# ---------------------------------------------------------------- 5. kernel
mapfile -t exps < <(in_cpu bench/kernel_bench.py list "${list_args[@]}" | cut -d' ' -f1)
((${#exps[@]})) || die "the grid selection is empty"
log "kernel: ${#exps[@]} experiments x ${processes} processes"
report_args_for() { printf '%s\n' --aa kernel "kernel#aa"; }
kernel_processes bench/kernel_bench.py "${kb_args[@]}"
kernel_reports bench/kernel_bench.py "${kb_args[@]}"

# ---------------------------------------------------------------- 6. gates
nvml_check "${exps[0]}"
gate_rc=0
in_cpu bench/kernel_gate.py "${res_rel}" | tee -a "${res}/SUMMARY.txt" || gate_rc=$?
((gate_rc != 2)) || die "kernel_gate.py refused the results (see above)"
[[ -s "${res}/k_gates.json" ]] || die "kernel_gate.py wrote no k_gates.json"
manifest
log "done: ${res_rel} (GATES.txt, k_gates.json, MANIFEST.sha256); gate exit ${gate_rc}"
exit "${gate_rc}"
