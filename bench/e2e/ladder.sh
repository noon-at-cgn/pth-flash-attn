#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Concurrency ladder for one running server, with the prompt salt pinned.
#
# Runs HyperQwen's bench/conc_ladder.py (https://github.com/syv-ai/HyperQwen) through
# bench/e2e/ladder_salt.py, which pins time.time() to SALT_EPOCH, so that two servers measured with
# the same SALT_EPOCH receive byte-identical prompts. conc_ladder.py keeps only its best repetition per
# invocation, so every repetition is a separate invocation with --reps 1, its own --label (8kr1,
# 8kr2, ...) and its own JSON file; the raw per-repetition data stays available.
#
# Per context depth (label: prompt tokens, repetitions):
#   8k: 8192, 5    32k: 32000, 3    128k: 128000, 3
# Each invocation: --n 1,2 --out 256 --reps 1 --ctx-tokens <tokens> --label <label>r<k>
#                  --json OUT_DIR/conc_ladder_<label>_r<k>.json
# The console output of each invocation goes to OUT_DIR/conc_ladder_<label>_r<k>.log.
#
# Usage:
#   HYPERQWEN_DIR=/path/to/HyperQwen bench/e2e/ladder.sh [options] [DEPTH...]
#
# DEPTH is one or more of 8k, 32k, 128k (default: all three, in that order).
#
# Options (each overrides the environment variable of the same meaning):
#   --hyperqwen-dir DIR   HyperQwen checkout containing bench/conc_ladder.py   [HYPERQWEN_DIR, required]
#   --port N              server port                                           [PORT, default 8000]
#   --python CMD          Python interpreter for the benchmark                  [PYTHON, default python3]
#   --salt-epoch N        integer the prompt-salt clock is pinned to            [SALT_EPOCH, default 1791800000]
#   --out-dir DIR         directory for the JSON and log files                  [OUT_DIR, default ./ladder-results]
#   --dry-run             print the commands without running them
#   -h, --help            this text
#
# Environment: REPS_8K, REPS_32K and REPS_128K override the repetition counts (5, 3, 3). The server
# address and key are resolved by HyperQwen's harness: VLLM_API (a server root; overrides PORT), else
# http://127.0.0.1:$PORT, and the key from the environment variables that harness reads (see its harness.py).
#
# Use the same SALT_EPOCH for every arm of a comparison; the default is the value of all published runs.
# Start each arm on a freshly launched server so that equal salts never meet a warm prefix cache.
#
# Exit status: 0 when every invocation succeeded, otherwise the number of failed invocations (at most
# 125); 64 for a usage error or a missing prerequisite.
set -euo pipefail

usage() { sed -n '/^# Concurrency ladder/,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"; }
die() { echo "ladder.sh: $*" >&2; exit 64; }
shell_word() {  # shell_word WORD: WORD quoted only when it contains characters special to the shell
  if [[ "$1" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s' "$1"; else printf '%q' "$1"; fi
}

PORT="${PORT:-8000}"
PYTHON="${PYTHON:-python3}"
SALT_EPOCH="${SALT_EPOCH:-1791800000}"
HYPERQWEN_DIR="${HYPERQWEN_DIR:-}"
OUT_DIR="${OUT_DIR:-./ladder-results}"
dry_run=0
depths=()

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=1 ;;
    --hyperqwen-dir|--port|--python|--salt-epoch|--out-dir)
      (( $# >= 2 )) || die "$1 needs a value"
      case "$1" in
        --hyperqwen-dir) HYPERQWEN_DIR="$2" ;;
        --port) PORT="$2" ;;
        --python) PYTHON="$2" ;;
        --salt-epoch) SALT_EPOCH="$2" ;;
        --out-dir) OUT_DIR="$2" ;;
      esac
      shift ;;
    8k|32k|128k) depths+=("$1") ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
(( ${#depths[@]} )) || depths=(8k 32k 128k)

[[ -n "${HYPERQWEN_DIR}" ]] || die "HYPERQWEN_DIR is not set: give a HyperQwen checkout (--hyperqwen-dir DIR)"
ladder_py="${HYPERQWEN_DIR}/bench/conc_ladder.py"
[[ -f "${ladder_py}" ]] || die "${ladder_py} not found: HYPERQWEN_DIR must be a HyperQwen checkout"
salt_py="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ladder_salt.py"
[[ -f "${salt_py}" ]] || die "${salt_py} not found"
[[ "${SALT_EPOCH}" =~ ^[0-9]+$ ]] || die "SALT_EPOCH must be a non-negative integer, got '${SALT_EPOCH}'"
[[ "${PORT}" =~ ^[0-9]+$ ]] || die "PORT must be an integer, got '${PORT}'"
if (( ! dry_run )); then
  command -v "${PYTHON}" >/dev/null 2>&1 || die "Python interpreter '${PYTHON}' not found (--python CMD)"
fi

tokens_for() {  # tokens_for DEPTH -> prompt tokens
  case "$1" in 8k) echo 8192 ;; 32k) echo 32000 ;; 128k) echo 128000 ;; esac
}
reps_for() {  # reps_for DEPTH -> repetitions
  case "$1" in
    8k) echo "${REPS_8K:-5}" ;;
    32k) echo "${REPS_32K:-3}" ;;
    128k) echo "${REPS_128K:-3}" ;;
  esac
}

export PORT
(( dry_run )) || mkdir -p "${OUT_DIR}"
failed=0
for depth in "${depths[@]}"; do
  tokens="$(tokens_for "${depth}")"
  reps="$(reps_for "${depth}")"
  [[ "${reps}" =~ ^[1-9][0-9]*$ ]] || die "repetition count for ${depth} must be a positive integer, got '${reps}'"
  for (( r = 1; r <= reps; r++ )); do
    base="${OUT_DIR}/conc_ladder_${depth}_r${r}"
    cmd=("${PYTHON}" "${salt_py}" "${SALT_EPOCH}" "${ladder_py}"
         --label "${depth}r${r}" --n "1,2" --out 256 --reps 1
         --ctx-tokens "${tokens}" --json "${base}.json")
    if (( dry_run )); then
      printf 'PORT=%s ' "${PORT}"
      for word in "${cmd[@]}"; do printf '%s ' "$(shell_word "${word}")"; done
      printf '> %s 2>&1\n' "$(shell_word "${base}.log")"
      continue
    fi
    echo "ladder.sh: ${depth} repetition ${r}/${reps}: ${base}.json"
    if ! "${cmd[@]}" >"${base}.log" 2>&1; then
      echo "ladder.sh: ${depth} repetition ${r} failed (see ${base}.log)" >&2
      failed=$((failed + 1))
    fi
  done
done
(( failed <= 125 )) || failed=125
exit "${failed}"
