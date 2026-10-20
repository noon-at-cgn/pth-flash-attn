#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# CPU limit of the CPU-only dev containers.
#
# Sourced by the scripts that start a container (`source tools/ci_cpus.sh`), which
# then pass `--cpus "$(pth_ci_cpus)"`; run directly, it prints the value.
#
# The limit is min(4, nproc), at least 1, because Docker refuses `--cpus N` with N
# above the number of CPUs of the machine (a 2-CPU CI runner rejects 4). The
# environment variable PTH_CI_CPUS overrides it with a positive integer; a machine
# with more CPUs can raise the limit that way, a smaller share of a large machine
# can lower it. The same number is the parallelism (`MAX_JOBS`) of the extension
# build.

pth_ci_cpus() {
  local n="${PTH_CI_CPUS:-}"
  if [[ -z "${n}" ]]; then
    n="$(nproc 2>/dev/null || echo 1)"
    if ((n > 4)); then
      n=4
    fi
  fi
  if [[ ! "${n}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PTH_CI_CPUS must be a positive integer, got '${n}'" >&2
    return 2
  fi
  printf '%s\n' "${n}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  pth_ci_cpus
fi
