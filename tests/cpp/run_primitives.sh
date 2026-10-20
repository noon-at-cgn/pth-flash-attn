#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Build commands for tests/cpp/test_primitives.cu, in one place. Run inside
# pth-flash-attn-dev (g++ 13.3, nvcc 13.0) from the repository root; the pytest
# wrapper tests/cpu/test_primitives_golden.py calls this script, so the
# commands a human runs and the ones CI runs are the same.
#
#   tests/cpp/run_primitives.sh host OUT_DIR [INCLUDE_DIR]
#       g++ build of the host mode, run it, write OUT_DIR/primitives.json.
#       INCLUDE_DIR (optional) is searched before csrc/pth_attn: the mutation
#       tests put a modified copy of a header there.
#   tests/cpp/run_primitives.sh device OUT_DIR
#       ptxas gate (0 spills, <= 255 registers) and SASS check (LDGSTS, LDSM,
#       HMMA present; no I2F/I2FP; no local memory) of the probe kernels for
#       sm_80/86/89, with JSON reports in OUT_DIR.
set -euo pipefail

mode="${1:?usage: run_primitives.sh host|device OUT_DIR [INCLUDE_DIR]}"
out="${2:?usage: run_primitives.sh host|device OUT_DIR [INCLUDE_DIR]}"
src="tests/cpp/test_primitives.cu"
mkdir -p "${out}"

case "${mode}" in
  host)
    includes=()
    [[ $# -ge 3 ]] && includes+=("-I$3")
    includes+=("-Icsrc/pth_attn")
    # -x c++: the .cu file is plain C++ in host mode (no __CUDACC__).
    # -Werror: a warning in a header is a defect the kernel would inherit.
    g++ -std=c++17 -O2 -Wall -Wextra -Werror -x c++ "${includes[@]}" \
      "${src}" -o "${out}/test_primitives"
    "${out}/test_primitives" "${out}/primitives.json"
    echo "wrote ${out}/primitives.json"
    ;;
  device)
    python tools/ptxas_gate.py "${src}" --min-kernels 5 \
      --out-dir "${out}/ptxas" --json "${out}/ptxas_gate.json"
    python tools/sass_check.py --cu "${src}" \
      --forbid LDL,STL,I2F,I2FP --require LDGSTS,LDSM,HMMA \
      --function 'probe_primitives' \
      --out-dir "${out}/sass" --json "${out}/sass_primitives.json"
    python tools/sass_check.py --cu "${src}" \
      --forbid LDL,STL,I2F,I2FP \
      --function 'probe_(cvt4|v_b_fragments|q_prescale)' \
      --out-dir "${out}/sass" --json "${out}/sass_counts.json"
    ;;
  *)
    echo "unknown mode ${mode}" >&2
    exit 2
    ;;
esac
