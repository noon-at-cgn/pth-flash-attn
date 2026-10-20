#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Proves that tools/export_upstream.py removes only code that compiles to
# nothing: the kernel translation units of the upstream subset are compiled
# from the repository and from the stripped export, with the flags of
# tools/build_attn_lib.sh for sm_80/86/89, each set linked into one library,
# and tools/sass_diff.py must find every kernel of every arch identical.
# launch.cu and torch_bindings.cpp are host code (no kernels) and are not
# compiled here.
#
#   tools/prove_export_sass.sh   -> build/export_proof/{repo,export}/libsubset.so
#
# Exit 0 identical, 1 a SASS difference, 2 a compile or tool failure. Needs no
# GPU; on the host it runs itself in a CPU-only, offline pth-flash-attn-dev
# container (like build_attn_lib.sh).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /.dockerenv ]]; then
  # shellcheck source=tools/ci_cpus.sh
  source "${repo}/tools/ci_cpus.sh"
  exec docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
    -v "${repo}:/w" -w /w -e CUDA_VISIBLE_DEVICES= \
    -e NVIDIA_VISIBLE_DEVICES=void --entrypoint nice pth-flash-attn-dev \
    -n 19 /bin/bash tools/prove_export_sass.sh "$@"
fi
py=python3
[[ -x /app/venv/bin/python ]] && py=/app/venv/bin/python
nvcc=/usr/local/cuda/bin/nvcc
out="${repo}/build/export_proof"
rm -rf "${out}"
mkdir -p "${out}/repo" "${out}/export"
"${py}" "${repo}/tools/export_upstream.py" --out "${out}/tree"
mapfile -t tus < <(cd "${repo}" && "${py}" -c '
import sys
sys.path.insert(0, "tools")
import build_flags
for s in build_flags.EXTENSION_SOURCES:
    if s.endswith(".cu") and not s.endswith("/launch.cu"):
        print(s.rsplit("/", 1)[1][:-3])
')
gencode=()
for arch in 80 86 89; do
  gencode+=("-gencode=arch=compute_${arch},code=sm_${arch}")
done
for side in repo export; do
  inc="${repo}/csrc/pth_attn"
  [[ ${side} == export ]] && inc="${out}/tree/csrc/pth_attn"
  pids=()
  for tu in "${tus[@]}"; do
    "${nvcc}" -std=c++17 -O3 -Xcompiler -fPIC -I"${inc}" "${gencode[@]}" \
      -c "${inc}/${tu}.cu" -o "${out}/${side}/${tu}.o" \
      >"${out}/${side}/${tu}.log" 2>&1 &
    pids+=("$!")
  done
  for i in "${!tus[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "== compile of ${side}/${tus[$i]}.cu failed:"
      cat "${out}/${side}/${tus[$i]}.log"
      exit 2
    fi
  done
  objs=()
  for tu in "${tus[@]}"; do objs+=("${out}/${side}/${tu}.o"); done
  "${nvcc}" -shared -cudart static "${gencode[@]}" "${objs[@]}" \
    -o "${out}/${side}/libsubset.so"
done
status=0
"${py}" "${repo}/tools/sass_diff.py" "${out}/repo/libsubset.so" \
  "${out}/export/libsubset.so" || status=$?
if [[ ${status} -eq 0 ]]; then
  echo "== prove_export_sass: PASS (every kernel identical)"
else
  echo "== prove_export_sass: FAIL"
fi
exit "${status}"
