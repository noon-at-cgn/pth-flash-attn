#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Builds the test-only reference libraries of csrc/pth_attn/ref_capi.h:
#
#   build/ref/libpth_ref.so       nvcc: SASS for sm_80/86/89 (no PTX, as the
#                                 extension), static cudart, plus
#                                 the host emulator entry points
#   build/ref/libpth_ref_host.so  g++: host emulator only (CPU tests; the
#                                 CUDA entry points return PTH_ATTN_ENOCUDA)
#   build/ref/ptxas_sm{80,86,89}.log   ptxas -v of the reference kernels
#
# Both compile with -fmad=false / no contraction: the reference kernel is an
# executable specification and every rounding in it must be the one written
# in the source (reference_kernel.cu header comment).
#
# Needs no GPU. Two ways to run it:
#   on the host:   tools/build_ref_lib.sh [--host-only]
#     starts its own CPU-only, offline container of pth-flash-attn-dev (like
#     tools/build_in_image.sh) and runs itself inside it;
#   in the image:  the same command (tests do this); /.dockerenv marks it.
# Exit: 0 built; 2 a build failed; 1 the ptxas gate rejected a kernel (spills or stack).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /.dockerenv ]]; then
  # shellcheck source=tools/ci_cpus.sh
  source "${repo}/tools/ci_cpus.sh"
  exec docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
    -v "${repo}:/w" -w /w -e CUDA_VISIBLE_DEVICES= \
    -e NVIDIA_VISIBLE_DEVICES=void --entrypoint nice pth-flash-attn-dev \
    -n 19 /bin/bash tools/build_ref_lib.sh "$@"
fi
out="${repo}/build/ref"
src="${repo}/csrc/pth_attn/ref_capi.cu"
inc="${repo}/csrc/pth_attn"
nvcc=/usr/local/cuda/bin/nvcc
mkdir -p "${out}"

echo "== g++ host library"
g++ -std=c++17 -O2 -fPIC -shared -ffp-contract=off -Wall -Wextra -Werror \
  -x c++ -I"${inc}" "${src}" -o "${out}/libpth_ref_host.so" -lpthread || exit 2

if [[ "${1:-}" == "--host-only" ]]; then
  echo "== build_ref_lib: host library only"
  exit 0
fi

echo "== nvcc CUDA library (sm_80/86/89)"
gencode=()
for arch in 80 86 89; do
  gencode+=("-gencode=arch=compute_${arch},code=sm_${arch}")
done
"${nvcc}" -std=c++17 -O3 -fmad=false -Xcompiler -fPIC -Xcompiler -ffp-contract=off \
  -shared -cudart static -I"${inc}" "${gencode[@]}" -Xptxas -v \
  "${src}" -o "${out}/libpth_ref.so" >"${out}/nvcc.log" 2>&1 || {
  cat "${out}/nvcc.log"
  exit 2
}

# Split ptxas -v output per arch and gate: no spills, no stack frame (the
# lanes' q and accumulator arrays must stay in registers).
status=0
for arch in 80 86 89; do
  grep -A3 "for 'sm_${arch}'" "${out}/nvcc.log" >"${out}/ptxas_sm${arch}.log" || true
  echo "-- sm_${arch}"
  sed 's/^/   /' "${out}/ptxas_sm${arch}.log"
  if grep -Eq "([1-9][0-9]* bytes spill)|([1-9][0-9]* bytes stack frame)" \
    "${out}/ptxas_sm${arch}.log"; then
    echo "   FAIL: spill or stack frame on sm_${arch}"
    status=1
  fi
done
echo "== build_ref_lib: $([[ ${status} -eq 0 ]] && echo PASS || echo FAIL)"
exit "${status}"
