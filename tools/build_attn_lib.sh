#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Builds the test library of csrc/pth_attn/attn_capi.h, which the kernel test
# ladder drives through ctypes:
#
#   build/attn/libpth_attn.so          nvcc: SASS for sm_80/86/89 (no PTX),
#                                      static cudart
#   build/attn/ptxas_<tu>.log          ptxas -v of every kernel, per TU
#
# The translation units are the extension's kernel library
# (tools/build_flags.py EXTENSION_SOURCES: attn_launch.cu, attn_combine.cu,
# instantiate_d{64,128,256}.cu) plus the test-only C ABI (attn_capi.cu) and
# the fragment unit tests (attn_test_kernels.cu), which the shipped extension
# does not contain. They compile in parallel, then link. The script fails
# (exit 1) if any kernel spills or has a stack frame (the ptxas gate),
# exit 2 if a compile fails.
#
# Needs no GPU. On the host it starts its own CPU-only, offline container of
# pth-flash-attn-dev (like tools/build_ref_lib.sh); inside the image (/.dockerenv)
# it builds directly.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /.dockerenv ]]; then
  # shellcheck source=tools/ci_cpus.sh
  source "${repo}/tools/ci_cpus.sh"
  exec docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
    -v "${repo}:/w" -w /w -e CUDA_VISIBLE_DEVICES= \
    -e NVIDIA_VISIBLE_DEVICES=void --entrypoint nice pth-flash-attn-dev \
    -n 19 /bin/bash tools/build_attn_lib.sh "$@"
fi
out="${repo}/build/attn"
defines=()
experiment=0
hooks=0
archs=(80 86 89)
# Options for test and measurement builds next to build/attn:
#   --out NAME     build into build/NAME instead of build/attn
#   --arch N       only sm_N (faster; the shipped library is always 80/86/89)
#   -DNAME[=V]     extra preprocessor definition for every translation unit;
#                  a -DPTH_ATTN_TEST_* macro also force-includes
#                  csrc/pth_attn/test_hooks.cuh (the handshake test hooks)
#   --lineinfo     nvcc -lineinfo (source-level attribution in Nsight Compute)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="${repo}/build/$2"; shift ;;
    --arch) archs=("$2"); shift ;;
    -DPTH_ATTN_TEST_PROTOCOL*)
      # The protocol checker prints and traps (a printf needs a stack frame),
      # so this test library cannot pass the ptxas gate; it is never shipped.
      defines+=("$1")
      experiment=1
      hooks=1
      ;;
    -DPTH_ATTN_TEST_*) defines+=("$1"); hooks=1 ;;
    -D*) defines+=("$1") ;;
    --lineinfo) defines+=(-lineinfo) ;;
    *) echo "build_attn_lib: unknown option $1"; exit 2 ;;
  esac
  shift
done
inc="${repo}/csrc/pth_attn"
if [[ ${hooks} -eq 1 ]]; then
  defines+=(-include "${inc}/test_hooks.cuh")
fi
nvcc=/usr/local/cuda/bin/nvcc
mkdir -p "${out}"
gencode=()
for arch in "${archs[@]}"; do
  gencode+=("-gencode=arch=compute_${arch},code=sm_${arch}")
done
tus=(attn_launch attn_capi attn_combine instantiate_d64 instantiate_d128
  instantiate_d256 attn_test_kernels)
start=$(date +%s)
pids=()
for tu in "${tus[@]}"; do
  "${nvcc}" -std=c++17 -O3 -Xcompiler -fPIC -I"${inc}" "${defines[@]}" \
    "${gencode[@]}" -Xptxas -v -c "${inc}/${tu}.cu" -o "${out}/${tu}.o" \
    >"${out}/ptxas_${tu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for i in "${!tus[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "== compile of ${tus[$i]}.cu failed:"
    cat "${out}/ptxas_${tus[$i]}.log"
    status=2
  fi
done
[[ ${status} -eq 0 ]] || exit "${status}"
objs=()
for tu in "${tus[@]}"; do objs+=("${out}/${tu}.o"); done
"${nvcc}" -shared -cudart static "${gencode[@]}" "${objs[@]}" \
  -o "${out}/libpth_attn.so"
echo "== built ${out}/libpth_attn.so ($(stat -c %s "${out}/libpth_attn.so") B)" \
  "in $(($(date +%s) - start)) s"
# Gate: no spill bytes and no stack frame in any kernel of any arch.
if grep -hE "([1-9][0-9]* bytes spill)|([1-9][0-9]* bytes stack frame)" \
  "${out}"/ptxas_*.log; then
  if [[ ${experiment} -eq 1 ]]; then
    echo "== build_attn_lib: experiment library, spills above are expected"
    exit 0
  fi
  echo "== build_attn_lib: FAIL (spill or stack frame above)"
  exit 1
fi
echo "== build_attn_lib: PASS (no spills, no stack frames)"
