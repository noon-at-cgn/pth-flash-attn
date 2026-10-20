#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Builds and runs the cache_geometry.cuh tests on the CPU (no GPU needed):
#   1. g++ build of tests/cpp/test_geometry.cpp with warnings as errors, run.
#   2. nvcc compile of the same file for sm_80/86/89 (-cubin, ptxas -v): the
#      __global__ probe instantiates every function in device code, so a
#      host-only construct in the header fails here.
#
# Run inside the dev image from the repository root (built with
# `docker build -f docker/dev.Dockerfile -t pth-flash-attn-dev .`):
#   docker run --rm --network none -v "$PWD":/w -w /w \
#     --entrypoint bash pth-flash-attn-dev tests/cpp/run_geometry.sh
# Exit: 0 all passed; 1 a test failed; 2 a build failed.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="${repo}/build/tests_cpp"
mkdir -p "${out}"
src="${repo}/tests/cpp/test_geometry.cpp"
inc="${repo}/csrc/pth_attn"
nvcc=/usr/local/cuda/bin/nvcc

echo "== g++ build"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -I"${inc}" "${src}" \
  -o "${out}/test_geometry" || exit 2
echo "== run"
"${out}/test_geometry" || exit 1

echo "== nvcc device compile (sm_80/86/89)"
for arch in 80 86 89; do
  rm -f "${out}/geometry_probe_sm${arch}.cubin"
  "${nvcc}" -std=c++17 -O3 -x cu -I"${inc}" \
    -gencode="arch=compute_${arch},code=sm_${arch}" -cubin \
    -Xptxas -v -o "${out}/geometry_probe_sm${arch}.cubin" "${src}" \
    >"${out}/nvcc_sm${arch}.log" 2>&1 || { cat "${out}/nvcc_sm${arch}.log"; exit 2; }
  grep -E "registers" "${out}/nvcc_sm${arch}.log" | sed "s/^/  sm_${arch}: /"
  test -s "${out}/geometry_probe_sm${arch}.cubin" || exit 2
done
echo "== run_geometry: PASS"
