#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Builds the test-only loader libraries (csrc/pth_attn/loader_capi.h):
#   OUT/libpth_loader.so       nvcc, sm_80/86/89 SASS (no PTX), static cudart:
#                              pth_loader_run (GPU) plus everything else
#   OUT/libpth_loader_host.so  g++ -x c++ of loader_capi.cu: host emulation,
#                              widths and lane maps only
# Needs no GPU. Run from the repository root, either on the host (the script
# then starts its own CPU-only, offline pth-flash-attn-dev container, like
# tools/build_in_image.sh) or inside that image:
#   tools/build_loader_lib.sh [OUT]
# OUT defaults to build/loader; give it relative to the repository root (from
# the host, the container sees only the repository).
# Exit status is non-zero on any failure. Compiler flags follow
# tools/build_flags.py (-O3, C++17, one -gencode per arch).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /.dockerenv ]]; then
  # shellcheck source=tools/ci_cpus.sh
  source "${repo}/tools/ci_cpus.sh"
  exec docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
    -v "${repo}:/w" -w /w -e CUDA_VISIBLE_DEVICES= \
    -e NVIDIA_VISIBLE_DEVICES=void --entrypoint nice pth-flash-attn-dev \
    -n 19 bash tools/build_loader_lib.sh "$@"
fi
out="${1:-${repo}/build/loader}"
inc="${repo}/csrc/pth_attn"
nvcc=/usr/local/cuda/bin/nvcc
mkdir -p "${out}"

gencode=()
for arch in 80 86 89; do
  gencode+=("-gencode=arch=compute_${arch},code=sm_${arch}")
done

"${nvcc}" -std=c++17 -O3 -shared -Xcompiler -fPIC -I"${inc}" "${gencode[@]}" \
  "${inc}/loader_capi.cu" "${inc}/loader_test_kernels.cu" \
  -o "${out}/libpth_loader.so"
echo "wrote ${out}/libpth_loader.so"

g++ -std=c++17 -O2 -Wall -Wextra -Werror -shared -fPIC -x c++ -I"${inc}" \
  "${inc}/loader_capi.cu" -o "${out}/libpth_loader_host.so"
echo "wrote ${out}/libpth_loader_host.so"
