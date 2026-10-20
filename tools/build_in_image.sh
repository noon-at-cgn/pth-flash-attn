#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Builds the op library (pth_flash_attn._C) inside the pinned dev image, CPU
# only: no --gpus, no network. The result is python/pth_flash_attn/_C.abi3.so.
#
# Usage: tools/build_in_image.sh [ARCH_LIST]
#   ARCH_LIST  torch spelling, default "8.0;8.6;8.9" (tools/build_flags.py).
#              A single arch such as "8.6" builds a one-cubin library.
#
# Why a container: the build must use the exact torch headers and nvcc of the
# image the kernel is validated in (nvcc 13.0.88, torch 2.13.0), not whatever
# the host has. Why --network none: the build must not depend on downloads.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
arch_list="${1:-8.0;8.6;8.9}"
# shellcheck source=tools/ci_cpus.sh
source "${repo}/tools/ci_cpus.sh"
cpus="$(pth_ci_cpus)"

# NVIDIA_VISIBLE_DEVICES=void: no GPU is ever exposed to this CPU-only step;
# nice: GPU timing sessions share the host.
docker run --rm --network none --cpus "${cpus}" --memory 16g \
  -v "${repo}:/w" -w /w \
  -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
  -e TORCH_CUDA_ARCH_LIST="${arch_list}" \
  -e MAX_JOBS="${cpus}" \
  --entrypoint nice pth-flash-attn-dev -n 19 /bin/bash -c '
    set -euo pipefail
    # The venv holds ninja; without it on PATH torch falls back to the slow,
    # serial distutils backend.
    export PATH="/app/venv/bin:${PATH}"
    start=$(date +%s%N)
    /app/venv/bin/python setup.py -q build_ext --inplace
    end=$(date +%s%N)
    so=python/pth_flash_attn/_C.abi3.so
    echo "built ${so} for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}:" \
      "$(stat -c %s "${so}") bytes in $(( (end - start) / 1000000 )) ms"
  '
