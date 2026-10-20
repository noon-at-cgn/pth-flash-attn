# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# GPU-run image: the dev image plus a build-time check that compute-sanitizer
# (memcheck, racecheck, synccheck, initcheck), Nsight Compute and the CUDA
# 13.0 binary tools are present. The NVIDIA CUDA 13.0 development base that
# docker/dev.Dockerfile starts from already ships all of them; this stage
# installs nothing and only fails the build if one goes missing, so the image
# tag used by the GPU scripts keeps meaning "sanitizer available". Build:
#   docker build -f docker/dev.Dockerfile --target dev -t pth-flash-attn-dev .
#   docker build -f docker/sanitizer.Dockerfile -t pth-flash-attn-sanitizer docker
#
# Running ncu in a container needs access to GPU performance counters (host
# driver option NVreg_RestrictProfilingToAdminUsers=0, or the container run
# with --cap-add SYS_ADMIN); that is a setting of the machine that runs it and
# is not made here.

ARG BASE=pth-flash-attn-dev
FROM ${BASE}

# Fail the build if a tool is missing where the GPU scripts call it.
RUN /usr/local/cuda/bin/compute-sanitizer --version \
 && /usr/local/cuda/bin/cuobjdump --version \
 && /usr/local/cuda/bin/nvdisasm --version \
 && /usr/local/cuda/bin/ncu --version
