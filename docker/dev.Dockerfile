# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Dev images for pth-flash-attn (CPU-only use: builds, tests, lint).
#
# Stage `dev` is the NVIDIA CUDA 13.0 development image (nvcc 13.0.88, g++
# 13.3, Ubuntu 24.04, Python 3.12) plus a virtual environment at /app/venv
# with torch 2.13.0 (cu130 build), vLLM 0.31.0 and the pinned test and lint
# Python packages. The repository scripts expect that layout: nvcc at
# /usr/local/cuda/bin and Python at /app/venv/bin. Build:
#   docker build -f docker/dev.Dockerfile --target dev -t pth-flash-attn-dev .
#
# Stage `lint` adds the tools that need binaries (clang-format, shellcheck,
# pre-commit, git) on top of an existing dev image. It is built FROM the
# local dev image rather than rebuilding `dev`, so lint and test runs use the
# identical Python environment:
#   docker build -f docker/dev.Dockerfile --target lint \
#     --build-arg LINT_BASE=pth-flash-attn-dev -t pth-flash-attn-lint .
#
# Network is needed for the build only (Ubuntu mirrors, PyPI and
# download.pytorch.org); every run uses --network none. No GPU is needed to
# build or to use these images; they run the CPU test suite and the
# compile-only checks.
#
# Pins: the base image by digest (tag 13.0.3-devel-ubuntu24.04, which carries
# nvcc 13.0.88), every Python package by exact version. Ubuntu packages (g++,
# Python, git) follow the distribution's security updates.

ARG CUDA_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
ARG LINT_BASE=dev

FROM ${CUDA_IMAGE} AS dev
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/app/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates python3 python3-dev python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /app/venv
# torch from the CUDA 13.0 index of the PyTorch project; its dependencies
# (nvidia-* runtime wheels, triton) come from PyPI.
RUN /app/venv/bin/pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu130 \
      --extra-index-url https://pypi.org/simple \
      torch==2.13.0
# Build tools and vLLM (the backend tests import it and skip without it).
RUN /app/venv/bin/pip install --no-cache-dir \
      setuptools==80.10.2 \
      ninja==1.13.0 \
      packaging==26.3 \
      numpy==2.3.5 \
      vllm==0.31.0
RUN /app/venv/bin/pip install --no-cache-dir \
      pytest==9.1.1 \
      pytest-timeout==2.4.0 \
      scipy==1.18.1 \
      jsonschema==4.26.0 \
      ruff==0.17.0 \
      mypy==2.4.0 \
      typos==1.51.1 \
      ast_serialize==0.12.1 \
      iniconfig==2.3.1 \
      librt==0.16.0 \
      mypy_extensions==1.1.0 \
      pathspec==1.1.1 \
      pluggy==1.6.0

FROM ${LINT_BASE} AS lint
# git: pre-commit needs it to list files; the repository is bind-mounted at
# run time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
# clang-format 23 and shellcheck 0.11 as PyPI binary wheels; pre-commit to run
# .pre-commit-config.yaml offline (all hooks are `repo: local`,
# `language: system`).
RUN /app/venv/bin/pip install --no-cache-dir \
      clang-format==23.1.3 \
      shellcheck-py==0.11.0.1 \
      pre-commit==4.6.2 \
      cfgv==3.5.0 \
      distlib==0.4.3 \
      identify==2.6.20 \
      nodeenv==1.11.0 \
      platformdirs==4.13.0 \
      python-discovery==1.6.2 \
      virtualenv==21.14.6
