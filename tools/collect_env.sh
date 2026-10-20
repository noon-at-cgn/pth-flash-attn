#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Read-only environment capture for benchmark and bug-report rows:
# driver, CUDA, torch, vLLM, git commit, GPU power limit and clocks.
#
# Usage: tools/collect_env.sh [OUTPUT_FILE]   (default: stdout)
#
# Read-only by construction: it only runs queries (`nvidia-smi --query-gpu`,
# `--version` flags, `git rev-parse`, package metadata). It never changes
# clocks, power limits, persistence or application clocks, never resets a
# GPU, and never imports torch or vllm (an import could initialise CUDA);
# versions come from package metadata instead. tests/cpu/test_tools_env.py
# enforces this with a scanner and a negative control.
#
# Every probe is optional: a missing tool prints "<tool>: not available" and
# the script still exits 0, so it works in the CPU-only dev container too.
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ge 1 ]]; then
  exec >"$1"
fi

section() { printf '\n## %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

section "time (UTC)"
date -u +%Y-%m-%dT%H:%M:%SZ

section "host"
uname -srm
if have systemd-detect-virt; then
  echo "virtualisation: $(systemd-detect-virt 2>/dev/null || true)"
fi

section "git"
if have git && git -C "${repo}" rev-parse HEAD >/dev/null 2>&1; then
  echo "commit: $(git -C "${repo}" rev-parse HEAD)"
  echo "dirty files: $(git -C "${repo}" status --porcelain | wc -l)"
else
  echo "git: not available"
fi

section "GPU (nvidia-smi, read-only queries)"
if have nvidia-smi; then
  # An unknown field fails the whole query, so the one field whose name
  # changed across driver versions is queried separately below.
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,power.default_limit,power.max_limit,clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv 2>&1 || true
  # Throttle reasons were renamed "clocks_event_reasons" in newer drivers.
  nvidia-smi --query-gpu=index,clocks_event_reasons.active --format=csv 2>/dev/null ||
    nvidia-smi --query-gpu=index,clocks_throttle_reasons.active --format=csv 2>&1 ||
    true
else
  echo "nvidia-smi: not available"
fi
if [[ -r /proc/driver/nvidia/version ]]; then
  head -n 1 /proc/driver/nvidia/version
fi

section "CUDA toolkit"
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  /usr/local/cuda/bin/nvcc --version | tail -n 2
elif have nvcc; then
  nvcc --version | tail -n 2
else
  echo "nvcc: not available"
fi

section "Python packages (metadata only, no import)"
for py in /app/venv/bin/python python3; do
  if have "${py}"; then
    echo "interpreter: ${py}"
    "${py}" - <<'EOF' 2>&1 || true
import importlib.metadata as md
import platform

print(f"python: {platform.python_version()}")
for name in ("torch", "vllm", "triton", "flashinfer-python", "numpy"):
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: not installed")
EOF
    break
  fi
done

section "container image (if run under docker)"
if [[ -f /.dockerenv ]]; then
  echo "inside a container; image digest is recorded by the caller"
else
  echo "not in a container"
fi
exit 0
