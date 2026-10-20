#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# One command for every CPU check: the same steps as .github/workflows/ci.yml,
# run in the pinned images (docker/dev.Dockerfile) with --network none and no
# GPU. Run it on any machine with Docker once the two images are built:
#
#   tools/run_ci_local.sh            # all steps
#   tools/run_ci_local.sh lint test  # only the named groups
#
# Groups, in order:
#   host   DCO sign-off of every commit (needs git, which only the host has)
#   lint   ruff check, ruff format --check, typos, mypy, clang-format, the
#          ShellCheck linter, SPDX headers, engine-facing check, pre-commit
#          config
#   build  the op extension for sm_80/86/89 via tools/build_in_image.sh, then
#          load it on CPU and compare its schema with docs/api.md; the
#          reference and loader test libraries (build/ref, build/loader),
#          which CPU tests load (e.g. the host emulator of build/ref)
#   test   pytest tests/cpu tests/sim; tests/cpp/run_geometry.sh
#   cost   compile_cost.py (per TU and per arch; .so bytes per arch) and
#          ptxas_gate.py on the op's host TU; reports in build/ci/
#   attn   the attention kernels (csrc/pth_attn/attn_*.cuh, built by
#          tools/build_attn_lib.sh): the library build (0 spills, no stack
#          frames), the ptxas gate on its logs (<= 255 registers, 0 spills,
#          every kernel family on every arch) and the SASS lint of the three
#          instantiation TUs (no local memory, no I2F in loops, LDGSTS, LDSM
#          and HMMA.16816.F32 in every kernel)
#
# The register probe is not a CI group: the attn group gates the real kernels.
#
# Images: pth-flash-attn-dev (build, tests), pth-flash-attn-lint (lint tools);
# both from docker/dev.Dockerfile. Every step runs even if an earlier one
# failed; the summary at the end lists each step's result and the script
# exits 1 if any step failed. Build the images once (network needed for the
# build only):
#   docker build -f docker/dev.Dockerfile --target dev -t pth-flash-attn-dev .
#   docker build -f docker/dev.Dockerfile --target lint \
#     --build-arg LINT_BASE=pth-flash-attn-dev -t pth-flash-attn-lint .
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dev_image="pth-flash-attn-dev"
lint_image="pth-flash-attn-lint"
groups=("$@")
[[ ${#groups[@]} -eq 0 ]] && groups=(host lint build test cost attn)

declare -a summary=()
failed=0

wants() {
  local g
  for g in "${groups[@]}"; do [[ "${g}" == "$1" ]] && return 0; done
  return 1
}

# CPU limit of every container: min(4, nproc), or PTH_CI_CPUS (tools/ci_cpus.sh).
# shellcheck source=tools/ci_cpus.sh
source "${repo}/tools/ci_cpus.sh"
ci_cpus="$(pth_ci_cpus)" || exit 2

# in_image IMAGE SCRIPT: run a bash script in a CPU-only, offline container.
# Called indirectly, as `step NAME in_image ...` ("$@" in step).
# shellcheck disable=SC2329
in_image() {
  docker run --rm --network none --cpus "${ci_cpus}" --memory 16g \
    -v "${repo}:/w" -w /w \
    -e PYTHONPATH=/w/python \
    -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
    -e PATH=/app/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --entrypoint nice "$1" -n 19 /bin/bash -c "set -euo pipefail; $2"
}

# step NAME COMMAND...: run, time and record one step.
step() {
  local name="$1"
  shift
  printf '\n=== %s\n' "${name}"
  local start end rc
  start=$(date +%s)
  "$@"
  rc=$?
  end=$(date +%s)
  if [[ ${rc} -eq 0 ]]; then
    summary+=("PASS  ${name} ($((end - start)) s)")
  else
    summary+=("FAIL  ${name} (exit ${rc}, $((end - start)) s)")
    failed=1
  fi
}

cpp_files='find . -path ./build -prune -o -path ./.git -prune -o \( -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.hpp" -o -name "*.cpp" \) -print'
sh_files='find . -path ./build -prune -o -path ./.git -prune -o -name "*.sh" -print'

if wants host; then
  step "host: DCO sign-off on every commit" \
    python3 "${repo}/tools/repo_checks.py" signoff HEAD
fi

if wants lint; then
  step "lint: ruff check" in_image "${lint_image}" "ruff check ."
  step "lint: ruff format --check" in_image "${lint_image}" "ruff format --check ."
  step "lint: typos" in_image "${lint_image}" "typos"
  step "lint: mypy" in_image "${lint_image}" "mypy"
  step "lint: clang-format (Google, --style=file)" in_image "${lint_image}" \
    "${cpp_files} | xargs -r clang-format --style=file --dry-run --Werror"
  step "lint: shellcheck" in_image "${lint_image}" \
    "${sh_files} | xargs -r shellcheck -s bash"
  step "lint: SPDX headers" in_image "${lint_image}" "python tools/repo_checks.py spdx"
  step "lint: engine-facing code (no torch.cuda, no env vars)" in_image "${lint_image}" \
    "python tools/repo_checks.py engine-facing"
  step "lint: pre-commit config" in_image "${lint_image}" \
    "pre-commit validate-config .pre-commit-config.yaml"
fi

if wants build; then
  step "build: op extension for 8.0;8.6;8.9" "${repo}/tools/build_in_image.sh" "8.0;8.6;8.9"
  step "build: load the op on CPU, schema = docs/api.md" in_image "${dev_image}" \
    "python -c '
import torch
torch.ops.load_library(\"python/pth_flash_attn/_C.abi3.so\")
op = torch.ops.pth_flash_attn.pth_flash_attn_fwd.default
print(op._schema)
'"
  step "build: reference libraries (build/ref)" "${repo}/tools/build_ref_lib.sh"
  step "build: loader test libraries (build/loader)" "${repo}/tools/build_loader_lib.sh"
fi

if wants test; then
  # -rs prints skip reasons (a skipped op test would mean a missing build);
  # --durations shows where the suite's time goes; build/ci/pytest.xml lets
  # two runs be compared test by test. "full and slow" is the emulator over
  # the whole GPU correctness grid (41 min at 16 workers): run by hand, never
  # in CI.
  step "test: pytest tests/cpu tests/sim" in_image "${dev_image}" \
    "mkdir -p build/ci && python -m pytest tests/cpu tests/sim -q -rs --durations=15 -m 'not (full and slow)' --junitxml=build/ci/pytest.xml"
  # cache_geometry.cuh: g++ test plus nvcc device instantiation for
  # sm_80/86/89 (exit 0 pass, 1 test failure, 2 build failure).
  step "test: tests/cpp/run_geometry.sh" in_image "${dev_image}" \
    "bash tests/cpp/run_geometry.sh"
fi

if wants cost; then
  step "cost: compile time and size per TU/arch, .so per arch" in_image "${dev_image}" \
    "mkdir -p build/ci && python tools/compile_cost.py \
       --tu csrc/pth_attn/launch.cu csrc/pth_attn/torch_bindings.cpp \
       --extension --json build/ci/compile_cost.json"
  step "cost: ptxas gate on the op's host TU" in_image "${dev_image}" \
    "mkdir -p build/ci && python tools/ptxas_gate.py csrc/pth_attn/launch.cu \
       --json build/ci/ptxas_gate.json"
  # The op's kernels are in the instantiation TUs; the attn group gates them.
fi

if wants attn; then
  # build_attn_lib.sh runs the library build directly inside the image and
  # exits 1 on a spill or stack frame.
  step "attn: build libpth_attn.so (ptxas: 0 spills, no stack frames)" \
    in_image "${dev_image}" "tools/build_attn_lib.sh"
  step "attn: ptxas gate on the library's kernels, sm_80/86/89" \
    in_image "${dev_image}" "python tools/attn_ci_gates.py"
  step "attn: SASS lint of the kernels, sm_80/86/89" in_image "${dev_image}" \
    "mkdir -p build/ci/attn && python tools/sass_check.py \
       --cu csrc/pth_attn/instantiate_d64.cu csrc/pth_attn/instantiate_d128.cu \
            csrc/pth_attn/instantiate_d256.cu \
       --out-dir build/ci/attn --require LDGSTS,LDSM,HMMA.16816.F32 \
       --json build/ci/attn_sass.json"
fi

printf '\n=== summary\n'
printf '%s\n' "${summary[@]}"
exit "${failed}"
