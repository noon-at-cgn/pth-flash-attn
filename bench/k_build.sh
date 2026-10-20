#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# k_build.sh: build everything a kernel benchmark run (bench/k_run.sh) needs, CPU only
# (offline containers, no --gpus), and record its hashes.
#
#   bench/k_build.sh [--kernel attn|ref] [--attn-commit REV] [--image IMAGE]
#
# Needs: docker, git, and the dev image (docker/dev.Dockerfile, default tag
# pth-flash-attn-dev, with nvcc 13, g++ and python3 on its PATH). No GPU is needed.
#
# Steps (each build script starts its own CPU-only container):
#   1. bench/ceilings_build.sh     sol_copy, mma_ceiling (the ceilings), with
#                                  their ptxas/SASS checks
#   2. tools/build_ref_lib.sh      the reference library (the `ref` arm)
#   3. the kernel under test       --kernel attn (default): the attention
#                                  library is built from a COMMIT
#                                  (--attn-commit, default HEAD): `git archive`
#                                  of that commit into build/k/src, its own
#                                  tools/build_attn_lib.sh there, and the
#                                  result copied to build/k/libpth_attn.so.
#                                  So the timed library is exactly a commit
#                                  (records can cite it), and a rebuild of
#                                  build/attn in the working tree cannot race
#                                  a running benchmark.
#                                  --kernel ref times the reference library
#                                  as a stand-in kernel instead.
#   4. nvcc bench/cupti_timer.cu   build/k/libpth_cupti_timer.so (the CUPTI
#                                  timer behind a C ABI; CUPTI is found as in
#                                  bench/ceilings_build.sh)
#   5. build/k/SHA256SUMS          sha256 of every file above (paths from the
#                                  repo root) and build/k/KERNEL (kind and
#                                  source commit); k_run.sh refuses to run
#                                  if any file differs. Written only after
#                                  all steps passed.
# Exit: 0 ok; 2 a build failed (the failing step's exit code is printed);
# 64 usage.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="pth-flash-attn-dev"
kernel="attn"
attn_commit="HEAD"
while (($#)); do
  case "$1" in
    --kernel)
      kernel="${2:?--kernel needs attn or ref}"
      shift
      ;;
    --attn-commit)
      attn_commit="${2:?--attn-commit needs a revision}"
      shift
      ;;
    --image)
      image="${2:?--image needs a value}"
      shift
      ;;
    -h | --help)
      sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 64
      ;;
  esac
  shift
done
[[ "${kernel}" == attn || "${kernel}" == ref ]] || {
  echo "--kernel must be attn or ref" >&2
  exit 64
}

sums="${repo}/build/k/SHA256SUMS"
rm -f "${sums}" # a stale file must never look like this run's result
mkdir -p "${repo}/build/k"

step() { # step NAME CMD...: run, stop with exit 2 on failure
  local name="$1" rc=0
  shift
  echo "=== ${name}"
  "$@" || rc=$?
  if ((rc != 0)); then
    echo "k_build: ${name} failed (exit ${rc})" >&2
    exit 2
  fi
}

step "ceilings_build" "${repo}/bench/ceilings_build.sh" --image "${image}"
step "build_ref_lib" "${repo}/tools/build_ref_lib.sh"
# A run times and hashes only copies in build/k: build/ceilings and build/ref may be
# rebuilt at any time (by CI or another checkout), and a hash mismatch in the middle
# of a long run voids it.
step "snapshot" cp "${repo}/build/ceilings/sol_copy" "${repo}/build/ceilings/mma_ceiling" \
  "${repo}/build/ref/libpth_ref.so" "${repo}/build/k/"
libs=(build/k/sol_copy build/k/mma_ceiling build/k/libpth_ref.so
  build/k/libpth_cupti_timer.so)
source_sha="-"
if [[ "${kernel}" == attn ]]; then
  source_sha="$(git -C "${repo}" rev-parse --verify --quiet "${attn_commit}^{commit}")" || {
    echo "k_build: --attn-commit ${attn_commit} is not a commit" >&2
    exit 2
  }
  src="${repo}/build/k/src"
  rm -rf "${src}"
  mkdir -p "${src}"
  echo "=== kernel source: git archive ${source_sha} -> build/k/src"
  git -C "${repo}" archive "${source_sha}" | tar -x -C "${src}" || {
    echo "k_build: git archive of ${source_sha} failed" >&2
    exit 2
  }
  [[ -x "${src}/tools/build_attn_lib.sh" ]] || {
    echo "k_build: tools/build_attn_lib.sh does not exist at ${source_sha} (use --kernel ref)" >&2
    exit 2
  }
  step "build_attn_lib" "${src}/tools/build_attn_lib.sh"
  [[ -s "${src}/build/attn/libpth_attn.so" ]] || {
    echo "k_build: the build of ${source_sha} produced no libpth_attn.so" >&2
    exit 2
  }
  cp "${src}/build/attn/libpth_attn.so" "${repo}/build/k/libpth_attn.so"
  libs+=(build/k/libpth_attn.so)
fi
# No stdin is fed to this container: the command is an argument, in single
# quotes because ${c} must expand inside the container, not here. CUPTI is looked
# up as in bench/ceilings_build.sh: $CUPTI_HOME, else the nvidia-cuda-cupti wheel of
# the image's Python, else the CUDA toolkit's extras/CUPTI.
# shellcheck source=tools/ci_cpus.sh
source "${repo}/tools/ci_cpus.sh"
# shellcheck disable=SC2016
step "cupti_timer" docker run --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
  -v "${repo}:/w" -w /w -e CUDA_VISIBLE_DEVICES= -e CUPTI_HOME \
  --entrypoint /bin/bash "${image}" -c '
    set -euo pipefail
    c="${CUPTI_HOME:-}"
    if [[ -z "${c}" ]]; then
      c="$(python3 -c "import importlib.util as u, pathlib as p
s = u.find_spec(\"nvidia\")
for b in (list(s.submodule_search_locations or []) if s else []):
    d = p.Path(b) / \"cu13\"
    if (d / \"include\" / \"cupti.h\").is_file():
        print(d)
        break" 2>/dev/null || true)"
    fi
    [[ -n "${c}" ]] || c=/usr/local/cuda/extras/CUPTI
    l="${c}/lib"
    [[ -d "${l}" ]] || l="${c}/lib64"
    /usr/local/cuda/bin/nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
      -I"${c}/include" -o build/k/libpth_cupti_timer.so bench/cupti_timer.cu \
      -L"${l}" -l:libcupti.so.13 -Xlinker -rpath="${l}"'

for f in "${libs[@]}"; do
  [[ -s "${repo}/${f}" ]] || {
    echo "k_build: ${f} missing after the build" >&2
    exit 2
  }
done
printf 'kernel %s\nsource %s\n' "${kernel}" "${source_sha}" >"${repo}/build/k/KERNEL"
(cd "${repo}" && sha256sum "${libs[@]}" build/k/KERNEL) >"${sums}.tmp"
mv "${sums}.tmp" "${sums}"
echo "k_build: ok (${kernel}, source ${source_sha}); hashes in build/k/SHA256SUMS"
