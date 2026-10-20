#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# ceilings_build.sh: build and statically check the hardware-ceiling programs
# (bench/sol_copy.cu, bench/mma_ceiling.cu).
#
#   bench/ceilings_build.sh [--image IMAGE] [--out DIR]
#
# Runs entirely in a CPU-only, offline container (no --gpus): it needs no GPU and
# does not touch one. Needs docker and the dev image (docker/dev.Dockerfile, default
# tag pth-flash-attn-dev, with nvcc 13 and python3 on its PATH). Steps:
#   1. nvcc builds build/ceilings/sol_copy and build/ceilings/mma_ceiling for sm_80,
#      sm_86 and sm_89 (SASS only, the extension's -O3 -std=c++17), linked against
#      the image's CUPTI (libcupti.so.13 from nvidia-cuda-cupti; rpath set so the
#      binaries find it inside the same image when bench/ceilings_run.sh runs
#      them). CUPTI is looked up in $CUPTI_HOME (a directory with include/ and
#      lib/ or lib64/), else in the nvidia-cuda-cupti wheel of the image's Python
#      (nvidia/cu13), else in /usr/local/cuda/extras/CUPTI.
#   2. tools/ptxas_gate.py: 0 spill bytes, <= 255 registers, every kernel.
#   3. tools/sass_check.py on per-arch cubins: no LDL/STL, and the
#      instructions each ceiling exists to measure are present:
#        separate-stream loader  LDGSTS.E (4 B), LDGSTS.E.64 (8 B),
#                                LDGSTS.E.BYPASS.128 (16 B), BAR.SYNC, LDS
#        whole-cell loader       LDGSTS.E.64
#        plain stream            LDG.E.128
#        mma f16f32 / f16f16 / bf16f32 / s8s32
#                                HMMA.16816.F32 / HMMA.16816.F16 /
#                                HMMA.16816.F32.BF16 / IMMA.16832.S8.S8
#      plus a negative control: the f16-accumulate kernels must NOT contain
#      HMMA.16816.F32 (if they did, the check could not tell the kinds apart).
#   4. sol_copy --self-test (host code: FastDivmod, widths, split partition).
#   5. SHA256SUMS of the binaries; ceilings_run.sh refuses binaries that differ.
# Output: build/ceilings/{sol_copy,mma_ceiling,SHA256SUMS,cubin/,sass/,ptxas/}.
# Exit: 0 all passed; 1 a check failed; 2 build or tool failure; 64 usage.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="pth-flash-attn-dev"
out="build/ceilings"
while (($#)); do
  case "$1" in
    --image)
      image="${2:?--image needs a value}"
      shift
      ;;
    --out)
      out="${2:?--out needs a directory relative to the repo}"
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
case "${out}" in
  /* | *..*)
    echo "--out must be a path inside the repository" >&2
    exit 64
    ;;
esac

# The build script runs inside the container: bash -s reads it from stdin,
# "$out" is passed as $1. `-i` is essential: without it docker does not
# attach stdin, bash -s reads EOF and exits 0, and the whole build is a
# silent no-op (tests/cpu/test_ceilings_scripts.py guards it). As a second guard, SHA256SUMS must be newer than a stamp taken
# here, i.e. written by this run, or the script fails with exit 2.
stamp="$(mktemp)"
trap 'rm -f "${stamp}"' EXIT
rc=0
# shellcheck source=tools/ci_cpus.sh
source "${repo}/tools/ci_cpus.sh"
docker run -i --rm --network none --cpus "$(pth_ci_cpus)" --memory 16g \
  -v "${repo}:/w" -w /w \
  -e PYTHONPATH=/w/python \
  -e CUDA_VISIBLE_DEVICES= \
  -e CUPTI_HOME \
  --entrypoint /bin/bash "${image}" -s -- "${out}" <<'IN_CONTAINER' || rc=$?
set -euo pipefail
out="$1"
rm -f "${out}/SHA256SUMS"  # a stale file must never look like this run's result
cupti="${CUPTI_HOME:-}"
if [[ -z "${cupti}" ]]; then
  cupti="$(python3 - <<'LOOKUP' 2>/dev/null || true
import importlib.util
import pathlib

spec = importlib.util.find_spec("nvidia")
for base in list(spec.submodule_search_locations or []) if spec else []:
    cand = pathlib.Path(base) / "cu13"
    if (cand / "include" / "cupti.h").is_file():
        print(cand)
        break
LOOKUP
)"
fi
[[ -n "${cupti}" ]] || cupti=/usr/local/cuda/extras/CUPTI
cupti_lib="${cupti}/lib"
[[ -d "${cupti_lib}" ]] || cupti_lib="${cupti}/lib64"
archs=(80 86 89)
mkdir -p "${out}/cubin" "${out}/sass"
gencode=()
for a in "${archs[@]}"; do gencode+=("-gencode=arch=compute_${a},code=sm_${a}"); done
common=(-O3 -std=c++17 "-I${cupti}/include")
link=("-L${cupti_lib}" -l:libcupti.so.13 "-Xlinker" "-rpath=${cupti_lib}")

echo "=== build"
for prog in sol_copy mma_ceiling; do
  nvcc "${common[@]}" "${gencode[@]}" -o "${out}/${prog}" "bench/${prog}.cu" "${link[@]}" || exit 2
  for a in "${archs[@]}"; do
    nvcc "${common[@]}" "-gencode=arch=compute_${a},code=sm_${a}" -cubin \
      -o "${out}/cubin/${prog}_sm${a}.cubin" "bench/${prog}.cu" || exit 2
  done
done

echo "=== ptxas gate (tools/ptxas_gate.py)"
python3 tools/ptxas_gate.py bench/sol_copy.cu bench/mma_ceiling.cu \
  "--nvcc-flag=-I${cupti}/include" --out-dir "${out}/ptxas" \
  --json "${out}/ptxas.json" --min-kernels 10

echo "=== SASS checks (tools/sass_check.py)"
fail=0
n=0
# check EXPECT ARCH PROGRAM FUNCTION_REGEX REQUIRE...
check() {
  local expect="$1" a="$2" prog="$3" fn="$4" rc
  shift 4
  local req=()
  for r in "$@"; do req+=(--require "${r}"); done
  n=$((n + 1))
  set +e
  python3 tools/sass_check.py --cubin "${out}/cubin/${prog}_sm${a}.cubin" \
    --arch "${a}" --function "${fn}" "${req[@]}" \
    --out-dir "${out}/sass" --json "${out}/sass/check_${n}.json" \
    >"${out}/sass/check_${n}.log" 2>&1
  rc=$?
  set -e
  if [[ "${rc}" == "${expect}" ]]; then
    echo "ok   sm_${a} ${fn} requires [$*] -> exit ${rc} (expected ${expect})"
  else
    echo "FAIL sm_${a} ${fn} requires [$*] -> exit ${rc} (expected ${expect}); see ${out}/sass/check_${n}.log"
    fail=1
  fi
}
for a in "${archs[@]}"; do
  check 0 "${a}" sol_copy 'sol_copy_pagedILi(64|128|256)ELb0E' \
    LDGSTS.E LDGSTS.E.64 LDGSTS.E.BYPASS.128 BAR.SYNC LDS
  check 0 "${a}" sol_copy 'sol_copy_pagedILi(64|128|256)ELb1E' LDGSTS.E.64 BAR.SYNC
  # The loader.cuh arms: same instruction families.
  check 0 "${a}" sol_copy 'sol_loader_pagedILi(64|128|256)ELb0E' \
    LDGSTS.E LDGSTS.E.64 LDGSTS.E.BYPASS.128 BAR.SYNC LDS
  check 0 "${a}" sol_copy 'sol_loader_pagedILi(64|128|256)ELb1E' LDGSTS.E.64 BAR.SYNC
  check 0 "${a}" sol_copy 'sol_stream_plainILi16E' LDG.E.128
  check 0 "${a}" mma_ceiling 'mma_ceiling_kernelILNS_4KindE0E' HMMA.16816.F32
  check 0 "${a}" mma_ceiling 'mma_ceiling_kernelILNS_4KindE1E' HMMA.16816.F16
  check 0 "${a}" mma_ceiling 'mma_ceiling_kernelILNS_4KindE2E' HMMA.16816.F32.BF16
  check 0 "${a}" mma_ceiling 'mma_ceiling_kernelILNS_4KindE3E' IMMA.16832.S8.S8
  # negative control: must fail (exit 1)
  check 1 "${a}" mma_ceiling 'mma_ceiling_kernelILNS_4KindE1E' HMMA.16816.F32
done

echo "=== host self-test"
"${out}/sol_copy" --self-test

echo "=== result"
if ((fail)); then
  echo "SASS checks FAILED"
  exit 1
fi
# Only binaries that passed every gate get checksums; ceilings_run.sh runs nothing
# without them.
(cd "${out}" && sha256sum sol_copy mma_ceiling >SHA256SUMS)
cat "${out}/SHA256SUMS"
echo "all checks passed; binaries and SHA256SUMS in ${out}"
IN_CONTAINER
if ((rc != 0)); then
  echo "ceilings_build: the in-container build failed (exit ${rc})" >&2
  exit "${rc}"
fi
if [[ ! "${repo}/${out}/SHA256SUMS" -nt "${stamp}" ]]; then
  echo "ceilings_build: no fresh ${out}/SHA256SUMS after the container exited 0:" \
    "the build did not run (stdin not attached?)" >&2
  exit 2
fi
