#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
#
# Proves that two trees of the vendored sources compile to the same machine
# code. Both trees are compiled by the same nvcc with the flags of their own
# tools/build_flags.py (which must be equal), for the compute capabilities of
# build_flags.ARCHS (sm_80, sm_86 and sm_89), and tools/sass_diff.py must find
# every kernel of every architecture identical: same instructions, same
# encodings, same order. That is the strongest statement that a source change
# is free: it needs no GPU and no timing run.
#
#   tools/prove_sass_equal.sh [options] LEFT RIGHT
#
# LEFT and RIGHT are trees that hold csrc/pth_attn/ and tools/build_flags.py:
# a vendoring project's copy of the sources, or a checkout of this repository.
#
#   --left-export, --right-export
#       that side is a repository checkout, not a vendored copy: its tree is
#       what tools/export_upstream.py --out writes (the test hooks stripped),
#       plus the five build files, which is what a vendoring project holds.
#   --extension
#       also build the op library pth_flash_attn/_C.abi3.so from both trees with
#       "python setup.py build_ext" (needs torch, ninja and g++) and compare
#       the kernels of the two libraries. Without it, the kernel translation
#       units are compiled directly and linked into one library per side; that
#       needs nvcc only. launch.cu and torch_bindings.cpp hold no kernels and are
#       compiled only in this mode.
#   --out DIR     keep the objects, libraries and logs here (default: a
#                 temporary directory, removed on exit)
#
# Environment: NVCC (default nvcc, or /usr/local/cuda/bin/nvcc), PYTHON
# (default /app/venv/bin/python if present, else python3), MAX_JOBS for the
# extension build.
#
# Exit 0 identical, 1 a SASS difference, 2 a usage, compile or tool failure.
# The compilers of both sides are the same binary, so the result says that the
# sources are equivalent for that nvcc; it does not depend on the nvcc version,
# but it is only a statement about the version that ran it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sass_diff="${here}/sass_diff.py"

usage() {
  echo "usage: tools/prove_sass_equal.sh [--left-export] [--right-export]" \
    "[--extension] [--out DIR] LEFT RIGHT" >&2
  exit 2
}

export_left=0
export_right=0
extension=0
out=""
trees=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --left-export) export_left=1 ;;
    --right-export) export_right=1 ;;
    --extension) extension=1 ;;
    --out) out="${2:?--out needs a directory}"; shift ;;
    -h | --help) usage ;;
    -*) echo "prove_sass_equal: unknown option $1" >&2; usage ;;
    *) trees+=("$1") ;;
  esac
  shift
done
[[ ${#trees[@]} -eq 2 ]] || usage

py=python3
[[ -x /app/venv/bin/python ]] && py=/app/venv/bin/python
py="${PYTHON:-${py}}"
nvcc="${NVCC:-nvcc}"
if ! command -v "${nvcc}" > /dev/null 2>&1 && [[ -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc=/usr/local/cuda/bin/nvcc
fi
command -v "${nvcc}" > /dev/null 2>&1 || { echo "prove_sass_equal: no nvcc" >&2; exit 2; }

if [[ -z "${out}" ]]; then
  out="$(mktemp -d "${TMPDIR:-/tmp}/sass-equal.XXXXXX")"
  trap 'rm -rf "${out}"' EXIT
else
  mkdir -p "${out}"
fi
out="$(cd "${out}" && pwd)"

echo "== prove_sass_equal: $("${nvcc}" --version | tail -n 1)"

# The vendored tree of one side, under ${out}/<side>/tree.
materialise() { # side path export
  local side="$1" path="$2" exported="$3" dest="${out}/$1/tree"
  mkdir -p "${dest}"
  if [[ "${exported}" -eq 1 ]]; then
    "${py}" "${path}/tools/export_upstream.py" --out "${dest}" > /dev/null
    mkdir -p "${dest}/tools"
    for f in setup.py pyproject.toml tools/build_flags.py LICENSE NOTICE; do
      cp "${path}/${f}" "${dest}/${f}"
    done
  else
    cp -r "${path}/." "${dest}/"
  fi
  [[ -f "${dest}/csrc/pth_attn/attn_kernel.cuh" && -f "${dest}/tools/build_flags.py" ]] \
    || { echo "prove_sass_equal: ${side}: ${path} is not a vendored tree" >&2; exit 2; }
}
materialise left "${trees[0]}" "${export_left}"
materialise right "${trees[1]}" "${export_right}"

# What one side compiles with: the nvcc flags, the translation units and the
# architectures, read from that side's own tools/build_flags.py.
describe() { # side
  (cd "${out}/$1/tree" && "${py}" -c '
import sys
sys.path.insert(0, "tools")
import build_flags as b
print("flags", *b.NVCC_FLAGS, *b.define_flags())
print("archs", *b.ARCHS)
for s in b.EXTENSION_SOURCES:
    print("tu", s)
')
}
describe left > "${out}/left/build.txt"
describe right > "${out}/right/build.txt"
if ! diff -u "${out}/left/build.txt" "${out}/right/build.txt"; then
  echo "prove_sass_equal: the two trees compile with different flags or sources" >&2
  exit 2
fi
mapfile -t flags < <(awk '$1 == "flags" { for (i = 2; i <= NF; i++) print $i }' "${out}/left/build.txt")
mapfile -t archs < <(awk '$1 == "archs" { for (i = 2; i <= NF; i++) print $i }' "${out}/left/build.txt")
mapfile -t tus < <(awk '$1 == "tu" && $2 ~ /\.cu$/ && $2 !~ /\/launch\.cu$/ { print $2 }' "${out}/left/build.txt")
gencode=()
for a in "${archs[@]}"; do gencode+=("-gencode=arch=compute_${a},code=sm_${a}"); done
echo "== flags: ${flags[*]} ${gencode[*]}"

compile_side() { # side
  local side="$1" tree="${out}/$1/tree" tu
  local -a pids=()
  mkdir -p "${out}/${side}/obj"
  for tu in "${tus[@]}"; do
    local name
    name="$(basename "${tu}" .cu)"
    "${nvcc}" "${flags[@]}" -Xcompiler -fPIC -I"${tree}/csrc/pth_attn" "${gencode[@]}" \
      -c "${tree}/${tu}" -o "${out}/${side}/obj/${name}.o" \
      > "${out}/${side}/obj/${name}.log" 2>&1 &
    pids+=("$!")
  done
  local i
  for i in "${!tus[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "== compile of ${side}/${tus[$i]} failed:" >&2
      cat "${out}/${side}/obj/$(basename "${tus[$i]}" .cu).log" >&2
      exit 2
    fi
  done
  local -a objs=()
  for tu in "${tus[@]}"; do objs+=("${out}/${side}/obj/$(basename "${tu}" .cu).o"); done
  "${nvcc}" -shared -cudart static "${gencode[@]}" "${objs[@]}" \
    -o "${out}/${side}/libsubset.so"
}

status=0
echo "== kernel translation units (${#tus[@]}): ${tus[*]}"
compile_side left
compile_side right
echo "== sass_diff: kernel translation units, archs ${archs[*]}"
"${py}" "${sass_diff}" "${out}/left/libsubset.so" "${out}/right/libsubset.so" \
  --json "${out}/subset.json" || status=$?

if [[ "${extension}" -eq 1 && ${status} -eq 0 ]]; then
  build_extension() { # side
    local tree="${out}/$1/tree"
    [[ -f "${tree}/README.md" ]] || : > "${tree}/README.md" # pyproject.toml names it
    (cd "${tree}" && "${py}" setup.py build_ext --inplace > "${out}/$1/setup.log" 2>&1) || {
      echo "== extension build of $1 failed:" >&2
      tail -n 40 "${out}/$1/setup.log" >&2
      exit 2
    }
  }
  echo "== building the op library from both trees (setup.py build_ext)"
  build_extension left
  build_extension right
  echo "== sass_diff: _C.abi3.so"
  "${py}" "${sass_diff}" \
    "${out}/left/tree/python/pth_flash_attn/_C.abi3.so" \
    "${out}/right/tree/python/pth_flash_attn/_C.abi3.so" \
    --json "${out}/extension.json" || status=$?
fi

if [[ ${status} -eq 0 ]]; then
  echo "== prove_sass_equal: PASS (every kernel identical)"
else
  echo "== prove_sass_equal: FAIL"
fi
exit "${status}"
