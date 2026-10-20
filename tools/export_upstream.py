#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The upstream subset: the files a downstream build vendors, and a copy of
them without the handshake test hooks.

What ships is defined by the build, not by a hand-kept list: the extension's
translation units (``tools/build_flags.py`` ``EXTENSION_SOURCES``) plus every
header they include with ``#include "..."``, followed transitively, plus the
modules of the ``pth_flash_attn`` package. Everything else (the reference
kernel, the loader and fragment test libraries, the C ABI of the test
libraries, ``test_hooks.cuh``, ``python/pth_testing``, ``bench/``) is test and
benchmark material that stays in this repository.
``tests/cpu/test_upstream_subset.py`` checks that it contains no test layer.

The kernels keep about twenty hook calls (``proto_*``, ``test_jitter``) and
three ``kTestMutate`` conditions: the only race test of the producer/consumer
handshake. In every build but the test libraries they
are empty inline functions and constant-false conditions, defined in the
``BEGIN/END pth-test-hooks`` block of ``attn_kernel.cuh``. ``--out`` writes the
subset with that block and every hook call removed and each mutation
condition unwrapped; ``tools/prove_export_sass.sh`` proves the copy compiles
to the same SASS as the repository (so the strip is a no-op by construction).

    tools/export_upstream.py --list
    tools/export_upstream.py --out build/upstream
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402

REPO = build_flags.REPO_ROOT
PACKAGE_DIR = REPO / "python" / "pth_flash_attn"

HOOKS = (
    "test_jitter",
    "proto_init",
    "proto_count",
    "proto_rewrite",
    "proto_landed",
    "proto_expect_landed",
    "proto_check_ledger",
    "proto_epoch_set",
    "proto_epoch_expect",
)
# Identifiers that must not survive the strip.
TEST_TOKENS = re.compile(
    r"\b(proto_\w+|test_jitter|kTestMutate|PTH_ATTN_TEST_\w*)\b|test_hooks"
)

_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
_BLOCK_BEGIN = "// BEGIN pth-test-hooks"
_BLOCK_END = "// END pth-test-hooks"
_HOOK_CALL = re.compile(
    r"^\s*(?:if (?:constexpr )?\(.*\) )?(?:" + "|".join(HOOKS) + r")\("
)
_MUTATE_IF = re.compile(r"^(\s*)if (?:constexpr )?\(kTestMutate != \d+\) \{$")


def shipped_csrc() -> list[str]:
    """Repo-relative paths of the extension's sources and their includes."""
    todo = [REPO / s for s in build_flags.EXTENSION_SOURCES]
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _INCLUDE.findall(path.read_text(encoding="utf-8")):
            included = path.parent / name
            if included.exists():
                todo.append(included)
    return sorted(str(p.relative_to(REPO)) for p in seen)


def shipped_python() -> list[str]:
    """Repo-relative paths of the shipped package's modules."""
    return sorted(str(p.relative_to(REPO)) for p in PACKAGE_DIR.glob("*.py"))


def shipped_files() -> list[str]:
    """The whole upstream subset, sorted."""
    return sorted(shipped_csrc() + shipped_python())


def strip_test_hooks(text: str) -> str:
    """`text` without the default-hook block, hook calls and mutation ifs.

    Raises ValueError if a test identifier survives (a hook used in a form
    this function does not know must fail loudly, not ship).
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == _BLOCK_BEGIN:
            while lines[i].strip() != _BLOCK_END:
                i += 1
            i += 1
            if i < len(lines) and not lines[i].strip() and out and not out[-1].strip():
                i += 1
            continue
        if _HOOK_CALL.match(line):
            while not lines[i].rstrip().endswith(";"):
                i += 1
            i += 1
            continue
        mutate = _MUTATE_IF.match(line)
        if mutate:
            indent = mutate.group(1)
            j = i + 1
            while lines[j] != indent + "}":
                j += 1
            out += [
                s[2:] if s.startswith(indent + "  ") else s for s in lines[i + 1 : j]
            ]
            i = j + 1
            continue
        out.append(line)
        i += 1
    stripped = "\n".join(out)
    left = TEST_TOKENS.search(stripped)
    if left:
        raise ValueError(f"test identifier left after the strip: {left.group(0)}")
    return stripped


def export(out_dir: Path) -> list[str]:
    """Writes the subset under `out_dir` (same relative paths); returns them."""
    files = shipped_files()
    for rel in files:
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith("csrc/"):
            src = (REPO / rel).read_text(encoding="utf-8")
            dst.write_text(strip_test_hooks(src), encoding="utf-8")
        else:
            shutil.copyfile(REPO / rel, dst)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="print the subset")
    mode.add_argument("--out", type=Path, help="write the stripped subset here")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(shipped_files()))
        return 0
    if args.out.exists():
        shutil.rmtree(args.out)
    files = export(args.out)
    print(f"export_upstream: {len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
