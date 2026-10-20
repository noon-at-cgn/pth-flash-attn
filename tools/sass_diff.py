#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""SASS identity of two builds: the proof that a source refactor is free.

A refactor of the kernel sources (moving code between headers, renaming a
local, replacing a macro by a template) is claimed to change no performance
and no numerics. Identical machine code is the strongest form of that claim:
same instructions, same encodings, same order, so same timing and same bits by
construction, with no GPU run needed. This tool checks it:

    tools/sass_diff.py build/before/libpth_attn.so build/after/libpth_attn.so

Both inputs are disassembled with ``cuobjdump -sass`` (any file cuobjdump
reads: .so, .o, .cubin). Every kernel is keyed by (arch, mangled name) and its
instruction lines, including the encodings, are compared after one
normalisation: whitespace is collapsed. Addresses are function-relative, so
they need no rewriting. In the key, an anonymous namespace
(``_GLOBAL__N__<hash>_<file>_<hash>``, unique per translation unit) is
replaced by ``(anonymous)``, so a kernel that moved to another .cu file is
still matched with its old self.

Output: one line per kernel that differs, is missing or is new, then a
summary. ``--allow-removed REGEX`` accepts kernels that are only in the first
build when their name matches (e.g. test kernels deliberately moved out of
the second library). ``--json`` writes the per-kernel results.

Exit codes: 0 identical (up to allowed removals), 1 a difference, 2 tool or
input failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sass_check import find_cuobjdump  # noqa: E402

# "arch = sm_86" in a fatbin listing (.so, .o), "code for sm_86" in a cubin.
_ARCH_RE = re.compile(r"^\s*(?:arch\s*=|code for)\s*(sm_\d+)")
_FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
_INSTR_RE = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/")
# Itanium mangling: <length><identifier>; the anonymous namespace's
# identifier starts with _GLOBAL__N_.
_ANON_RE = re.compile(r"(\d+)(?=_GLOBAL__N_)")


def normalise_name(mangled: str) -> str:
    """`mangled` with every anonymous-namespace identifier replaced."""
    out, i = [], 0
    for m in _ANON_RE.finditer(mangled):
        if m.start() < i:
            continue
        out.append(mangled[i : m.start()])
        out.append("(anonymous)")
        i = m.end() + int(m.group(1))
    out.append(mangled[i:])
    return "".join(out)


def kernels(cuobjdump: str, path: Path) -> dict[tuple[str, str], list[str]]:
    """Returns {(arch, kernel): normalised SASS lines} of one binary."""
    text = subprocess.run(
        [cuobjdump, "-sass", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    out: dict[tuple[str, str], list[str]] = {}
    arch = "?"
    current: list[str] | None = None
    for line in text.splitlines():
        if m := _ARCH_RE.match(line):
            arch = m.group(1)
            current = None
        elif m := _FUNC_RE.match(line):
            key = (arch, normalise_name(m.group(1)))
            if key in out:
                raise SystemExit(f"sass_diff: {path}: duplicate kernel {key}")
            current = out[key] = []
        elif current is not None and (
            _INSTR_RE.match(line) or line.strip().startswith("/* 0x")
        ):
            # Instruction lines and the continuation line of their encoding.
            current.append(" ".join(line.split()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--allow-removed", default=None, metavar="REGEX")
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        tool = find_cuobjdump(args.cuobjdump)
        a = kernels(tool, args.before)
        b = kernels(tool, args.after)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"sass_diff: {e}", file=sys.stderr)
        return 2
    if not a:
        print(f"sass_diff: no kernels in {args.before}", file=sys.stderr)
        return 2
    allowed = re.compile(args.allow_removed) if args.allow_removed else None
    results: dict[str, str] = {}
    for key in sorted(a.keys() | b.keys()):
        name = f"{key[0]} {key[1]}"
        if key not in b:
            removed_ok = allowed is not None and allowed.search(key[1])
            results[name] = "removed (allowed)" if removed_ok else "removed"
        elif key not in a:
            results[name] = "new"
        else:
            results[name] = "identical" if a[key] == b[key] else "differs"
    bad = {k: v for k, v in results.items() if v in ("removed", "new", "differs")}
    for name, verdict in bad.items():
        print(f"{verdict:8s} {name}")
    counts: dict[str, int] = {}
    for v in results.values():
        counts[v] = counts.get(v, 0) + 1
    print(
        "sass_diff: "
        + ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
        + f" ({len(a)} kernels before, {len(b)} after)"
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
