#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Repository policy checks used by pre-commit, run_ci_local.sh and CI.

Subcommands (each exits 1 on a violation and prints one line per finding):

``spdx [FILES...]``
    Every source file starts with the two SPDX lines:
    ``SPDX-License-Identifier: Apache-2.0`` and a
    ``SPDX-FileCopyrightText:`` line naming ``noon-at-cgn``, both within the
    first ``SPDX_HEADER_LINES`` lines (room for a shebang, a ``# syntax=``
    line and a blank). Without FILES, walks the repository. Source files are
    those with a suffix in ``SPDX_SUFFIXES`` or named ``*Dockerfile``;
    markdown and data files are not checked (vLLM checks Python only; this
    project also covers C++/CUDA/shell/config because everything here is
    hand-written).
``engine-facing [FILES...]``
    The package's top-level modules
    (``python/pth_flash_attn/*.py``, the code an inference engine imports)
    neither call ``torch.cuda`` (use ``current_platform``) nor read
    environment variables (``os.environ``, ``os.getenv``). Comments and
    strings are ignored. The ``bench`` subpackage is not engine-facing and is
    exempt.
``signoff [REV_RANGE]``
    Every commit in REV_RANGE (default ``HEAD``: all reachable commits) has a
    ``Signed-off-by:`` trailer (Developer Certificate of Origin). Needs git.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPDX_HEADER_LINES = 5
SPDX_SUFFIXES = {
    ".py", ".sh", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".toml", ".yaml", ".yml",
}  # fmt: skip
SPDX_EXTRA_NAMES = {".clang-format"}
LICENSE_LINE = "SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_RE = re.compile(r"SPDX-FileCopyrightText:.*noon-at-cgn")
SKIP_DIRS = {".git", "build", "__pycache__", "results", ".mypy_cache",
             ".ruff_cache", ".pytest_cache"}  # fmt: skip


def is_source(path: Path) -> bool:
    """True for files that must carry the SPDX header."""
    return (
        path.suffix in SPDX_SUFFIXES
        or path.name in SPDX_EXTRA_NAMES
        or path.name.endswith("Dockerfile")
    )


def spdx_problems(path: Path) -> list[str]:
    """Returns the missing SPDX lines of one file (empty if compliant)."""
    try:
        head = path.read_text(encoding="utf-8").splitlines()[:SPDX_HEADER_LINES]
    except UnicodeDecodeError:
        return []
    text = "\n".join(head)
    problems = []
    if LICENSE_LINE not in text:
        problems.append(f"{path}: missing '{LICENSE_LINE}' in first lines")
    if not COPYRIGHT_RE.search(text):
        problems.append(f"{path}: missing 'SPDX-FileCopyrightText: ... noon-at-cgn'")
    return problems


def walk_sources(root: Path) -> list[Path]:
    """All source files below root, skipping build output and caches."""
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.is_file() and is_source(path):
            out.append(path)
    return out


_FORBIDDEN_NAMES = (
    ("torch", "cuda", "torch.cuda (use vllm.platforms.current_platform)"),
    ("os", "environ", "os.environ (no env vars)"),
    ("os", "getenv", "os.getenv (no env vars)"),
)


def engine_facing_problems(path: Path) -> list[str]:
    """Finds ``torch.cuda`` and env-var reads in code tokens of one file."""
    problems = []
    toks = [
        t
        for t in tokenize.generate_tokens(
            io.StringIO(path.read_text(encoding="utf-8")).readline
        )
        if t.type in (tokenize.NAME, tokenize.OP)
    ]
    for a, dot, b in zip(toks, toks[1:], toks[2:]):
        if dot.string != ".":
            continue
        for left, right, what in _FORBIDDEN_NAMES:
            if a.string == left and b.string == right:
                problems.append(f"{path}:{a.start[0]}: {what}")
    return problems


def engine_facing_files(root: Path) -> list[Path]:
    """The package's top-level modules (bench/ subpackage excluded)."""
    return sorted((root / "python" / "pth_flash_attn").glob("*.py"))


def signoff_problems(rev_range: str) -> list[str]:
    """Commits in ``rev_range`` without a Signed-off-by trailer."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "--format=%H%x00%B%x01", rev_range],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    problems = []
    for entry in out.split("\x01"):
        if not entry.strip():
            continue
        sha, _, body = entry.strip().partition("\x00")
        if not re.search(r"^Signed-off-by: .+ <.+>$", body, re.MULTILINE):
            subject = body.strip().splitlines()[0] if body.strip() else ""
            problems.append(f"{sha[:10]} {subject!r}: no Signed-off-by trailer")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_spdx = sub.add_parser("spdx")
    p_spdx.add_argument("files", nargs="*", type=Path)
    p_ef = sub.add_parser("engine-facing")
    p_ef.add_argument("files", nargs="*", type=Path)
    p_so = sub.add_parser("signoff")
    p_so.add_argument("rev_range", nargs="?", default="HEAD")
    args = ap.parse_args(argv)

    if args.cmd == "spdx":
        files = [f for f in args.files if is_source(f)] or (
            [] if args.files else walk_sources(REPO)
        )
        problems = [p for f in files for p in spdx_problems(f)]
    elif args.cmd == "engine-facing":
        files = args.files or engine_facing_files(REPO)
        problems = [p for f in files for p in engine_facing_problems(f)]
    else:
        problems = signoff_problems(args.rev_range)
    for p in problems:
        print(p)
    print(f"repo_checks {args.cmd}: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
