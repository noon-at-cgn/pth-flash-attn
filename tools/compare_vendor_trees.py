#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Compares two vendored trees and proves that they differ in comments only.

A project that vendors ``pth_flash_attn`` (the files printed by
``tools/export_upstream.py --list``) holds a
copy of a fixed set of files. When the copy is re-vendored from another
revision, this tool answers one question: does the new copy differ from the old
one only in comments and docstrings, so that it builds to the same code and
behaves the same? Every file present in either tree is compared by its kind:

* ``.py``: ``ast.dump`` of the module with every docstring (the first string
  statement of a module, class or function) removed. Comments are not in the
  AST. Comment pragmas (``# noqa``, ``# type:``, ``# pragma``, ``# fmt:``,
  ``# ruff:``) are compared as a separate note, because they steer tools and
  not the program.
* ``.h``, ``.cuh``, ``.cu``, ``.cpp``: ``//`` and ``/* */`` comments are removed
  outside string, raw-string and character literals by a small lexer; the
  remainder must be equal as a whitespace-collapsed token stream and as lines
  (trailing whitespace stripped, blank lines dropped).
* ``.toml``: the parsed tables must be equal, except for the dotted key paths
  that ``--allow-toml`` names. ``build-system``, ``project.name``,
  ``project.dependencies``, ``project.entry-points`` and
  ``project.requires-python`` can never be allowed to differ.
* anything else (``LICENSE``, ``NOTICE``, ``SHA256SUMS`` and so on): byte for
  byte.

A difference that is not a comment is a failure, shown as a diff of the
comment-free forms, unless it is listed on the command line:
``--allow-str-changes PATH N`` accepts exactly N Python string constants of
PATH that differ (an error message, say), ``--allow-toml KEY`` accepts a
changed key of a ``.toml`` file, and ``--allow-text-diff PATH`` accepts any
text difference in one byte-compared file (``NOTICE``, say; the diff is
printed). An allowance that matches nothing, or fewer or
more differences than were allowed, is an error, so that a stale list cannot
hide a change. Every allowed difference is printed.

    tools/compare_vendor_trees.py OLD_DIR NEW_DIR

``--old-export`` and ``--new-export`` mean that the directory is a repository
checkout; the vendored tree is then built from it first, the way a vendoring
project does it: ``tools/export_upstream.py --out`` followed by the five build
files (``setup.py``, ``pyproject.toml``, ``tools/build_flags.py``, ``LICENSE``,
``NOTICE``). ``--exclude NAME`` skips a top-level entry of either tree (a
vendoring project's own files). ``--only LIST`` restricts the comparison to the
paths named in a file. ``--self-test`` runs only the negative controls, which
also run before every comparison.

Exit codes: 0 comment and docstring only (plus the listed allowances), 1 a code
difference, 2 usage or tool failure.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

CXX_SUFFIXES = (".h", ".cuh", ".cu", ".cpp", ".cc", ".hpp")
BUILD_FILES = (
    "setup.py",
    "pyproject.toml",
    "tools/build_flags.py",
    "LICENSE",
    "NOTICE",
)
# Never compared: generated or cache directories.
SKIP_DIRS = {"build", "__pycache__"}
SKIP_SUFFIXES = (".egg-info", ".pyc")
# Tables of pyproject.toml whose change is never a metadata change.
TOML_PROTECTED = (
    "build-system",
    "project.name",
    "project.dependencies",
    "project.entry-points",
    "project.requires-python",
)
_PRAGMA = re.compile(r"#\s*(noqa|type:|pragma|fmt:|ruff:)")


# --------------------------------------------------------------------- Python
def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


def heal_strings(old: ast.AST, new: ast.AST) -> list[tuple[str, str]]:
    """Makes the string constants of `old` that differ from `new` equal to them.

    The two trees are walked in step. If they differ anywhere else, the walk
    is misaligned and the comparison of the trees that follows still fails, so
    healing a string never hides another kind of difference. Returns the
    (old, new) pairs that were healed.
    """
    healed = []
    for a, b in zip(ast.walk(old), ast.walk(new)):
        if (
            isinstance(a, ast.Constant)
            and isinstance(b, ast.Constant)
            and isinstance(a.value, str)
            and isinstance(b.value, str)
            and a.value != b.value
        ):
            healed.append((a.value, b.value))
            a.value = b.value
    return healed


def py_pragmas(src: str) -> list[str]:
    """The comment pragmas of `src`, in order."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and _PRAGMA.match(tok.string):
            out.append(" ".join(tok.string.split()))
    return out


# ------------------------------------------------------------------- C++/CUDA
def strip_cxx_comments(s: str) -> str:
    """`s` without comments; literals are kept verbatim.

    Block comments keep their newlines (so the line structure stays) and become
    one space. A ``//`` comment ends at a newline that is not preceded by a
    backslash.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        nxt = s[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and s[i] != "\n":
                i += 2 if s[i] == "\\" and i + 1 < n else 1
        elif c == "/" and nxt == "*":
            j = s.find("*/", i + 2)
            if j < 0:
                raise ValueError("unterminated /* comment")
            out.append(" " + "\n" * s.count("\n", i, j + 2))
            i = j + 2
        elif c == "R" and s.startswith('R"', i) and not (i and s[i - 1].isalnum()):
            m = re.match(r'R"([^(\s]*)\(', s[i:])
            if not m:
                out.append(c)
                i += 1
                continue
            end = s.find(")" + m.group(1) + '"', i)
            if end < 0:
                raise ValueError("unterminated raw string")
            stop = end + len(m.group(1)) + 2
            out.append(s[i:stop])
            i = stop
        elif c == "'" and i and s[i - 1].isalnum() and nxt.isalnum():
            out.append(c)  # C++14 digit separator, as in 1'000
            i += 1
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                if s[j] == "\n":
                    break
                j += 2 if s[j] == "\\" else 1
            out.append(s[i : j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def cxx_keys(src: str) -> tuple[str, list[str]]:
    """(whitespace-collapsed code, code lines without blanks) of `src`."""
    text = strip_cxx_comments(src)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return " ".join(text.split()), lines


# ----------------------------------------------------------------------- TOML
def _flatten(table: object, prefix: str = "") -> dict[str, object]:
    """Dotted key path -> value; lists and scalars are leaves."""
    if not isinstance(table, dict):
        return {prefix: table}
    out: dict[str, object] = {}
    for key, value in table.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.update(_flatten(value, path) if isinstance(value, dict) else {path: value})
    return out


def toml_delta(old: str, new: str) -> dict[str, tuple[object, object]]:
    """Dotted key paths whose value differs (missing = None marker)."""
    a, b = _flatten(tomllib.loads(old)), _flatten(tomllib.loads(new))
    missing = "<absent>"
    return {
        k: (a.get(k, missing), b.get(k, missing))
        for k in sorted(a.keys() | b.keys())
        if a.get(k, missing) != b.get(k, missing)
    }


def _covered(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + ".") for p in prefixes)


# -------------------------------------------------------------------- compare
class Result:
    """Verdict of one file: ``ok``, a short ``how`` and detail lines."""

    def __init__(self, ok: bool, how: str, detail: list[str] | None = None) -> None:
        self.ok, self.how, self.detail = ok, how, detail or []


def _diff(a: list[str], b: list[str], limit: int = 30) -> list[str]:
    lines = list(difflib.unified_diff(a, b, "old", "new", lineterm="", n=1))
    return lines[:limit] + (["..."] if len(lines) > limit else [])


def compare(
    old: str,
    new: str,
    name: str,
    allow_str_changes: int = 0,
    allow_toml: tuple[str, ...] = (),
    allow_text: bool = False,
) -> Result:
    """Compares one file's two texts by the rules of the module docstring."""
    if old == new:
        return Result(True, "identical")
    if allow_text:
        if name.endswith((".py", ".toml") + CXX_SUFFIXES):
            return Result(False, "--allow-text-diff is for byte-compared files only")
        diff = list(
            difflib.unified_diff(
                old.splitlines(), new.splitlines(), "old", "new", n=0, lineterm=""
            )
        )
        return Result(True, "text differs (allowed)", diff)
    if name.endswith(".py"):
        try:
            ta, tb = (
                _strip_docstrings(ast.parse(old)),
                _strip_docstrings(ast.parse(new)),
            )
        except SyntaxError as exc:
            return Result(False, f"does not parse: {exc}")
        healed = heal_strings(ta, tb) if allow_str_changes else []
        if len(healed) != allow_str_changes:
            return Result(
                False,
                f"expected {allow_str_changes} changed string constant(s), "
                f"found {len(healed)}",
                [f"{o!r} -> {n!r}" for o, n in healed],
            )
        ka = ast.dump(ta, include_attributes=False)
        kb = ast.dump(tb, include_attributes=False)
        if ka != kb:
            da = ka.replace(", ", ",\n").splitlines()
            db = kb.replace(", ", ",\n").splitlines()
            return Result(False, "code differs (AST)", _diff(da, db))
        notes = []
        if py_pragmas(old) != py_pragmas(new):
            notes.append(
                "note: comment pragmas differ: "
                f"{sorted(set(py_pragmas(old)) ^ set(py_pragmas(new)))}"
            )
        for o, n in healed:
            notes.append(f"allowed: string constant {o!r} -> {n!r}")
        how = "docstring/comment only (AST equal)"
        return Result(True, how + (" + allowed" if healed else ""), notes)
    if name.endswith(CXX_SUFFIXES):
        try:
            ca, la = cxx_keys(old)
            cb, lb = cxx_keys(new)
        except ValueError as exc:
            return Result(False, f"cannot lex: {exc}")
        if ca != cb or la != lb:
            return Result(False, "code differs (tokens)", _diff(la, lb))
        return Result(True, "comment only (tokens and code lines equal)")
    if name.endswith(".toml"):
        delta = toml_delta(old, new)
        bad = {k: v for k, v in delta.items() if _covered(k, TOML_PROTECTED)}
        if bad:
            return Result(
                False, "protected keys differ", [f"{k}: {v}" for k, v in bad.items()]
            )
        loose = {k: v for k, v in delta.items() if not _covered(k, allow_toml)}
        if loose:
            return Result(
                False,
                "keys differ that are not allowed",
                [f"{k}: {v}" for k, v in loose.items()],
            )
        unused = [p for p in allow_toml if not any(_covered(k, (p,)) for k in delta)]
        if unused:
            return Result(False, f"allowed keys without a difference: {unused}")
        return Result(
            True,
            "parsed tables equal except the allowed keys",
            [f"allowed: {k}" for k in delta],
        )
    return Result(False, "differs (not a source file)")


def selftest() -> bool:
    """Negative and positive controls of the rules above."""
    py0 = 'def f(x):\n    """doc"""\n    # c\n    return x + 1\n'
    cxx0 = "int a = 1; // one\n/* block\n */ int b;\n"
    toml0 = '[project]\nname = "x"\nversion = "1"\ndependencies = ["torch"]\n'
    cases = [
        (
            "a.py",
            py0,
            py0.replace('"""doc"""', '"""other"""').replace("# c", "# d"),
            True,
        ),
        ("a.py", py0, py0.replace("x + 1", "x + 2"), False),
        ("a.py", py0 + 'y = "s"\n', py0 + 'y = "t"\n', False),
        ("a.py", py0, py0.replace("    # c\n", ""), True),
        ("a.py", "x = 1  # noqa\n", "x = 1\n", True),  # pragma: a note, not a failure
        ("a.cuh", cxx0, "int a = 1; // uno\n/* other */\n int b;\n", True),
        ("a.cuh", "int a = 1; // one\n", "int a = 2; // one\n", False),
        (
            "a.cuh",
            'const char* s = "//x"; // c\n',
            'const char* s = "//y"; // c\n',
            False,
        ),
        ("a.cuh", "int a;\n// int b;\n", "int a;\nint b;\n", False),
        (
            "a.cuh",
            'auto s = R"x(// not a comment)x";\n',
            'auto s = R"x(// other)x";\n',
            False,
        ),
        ("a.cuh", "int a = 0x1'0000; // c\n", "int a = 0x1'0000;\n", True),
        ("a.toml", toml0, toml0.replace('"1"', '"2"'), False),
        ("a.toml", toml0, toml0.replace('"torch"', '"torch>=2"'), False),
        ("NOTICE", "a\n", "b\n", False),
    ]
    ok = True
    for name, a, b, want in cases:
        got = compare(a, b, name).ok
        ok &= got == want
        print(
            f"  self-test {'PASS' if got == want else 'FAIL'}  {name}: "
            f"want {'equal' if want else 'different'}, "
            f"got {'equal' if got else 'different'}"
        )
    allowed = compare(
        toml0, toml0.replace('"1"', '"2"'), "a.toml", allow_toml=("project.version",)
    )
    stale = compare(
        toml0,
        toml0.replace("1", "2"),
        "a.toml",
        allow_toml=("project.version", "project.description"),
    )
    str_ok = compare(py0 + 'y = "s"\n', py0 + 'y = "t"\n', "a.py", allow_str_changes=1)
    str_stale = compare(
        py0 + 'y = "s"\n',
        py0.replace("doc", "d2") + 'y = "s"\n',
        "a.py",
        allow_str_changes=1,
    )
    str_more = compare(
        py0 + 'y = "s"\nz = "u"\n',
        py0 + 'y = "t"\nz = "v"\n',
        "a.py",
        allow_str_changes=1,
    )
    str_code = compare(
        py0 + 'y = "s"\n', py0 + 'y = "t"\nw = 1\n', "a.py", allow_str_changes=1
    )
    for label, res, want in (
        ("allowed toml key", allowed, True),
        ("allowed string", str_ok, True),
        ("stale string allowance", str_stale, False),
        ("more changed strings than allowed", str_more, False),
        ("changed string plus a code change", str_code, False),
        ("stale toml allowance", stale, False),
    ):
        ok &= res.ok == want
        print(f"  self-test {'PASS' if res.ok == want else 'FAIL'}  {label}")
    text_ok = compare("a\n", "b\n", "NOTICE", allow_text=True)
    text_same = compare("a\n", "a\n", "NOTICE", allow_text=True)
    text_py = compare("x = 1\n", "x = 2\n", "a.py", allow_text=True)
    text_toml = compare(toml0, toml0.replace('"1"', '"2"'), "a.toml", allow_text=True)
    text_cxx = compare("int a;\n", "int b;\n", "a.cuh", allow_text=True)
    for label, res, want in (
        ("allowed text difference", text_ok, True),
        ("allowed text, files equal (main() rejects this)", text_same, True),
        ("allowed text on a .py file is refused", text_py, False),
        ("allowed text on a .toml file is refused", text_toml, False),
        ("allowed text on a C++ file is refused", text_cxx, False),
    ):
        ok &= res.ok == want
        print(f"  self-test {'PASS' if res.ok == want else 'FAIL'}  {label}")
    return ok


# ------------------------------------------------------------------- the trees
def listing(root: Path, exclude: set[str]) -> set[str]:
    """Relative paths of the files under `root` that take part."""
    files = set()
    for d, dirs, names in os.walk(root):
        rel_dir = Path(d).relative_to(root)
        dirs[:] = [
            x
            for x in dirs
            if x not in SKIP_DIRS
            and not x.endswith(SKIP_SUFFIXES)
            and (rel_dir / x).as_posix() not in exclude
        ]
        for name in names:
            rel = (rel_dir / name).as_posix()
            if rel not in exclude and not name.endswith(SKIP_SUFFIXES):
                files.add(rel)
    return files


def materialise(repo: Path, out: Path) -> None:
    """Builds the vendored tree of repository `repo` under `out`."""
    tool = repo / "tools" / "export_upstream.py"
    if not tool.is_file():
        raise SystemExit(
            f"compare_vendor_trees: {repo} has no tools/export_upstream.py"
        )
    subprocess.run(
        [sys.executable, str(tool), "--out", str(out)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (out / "tools").mkdir(exist_ok=True)
    for rel in BUILD_FILES:
        shutil.copyfile(repo / rel, out / rel)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("old", type=Path, nargs="?")
    ap.add_argument("new", type=Path, nargs="?")
    ap.add_argument(
        "--old-export", action="store_true", help="OLD is a repository checkout"
    )
    ap.add_argument(
        "--new-export", action="store_true", help="NEW is a repository checkout"
    )
    ap.add_argument("--exclude", action="append", default=[], metavar="PATH")
    ap.add_argument(
        "--only",
        type=Path,
        metavar="LIST",
        help="compare only the relative paths named in this file, one per line",
    )
    ap.add_argument(
        "--allow-str-changes",
        action="append",
        default=[],
        nargs=2,
        metavar=("PATH", "N"),
        help="PATH has exactly N string constants that differ (printed)",
    )
    ap.add_argument("--allow-toml", action="append", default=[], metavar="KEY")
    ap.add_argument(
        "--allow-text-diff",
        action="append",
        default=[],
        metavar="PATH",
        help="PATH, a byte-compared file, may differ in text (the diff is printed)",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    ok = selftest()
    if args.self_test:
        print("self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    if not ok:
        print("compare_vendor_trees: the self-test failed", file=sys.stderr)
        return 2
    if args.old is None or args.new is None:
        ap.error("OLD_DIR and NEW_DIR are required")
    checked = [
        p for p in args.allow_text_diff if p.endswith((".py", ".toml") + CXX_SUFFIXES)
    ]
    if checked:
        ap.error(f"--allow-text-diff is for byte-compared files only: {checked}")
    exclude = set(args.exclude)
    with tempfile.TemporaryDirectory(prefix="vendor-trees-") as tmp:
        roots = []
        for label, path, export in (
            ("old", args.old, args.old_export),
            ("new", args.new, args.new_export),
        ):
            if not path.is_dir():
                print(
                    f"compare_vendor_trees: {path} is not a directory", file=sys.stderr
                )
                return 2
            if export:
                out = Path(tmp) / label
                materialise(path, out)
                path = out
            roots.append(path)
        old_root, new_root = roots
        files = sorted(listing(old_root, exclude) | listing(new_root, exclude))
        if args.only:
            wanted = {
                ln.strip() for ln in args.only.read_text().splitlines() if ln.strip()
            }
            files = [f for f in files if f in wanted]
            missing = sorted(wanted - set(files))
            if missing:
                print(
                    "compare_vendor_trees: --only names files in neither tree: "
                    f"{missing}",
                    file=sys.stderr,
                )
                return 2
        changed = failed = 0
        for rel in files:
            pa, pb = old_root / rel, new_root / rel
            if not (pa.is_file() and pb.is_file()):
                print(f"  FAIL  {rel}: only in {'new' if pb.is_file() else 'old'}")
                failed += 1
                continue
            allow_str = int(dict(args.allow_str_changes).get(rel, 0))
            try:
                a = pa.read_text(encoding="utf-8")
                b = pb.read_text(encoding="utf-8")
                res = compare(
                    a,
                    b,
                    rel,
                    allow_str,
                    tuple(args.allow_toml),
                    rel in args.allow_text_diff,
                )
                if rel in args.allow_text_diff and a == b:
                    res = Result(False, "--allow-text-diff names an identical file")
            except UnicodeDecodeError:
                res = Result(pa.read_bytes() == pb.read_bytes(), "binary file")
            if res.how != "identical":
                changed += 1
                print(f"  {'PASS' if res.ok else 'FAIL'}  {rel}: {res.how}")
                for line in res.detail:
                    print(f"        {line}")
            failed += not res.ok
    for p, _ in args.allow_str_changes:
        if p not in files:
            print(f"  FAIL  --allow-str-changes names {p}, which is in neither tree")
            failed += 1
    for p in args.allow_text_diff:
        if p not in files:
            print(f"  FAIL  --allow-text-diff names {p}, which is in neither tree")
            failed += 1
    print(f"{len(files)} files, {changed} changed, {len(files) - changed} identical")
    print("comment/docstring-only:", "YES" if failed == 0 else "NO")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
