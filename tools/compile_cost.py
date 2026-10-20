#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Compile-cost report: nvcc time and binary size per TU and per arch.

Why it exists: build time and binary size are acceptance criteria for kernel
changes (about 0.9 s and 62 KB of cubin per kernel per arch is the budget),
and downstream maintainers weigh build time and wheel size. This
tool turns those into measured, committed numbers.

Two measurements, both with the extension's own flags (``build_flags.py``):

``--tu FILE...``  For each ``.cu`` and each arch: wall time of
    ``nvcc -cubin`` (device compilation for that one arch) and cubin bytes.
    For each ``.cpp``: wall time of one host ``c++ -c`` and object bytes
    (arch-independent).
``--extension``   For each single arch and for the full arch list: wall time
    of ``setup.py build_ext`` into a fresh temporary directory (so nothing is
    cached) and the resulting ``.so`` bytes. Run with ``MAX_JOBS=1`` so the
    time is a sum over TUs, not a parallel-schedule artefact.

Times are wall-clock seconds on whatever CPU runs it (the dev container is
limited to ``min(4, nproc)`` CPUs); they are comparable within one report,
indicative across hosts.

Output: a table on stdout and, with ``--json``, a machine-readable report.
Exit codes: 0 all compiles succeeded, 2 a compile failed (recorded in JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402


def _timed(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None):
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env, cwd=cwd
    )
    return time.monotonic() - t0, proc


def measure_tu(nvcc: str, cxx: str, source: Path, archs: list[str], out_dir: Path):
    """Measures one translation unit; returns a list of result rows."""
    rows = []
    if source.suffix == ".cu":
        for arch in archs:
            cubin = out_dir / f"{source.stem}.sm_{arch}.cubin"
            cmd = build_flags.nvcc_compile_command(
                nvcc, source, arch, ["-cubin", "-o", str(cubin)]
            )
            secs, proc = _timed(cmd)
            rows.append(
                {
                    "tu": str(source),
                    "arch": f"sm_{arch}",
                    "kind": "nvcc -cubin",
                    "ok": proc.returncode == 0,
                    "seconds": round(secs, 3),
                    "bytes": cubin.stat().st_size if proc.returncode == 0 else None,
                    "error": proc.stderr[-2000:] if proc.returncode else "",
                }
            )
    else:
        obj = out_dir / f"{source.stem}.o"
        includes = [
            f"-I{d}"
            for d in [str(build_flags.CSRC_DIR), *build_flags.torch_include_dirs()]
        ]
        cmd = [
            cxx,
            *build_flags.CXX_FLAGS,
            *build_flags.define_flags(),
            *includes,
            "-fPIC",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        secs, proc = _timed(cmd)
        rows.append(
            {
                "tu": str(source),
                "arch": "host",
                "kind": "c++ -c",
                "ok": proc.returncode == 0,
                "seconds": round(secs, 3),
                "bytes": obj.stat().st_size if proc.returncode == 0 else None,
                "error": proc.stderr[-2000:] if proc.returncode else "",
            }
        )
    return rows


def measure_extension(python: str, arch_lists: list[str]):
    """Builds the extension once per arch list in a fresh temp dir."""
    rows = []
    for arch_list in arch_lists:
        with tempfile.TemporaryDirectory(prefix="pth_cost_") as tmp:
            env = dict(os.environ)
            env.update(TORCH_CUDA_ARCH_LIST=arch_list, MAX_JOBS="1")
            cmd = [
                python,
                "setup.py",
                "-q",
                "build_ext",
                "--build-lib",
                f"{tmp}/lib",
                "--build-temp",
                f"{tmp}/tmp",
            ]
            secs, proc = _timed(cmd, env=env, cwd=build_flags.REPO_ROOT)
            sos = list(Path(tmp, "lib").rglob("_C*.so"))
            ok = proc.returncode == 0 and len(sos) == 1
            rows.append(
                {
                    "arch_list": arch_list,
                    "kind": "setup.py build_ext (MAX_JOBS=1)",
                    "ok": ok,
                    "seconds": round(secs, 3),
                    "so_bytes": sos[0].stat().st_size if ok else None,
                    "error": "" if ok else (proc.stdout + proc.stderr)[-2000:],
                }
            )
    return rows


def environment() -> dict[str, str]:
    """Records the tool versions that the numbers depend on."""

    def first_line(cmd: list[str]) -> str:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except FileNotFoundError:
            return "absent"
        lines = (out.stdout or out.stderr).strip().splitlines()
        return lines[-1] if lines else "unknown"

    import torch

    return {
        "nvcc": first_line(["/usr/local/cuda/bin/nvcc", "--version"]),
        "cxx": first_line(["c++", "--version"]).strip(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpus_visible": str(os.cpu_count()),
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tu", type=Path, nargs="*", default=[])
    ap.add_argument("--extension", action="store_true")
    ap.add_argument("--arch", nargs="+", default=list(build_flags.ARCHS))
    ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    ap.add_argument("--cxx", default="c++")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", type=Path, default=Path("build/compile_cost"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    if not args.tu and not args.extension:
        ap.error("nothing to measure: pass --tu FILES and/or --extension")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "environment": environment(),
        "tu": [],
        "extension": [],
    }
    tu_rows = [
        r
        for src in args.tu
        for r in measure_tu(args.nvcc, args.cxx, src, args.arch, args.out_dir)
    ]
    report["tu"] = tu_rows
    for r in tu_rows:
        size = f"{r['bytes']} B" if r["ok"] else "FAILED"
        print(
            f"{r['tu']:<40} {r['arch']:<6} {r['kind']:<12} "
            f"{r['seconds']:>8.2f} s  {size}"
        )
        if not r["ok"]:
            print(r["error"], file=sys.stderr)

    ext_rows = []
    if args.extension:
        lists = [f"{a[0]}.{a[1]}" for a in args.arch]
        if len(lists) > 1:
            lists.append(";".join(lists))
        ext_rows = measure_extension(args.python, lists)
        report["extension"] = ext_rows
        for r in ext_rows:
            size = f"{r['so_bytes']} B" if r["ok"] else "FAILED"
            print(
                f"_C.abi3.so  TORCH_CUDA_ARCH_LIST={r['arch_list']:<12} "
                f"{r['seconds']:>8.2f} s  {size}"
            )
            if not r["ok"]:
                print(r["error"], file=sys.stderr)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    all_ok = all(r["ok"] for r in [*tu_rows, *ext_rows])
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
