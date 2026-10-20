#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""ptxas resource gate: registers, spills and static shared memory per kernel.

For each ``.cu`` file and each arch (default sm_80/86/89) this compiles a
cubin with exactly the extension's flags (``tools/build_flags.py``) plus
``-Xptxas -v``, parses ptxas' per-kernel report and fails (exit 1) if any
kernel violates:

* registers per thread <= 255 (``MAX_REGISTERS``; the hardware limit; ptxas
  normally spills before exceeding it, so this mostly guards ``maxrregcount``
  misuse and parser drift);
* spill stores == 0 and spill loads == 0 (the D=256 probe sits 7
  registers below the limit, and a spill in the main loop costs more than
  the whole loader budget);
* static shared memory <= 101376 bytes (``SMEM_PER_BLOCK_LIMIT``, sm_86
  per-block opt-in maximum).

What this gate cannot see, and who covers it:

* Dynamic shared memory (``extern __shared__``) is sized at launch; ptxas
  reports only static ``smem``. The kernel's dynamic budget is enforced by
  ``static_assert`` in ``smem_layout.cuh``.
* Local-memory arrays that are not spills show up as "stack frame" bytes, not
  as spills. They are reported here (``stack_bytes``) and gated by
  ``sass_check.py`` (no ``LDL``/``STL``).
* Runtime occupancy (CTAs per SM) depends on the launch; verified against the
  occupancy API on a GPU.

A file with no kernels passes with ``"kernels": 0`` in the report; pass
``--min-kernels`` to make that a failure.

Exit codes: 0 pass, 1 gate violation, 2 compile or parse failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402

MAX_REGISTERS = 255
SMEM_PER_BLOCK_LIMIT = 101376

_ENTRY_RE = re.compile(r"Compiling entry function '([^']+)' for '(sm_\d+)'")
_PROPS_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)
_USED_RE = re.compile(r"Used (\d+) registers")
_SMEM_RE = re.compile(r"(\d+) bytes smem")
_PROPS_FOR_RE = re.compile(r"Function properties for (\S+)")


@dataclass
class KernelResources:
    """ptxas' resource report for one kernel on one arch."""

    name: str
    arch: str
    registers: int = -1
    stack_bytes: int = 0
    spill_stores: int = 0
    spill_loads: int = 0
    smem_bytes: int = 0
    violations: list[str] = field(default_factory=list)


def parse_ptxas_log(text: str) -> list[KernelResources]:
    """Parses ``ptxas -v`` output into one record per entry function.

    Args:
        text: Combined stderr of an nvcc ``-Xptxas -v`` compile.

    Returns:
        Records in the order ptxas reported them. A kernel whose "Used N
        registers" line is missing keeps ``registers == -1``, which
        ``evaluate`` treats as a parse failure rather than a pass.
    """
    kernels: list[KernelResources] = []
    current: KernelResources | None = None
    props_for = ""
    for line in text.splitlines():
        if m := _ENTRY_RE.search(line):
            current = KernelResources(name=m.group(1), arch=m.group(2))
            kernels.append(current)
            continue
        if current is None:
            continue
        if m := _PROPS_FOR_RE.search(line):
            props_for = m.group(1)
            continue
        if m := _PROPS_RE.search(line):
            if props_for == current.name:
                current.stack_bytes = int(m.group(1))
                current.spill_stores = int(m.group(2))
                current.spill_loads = int(m.group(3))
            else:
                # A non-inlined callee (its own "Function properties" block).
                # Its spills execute inside this kernel, so they count here.
                current.spill_stores += int(m.group(2))
                current.spill_loads += int(m.group(3))
        if m := _USED_RE.search(line):
            current.registers = int(m.group(1))
            if s := _SMEM_RE.search(line):
                current.smem_bytes = int(s.group(1))
    return kernels


def evaluate(kernels: list[KernelResources]) -> bool:
    """Fills ``violations`` for every kernel; returns True if all pass."""
    ok = True
    for k in kernels:
        if k.registers < 0:
            k.violations.append("register count not found in ptxas output")
        elif k.registers > MAX_REGISTERS:
            k.violations.append(f"registers {k.registers} > {MAX_REGISTERS}")
        if k.spill_stores or k.spill_loads:
            k.violations.append(
                f"spills: {k.spill_stores} B stores, {k.spill_loads} B loads"
            )
        if k.smem_bytes > SMEM_PER_BLOCK_LIMIT:
            k.violations.append(
                f"static smem {k.smem_bytes} B > {SMEM_PER_BLOCK_LIMIT} B"
            )
        ok = ok and not k.violations
    return ok


def compile_and_report(
    nvcc: str, source: Path, arch: str, out_dir: Path, nvcc_flags: list[str]
) -> tuple[list[KernelResources], str]:
    """Compiles ``source`` to a cubin for ``sm_<arch>`` and parses ptxas -v.

    Raises:
        RuntimeError: nvcc failed; the message carries its output.
    """
    cubin = out_dir / f"{source.stem}.sm_{arch}.cubin"
    cmd = build_flags.nvcc_compile_command(
        nvcc, source, arch, ["-cubin", "-Xptxas", "-v", *nvcc_flags, "-o", str(cubin)]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed for {source} sm_{arch}:\n{log}")
    return parse_ptxas_log(log), log


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="+", type=Path, help=".cu files")
    ap.add_argument("--arch", nargs="+", default=list(build_flags.ARCHS))
    ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    ap.add_argument(
        "--nvcc-flag",
        action="append",
        default=[],
        help="extra nvcc flag (repeatable), e.g. --nvcc-flag=-maxrregcount=32",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("build/ptxas_gate"))
    ap.add_argument("--json", type=Path, help="write the report here")
    ap.add_argument("--min-kernels", type=int, default=0)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "limits": {
            "max_registers": MAX_REGISTERS,
            "smem_per_block_limit": SMEM_PER_BLOCK_LIMIT,
        },
        "files": [],
    }
    all_ok = True
    for src in args.sources:
        for arch in args.arch:
            try:
                kernels, _ = compile_and_report(
                    args.nvcc, src, arch, args.out_dir, args.nvcc_flag
                )
            except RuntimeError as e:
                print(e, file=sys.stderr)
                return 2
            ok = evaluate(kernels)
            if len(kernels) < args.min_kernels:
                ok = False
                print(
                    f"FAIL {src} sm_{arch}: {len(kernels)} kernels "
                    f"< --min-kernels {args.min_kernels}"
                )
            all_ok = all_ok and ok
            report["files"].append(  # type: ignore[attr-defined]
                {
                    "source": str(src),
                    "arch": f"sm_{arch}",
                    "kernels": len(kernels),
                    "pass": ok,
                    "records": [asdict(k) for k in kernels],
                }
            )
            print(f"{'PASS' if ok else 'FAIL'} {src} sm_{arch}: {len(kernels)} kernels")
            for k in kernels:
                status = "; ".join(k.violations) if k.violations else "ok"
                print(
                    f"  {k.name}: {k.registers} regs, {k.spill_stores}/"
                    f"{k.spill_loads} B spill st/ld, {k.stack_bytes} B stack, "
                    f"{k.smem_bytes} B smem -> {status}"
                )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
