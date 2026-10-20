#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Static SASS lint: forbidden and required instructions per kernel.

Disassembles a cubin with ``cuobjdump -sass`` (the CUDA toolkit's if present,
else the copy shipped in Triton's ``backends/nvidia/bin``) and checks, for
every kernel whose name matches
``--function`` (default: all):

* ``--forbid OPC``: OPC never appears (default ``LDL`` and ``STL``: no local
  memory traffic, which is how spills and local arrays reach DRAM);
* ``--forbid-in-loops OPC``: OPC never appears inside a loop body (default
  ``I2F`` and ``I2FP``: the int8 path converts through the PRMT/LOP3/HADD2
  magic-number sequence; an int-to-float conversion in the main loop means
  that sequence was lost, and I2F runs on the quarter-rate unit);
* ``--require OPC``: OPC appears at least once (e.g. ``LDGSTS`` for
  ``cp.async``, ``HMMA`` for ``mma.sync``, ``LDSM`` for ``ldmatrix``).

Opcode matching: a spec matches an instruction whose opcode equals it or
starts with it followed by ``.``; ``HMMA`` matches ``HMMA.16816.F32`` and
``I2F`` does not match ``I2FP.F32.S32`` (hence both are listed).

Loop detection: a branch (``BRA``) whose target address is before its own
address closes a loop; the instructions from the target to the branch form
the body. A ``BRA`` to its own address is not a loop: ptxas ends every
kernel with one after ``EXIT`` as a trap for runaway execution, and it is
never reached. Every real loop is checked, which is a superset of "the main
loop": an I2F in a prologue loop is also reported.

What this tool cannot check (printed under ``unchecked``):

* whether a required instruction is on the hot path (presence only);
* loops formed by indirect branches (``BRX``/``JMX``) or calls;
* predication: a forbidden instruction that is always predicated off still
  counts (conservative);
* kernels with no backward branch: the loop rule holds vacuously and is
  listed as such so a reader does not mistake it for evidence.

Exit codes: 0 pass, 1 rule violation, 2 tool or input failure.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402

DEFAULT_FORBID = ("LDL", "STL")
DEFAULT_FORBID_IN_LOOPS = ("I2F", "I2FP")

CUOBJDUMP_CANDIDATES = (
    "/usr/local/cuda/bin/cuobjdump",
    "/app/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump",
)

_FUNC_RE = re.compile(r"^\s*Function\s*:\s*(\S+)")
# "        /*0050*/              @!P0 BRA 0x870 ;     /* 0x... */"
_INSTR_RE = re.compile(
    r"^\s*/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\s*([^;]*);"
)
_BRA_TARGET_RE = re.compile(r"(0x[0-9a-f]+)")


@dataclass
class Instr:
    """One SASS instruction: byte address, full opcode, operand text."""

    addr: int
    opcode: str
    operands: str


@dataclass
class KernelReport:
    """Findings for one kernel."""

    name: str
    instructions: int
    loops: list[tuple[int, int]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)
    opcode_counts: dict[str, int] = field(default_factory=dict)


def find_cuobjdump(explicit: str | None = None) -> str:
    """Returns the cuobjdump to use: explicit, CUDA toolkit, Triton's, PATH.

    Raises:
        FileNotFoundError: none of the candidates exists.
    """
    for cand in ([explicit] if explicit else []) + list(CUOBJDUMP_CANDIDATES):
        if cand and Path(cand).is_file():
            return cand
    found = shutil.which("cuobjdump")
    if found:
        return found
    raise FileNotFoundError("cuobjdump not found (CUDA toolkit or Triton bin)")


def parse_sass(text: str) -> dict[str, list[Instr]]:
    """Splits ``cuobjdump -sass`` output into instructions per kernel."""
    kernels: dict[str, list[Instr]] = {}
    current: list[Instr] | None = None
    for line in text.splitlines():
        if m := _FUNC_RE.match(line):
            current = kernels.setdefault(m.group(1), [])
            continue
        if current is None:
            continue
        if m := _INSTR_RE.match(line):
            current.append(Instr(int(m.group(1), 16), m.group(2), m.group(3)))
    return kernels


def opcode_matches(opcode: str, spec: str) -> bool:
    """True if ``spec`` names ``opcode`` exactly or as a dotted prefix."""
    return opcode == spec or opcode.startswith(spec + ".")


def find_loops(instrs: list[Instr]) -> list[tuple[int, int]]:
    """Returns (start, end) byte ranges closed by backward ``BRA`` branches."""
    loops = []
    for ins in instrs:
        if not opcode_matches(ins.opcode, "BRA"):
            continue
        if m := _BRA_TARGET_RE.search(ins.operands):
            target = int(m.group(1), 16)
            if target < ins.addr:
                loops.append((target, ins.addr))
    return loops


def check_kernel(
    name: str,
    instrs: list[Instr],
    forbid: list[str],
    forbid_in_loops: list[str],
    require: list[str],
) -> KernelReport:
    """Applies the three rule kinds to one kernel's instructions."""
    rep = KernelReport(name=name, instructions=len(instrs))
    for ins in instrs:
        base = ins.opcode.split(".")[0]
        rep.opcode_counts[base] = rep.opcode_counts.get(base, 0) + 1
        if opc := next((s for s in forbid if opcode_matches(ins.opcode, s)), None):
            rep.violations.append(f"forbidden {opc} at 0x{ins.addr:04x}: {ins.opcode}")
        if any(opcode_matches(ins.opcode, s) for s in ("BRX", "JMX", "CALL")):
            rep.unchecked.append(
                f"{ins.opcode} at 0x{ins.addr:04x}: loops through it are not traced"
            )
    rep.loops = find_loops(instrs)
    if forbid_in_loops:
        if not rep.loops:
            rep.unchecked.append(
                "no backward branch: forbid-in-loops "
                f"{','.join(forbid_in_loops)} holds vacuously"
            )
        for start, end in rep.loops:
            for ins in instrs:
                if start <= ins.addr <= end and (
                    opc := next(
                        (s for s in forbid_in_loops if opcode_matches(ins.opcode, s)),
                        None,
                    )
                ):
                    rep.violations.append(
                        f"{opc} inside loop 0x{start:04x}-0x{end:04x} at "
                        f"0x{ins.addr:04x}: {ins.opcode}"
                    )
    for spec in require:
        if not any(opcode_matches(i.opcode, spec) for i in instrs):
            rep.violations.append(f"required {spec} not present")
    if require:
        rep.unchecked.append("required opcodes: presence only, not hot-path use")
    return rep


def disassemble(cuobjdump: str, cubin: Path) -> str:
    """Runs ``cuobjdump -sass`` on a cubin and returns its text."""
    proc = subprocess.run(
        [cuobjdump, "-sass", str(cubin)], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cuobjdump failed on {cubin}:\n{proc.stderr}")
    return proc.stdout


def compile_cubin(nvcc: str, source: Path, arch: str, out_dir: Path) -> Path:
    """Compiles a ``.cu`` to a cubin with the extension's flags."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cubin = out_dir / f"{source.stem}.sm_{arch}.cubin"
    cmd = build_flags.nvcc_compile_command(
        nvcc, source, arch, ["-cubin", "-o", str(cubin)]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed for {source} sm_{arch}:\n{proc.stderr}")
    return cubin


def _split(values: list[str]) -> list[str]:
    return [v for item in values for v in item.split(",") if v]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cubin", type=Path, nargs="+")
    src.add_argument("--cu", type=Path, nargs="+")
    ap.add_argument("--arch", nargs="+", default=list(build_flags.ARCHS))
    ap.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    ap.add_argument("--cuobjdump")
    ap.add_argument("--function", default=".*", help="regex on kernel names")
    ap.add_argument("--forbid", action="append", default=None)
    ap.add_argument("--forbid-in-loops", action="append", default=None)
    ap.add_argument("--require", action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=Path("build/sass_check"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    forbid = _split(args.forbid) if args.forbid is not None else list(DEFAULT_FORBID)
    forbid_loops = (
        _split(args.forbid_in_loops)
        if args.forbid_in_loops is not None
        else list(DEFAULT_FORBID_IN_LOOPS)
    )
    require = _split(args.require)
    try:
        cuobjdump = find_cuobjdump(args.cuobjdump)
        if args.cu:
            cubins = [
                compile_cubin(args.nvcc, cu, arch, args.out_dir)
                for cu in args.cu
                for arch in args.arch
            ]
        else:
            cubins = list(args.cubin)
        texts = {c: disassemble(cuobjdump, c) for c in cubins}
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 2

    pattern = re.compile(args.function)
    all_ok = True
    out: dict[str, object] = {
        "cuobjdump": cuobjdump,
        "rules": {
            "forbid": forbid,
            "forbid_in_loops": forbid_loops,
            "require": require,
        },
        "cubins": [],
    }
    for cubin, text in texts.items():
        kernels = {n: i for n, i in parse_sass(text).items() if pattern.search(n)}
        reports = [
            check_kernel(n, i, forbid, forbid_loops, require)
            for n, i in kernels.items()
        ]
        if not reports:
            print(f"FAIL {cubin}: no kernel matches --function {args.function!r}")
            all_ok = False
        for r in reports:
            ok = not r.violations
            all_ok = all_ok and ok
            print(
                f"{'PASS' if ok else 'FAIL'} {cubin} {r.name}: "
                f"{r.instructions} instrs, {len(r.loops)} loops"
            )
            for v in r.violations:
                print(f"  violation: {v}")
            for u in r.unchecked:
                print(f"  unchecked: {u}")
        out["cubins"].append(  # type: ignore[attr-defined]
            {"cubin": str(cubin), "kernels": [asdict(r) for r in reports]}
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
