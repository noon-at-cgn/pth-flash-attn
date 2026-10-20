#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""ptxas gate over the tensor-core attention library's own build logs.

``tools/build_attn_lib.sh`` compiles the five translation units of
``libpth_attn.so`` with ``-Xptxas -v`` and keeps ``build/attn/ptxas_<tu>.log``.
This reads those logs with ``ptxas_gate``'s parser and limits, so the gate is
the one of the probe (<= 255 registers, 0 spill bytes, static shared memory
within the block limit) applied to the real kernels, without compiling them a
second time. It also requires that every kernel family is present for every
architecture of the build (a family that silently stopped being instantiated
would otherwise pass):

* ``attn_fwd_kernel``  the 16-row kernel (D = 64, 128, 256),
* ``attn_fwd_window_kernel``  its sliding-window / non-causal variant
  (D = 128),
* ``attn_dec_kernel``  the decode kernel (swap-AB, D = 256),
* ``combine``          the split-combine kernels.

Exit codes: 0 pass, 1 gate violation, 2 missing logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_flags  # noqa: E402
import ptxas_gate  # noqa: E402

FAMILIES = ("attn_fwd_kernel", "attn_fwd_window_kernel", "attn_dec_kernel", "combine")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, default=Path("build/attn"))
    args = ap.parse_args(argv)
    logs = sorted(args.logs.glob("ptxas_*.log"))
    if not logs:
        print(f"no ptxas logs in {args.logs}: run tools/build_attn_lib.sh")
        return 2
    kernels = [k for log in logs for k in ptxas_gate.parse_ptxas_log(log.read_text())]
    ok = ptxas_gate.evaluate(kernels)
    for arch in (f"sm_{a}" for a in build_flags.ARCHS):
        of_arch = [k for k in kernels if k.arch == arch]
        for family in FAMILIES:
            n = sum(family in k.name for k in of_arch)
            print(f"{arch} {family}: {n} kernels")
            if n == 0:
                print(f"FAIL {arch}: no {family} kernel in the build logs")
                ok = False
    worst = max(kernels, key=lambda k: k.registers)
    spill = sum(k.spill_stores + k.spill_loads for k in kernels)
    print(
        f"{len(kernels)} kernel records, max {worst.registers} registers "
        f"({worst.arch}), spill bytes {spill}"
    )
    for k in kernels:
        if k.violations:
            print(f"FAIL {k.arch} {k.name}: {'; '.join(k.violations)}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
