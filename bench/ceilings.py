# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The hardware ceilings of one run as the JSON ``attn_dev_timing.py --ceilings`` reads.

Dev timing (``tests/gpu/attn_dev_timing.py``, ``attn_ab.py``) prints eta
columns against copy and MMA ceilings; they must be the ceilings of the same
GPU at the same power limit. This reads them from the arm records of a
``bench/ceilings_run.sh`` result directory instead of copying numbers by hand:

* ``copy_gbs``: the median throughput of ``copy-L<n>k__sep-nhd`` (B = 1, block
  816, NHD: the loader's address stream) per KV length;
* ``mma_tflops``: ``mma-f16f32__f16f32_4_1_8`` (this kernel's occupancy,
  chain 8);
* ``power_limit_w``: the power limit in watts recorded in ``env.json``
  (``null`` if it could not be read).

Usage: ``python3 bench/ceilings.py results/<ceilings dir> > ceilings.json``, the
result directory being ``results/<date>-ceilings-<W>w-gpu<N>``.
Needs no GPU (reads files only). Standard library only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# The KV lengths of the copy arms (copy-L<n>k) in tokens per k.
TOKENS_PER_K = 1024


def main(argv: list[str] | None = None) -> int:
    """Print the JSON; exit 1 if a required record is missing."""
    args = sys.argv[1:] if argv is None else argv
    if args in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if len(args) != 1:
        print(__doc__.split("\n")[0], file=sys.stderr)
        return 64
    d = Path(args[0])
    env = json.loads((d / "env.json").read_text())
    power_limit_w = env.get("power_limit_w")
    copy = {}
    for p in sorted(d.glob("copy-L*k__sep-nhd.json")):
        m = re.fullmatch(r"copy-L(\d+)k__sep-nhd\.json", p.name)
        if m:
            rec = json.loads(p.read_text())
            copy[str(int(m.group(1)) * TOKENS_PER_K)] = rec["summary"]["throughput"][
                "estimate"
            ]
    mma_p = d / "mma-f16f32__f16f32_4_1_8.json"
    if not copy or not mma_p.is_file():
        print(f"ceilings: {d} lacks copy-L*k__sep-nhd or {mma_p.name}", file=sys.stderr)
        return 1
    mma = json.loads(mma_p.read_text())["summary"]["throughput"]["estimate"]
    out = {
        "power_limit_w": power_limit_w,
        "source": str(d),
        "copy_gbs": copy,
        "mma_tflops": mma,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
