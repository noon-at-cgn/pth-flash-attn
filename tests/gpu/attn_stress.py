# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Determinism stress of the attention kernels' producer/consumer handshake.

Runs the same launch many times and prints the distinct SHA-256 digests of the
output (and, with splits, of the fp32 partials). A correct handshake gives one
digest. Run it on the shipped library and on a ``-DPTH_ATTN_TEST_JITTER`` build
(a random 0-255 ns ``__nanosleep`` at every handshake point of producers and
consumers, ``tools/build_attn_lib.sh --out NAME -DPTH_ATTN_TEST_JITTER``): the
digests must be identical. compute-sanitizer racecheck cannot model the named
``bar.arrive`` / ``bar.sync`` handshake, so this and the
mutation builds are the evidence instead.

    python tests/gpu/attn_stress.py --lib build/NAME/libpth_attn.so --launches 300
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import attn_case
import torch
from attn_lib import ATTN_LIB, AttnLib


def digest(c) -> str:
    h = hashlib.sha256(c.out.view(torch.int16).cpu().numpy().tobytes())
    if c.ws_o is not None:
        h.update(c.ws_o.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", default=str(ATTN_LIB))
    ap.add_argument("--launches", type=int, default=300)
    ap.add_argument("--L", type=int, default=16384)
    ap.add_argument(
        "--graph-batch", type=int, default=10, help="back-to-back replays per check"
    )
    args = ap.parse_args(argv)
    lib = AttnLib(Path(args.lib))
    stream = torch.cuda.current_stream()
    bad = 0
    for q_len, S in [(1, 1), (1, None), (4, None), (10, None)]:
        c = attn_case.build(q_len, 1, args.L, num_splits=S)
        seen: dict[str, int] = {}
        for _ in range(args.launches):
            attn_case.clear(c)
            attn_case.run(lib, c, stream.cuda_stream)
            torch.cuda.synchronize()
            d = digest(c)
            seen[d] = seen.get(d, 0) + 1
        print(f"q={q_len} S={c.S} rows={c.rows}: {len(seen)} distinct digest(s) {seen}")
        bad += len(seen) != 1
        # The same launch replayed from a CUDA graph, back to back (no
        # synchronisation in between): the digest must be the eager one.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attn_case.run(lib, c, torch.cuda.current_stream().cuda_stream)
        gseen: dict[str, int] = {}
        for _ in range(args.launches // args.graph_batch):
            attn_case.clear(c)
            for _ in range(args.graph_batch):
                graph.replay()
            torch.cuda.synchronize()
            d = digest(c)
            gseen[d] = gseen.get(d, 0) + 1
        print(f"  graph replays: {gseen}")
        bad += set(gseen) != set(seen)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
