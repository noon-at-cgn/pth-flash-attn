# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Prefill and mixed launches at the production geometry, checked on windows.

The shared grid has prefill rows only at q = 512 / 2048 and at one prefix
length; its fp64 reference and bound cost ~30 min per q = 2048 case on this
host (docs: cached ``seconds`` in build/grid_expected) and grow with q^2, so a
q = 8192 case would take hours. A request's row tiles are independent (every
tile has its own causal limit and its own softmax state), so the bound of a
*window* of whole row tiles equals the bound of those tiles in the full
request: the window [t_start, t_end) of a request of q tokens and L keys is itself a
request of q' = t_end - t_start tokens and L' = L - q + t_end keys (bottom-right causal:
token j' of the window has limit L' - q' + j' = L - q + t_start + j'), and with
t_start a multiple of bq the window's tiles start where the full request's do.

Cases here: q = 2048 and 8192 after a prefix that ends inside a page (block
816), a prefill without prefix, and one launch that mixes a prefill chunk,
decode and verify requests, an empty request and padding rows (the planner's
launch-wide bq gives every request the 60-row tile), at S = 1 and S = 2; all
with sentinel keys at the tile, page and split edges. Requests of up to
``WHOLE`` tokens are checked whole; longer ones on three windows of
``WINDOW_TILES`` tiles (start, middle, end, the end window includes the
ragged last tile).

    python tests/gpu/attn_prefill.py expected --jobs 8     # fill the CPU cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass

import grid_cases as grid
import numpy as np
from pth_testing.bound import bound
from pth_testing.reference import DenseKV

WHOLE = 128  # requests up to this many query tokens are checked whole
WINDOW_TILES = 4  # row tiles per window of a longer request


@dataclass(frozen=True)
class Window:
    """Query tokens [t_start, t_end) of request ``req``."""

    req: int
    t_start: int
    t_end: int


def cases() -> list[grid.GridCase]:
    """The prefill and mixed cases (module doc)."""
    out = [
        grid.GridCase(
            "pfx-q2048-prefix3000", "prefill", (2048,), (2048 + 3000,), sentinel=True
        ),
        grid.GridCase("pfx-q8192-noprefix", "prefill", (8192,), (8192,), sentinel=True),
        grid.GridCase(
            "pfx-q8192-prefix4000", "prefill", (8192,), (8192 + 4000,), sentinel=True
        ),
    ]
    for S in (1, 2):
        out.append(
            grid.GridCase(
                f"pfx-mixed-b6-S{S}",
                "prefill",
                (2048, 1, 4, 10, 0, 1),
                (2048 + 1633, 65536, 5000, 9000, 0, 817),
                num_splits=S,
                pad_rows=3,
                sentinel=True,
            )
        )
    return out


def case_by_name(name: str) -> grid.GridCase:
    for c in cases():
        if c.name == name:
            return c
    raise KeyError(name)


def windows(case: grid.GridCase) -> list[Window]:
    """Whole short requests, three windows of a long one."""
    bq = case.bq
    out = []
    for r, q in enumerate(case.q_lens):
        if q == 0:
            continue
        if q <= WHOLE:
            out.append(Window(r, 0, q))
            continue
        tiles = -(-q // bq)
        for first in (0, (tiles - WINDOW_TILES) // 2, tiles - WINDOW_TILES):
            out.append(Window(r, first * bq, min((first + WINDOW_TILES) * bq, q)))
    return out


def window_name(w: Window) -> str:
    return f"r{w.req}-{w.t_start}-{w.t_end}"


def _path(case: grid.GridCase, w: Window):
    key = hashlib.sha256(
        (
            json.dumps(case.describe(), sort_keys=True)
            + window_name(w)
            + grid._code_key()
        ).encode()
    ).hexdigest()[:16]
    return grid.EXPECTED_DIR / f"{case.name}-{window_name(w)}-{key}.npz"


def expected(case: grid.GridCase, w: Window, inp=None):
    """(ref, bound, valid, seconds) of one window, from the cache or computed."""
    path = _path(case, w)
    if path.exists():
        with np.load(path) as z:
            return z["ref"], z["bound"], z["valid"], float(z["seconds"])
    inp = inp or grid.build_inputs(case)
    q_total = case.q_lens[w.req]
    start = int(inp.qsl[w.req])
    kv = inp.kv[w.req]
    seq = case.seq_lens[w.req] - q_total + w.t_end
    sub_kv = DenseKV(kv.k_int[:seq], kv.k_scale[:seq], kv.v_int[:seq], kv.v_scale[:seq])
    t = time.perf_counter()
    b = bound(
        inp.q[start + w.t_start : start + w.t_end],
        [sub_kv],
        np.array([0, w.t_end - w.t_start], dtype=np.int32),
        inp.softmax_scale,
        tiling=case.tiling,
        out_dtype=case.torch_out_dtype,
    )
    secs = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, ref=b.ref, bound=b.bound, valid=b.valid, seconds=secs)
    tmp.replace(path)
    return b.ref, b.bound, b.valid, secs


def _worker(args: tuple[str, int]) -> tuple[str, float]:
    import torch

    torch.set_num_threads(1)
    name, i = args
    case = case_by_name(name)
    w = windows(case)[i]
    _, _, _, secs = expected(case, w)
    return f"{name} {window_name(w)}", secs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("list", "expected"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    work = [
        (c.name, i)
        for c in cases()
        if args.only in c.name
        for i in range(len(windows(c)))
    ]
    if args.command == "list":
        for name, i in work:
            print(name, windows(case_by_name(name))[i])
        return 0
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for label, secs in pool.imap_unordered(_worker, work):
            print(f"{label}: {secs:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
