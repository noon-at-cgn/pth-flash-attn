# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The correctness grid, run by the tensor-core kernel.

Every ``grid_cases`` case becomes an :class:`attn_lib.AttnCase` (same inputs,
the kernel's tiling for the bound) and runs through ``grid_cuda.ref_kernel``
with :class:`attn_lib.AttnLib` in place of the reference library: the same
gate (every valid element within ``bound()``, no NaN/inf, canaries intact).

The fp64 reference and the bound of a case are CPU work (seconds to minutes)
and are cached in ``build/grid_expected/`` under a key that includes the
tiling. Fill the cache before a GPU run (CPU-only container)::

    python tests/gpu/attn_grid.py expected --tier full --jobs 12

A/B builds (``tools/build_attn_lib.sh --out NAME -D...``) are graded with the
same grid by loading this module as a pytest plugin::

    pytest -p attn_grid tests/gpu/test_attn_grid.py --attn-lib build/NAME/libpth_attn.so
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import grid_cases as grid
from attn_lib import ATTN_LIB, AttnCase, AttnLib, attn_case

_LIB: AttnLib | None = None
_LIB_PATH: Path | None = None


def pytest_addoption(parser) -> None:  # plugin hook, see the module docstring
    parser.addoption(
        "--attn-lib", default=None, help="library to grade (default build/attn)"
    )


def pytest_configure(config) -> None:  # plugin hook
    global _LIB_PATH
    path = config.getoption("--attn-lib", default=None)
    if path:
        _LIB_PATH = Path(path).resolve()


def lib() -> AttnLib:
    """The process's library (loaded once; workers load their own)."""
    global _LIB
    if _LIB is None:
        _LIB = AttnLib(_LIB_PATH or ATTN_LIB)
    return _LIB


def cases(tier: str) -> list[AttnCase]:
    """The grid's cases with the kernel's tiling."""
    return [attn_case(c, lib()) for c in grid.cases(tier)]


def case_by_name(name: str) -> AttnCase:
    return attn_case(grid.case_by_name(name), lib())


def _expected_worker(name: str) -> tuple[str, float]:
    return name, grid.load_expected(case_by_name(name)).seconds


def _init(lib_path: str | None = None) -> None:
    global _LIB_PATH
    if lib_path:
        _LIB_PATH = Path(lib_path).resolve()
    grid._init_worker()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("list", "expected"))
    ap.add_argument("--tier", choices=("quick", "full"), default="full")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--lib", help="library to grade (default build/attn)")
    ap.add_argument("--only", default="", help="substring filter on case names")
    args = ap.parse_args(argv)
    _init(args.lib)
    cs = [c for c in cases(args.tier) if args.only in c.name]
    if args.command == "list":
        for c in cs:
            print(c.name, c.tiling)
        return 0
    names = [c.name for c in sorted(cs, key=grid._cost, reverse=True)]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init, initargs=(args.lib,)) as pool:
        for name, secs in pool.imap_unordered(_expected_worker, names):
            print(f"{name}: {secs:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
