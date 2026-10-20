#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Run an unmodified benchmark script with ``time.time()`` pinned to an integer EPOCH.

    python3 bench/e2e/ladder_salt.py EPOCH SCRIPT [ARGS...]

Purpose: HyperQwen's ``bench/conc_ladder.py`` (https://github.com/syv-ai/HyperQwen)
salts every prompt with ``int(time.time())`` so that no request is served from the
prefix cache. Two runs of an A/B comparison therefore send different prompts, and
prompt-dependent quantities (for example speculative decoding acceptance) differ for
that reason alone. Pinning the clock that the script reads for the salt makes both arms
send byte-identical prompts. The script's timings use ``time.perf_counter``, which is
left alone, and the salt still contains the script's own ``--label``, so repetitions
stay distinct when each is run with its own label (``bench/e2e/ladder.sh`` does this).
Each arm runs against a freshly started server, so an identical salt across arms never
meets a warm prefix cache.

EPOCH must be a non-negative integer; anything else prints a usage message and exits
with status 1. The script runs as ``__main__`` with ``sys.argv`` set to
``[SCRIPT, ARGS...]`` and its own directory first on ``sys.path``, exactly as
``python3 SCRIPT ARGS...`` would, so sibling imports (the ``harness`` module next to
``conc_ladder.py``) resolve.
"""

import os
import runpy
import sys
import time


def main() -> None:
    if len(sys.argv) < 3 or not sys.argv[1].isdigit():
        sys.exit("usage: ladder_salt.py EPOCH SCRIPT [ARGS...]")
    epoch = float(sys.argv[1])
    script = sys.argv[2]
    sys.argv = [script, *sys.argv[3:]]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    time.time = lambda: epoch  # the pinned salt clock (see the module docstring)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
