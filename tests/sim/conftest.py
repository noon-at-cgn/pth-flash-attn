# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Make the `ptxsim` simulator package importable from the sim tests.

The simulators live next to their tests (tests/sim/ptxsim) because they are
test infrastructure, not part of the shipped `pth_flash_attn` package. Adding
this directory to sys.path here works under every pytest import mode
(prepend, append, importlib), which a relative import would not.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
