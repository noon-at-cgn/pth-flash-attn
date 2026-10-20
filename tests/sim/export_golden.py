# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Write the simulator's golden tables (pi, magic, swizzles, loader map) as
JSON for the header tests.

    python tests/sim/export_golden.py build/sim_golden.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ptxsim.golden import golden_tables  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_golden.py OUT.json")
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden_tables(), sort_keys=True) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
