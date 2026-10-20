# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for tools/repo_checks.py (SPDX, engine-facing, sign-off).

These checks enforce that every source file names its licence and
copyright holder, and that engine-facing code neither calls torch.cuda
nor reads environment variables. Each check is tested on a compliant file and
on a violating one.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import repo_checks  # noqa: E402

GOOD_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn\n"
)


def test_spdx_header_present_and_missing(tmp_path):
    """A compliant file passes; dropping either line is reported."""
    good = tmp_path / "good.py"
    good.write_text("#!/usr/bin/env python3\n" + GOOD_HEADER + "x = 2\n")
    assert repo_checks.spdx_problems(good) == []

    no_copyright = tmp_path / "nocopy.cu"
    no_copyright.write_text("// SPDX-License-Identifier: Apache-2.0\n")
    assert "noon-at-cgn" in repo_checks.spdx_problems(no_copyright)[0]

    wrong_licence = tmp_path / "mit.sh"
    wrong_licence.write_text(
        "# SPDX-License-Identifier: MIT\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn\n"
    )
    assert "Apache-2.0" in repo_checks.spdx_problems(wrong_licence)[0]


def test_spdx_header_too_late_is_missing(tmp_path):
    """The header must be at the top, not buried after the code starts."""
    late = tmp_path / "late.py"
    late.write_text("a = 2\n" * 6 + GOOD_HEADER)
    assert len(repo_checks.spdx_problems(late)) == 2


def test_which_files_are_sources():
    """Code, shell, config and Dockerfiles are sources; markdown is not."""
    for name in ("a.py", "b.cu", "c.cuh", "d.sh", "e.toml", "dev.Dockerfile",
                 ".clang-format", "ci.yml"):  # fmt: skip
        assert repo_checks.is_source(Path(name)), name
    for name in ("README.md", "LICENSE", "x.json", "y.cubin"):
        assert not repo_checks.is_source(Path(name)), name


def test_repository_is_spdx_clean():
    """Every source file in this repository carries the header."""
    problems = [
        p for f in repo_checks.walk_sources(REPO) for p in repo_checks.spdx_problems(f)
    ]
    assert problems == []


def test_engine_facing_rule(tmp_path):
    """torch.cuda and env-var reads are flagged in code, not in strings.

    current_platform is vLLM's device abstraction; os.environ reads are
    forbidden there. Mentions in comments and docstrings are allowed so
    the rule can be documented in the code it governs.
    """
    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""Never call torch.cuda or os.environ here."""\n'
        "# torch.cuda is forbidden\n"
        "from vllm.platforms import current_platform\n"
        "cap = current_platform.get_device_capability()\n"
    )
    assert repo_checks.engine_facing_problems(clean) == []

    dirty = tmp_path / "dirty.py"
    dirty.write_text(
        "import os\nimport torch\n"
        "n = torch.cuda.device_count()\n"
        'x = os.environ["VLLM_X"]\n'
        'y = os.getenv("Y")\n'
    )
    found = repo_checks.engine_facing_problems(dirty)
    assert [p.split(": ", 1)[1].split(" ")[0] for p in found] == [
        "torch.cuda",
        "os.environ",
        "os.getenv",
    ]


def test_package_top_level_is_engine_facing_clean():
    """The package's own modules obey the engine-facing rule today."""
    files = repo_checks.engine_facing_files(REPO)
    assert [p for f in files for p in repo_checks.engine_facing_problems(f)] == []
