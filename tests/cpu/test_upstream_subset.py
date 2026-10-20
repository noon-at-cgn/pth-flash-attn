# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The vendored subset is a file selection with no test layer in it.

A downstream copy of the kernel carries the extension's sources and the headers
they include, plus the ``pth_flash_attn`` package (tools/export_upstream.py
derives the list from the build). Two things can silently go wrong:

* a shipped file starts to include a test layer (the reference library, a
  C ABI of a test library, ``test_hooks.cuh``), and the cut is no longer a
  file selection;
* a new file in csrc/pth_attn is neither shipped nor known to be test-only.

The strip of the default hook block and the hook calls is checked here on
text (nothing test-only survives, unknown hook forms fail loudly); that it is
a no-op for the machine code is tools/prove_export_sass.sh (needs nvcc).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import export_upstream  # noqa: E402

# Test-only sources of csrc/pth_attn: built into the test libraries
# (tools/build_attn_lib.sh, build_ref_lib.sh, build_loader_lib.sh) only.
TEST_ONLY_CSRC = {
    "csrc/pth_attn/attn_capi.cu",
    "csrc/pth_attn/attn_capi.h",
    "csrc/pth_attn/attn_test_kernels.cu",
    "csrc/pth_attn/loader_capi.cu",
    "csrc/pth_attn/loader_capi.h",
    "csrc/pth_attn/loader_test_kernels.cu",
    "csrc/pth_attn/loader_test_kernels.h",
    "csrc/pth_attn/ref_capi.cu",
    "csrc/pth_attn/ref_capi.h",
    "csrc/pth_attn/reference_kernel.cu",
    "csrc/pth_attn/test_hooks.cuh",
}


def test_subset_holds_the_op_kernels_and_package():
    """Sanity of the closure: the op, every kernel header, the package."""
    files = set(export_upstream.shipped_files())
    for f in (
        "csrc/pth_attn/launch.cu",
        "csrc/pth_attn/attn_launch.h",
        "csrc/pth_attn/attn_args.h",
        "csrc/pth_attn/attn_kernel.cuh",
        "csrc/pth_attn/attn_decode.cuh",
        "csrc/pth_attn/combine_kernel.cuh",
        "python/pth_flash_attn/vllm_backend.py",
        "python/pth_flash_attn/planner.py",
        "python/pth_flash_attn/ops.py",
    ):
        assert f in files, f


def test_no_test_layer_in_the_subset():
    """No test-only file is shipped, and no shipped file includes one."""
    files = export_upstream.shipped_files()
    assert not TEST_ONLY_CSRC & set(files)
    test_only_names = {Path(f).name for f in TEST_ONLY_CSRC}
    for rel in files:
        if rel.startswith("csrc/"):
            text = (REPO / rel).read_text(encoding="utf-8")
            included = set(re.findall(r'#\s*include\s+"([^"]+)"', text))
            assert not included & test_only_names, (rel, included & test_only_names)


def test_every_csrc_file_is_classified():
    """A new file in csrc/pth_attn must be shipped or listed as test-only."""
    shipped = set(export_upstream.shipped_csrc())
    for path in sorted((REPO / "csrc" / "pth_attn").iterdir()):
        rel = str(path.relative_to(REPO))
        assert (rel in shipped) != (rel in TEST_ONLY_CSRC), rel


def test_export_has_no_test_identifier(tmp_path):
    """The exported copy names no hook, mutation, test macro or test_hooks."""
    files = export_upstream.export(tmp_path)
    assert files == export_upstream.shipped_files()
    for rel in files:
        text = (tmp_path / rel).read_text(encoding="utf-8")
        assert not export_upstream.TEST_TOKENS.search(text), rel
    kernel = (tmp_path / "csrc/pth_attn/attn_kernel.cuh").read_text(encoding="utf-8")
    assert "named_bar_sync(kBarEmpty0 + sj, kAttnThreadsWs)" in kernel


def test_strip_unwraps_a_mutation_and_drops_hook_calls():
    """Teaching example of the three strip rules."""
    src = "\n".join(
        [
            "#include <x>",
            "// BEGIN pth-test-hooks",
            "constexpr int kTestMutate = 0;",
            "// END pth-test-hooks",
            "",
            "void f() {",
            "  proto_init();",
            '  proto_expect_landed("long message", buf,',
            "                      i - tb);",
            "  if (kTestMutate != 3) {",
            "    sync();  // keep",
            "  }",
            "  if constexpr (kProducers > 0) proto_check_ledger(tb, te);",
            "  work();",
            "}",
        ]
    )
    assert export_upstream.strip_test_hooks(src) == "\n".join(
        ["#include <x>", "", "void f() {", "  sync();  // keep", "  work();", "}"]
    )


def test_strip_fails_on_an_unknown_hook_form_negative_control():
    """A hook used inside an expression is not stripped: it must not ship."""
    with pytest.raises(ValueError, match="test_jitter"):
        export_upstream.strip_test_hooks("int x = (test_jitter(1), 2);")
