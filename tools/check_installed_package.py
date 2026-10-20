#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Checks the INSTALLED pth-flash-attn package (run from outside the source tree).

Run it after ``pip install`` (``docs/build.md``, "Checking an installed copy"): it
proves the environment carries a working copy and not just a source tree that
happens to be importable.

1. ``pth_flash_attn`` resolves to site-packages, not to a checkout;
2. the op library ``_C.abi3.so`` is inside the installed package and loads,
   with the op and its fake registered;
3. exactly one ``vllm.general_plugins`` entry point named ``pth_flash_attn``
   exists and, when called the way vLLM calls it, registers the backend under
   ``AttentionBackendEnum.CUSTOM``;
4. the backend class imports, and reports the name ``CUSTOM``.

Exit status 0 on success; every failure is an AssertionError with the facts.
"""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path


def main() -> int:
    """Run the four checks; print one line per check."""
    import pth_flash_attn
    import pth_flash_attn.ops as ops

    pkg = Path(pth_flash_attn.__file__).resolve().parent
    assert "site-packages" in pkg.parts, f"not an installed package: {pkg}"
    print(f"1. installed at {pkg}")

    assert ops.is_available(), f"op library {ops.LIBRARY_PATH} not loaded"
    assert Path(ops.LIBRARY_PATH).resolve().parent == pkg, ops.LIBRARY_PATH
    print(f"2. op library loaded: {ops.LIBRARY_PATH}")

    eps = [
        e
        for e in metadata.entry_points(group="vllm.general_plugins")
        if e.name == "pth_flash_attn"
    ]
    assert len(eps) == 1, eps
    assert eps[0].value == "pth_flash_attn.plugin:register", eps[0].value
    eps[0].load()()
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    path = AttentionBackendEnum.CUSTOM.get_path()
    assert path == "pth_flash_attn.vllm_backend.PthFlashAttnBackend", path
    print(f"3. entry point registers CUSTOM -> {path}")

    cls = AttentionBackendEnum.CUSTOM.get_class()
    assert cls.get_name() == "CUSTOM", cls.get_name()
    print(f"4. backend class imports: {cls.__module__}.{cls.__qualname__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
