# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Test oracles of the kernel; not part of the shipped package.

* :mod:`pth_testing.reference`: the fp64 reference attention and vLLM-layout
  int8 cache builders (the exact answer);
* :mod:`pth_testing.bound`: the data-dependent error bound every kernel
  output must satisfy;
* :mod:`pth_testing.emulate`: a bit-level CPU model of the kernel's
  arithmetic, which the bound is tested against;
* :mod:`pth_testing.ref_capi`: ctypes mirror of the C-ABI test libraries
  (``csrc/pth_attn/ref_capi.h``, ``attn_capi.h``).

They live next to ``pth_flash_attn`` under ``python/`` so that every tool and
container that puts ``python/`` on ``PYTHONPATH`` can import them, but the
wheel packages ``pth_flash_attn`` only (``pyproject.toml``, ``setup.py``).
"""
