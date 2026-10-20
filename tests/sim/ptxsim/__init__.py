# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""numpy simulators of the kernel's fragment and shared-memory data path.

Modules (each cites the PTX ISA / CUDA documentation it models):
- fp:        bit-exact fp16/bf16 helpers and an independent RNE rounding.
- isa:       prmt, __byte_perm, lop3, sub.f16x2, ldmatrix, mma.m16n8k16,
             LDS.32 and cp.async, per lane.
- dequant:   the magic int8->fp16 conversion (cvt4) and V byte pairing.
- layouts:   stage layout, XOR swizzles, pi, V d-permutation, loader map.
- banks:     shared-memory bank-conflict counter.
- patterns:  the kernel's exact shared-memory access patterns.
- paths:     one warp's QK and PV path executed instruction by instruction.
- prescale:  per-row power-of-two Q prescale into fp16.
- golden:    JSON golden tables for checking the CUDA headers.
"""
