# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The exact shared-memory access patterns of the kernel, as bank-model input.

Each generator yields `banks.Access` objects, one per warp instruction (or,
for ldmatrix, one per 8x8 matrix, which is one phase). The patterns are:
- K: ldmatrix (b16 view of int8) of 8 consecutive tokens x one 16-B chunk
  per matrix (one MMA n-tile x one k16 step).
- Q: ldmatrix of 8 consecutive Q rows x one 16-B chunk (A operand).
- V: LDS.32 by lane (g, q) of V[t][32a+4g .. +3] for t in
  {2q, 2q+1, 2q+8, 2q+9} + 16*kt (one instruction per t-offset).
- ks/vs: LDS.64 of scales for tokens (2q, 2q+1) and (2q+8, 2q+9).
- cp.async destination writes for each stream and width.
"""

from __future__ import annotations

from collections.abc import Iterator

from .banks import Access
from .layouts import (
    N_TOKENS,
    cell_loader_instructions,
    cell_pitch,
    k_offset,
    loader_instructions,
    q_offset,
    scale_pair_instructions,
    stage_offsets,
    v_offset,
)

V_TOKEN_OFFSETS = (0, 1, 8, 9)  # B-fragment k rows 2q, 2q+1, 2q+8, 2q+9 minus 2q
Q_ROWS_MAX = 64  # WM=4 warps x 16 rows


def k_ldmatrix(D: int, swizzled: bool = True, offset_fn=None) -> Iterator[Access]:
    fn = offset_fn or (lambda t, b: k_offset(t, b, D, swizzled))
    for nt in range(N_TOKENS // 8):
        for s in range(D // 16):
            addrs = tuple(int(fn(8 * nt + r, 16 * s)) for r in range(8))
            yield Access(addrs, 16, kind="ldmatrix")


def q_ldmatrix(D: int, swizzled: bool = True) -> Iterator[Access]:
    for r0 in range(0, Q_ROWS_MAX, 8):
        for c in range(2 * D // 16):
            addrs = tuple(int(q_offset(r0 + r, 8 * c, D, swizzled)) for r in range(8))
            yield Access(addrs, 16, kind="ldmatrix")


def v_lds32(D: int, swizzled: bool = True, offset_fn=None) -> Iterator[Access]:
    fn = offset_fn or (lambda t, b: v_offset(t, b, D, swizzled))
    for kt in range(N_TOKENS // 16):
        for a in range(D // 32):
            for off in V_TOKEN_OFFSETS:
                addrs = []
                for lane in range(32):
                    g, q = lane >> 2, lane & 3
                    addrs.append(int(fn(16 * kt + 2 * q + off, 32 * a + 4 * g)))
                yield Access(tuple(addrs), 4)


def scale_lds64(D: int, stream: str = "vs") -> Iterator[Access]:
    base = stage_offsets(D)[stream]
    for kt in range(N_TOKENS // 16):
        for half in (0, 8):
            addrs = tuple(
                base + 4 * (16 * kt + 2 * (ln & 3) + half) for ln in range(32)
            )
            yield Access(addrs, 8)


def cp_async_writes(
    D: int, stream: str, width: int, swizzled: bool = True
) -> Iterator[Access]:
    """Destination side of the loader's cp.async stream for one tile.

    stream: "K" or "V" (any width), or "scales" (the paired 4-B instruction
    that writes ks and vs together, `scale_pair_instructions`).
    """
    offs = stage_offsets(D)
    if stream == "scales":
        assert width == 4
        for ins in scale_pair_instructions():
            lanes = tuple(ln for (_, ln, _, _, _) in ins)
            addrs = tuple(offs[s] + 4 * t + b for (_, _, s, t, b) in ins)
            yield Access(addrs, width, lanes=lanes)
        return
    fn = k_offset if stream == "K" else v_offset
    for ins in loader_instructions(D, width):
        lanes = tuple(ln for (_, ln, _, _) in ins)
        addrs = tuple(offs[stream] + int(fn(t, b, D, swizzled)) for (_, _, t, b) in ins)
        yield Access(addrs, width, lanes=lanes)


def scale_pair_same_token_writes(D: int) -> Iterator[Access]:
    """Negative control for the scale pairing: lanes 16..31 writing vs of the
    warp's *own* tokens (a pairing the kernel must not use) instead of warp w ^ 1."""
    offs = stage_offsets(D)
    for w in range(4):
        addrs = tuple(offs["ks"] + 4 * (16 * w + ln) for ln in range(16))
        addrs += tuple(offs["vs"] + 4 * (16 * w + ln) for ln in range(16))
        yield Access(addrs, 4)


def cell_writes(D: int, width: int) -> Iterator[Access]:
    """Destination side of the whole-cell challenger's cp.async stream."""
    pitch = cell_pitch(D)
    for ins in cell_loader_instructions(D, width):
        lanes = tuple(ln for (_, ln, _, _) in ins)
        addrs = tuple(t * pitch + b for (_, _, t, b) in ins)
        yield Access(addrs, width, lanes=lanes)
