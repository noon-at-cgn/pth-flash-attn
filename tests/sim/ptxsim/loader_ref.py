# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Expected shared-memory stage images of the tile loader.

`csrc/pth_attn/loader.cuh` copies one 64-token KV tile of one request and one
KV head into a pipeline stage. This module says, independently of that code,
which byte must end up where, so the host emulation (g++) and the GPU run of
the loader can be compared with it byte for byte:

- Which cache bytes: read through the torch views the op receives
  (`k_cache[page, slot, head]` etc.), i.e. with torch's own stride handling,
  never with the project's stride code (cache_geometry.cuh), so a stride bug
  there cannot cancel out here.
- Where they go: the simulator's layout functions (`layouts.k_offset`,
  `v_offset`, `stage_offsets`, `cell_pitch`), proved in tests/sim.
- Tokens at or past `tok_end` (outside the CTA's KV range) are zero, and so
  are tokens below `tok_begin` (the windowed kernels' range begin)
  and every byte the loader never writes (the test kernel zeroes the stage
  first): the cell pitch padding.
"""

from __future__ import annotations

import numpy as np
import torch

from .layouts import (
    N_TOKENS,
    SCALE_BYTES,
    cell_bytes,
    cell_pitch,
    k_offset,
    stage_offsets,
    v_offset,
)

VARIANT_SEPARATE = 0  # loader_capi.h PTH_LOADER_VARIANT_SEPARATE
VARIANT_CELL = 1  # loader_capi.h PTH_LOADER_VARIANT_CELL


def stage_bytes(variant: int, D: int) -> int:
    """Size of one stage image (equals pth_loader_stage_bytes)."""
    if variant == VARIANT_SEPARATE:
        return stage_offsets(D)["bytes"]
    return N_TOKENS * cell_pitch(D)


def _token_rows(k_cache, v_cache, k_scale, v_scale, block_table, req, kvh, pos):
    """Bytes of K, V, ks, vs of KV position `pos` via torch indexing."""
    bs = k_cache.shape[1]
    page = int(block_table[req, pos // bs])
    slot = pos % bs

    def raw(t):
        return t.contiguous().view(torch.uint8).numpy()

    return (
        raw(k_cache[page, slot, kvh]),
        raw(v_cache[page, slot, kvh]),
        raw(k_scale[page, slot, kvh].reshape(1)),
        raw(v_scale[page, slot, kvh].reshape(1)),
    )


def expected_images(
    variant: int,
    D: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_table: torch.Tensor,
    req: int,
    kv_head: int,
    tile_begin: int,
    tile_end: int,
    tok_end: int,
    tok_begin: int = 0,
) -> np.ndarray:
    """uint8 [tile_end - tile_begin, stage_bytes] expected stage images.

    Separate streams (variant 0): K [64][D] and V [64][D] swizzled per
    the stage layout, then ks f32[64], vs f32[64]. Whole cell (variant 1): token t's
    cell [K D | ks | V D | vs] at t * (2D + 16), padding zero.
    """
    k_cache, v_cache, k_scale, v_scale, block_table = (
        x.cpu() for x in (k_cache, v_cache, k_scale, v_scale, block_table)
    )
    nb = stage_bytes(variant, D)
    out = np.zeros((tile_end - tile_begin, nb), dtype=np.uint8)
    offs = stage_offsets(D)
    b = np.arange(D)
    for i, tile in enumerate(range(tile_begin, tile_end)):
        img = out[i]
        for t in range(N_TOKENS):
            pos = tile * N_TOKENS + t
            if pos >= tok_end or pos < tok_begin:
                continue  # zero-filled
            k, v, ks, vs = _token_rows(
                k_cache, v_cache, k_scale, v_scale, block_table, req, kv_head, pos
            )
            if variant == VARIANT_SEPARATE:
                img[offs["K"] + k_offset(t, b, D)] = k
                img[offs["V"] + v_offset(t, b, D)] = v
                img[
                    offs["ks"] + SCALE_BYTES * t : offs["ks"] + SCALE_BYTES * (t + 1)
                ] = ks
                img[
                    offs["vs"] + SCALE_BYTES * t : offs["vs"] + SCALE_BYTES * (t + 1)
                ] = vs
            else:
                cell = np.concatenate([k, ks, v, vs])
                assert cell.size == cell_bytes(D)
                img[t * cell_pitch(D) : t * cell_pitch(D) + cell.size] = cell
    return out
