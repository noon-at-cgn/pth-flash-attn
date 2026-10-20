# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The paged tile loader on the GPU, byte-exact, both variants.

``pth_loader_run`` (csrc/pth_attn/loader_capi.h) runs one CTA of
the real two-stage cp.async pipeline over KV tiles [tile_begin, tile_end) of
one request and KV head and dumps every stage image. The expected images come
from ``tests/sim/ptxsim/loader_ref.expected_images``: cache bytes read through
torch's own view indexing (never through cache_geometry.cuh) and placed with
the simulator's proven layout functions (line-form swizzle for the
separate-stream variant; pitch 2D+16 cells for the whole-cell variant).

Why it matters: the loader is the only code that turns (request, position,
head) into cache addresses. A wrong page, slot, head, stride, width, swizzle,
zero fill or a race between the two stages shows up here as a wrong byte, on
the hardware that will run the kernel (the CPU emulation in
tests/cpu/test_loader_host.py cannot see cp.async ordering or races).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from grid_cuda import (
    LOADER_ERR_NOT_CELL,
    LOADER_ERR_WIDTH,
    VARIANT_CELL,
    VARIANT_SEPARATE,
    cache_to,
)
from pth_testing.reference import make_int8_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from ptxsim.loader_ref import expected_images  # noqa: E402

TILE = 64  # tokens per KV tile (tile_traits.cuh kLoadTileTokens)
# Page padding as in tests/cpu/test_loader_host.py: 72 B moves every page off
# the 16-B grid of the dense page, so widths and page strides are exercised.
PAD_PAGE_BYTES = 72
# Tokens past the last page edge, and the causal-range cut for the second
# run (tok_end mid-tile, so the zero fill is exercised): CPU test's values.
TAIL_TOKENS = 37
CUT_TOKENS = 29
VARIANTS = {"sep": VARIANT_SEPARATE, "cell": VARIANT_CELL}


def _views(c):
    return c.k_cache, c.v_cache, c.k_scale, c.v_scale


def _run(lib, variant, D, views, bt, req, kvh, tile_end, tok_end, override=None):
    out = torch.zeros(
        tile_end * lib.stage_bytes(variant, D),
        dtype=torch.uint8,
        device=views[0].device,
    )
    rc = lib.run(
        variant,
        D,
        views,
        bt,
        req,
        kvh,
        0,
        tile_end,
        tok_end,
        out,
        override=override,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return rc, out.cpu().numpy().reshape(tile_end, -1)


def _compare(got: np.ndarray, want: np.ndarray, what: str) -> None:
    if not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        t, b = (int(x) for x in bad[0])
        pytest.fail(
            f"{what}: {len(bad)} bytes differ; first at tile {t} byte {b}: "
            f"got {got[t, b]}, want {want[t, b]}"
        )


def _check_all_tiles(
    lib,
    host,
    dev,
    variant,
    D,
    req,
    kvh,
    tok_end,
    what,
    views_host=None,
    views_dev=None,
    override=None,
):
    vh = views_host or _views(host)
    vd = views_dev or _views(dev)
    tiles = -(-tok_end // TILE)
    want = expected_images(
        variant, D, *vh, host.block_table, req, kvh, 0, tiles, tok_end
    )
    rc, got = _run(
        lib, variant, D, vd, dev.block_table, req, kvh, tiles, tok_end, override
    )
    assert rc == 0, f"{what}: pth_loader_run returned {rc}"
    _compare(got, want, what)


def _configs():
    for D in (64, 128, 256):
        for bs in (16, 64, 528, 816):
            for layout in ("NHD", "HND"):
                for vname in VARIANTS:
                    marks = []
                    if D == 256 and bs in (16, 816):
                        marks.append(pytest.mark.quick)
                    if D == 128 and bs == 16 and layout == "NHD":
                        marks.append(pytest.mark.sanitizer)
                    yield pytest.param(
                        D,
                        bs,
                        layout,
                        vname,
                        marks=marks,
                        id=f"D{D}-bs{bs}-{layout}-{vname}",
                    )


@pytest.mark.parametrize(("D", "bs", "layout", "vname"), list(_configs()))
def test_loader_images_byte_exact(loader_lib, device, record, D, bs, layout, vname):
    """Every tile of a request spanning two page edges, both KV heads, the
    full KV range and a range ending mid-tile: GPU images == reference."""
    L = 2 * bs + TAIL_TOKENS
    host = make_int8_cache(
        [5, L],
        Hkv=2,
        D=D,
        block_size=bs,
        layout=layout,
        pad_page_bytes=PAD_PAGE_BYTES,
        seed=D + bs,
    )
    dev = cache_to(host, device)
    v = VARIANTS[vname]
    for kvh, tok_end in ((1, L), (0, L - CUT_TOKENS)):
        _check_all_tiles(
            loader_lib,
            host,
            dev,
            v,
            D,
            1,
            kvh,
            tok_end,
            f"D{D} bs{bs} {layout} {vname} kvh{kvh} tok_end{tok_end}",
        )
    record(
        test="loader",
        case=f"D{D}-bs{bs}-{layout}-{vname}",
        passed=True,
        tiles=-(-L // TILE),
    )


@pytest.mark.parametrize("vname", list(VARIANTS))
def test_loader_storage_offset_and_high_block_ids(loader_lib, device, record, vname):
    """A 4-B storage offset (narrower widths) and byte offsets above 2^31
    (int64 addressing) on the GPU."""
    v = VARIANTS[vname]
    off = make_int8_cache(
        [90], Hkv=1, D=128, block_size=16, storage_offset_bytes=4, seed=5
    )
    _check_all_tiles(
        loader_lib, off, cache_to(off, device), v, 128, 0, 0, 90, f"offset4 {vname}"
    )
    high = make_int8_cache(
        [100, 37], Hkv=1, D=64, block_size=16, high_block_ids=True, seed=3
    )
    assert high.max_block_id * high.page_bytes > 2**31
    _check_all_tiles(
        loader_lib,
        high,
        cache_to(high, device),
        v,
        64,
        0,
        0,
        100,
        f"high block ids {vname}",
    )
    record(test="loader", case=f"offset-high-{vname}", passed=True)


@pytest.mark.parametrize("bs", [16, 816, 864])
@pytest.mark.parametrize("D", [64, 128])
def test_loader_lower_bound_byte_exact(loader_lib, device, record, D, bs):
    """The windowed kernels' loader: tokens below tok_begin are
    zero in the stage, the rest are the reference bytes, and pages wholly
    below tok_begin are never read: their block-table entries point to a
    block of all-0xFF bytes (NaN scales, the evicted-window null block), so a
    read would show up as 0xFF bytes in the image."""
    # At least 3 tiles before the tail, so every tok_begin below lies inside
    # (0, L) for every block size (bs = 16 would give L = 85 otherwise).
    L = 3 * max(bs, TILE) + TAIL_TOKENS
    host = make_int8_cache([L], Hkv=2, D=D, block_size=bs, seed=D + bs + 1)
    used = set(host.block_table.flatten().tolist())
    spare = next(b for b in range(host.k_cache.shape[0]) if b not in used)
    host.kv_packed[spare].fill_(-1)
    for tok_begin in (70, 80, 128, L - 100):
        assert 0 < tok_begin < L
        bt = host.block_table.clone()
        bt[0, : tok_begin // bs] = spare
        h = dataclasses.replace(host, block_table=bt)
        dev = cache_to(h, device)
        first = tok_begin // TILE
        tiles = -(-L // TILE)
        want = expected_images(
            VARIANT_SEPARATE, D, *_views(h), bt, 0, 1, first, tiles, L, tok_begin
        )
        out = torch.zeros(
            (tiles - first) * loader_lib.stage_bytes(VARIANT_SEPARATE, D),
            dtype=torch.uint8,
            device=device,
        )
        rc = loader_lib.run(
            VARIANT_SEPARATE,
            D,
            _views(dev),
            dev.block_table,
            0,
            1,
            first,
            tiles,
            L,
            out,
            stream=torch.cuda.current_stream().cuda_stream,
            tok_begin=tok_begin,
        )
        torch.cuda.synchronize()
        assert rc == 0, f"pth_loader_run returned {rc}"
        got = out.cpu().numpy().reshape(tiles - first, -1)
        _compare(got, want, f"D{D} bs{bs} tok_begin{tok_begin}")
    record(test="loader", case=f"lower-D{D}-bs{bs}", passed=True)


@pytest.mark.parametrize("D", [64, 128, 256])
def test_loader_every_width_on_decoupled_caches(loader_lib, device, record, D):
    """Contiguous (decoupled, #56164-style) K/V views: widths 16, 8 and 4 all
    give the reference image; a width above the legal one is refused, and
    the whole-cell variant refuses views that are not one inline cell."""
    host = make_int8_cache([3, 300], Hkv=2, D=D, block_size=64, seed=11)
    dec_h = tuple(t.contiguous() for t in _views(host))
    dev = cache_to(host, device)
    dec_d = tuple(t.to(device) for t in dec_h)
    for w in (16, 8, 4):
        _check_all_tiles(
            loader_lib,
            host,
            dev,
            VARIANT_SEPARATE,
            D,
            1,
            1,
            300,
            f"decoupled w{w}",
            views_host=dec_h,
            views_dev=dec_d,
            override=[w, w, 4, 4],
        )
    rc, _ = _run(
        loader_lib,
        VARIANT_SEPARATE,
        D,
        _views(dev),
        dev.block_table,
        1,
        0,
        1,
        300,
        override=[16, 4, 4, 4],
    )
    assert rc == LOADER_ERR_WIDTH
    rc, _ = _run(loader_lib, VARIANT_CELL, D, dec_d, dev.block_table, 1, 0, 1, 300)
    assert rc == LOADER_ERR_NOT_CELL
    record(test="loader", case=f"decoupled-D{D}", passed=True)


@pytest.mark.parametrize("vname", list(VARIANTS))
def test_loader_comparison_sees_one_byte(loader_lib, device, vname):
    """Negative control: one K byte changed on the device after the expected
    image was built must make the comparison fail.

    Why: a comparison that cannot fail (wrong buffer, all-zero images on
    both sides) would make every test above vacuous.
    """
    v = VARIANTS[vname]
    host = make_int8_cache([130], Hkv=1, D=64, block_size=16, seed=7)
    dev = cache_to(host, device)
    want = expected_images(v, 64, *_views(host), host.block_table, 0, 0, 0, 3, 130)
    page = int(host.block_table[0, 70 // 16])
    dev.k_cache[page, 70 % 16, 0, 5] += 1  # token 70 = tile 1, byte 5
    rc, got = _run(loader_lib, v, 64, _views(dev), dev.block_table, 0, 0, 3, 130)
    assert rc == 0
    assert np.count_nonzero(got != want) == 1
    assert np.count_nonzero(got[1] != want[1]) == 1
