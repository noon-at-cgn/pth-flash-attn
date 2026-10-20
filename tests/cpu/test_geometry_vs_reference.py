# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""cache_geometry.cuh against the Python reference, the planner and the bound.

The header's functions are compiled with g++ into a small shared library
(``tests/cpp/test_geometry.cpp`` with ``-DPTH_GEOMETRY_SHARED``) and called
through ctypes with the *real* data pointers and view strides of caches built
by ``pth_testing.reference.make_int8_cache`` (vLLM's packed layout) and of a
#56164-style decoupled layout. The expected bytes come from the writer mirror
applied to the source K/V (``quantize_rows``), which never touches a stride,
so a correct gather proves the address math reads exactly what vLLM wrote.
"""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from pth_flash_attn import planner
from pth_testing.bound import Tiling, token_plan
from pth_testing.reference import (
    make_int8_cache,
    quantize_rows,
    ref_attention,
    visible_bounds,
)

REPO = Path(__file__).resolve().parents[2]
I64P = ctypes.POINTER(ctypes.c_int64)
I32P = ctypes.POINTER(ctypes.c_int32)


@pytest.fixture(scope="module")
def geom(tmp_path_factory):
    """The header compiled as a ctypes library (skipped if g++ is missing)."""
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("g++ not available")
    lib = tmp_path_factory.mktemp("geom") / "libgeom.so"
    subprocess.run(
        [
            gxx,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            "-DPTH_GEOMETRY_SHARED",
            f"-I{REPO / 'csrc' / 'pth_attn'}",
            str(REPO / "tests" / "cpp" / "test_geometry.cpp"),
            "-o",
            str(lib),
        ],
        check=True,
    )
    g = ctypes.CDLL(str(lib))
    g.geom_gather.restype = ctypes.c_int
    g.geom_kv_limit.restype = ctypes.c_int32
    g.geom_bq_capacity.restype = ctypes.c_int32
    g.geom_row_tile.restype = ctypes.c_int32
    g.geom_tile_needs_mask.restype = ctypes.c_int32
    return g


def _i64(vals):
    return (ctypes.c_int64 * len(vals))(*[int(v) for v in vals])


def _bounds(*views):
    """[begin, end) byte range of each view's storage, for the C side's check."""
    vals = []
    for t in views:
        s = t.untyped_storage()
        vals += [s.data_ptr(), s.data_ptr() + s.nbytes()]
    return (ctypes.c_uint64 * len(vals))(*vals)


def _gather(geom, k, v, ks, vs, block_table, seq_lens, D, block_size, strides=None):
    """Run geom_gather on four views; returns (k, v, ks, vs, widths, rc).

    rc is 0, or -2 when an address fell outside a view's storage (only the
    negative control may see that; the other callers assert rc == 0).
    """
    total = int(sum(seq_lens))
    H = k.shape[2]
    out_k = torch.empty(total, H, D, dtype=torch.int8)
    out_v = torch.empty(total, H, D, dtype=torch.int8)
    out_ks = torch.empty(total, H, dtype=torch.float32)
    out_vs = torch.empty(total, H, dtype=torch.float32)
    widths = (ctypes.c_int32 * 4)()
    st = strides or [t.stride()[:3] for t in (k, v, ks, vs)]
    lens = torch.tensor(list(seq_lens), dtype=torch.int32)
    rc = geom.geom_gather(
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(ks.data_ptr()),
        ctypes.c_void_p(vs.data_ptr()),
        _i64(k.shape[:3]),
        _i64(st[0]),
        _i64(st[1]),
        _i64(st[2]),
        _i64(st[3]),
        _bounds(k, v, ks, vs),
        ctypes.cast(block_table.data_ptr(), I32P),
        ctypes.c_int64(block_table.stride(0)),
        ctypes.cast(lens.data_ptr(), I32P),
        ctypes.c_int32(len(seq_lens)),
        ctypes.c_int32(D),
        ctypes.c_int32(block_size),
        ctypes.cast(out_k.data_ptr(), ctypes.POINTER(ctypes.c_int8)),
        ctypes.cast(out_v.data_ptr(), ctypes.POINTER(ctypes.c_int8)),
        ctypes.cast(out_ks.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(out_vs.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        widths,
    )
    assert rc in (0, -2), rc
    if strides is None:
        assert rc == 0, "address outside the view's storage"
    return out_k, out_v, out_ks, out_vs, list(widths), rc


def _expected(c):
    kq, ks = zip(*(quantize_rows(x) for x in c.k_src))
    vq, vs = zip(*(quantize_rows(x) for x in c.v_src))
    return torch.cat(kq), torch.cat(vq), torch.cat(ks), torch.cat(vs)


@pytest.mark.parametrize("block_size", [1, 16, 64, 528, 816, 1056])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_gather_matches_writer(geom, block_size, layout, D):
    """Every byte the header addresses is the byte vLLM's writer stored there.

    Why: this is the kernel's whole read path (paging, slot, head, cell
    offsets, int64 strides) checked on the real packed cache: NHD and HND,
    padded pages (72 B), a 4-byte storage offset, shuffled block ids, empty and
    mid-block requests, every block size of the test matrix (1 included).
    Widths with the 4-byte storage offset: K 4 (base is only 4-aligned) and V
    8 (base + D + 4 + 4 is 8-aligned): the offset moves the alignment from K
    to V. Without the offset it is K 8, V 4 (next test); scales are always 4.
    """
    lens = [3, 0, block_size + 5, 2 * block_size - 1]
    c = make_int8_cache(
        lens,
        Hkv=2,
        D=D,
        block_size=block_size,
        layout=layout,
        pad_page_bytes=72,
        seed=D + block_size,
        storage_offset_bytes=4,
    )
    got = _gather(
        geom,
        c.k_cache,
        c.v_cache,
        c.k_scale,
        c.v_scale,
        c.block_table,
        lens,
        D,
        block_size,
    )
    for a, b in zip(got[:4], _expected(c)):
        assert torch.equal(a, b)
    assert got[4] == [4, 8, 4, 4]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_widths_today_and_decoupled(geom, layout):
    """Today's cell gives K 8, V 4, scales 4; #56164's decoupled layout gives K/V 16.

    Why: the loader's per-stream widths are derived on the host
    from the actual pointers and strides; this pins the two layouts. The
    decoupled caches are separate contiguous [NB, BS, Hkv, D] int8
    and [NB, BS, Hkv] f32 tensors filled through the block table.
    """
    lens, D, BS = [40, 17], 256, 16
    c = make_int8_cache(lens, Hkv=2, D=D, block_size=BS, layout=layout, seed=1)
    w = _gather(
        geom, c.k_cache, c.v_cache, c.k_scale, c.v_scale, c.block_table, lens, D, BS
    )[4]
    assert w == [8, 4, 4, 4]
    nb = c.k_cache.shape[0]
    kd = torch.zeros(nb, BS, 2, D, dtype=torch.int8)
    vd = torch.zeros_like(kd)
    ksd = torch.zeros(nb, BS, 2)
    vsd = torch.zeros_like(ksd)
    for r, s in enumerate(lens):
        pos = torch.arange(s)
        blk = c.block_table[r].long()[pos // BS]
        kd[blk, pos % BS], ksd[blk, pos % BS] = quantize_rows(c.k_src[r])
        vd[blk, pos % BS], vsd[blk, pos % BS] = quantize_rows(c.v_src[r])
    got = _gather(geom, kd, vd, ksd, vsd, c.block_table, lens, D, BS)
    for a, b in zip(got[:4], _expected(c)):
        assert torch.equal(a, b)
    assert got[4] == [16, 16, 4, 4]


def test_high_block_ids_and_padded_table(geom):
    """Pages with byte offsets beyond 2^31, read via a block table with padded rows.

    Why: int64 offsets (vLLM #49716) on real memory: every block id
    is above 2^31 / page_bytes, so an int32 offset would wrap. The block table
    is a column slice of a wider table, so its row stride differs from its
    width (strides come from views).
    """
    lens, D, BS = [100, 37], 64, 16
    c = make_int8_cache(lens, Hkv=1, D=D, block_size=BS, high_block_ids=True, seed=3)
    assert c.max_block_id * c.page_bytes > 2**31
    wide = torch.full((2, c.block_table.shape[1] + 9), -1, dtype=torch.int32)
    wide[:, 3 : 3 + c.block_table.shape[1]] = c.block_table
    bt = wide[:, 3:]
    assert bt.stride(0) != bt.shape[1]
    got = _gather(geom, c.k_cache, c.v_cache, c.k_scale, c.v_scale, bt, lens, D, BS)
    for a, b in zip(got[:4], _expected(c)):
        assert torch.equal(a, b)


@pytest.mark.parametrize(("layout", "pad"), [("HND", 0), ("NHD", 64)])
def test_shape_derived_strides_fail(geom, layout, pad):
    """Negative control: strides derived from the shapes read the wrong bytes.

    The "shape-derived" strides assume the cache is a contiguous
    [NB, BS, Hkv, 2 (D + 4)] tensor (unpadded NHD), the vLLM #49716 bug class.
    On HND or padded pages they must fail the gather check that the view
    strides pass (the reads stay inside the storage: same page set, smaller
    or permuted offsets).
    """
    lens, D, BS = [50, 20], 64, 16
    c = make_int8_cache(
        lens, Hkv=2, D=D, block_size=BS, layout=layout, pad_page_bytes=pad, seed=9
    )
    C = 2 * (D + 4)
    shape_kv = (BS * 2 * C, 2 * C, C)
    shape_sc = tuple(s // 4 for s in shape_kv)
    bad = _gather(
        geom,
        c.k_cache,
        c.v_cache,
        c.k_scale,
        c.v_scale,
        c.block_table,
        lens,
        D,
        BS,
        strides=[shape_kv, shape_kv, shape_sc, shape_sc],
    )
    exp = _expected(c)
    assert bad[5] == -2 or not all(torch.equal(a, b) for a, b in zip(bad[:4], exp))
    good = _gather(
        geom, c.k_cache, c.v_cache, c.k_scale, c.v_scale, c.block_table, lens, D, BS
    )
    assert all(torch.equal(a, b) for a, b in zip(good[:4], exp))


def test_split_partition_equals_planner_and_bound(geom):
    """The header's split partition is planner.split_tile_range and bound.token_plan's.

    Why: the kernel, the planner (which sizes the workspace) and the error
    bound (which counts accumulation lengths) must agree on which tiles each
    split owns; a disagreement would make the bound or the combine wrong.
    """
    b, e = ctypes.c_int32(), ctypes.c_int32()
    for T in range(0, 301):
        for S in range(1, 71):
            counts = []
            for s in range(S):
                geom.geom_split_tile_range(s, S, T, ctypes.byref(b), ctypes.byref(e))
                assert (b.value, e.value) == planner.split_tile_range(
                    split=s, num_splits=S, num_tiles=T
                )
                counts.append(e.value - b.value)
            if T and S <= 40 and T % 7 == 0:
                tp = token_plan(T * 64, Tiling(64, 16, 4, S, 1))
                assert tp.n_tiles_split.tolist() == counts


def _row_tiles(geom, qsl, lens, bq, grid_x):
    out6 = (ctypes.c_int32 * 6)()
    qa = (ctypes.c_int32 * len(qsl))(*qsl)
    la = (ctypes.c_int32 * len(lens))(*lens)
    tiles = []
    for x in range(grid_x + 2):
        if geom.geom_row_tile(qa, la, len(lens), bq, x, out6):
            tiles.append((x, *out6))
    return tiles


@pytest.mark.parametrize("G", [1, 4, 6, 8, 16, 64])
def test_kernel_bq_with_planner_grid(geom, G):
    """BQ = floor(16 wm / G) tiles fit the planner's grid and cover every token.

    Why: the op gets the planner's ``wm`` but not its ``bq`` (frozen schema), so
    the kernel packs floor(16 wm / G) tokens per row tile. That is >= the
    planner's bq, so every tile with work has x < T // bq + num_reqs (grid.x),
    and for decode ceil(q / BQ) equals the planner's rt (so S is
    unchanged). Checked for decode and prefill descriptors.
    """
    rng = np.random.default_rng(G)
    for _ in range(60):
        R = int(rng.integers(1, 6))
        prefill = bool(rng.integers(0, 2))
        qmax = int(rng.integers(1, 200 if prefill else 17))
        qs = [int(rng.integers(0, qmax + 1)) for _ in range(R)]
        qs[0] = qmax
        T = sum(qs)
        p = planner.plan_launch(
            num_tokens=T,
            num_reqs=R,
            max_query_len=qmax,
            G=G,
            Hkv=2,
            D=128,
            num_sms=68,
            is_prefill=prefill,
            ws_token_splits=0,
        )
        bq = geom.geom_bq_capacity(p.wm, G)
        assert bq >= p.bq and bq * G <= 16 * p.wm
        if not prefill:
            assert math.ceil(qmax / bq) == math.ceil(qmax / p.bq)
        qsl = [0, *np.cumsum(qs).tolist()]
        lens = [q + 5 for q in qs]
        covered = []
        for x, req, tb, nt, ql, sl, q0 in _row_tiles(
            geom, qsl, lens, bq, p.q_tiles_upper
        ):
            assert x < p.q_tiles_upper
            assert (ql, sl, q0) == (qs[req], lens[req], qsl[req])
            covered += list(range(q0 + tb, q0 + tb + nt))
        assert sorted(covered) == list(range(T))


def test_causal_limit_matches_reference(geom):
    """kv_limit and the row tile's KV range are ref_attention's bottom-right mask.

    Why: the reference is the oracle; with K = 0 every visible key has the
    same weight and V = one-hot(token), so the visible set of each query row
    can be read off ref_attention's output and compared with the header's
    limit. Also checks the range end and min limit the bound and emulator use
    (L - q_len + last + 1) and that the mask predicate is exact.
    """
    rng = np.random.default_rng(4)
    for _ in range(20):
        ql = int(rng.integers(1, 12))
        L = ql + int(rng.integers(0, 70))
        k = torch.zeros(L, 1, L, dtype=torch.float64)
        v = torch.zeros(L, 1, L, dtype=torch.float64)
        v[torch.arange(L), 0, torch.arange(L)] = 1.0
        out = ref_attention(torch.zeros(ql, 1, L), [k], [v], [0, ql], [L], 1.0).out
        for i in range(ql):
            visible = int((out[i, 0] > 0).sum())
            assert geom.geom_kv_limit(ql, L, i) == visible - 1
        out2 = (ctypes.c_int32 * 2)()
        for bq in (1, 3, 10):
            for tb in range(0, ql, bq):
                nt = min(bq, ql - tb)
                geom.geom_tile_range(ql, L, tb, nt, out2)
                assert out2[0] == L - ql + tb + nt  # bound/emulate range_end
                assert out2[1] == L - ql + tb
                for tile in range(-(-out2[0] // 64)):
                    need = geom.geom_tile_needs_mask(tile, 64, out2[1])
                    masked = any(
                        j > L - ql + i
                        for i in range(tb, tb + nt)
                        for j in range(tile * 64, tile * 64 + 64)
                    )
                    assert bool(need) == masked


def test_window_matches_reference_bound_and_planner(geom):
    """The window geometry is ref_attention's visible set; tiles and splits agree.

    Why: the kernel's kv_window_lo/hi must be the reference's
    (vLLM's) window for every row, read off ref_attention with K = 0 and
    one-hot V; the CTA range [begin, end) must be the one the bound and the
    emulator use (lo of the first token, hi of the last + 1); and the split
    partition must start at the tile holding begin and equal
    planner.split_tile_range offset by that tile, and bound.token_plan's
    per-split tile counts.
    """
    rng = np.random.default_rng(8)
    out2 = (ctypes.c_int32 * 2)()
    out4 = (ctypes.c_int32 * 4)()
    b, e = ctypes.c_int32(), ctypes.c_int32()
    for _ in range(40):
        ql = int(rng.integers(1, 12))
        L = ql + int(rng.integers(0, 200))
        window = int(rng.choice([0, 1, 2, int(rng.integers(1, L + 20))]))
        causal = bool(rng.integers(0, 2))
        k = torch.zeros(L, 1, L, dtype=torch.float64)
        v = torch.zeros(L, 1, L, dtype=torch.float64)
        v[torch.arange(L), 0, torch.arange(L)] = 1.0
        out = ref_attention(
            torch.zeros(ql, 1, L), [k], [v], [0, ql], [L], 1.0,
            window=window, causal=causal,
        ).out  # fmt: skip
        for i in range(ql):
            seen = (out[i, 0] > 0).nonzero().flatten().tolist()
            geom.geom_kv_window(ql, L, i, window, int(causal), out2)
            assert seen == list(range(out2[0], out2[1] + 1))
        for bq in (1, 3, 10):
            for tb in range(0, ql, bq):
                nt = min(bq, ql - tb)
                geom.geom_window_tile_range(ql, L, tb, nt, window, int(causal), out4)
                lo, hi = visible_bounds(
                    [L - ql + i for i in range(tb, tb + nt)], L, window, causal
                )
                assert list(out4) == [lo[0], hi[-1] + 1, lo[-1], hi[0]]
                first = lo[0] // 64
                T = -(-(hi[-1] + 1) // 64) - first
                for S in (1, 2, 5):
                    counts = []
                    for s in range(S):
                        geom.geom_window_split_tile_range(
                            s, S, out4[0], out4[1], 64, ctypes.byref(b), ctypes.byref(e)
                        )
                        pb, pe = planner.split_tile_range(
                            split=s, num_splits=S, num_tiles=T
                        )
                        assert (b.value, e.value) == (first + pb, first + pe)
                        counts.append(e.value - b.value)
                    tp = token_plan(hi[-1] + 1, Tiling(64, 16, 4, S, nt), lo[0])
                    assert tp.n_tiles_split.tolist() == counts
