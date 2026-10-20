# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Host checks of the tile loader (csrc/pth_attn/loader.cuh).

The loader's per-lane code is compiled by g++ (loader_capi.cu, host
emulation: cp.async becomes memcpy plus zero fill, the shuffle becomes a
lookup in the 32 lanes' values) and run on real caches built by
`pth_testing.reference.make_int8_cache`. Two independent references:

- the simulator's golden loader lane map (tests/sim/ptxsim/golden.py), for
  which lane copies which (token, byte);
- `ptxsim.loader_ref.expected_images`, the byte image a stage must hold,
  built with torch view indexing (never the project's stride code) and the
  simulator's swizzle functions.

Mutations of temp copies of the headers (wrong swizzle, wrong V region,
missing zero fill, wrong scale pairing) must each be caught. The device
build of the same code is checked by tests/cpp + the SASS gate here, and
byte for byte on the GPU by tests/gpu.
"""

import ctypes
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pth_testing.reference import make_int8_cache

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "sim"))

from ptxsim.golden import golden_tables  # noqa: E402
from ptxsim.loader_ref import (  # noqa: E402
    VARIANT_CELL,
    VARIANT_SEPARATE,
    expected_images,
    stage_bytes,
)

CSRC = REPO / "csrc" / "pth_attn"
SOURCES = (
    "loader_capi.cu",
    "loader_capi.h",
    "loader.cuh",
    "loader_test_kernels.h",
    "cache_geometry.cuh",
    "ptx.cuh",
    "smem_layout.cuh",
    "tile_traits.cuh",
)
DS = (64, 128, 256)
TILE = 64  # tokens per KV tile (tile_traits.cuh kLoadTileTokens)
MAP_FIELDS = 5  # warp, lane, stream, token, byte
MAP_K, MAP_V, MAP_SCALES, MAP_CELL = 0, 1, 2, 3
ERR_ARGS, ERR_WIDTH, ERR_NOT_CELL = -1, -2, -3
I64P = ctypes.POINTER(ctypes.c_int64)

needs_gxx = pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")


def _build(out_dir: Path, src_dir: Path = CSRC) -> ctypes.CDLL:
    lib = out_dir / "libpth_loader_host.so"
    cmd = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-shared"]
    cmd += ["-fPIC", "-x", "c++", f"-I{src_dir}", str(src_dir / "loader_capi.cu")]
    subprocess.run(cmd + ["-o", str(lib)], check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("needs g++")
    return _build(tmp_path_factory.mktemp("loader"))


@pytest.fixture(scope="module")
def golden():
    return json.loads(json.dumps(golden_tables()))


def _i64(vals):
    return (ctypes.c_int64 * len(vals))(*[int(x) for x in vals])


def _views(c):
    return c.k_cache, c.v_cache, c.k_scale, c.v_scale


def _widths(lib, variant, D, views):
    k, v, ks, vs = views
    out = (ctypes.c_int32 * 4)()
    rc = lib.pth_loader_widths(
        variant,
        D,
        *(ctypes.c_void_p(t.data_ptr()) for t in views),
        _i64(k.shape[:3]),
        *(_i64(t.stride()[:3]) for t in views),
        out,
    )
    return rc, list(out)


def _emulate(
    lib,
    variant,
    D,
    views,
    block_table,
    req,
    kvh,
    tb,
    te,
    tok_end,
    override=None,
    tok_begin=0,
):
    """Runs pth_loader_emulate; returns (rc, uint8 [te - tb, stage] images)."""
    k = views[0]
    nb = stage_bytes(variant, D)
    out = np.zeros((te - tb, nb), dtype=np.uint8)
    ov = (ctypes.c_int32 * 4)(*override) if override is not None else None
    bt = block_table.contiguous()
    rc = lib.pth_loader_emulate(
        variant,
        D,
        *(ctypes.c_void_p(t.data_ptr()) for t in views),
        _i64(k.shape[:3]),
        *(_i64(t.stride()[:3]) for t in views),
        ov,
        ctypes.cast(bt.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(bt.stride(0)),
        ctypes.c_int32(k.shape[1]),
        req,
        kvh,
        tb,
        te,
        tok_begin,
        tok_end,
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(out.size),
    )
    return rc, out


def _lane_map(lib, which, D, width):
    cap = 4 * 32 * 200  # >= 4 warps x 65 instructions x 32 lanes
    buf = (ctypes.c_int32 * (cap * MAP_FIELDS))()
    n = lib.pth_loader_lane_map(which, D, width, buf, ctypes.c_int64(cap))
    assert n >= 0, n
    return np.frombuffer(buf, dtype=np.int32)[: n * MAP_FIELDS].reshape(n, MAP_FIELDS)


@needs_gxx
@pytest.mark.parametrize("D", DS)
def test_lane_maps_equal_simulator_golden(lib, golden, D):
    """For every stream and width, the loader's lane -> (token, byte) map,
    in instruction order, is the simulator's golden loader map.

    Why it matters: the simulator proved this map conflict-free on the
    swizzled stage and complete (every byte once; tests/sim
    test_swizzle_conflict_free,
    test_loader_fills_stage_exactly_once_with_swizzled_image...). The
    header's own index functions produce it here, so those proofs are about
    the code.
    """
    gold = golden["per_d"][str(D)]["loader"]
    for which, name in ((MAP_K, "K"), (MAP_V, "V")):
        for w in (4, 8, 16):
            got = _lane_map(lib, which, D, w)
            assert np.all(got[:, 2] == which)
            assert got[:, [0, 1, 3, 4]].tolist() == gold[f"{name}_w{w}"], (name, w)
    assert _lane_map(lib, MAP_SCALES, D, 4).tolist() == gold["scales_paired_w4"]
    for w in (8, 4):
        got = _lane_map(lib, MAP_CELL, D, w)
        assert got[:, [0, 1, 3, 4]].tolist() == gold[f"cell_w{w}"], w


@needs_gxx
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_host_picked_widths(lib, layout):
    """Today's inline cell: K 8, V 4, scales 4, whole cell 8. #56164-style
    decoupled caches: K/V 16, scales 4, and the whole-cell variant refuses
    them (they are not one inline cell).

    Why it matters: the widths decide which LDGSTS forms the loader issues;
    the cell check keeps the challenger from reading a layout it
    does not understand.
    """
    c = make_int8_cache([40, 17], Hkv=2, D=256, block_size=16, layout=layout, seed=1)
    assert _widths(lib, VARIANT_SEPARATE, 256, _views(c)) == (0, [8, 4, 4, 4])
    assert _widths(lib, VARIANT_CELL, 256, _views(c)) == (0, [8, 0, 0, 0])
    dec = tuple(t.contiguous() for t in _views(c))
    assert _widths(lib, VARIANT_SEPARATE, 256, dec) == (0, [16, 16, 4, 4])
    assert _widths(lib, VARIANT_CELL, 256, dec)[0] == ERR_NOT_CELL


def _interesting_tiles(block_size, seq_len):
    """Tile ranges [t, t + 2) around the first page boundaries, the first
    tile and the last (partial) tiles."""
    last = (seq_len - 1) // TILE
    starts = {0, last - 1}
    for m in (1, 2):
        b = m * block_size
        if b < seq_len:
            starts |= {max(0, b // TILE - 1), b // TILE}
    return sorted({min(max(s, 0), max(last - 1, 0)) for s in starts})


def _check(
    lib, c, variant, D, req, kvh, tok_end, override=None, tiles=None, tok_begin=0
):
    views = _views(c)
    seq_len = int(c.seq_lens[req])
    for tb in tiles or _interesting_tiles(c.block_size, seq_len):
        te = tb + 2  # tiles past tok_end are legal: all zero-filled
        rc, got = _emulate(
            lib,
            variant,
            D,
            views,
            c.block_table,
            req,
            kvh,
            tb,
            te,
            tok_end,
            override,
            tok_begin,
        )
        assert rc == 0, rc
        want = expected_images(
            variant, D, *views, c.block_table, req, kvh, tb, te, tok_end, tok_begin
        )
        if not np.array_equal(got, want):
            i, b = np.argwhere(got != want)[0]
            raise AssertionError(
                f"variant {variant} D {D} tiles [{tb},{te}) tile {tb + i} byte {b}: "
                f"got {got[i, b]} want {want[i, b]}"
            )


@needs_gxx
@pytest.mark.parametrize("variant", [VARIANT_SEPARATE, VARIANT_CELL])
@pytest.mark.parametrize("block_size", [16, 64, 528, 816, 1056])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("D", DS)
def test_emulated_stage_images_match_reference(lib, variant, block_size, layout, D):
    """The emulated loader fills every stage exactly as the reference image
    says: paged NHD/HND caches with padded pages (72 B) and shuffled block
    ids, every block size of the test matrix (816: a tile straddles a page
    every 12.75 tiles), tiles around the first two page boundaries, the KV
    end mid-tile (zero fill), the second request and the second KV head.

    Why it matters: this is the loader's whole job; a wrong page, slot, head,
    width, swizzle or zero fill shows up as a wrong byte.
    """
    L = 2 * block_size + 37
    c = make_int8_cache(
        [5, L],
        Hkv=2,
        D=D,
        block_size=block_size,
        layout=layout,
        pad_page_bytes=72,
        seed=D + block_size,
    )
    _check(lib, c, variant, D, req=1, kvh=1, tok_end=L)
    _check(lib, c, variant, D, req=1, kvh=0, tok_end=L - 29)  # causal range end


@needs_gxx
@pytest.mark.parametrize("D", DS)
def test_every_width_on_decoupled_caches(lib, D):
    """On #56164-style decoupled caches (picked K/V width 16) the overrides
    16, 8 and 4 all give the reference image; an override above the picked
    width is refused.

    Why it matters: covers all three runtime width branches of the K and V
    streams with data (the width switch is warp-uniform; each branch is its
    own code).
    """
    c = make_int8_cache([3, 300], Hkv=2, D=D, block_size=64, seed=11)
    dec = tuple(t.contiguous() for t in _views(c))

    class Dec:
        k_cache, v_cache, k_scale, v_scale = dec
        block_table, seq_lens, block_size = c.block_table, c.seq_lens, 64

    for w in (16, 8, 4):
        _check(
            lib,
            Dec,
            VARIANT_SEPARATE,
            D,
            1,
            1,
            300,
            override=[w, w, 4, 4],
            tiles=[0, 3],
        )
    rc, _ = _emulate(
        lib,
        VARIANT_SEPARATE,
        256,
        _views(c),
        c.block_table,
        1,
        0,
        0,
        1,
        300,
        override=[16, 4, 4, 4],
    )
    assert rc == ERR_WIDTH


@needs_gxx
def test_storage_offset_and_high_block_ids(lib):
    """A 4-B storage offset (K width 4, V width 8, cell width 4) and block
    ids whose byte offsets exceed 2^31 (int64 addressing).

    Why it matters: the alignment moves between streams with the base
    pointer, and 32-bit offset arithmetic would wrap on large pools.
    """
    c = make_int8_cache(
        [90], Hkv=1, D=128, block_size=16, storage_offset_bytes=4, seed=5
    )
    assert _widths(lib, VARIANT_SEPARATE, 128, _views(c))[1] == [4, 8, 4, 4]
    assert _widths(lib, VARIANT_CELL, 128, _views(c))[1] == [4, 0, 0, 0]
    for variant in (VARIANT_SEPARATE, VARIANT_CELL):
        _check(lib, c, variant, 128, 0, 0, 90, tiles=[0])
    h = make_int8_cache(
        [100, 37], Hkv=1, D=64, block_size=16, high_block_ids=True, seed=3
    )
    assert h.max_block_id * h.page_bytes > 2**31
    for variant in (VARIANT_SEPARATE, VARIANT_CELL):
        _check(lib, h, variant, 64, 0, 0, 100, tiles=[0])


@needs_gxx
@pytest.mark.parametrize("block_size", [16, 64, 816, 864])
@pytest.mark.parametrize("D", (64, 128))
def test_lower_bound_zero_fill(lib, block_size, D):
    """The windowed kernels' loader: tokens below tok_begin are
    zero-filled like those past tok_end, and their pages are never read.

    Range begins mid-warp, at a warp edge and at a tile edge, in the first
    tile of the range. Every block-table entry whose page lies wholly below
    tok_begin is poisoned with an id far outside the pool (vLLM maps evicted
    window pages to the null block): a read through it would either fail the
    image or crash. Also: the whole-cell variant and tok_begin > tok_end are
    refused.

    Why it matters: evicted window positions hold whatever the null block
    holds; a NaN scale there must never reach the PV product.
    """
    L = 3 * block_size + 150
    c = make_int8_cache([L], Hkv=2, D=D, block_size=block_size, seed=7 + D)
    for tok_begin in (70, 80, 128, L - 100):
        tile = tok_begin // TILE
        bt = c.block_table.clone()
        bt[0, : tok_begin // block_size] = 2**30  # never a valid page id
        poisoned = dataclasses.replace(c, block_table=bt)
        _check(
            lib,
            poisoned,
            VARIANT_SEPARATE,
            D,
            0,
            1,
            L,
            tiles=[tile],
            tok_begin=tok_begin,
        )
    views = _views(c)
    rc, _ = _emulate(
        lib, VARIANT_CELL, D, views, c.block_table, 0, 0, 1, 2, L, None, 70
    )
    assert rc == ERR_ARGS
    rc, _ = _emulate(
        lib, VARIANT_SEPARATE, D, views, c.block_table, 0, 0, 1, 2, 60, None, 70
    )
    assert rc == ERR_ARGS


MUTATIONS = {
    # id: (file, [(old, new)], what must fail: "image" or "lane_map")
    "k_swizzle": (
        "smem_layout.cuh",
        [("kKSwizzleMask = 7;", "kKSwizzleMask = 3;")],
        "image",
    ),
    "v_region": ("loader.cuh", [("St::kV + v_swz", "St::kV + 16 + v_swz")], "image"),
    "no_zero_fill": (
        "loader.cuh",
        [("ok ? static_cast<uint32_t>(kW) : 0u", "static_cast<uint32_t>(kW)")],
        "image",
    ),
    "scale_same_warp": (
        "loader.cuh",
        [
            (
                "lane < kLoaderTokensPerWarp ? warp : (warp ^ 1)",
                "lane < kLoaderTokensPerWarp ? warp : warp",
            )
        ],
        "lane_map",
    ),
    "no_lower_fill": (
        "loader.cuh",
        [("(!kLower || c.token >= skip)", "(!kLower || c.token + skip >= skip)")],
        "lower_image",
    ),
}


@needs_gxx
@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutated_loader_is_caught(tmp_path, golden, name):
    """Negative controls on temp copies of the sources: a wrong K swizzle, a V
    region shifted by 16 B, the zero fill removed, and v_scale paired with the
    warp's own tokens each fail the check that should catch them (the first
    three the image comparison, the last only the lane map: it moves the
    same bytes with a bank conflict, so the image stays right). The windowed
    loader's zero fill below tok_begin removed fails the image
    with a lower bound and leaves the causal image right.

    Why it matters: shows both comparisons above can fail.
    """
    src = tmp_path / "src"
    src.mkdir()
    for f in SOURCES:
        shutil.copy(CSRC / f, src / f)
    file, edits, kind = MUTATIONS[name]
    text = (src / file).read_text()
    for old, new in edits:
        assert old in text, f"mutation anchor missing in {file}: {old}"
        text = text.replace(old, new)
    (src / file).write_text(text)
    m = _build(tmp_path, src)
    D = 64
    c = make_int8_cache([5, 150], Hkv=2, D=D, block_size=16, seed=21)
    if kind == "lane_map":
        got = _lane_map(m, MAP_SCALES, D, 4).tolist()
        assert got != golden["per_d"][str(D)]["loader"]["scales_paired_w4"]
        _check(m, c, VARIANT_SEPARATE, D, 1, 1, 150, tiles=[1])  # image still right
        return
    if kind == "lower_image":
        _check(m, c, VARIANT_SEPARATE, D, 1, 1, 150, tiles=[1])  # causal path fine
        with pytest.raises(AssertionError):
            _check(m, c, VARIANT_SEPARATE, D, 1, 1, 150, tiles=[1], tok_begin=80)
        return
    with pytest.raises(AssertionError):
        _check(m, c, VARIANT_SEPARATE, D, 1, 1, 150, tiles=[1])


# ---------------------------------------------------------------------------
# Device build: ptxas gate and SASS forms (needs nvcc, no GPU)
# ---------------------------------------------------------------------------

NVCC = Path("/usr/local/cuda/bin/nvcc")
needs_nvcc = pytest.mark.skipif(not NVCC.exists(), reason="needs nvcc")
KERNEL_SRC = CSRC / "loader_test_kernels.cu"


def _tool(args, out_dir):
    path = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": path},
    )


@pytest.fixture(scope="module")
def sass(tmp_path_factory):
    out = tmp_path_factory.mktemp("loader_sass")
    gate = _tool(
        [
            "tools/ptxas_gate.py",
            str(KERNEL_SRC),
            "--min-kernels",
            "6",
            "--out-dir",
            str(out / "ptxas"),
            "--json",
            str(out / "ptxas.json"),
        ],
        out,
    )
    chk = _tool(
        [
            "tools/sass_check.py",
            "--cu",
            str(KERNEL_SRC),
            "--forbid",
            "LDL,STL,LDG.E.U8,LDG.E.S8",
            "--require",
            "LDGSTS",
            "--out-dir",
            str(out / "sass"),
            "--json",
            str(out / "sass.json"),
        ],
        out,
    )
    sys.path.insert(0, str(REPO / "tools"))
    import sass_check

    cuobjdump = sass_check.find_cuobjdump()
    listings = {}
    for arch in ("80", "86", "89"):
        cub = out / "sass" / f"loader_test_kernels.sm_{arch}.cubin"
        if cub.exists():
            listings[arch] = sass_check.parse_sass(
                sass_check.disassemble(cuobjdump, cub)
            )
    return gate, chk, listings


@needs_nvcc
def test_loader_kernels_pass_ptxas_gate_and_sass_check(sass):
    """All six test kernels (D 64/128/256 x separate/cell) compile for
    sm_80/86/89 with 0 spills, no local memory and no byte loads, and use
    LDGSTS (cp.async)."""
    gate, chk, listings = sass
    assert gate.returncode == 0, gate.stdout + gate.stderr
    assert chk.returncode == 0, chk.stdout + chk.stderr
    assert set(listings) == {"80", "86", "89"}


def _ldgsts_forms(instrs):
    return {i.opcode for i in instrs if i.opcode.startswith("LDGSTS")}


@needs_nvcc
def test_ldgsts_forms_per_width(sass):
    """Each width lowers to its cp.async form on every arch: the separate
    kernels contain LDGSTS.E.BYPASS.128 (16 B, .cg), LDGSTS.E.64 (8 B) and
    LDGSTS.E (4 B: V today and the scales); the cell kernels only .64 and
    the 4-B form (16 B cannot divide a 2D + 8 cell).

    Why it matters: a width that silently lowered to narrower copies would
    multiply the loader's instruction count (the loader's LSU budget).
    """
    _, _, listings = sass
    for arch, kernels in listings.items():
        for name, instrs in kernels.items():
            forms = {f.replace(".ZFILL", "") for f in _ldgsts_forms(instrs)}
            if "Lb0E" in name:  # loader_test_kernel<D, false>: separate streams
                assert {"LDGSTS.E.BYPASS.128", "LDGSTS.E.64", "LDGSTS.E"} <= forms, (
                    arch,
                    name,
                    forms,
                )
            else:
                assert "LDGSTS.E.64" in forms and "LDGSTS.E" in forms, (
                    arch,
                    name,
                    forms,
                )
                assert "LDGSTS.E.BYPASS.128" not in forms, (arch, name, forms)
