# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for the stage layout, swizzles, bank conflicts and the
cp.async loader's destination mapping.

Bank model: ptxsim.banks (32 banks x 4 B; phase split per access width is
assumed, see that module). "1-way" = conflict-free.
"""

import numpy as np
import pytest
from ptxsim.banks import worst
from ptxsim.isa import cp_async
from ptxsim.layouts import (
    N_TOKENS,
    cell_bytes,
    cell_loader_instructions,
    cell_pitch,
    k_offset,
    k_offset_row_chunks,
    loader_instructions,
    scale_pair_instructions,
    stage_offsets,
    stream_width,
    v_offset,
    v_offset_row_chunks,
)
from ptxsim.paths import build_stage
from ptxsim.patterns import (
    cp_async_writes,
    k_ldmatrix,
    q_ldmatrix,
    scale_lds64,
    scale_pair_same_token_writes,
    v_lds32,
)

DS = (64, 128, 256)
RNG_SEED = 20261013


def _is_permutation(offsets, size):
    offsets = np.asarray(offsets).ravel()
    return (
        offsets.min() == 0
        and offsets.max() == size - 1
        and len(np.unique(offsets)) == size
    )


@pytest.mark.parametrize("D", DS)
def test_swizzles_are_permutations_of_the_tile(D):
    """K and V swizzles map the N x D tile onto itself one-to-one, and keep
    every byte inside its own 128-B line and its 16-B chunk's internal order.

    Why it matters: a swizzle that is not a bijection overwrites data (two
    bytes to one address); one that leaves the 128-B line would break the
    cp.async destination analysis and the 16-B alignment ldmatrix needs.
    """
    t, b = np.meshgrid(np.arange(N_TOKENS), np.arange(D), indexing="ij")
    for fn in (k_offset, v_offset):
        off = fn(t, b, D)
        assert _is_permutation(off, N_TOKENS * D)
        assert np.array_equal(off // 128, (t * D + b) // 128)
        assert np.array_equal(off % 16, b % 16)


def test_chunk_swizzle_equals_line_form_for_d_ge_128():
    """For D >= 128 the documented chunk swizzle `c ^ (t&7)` / `c ^ 2((t>>1)&3)`
    on row chunks is identical to the 128-B-line form used here.

    Why it matters: the swizzle is stated on 128-B lines so that one
    formula also covers D = 64; this test shows it changes nothing for the
    production D = 256 (and 128).
    """
    for D in (128, 256):
        t, b = np.meshgrid(np.arange(N_TOKENS), np.arange(D), indexing="ij")
        assert np.array_equal(k_offset(t, b, D), k_offset_row_chunks(t, b, D))
        assert np.array_equal(v_offset(t, b, D), v_offset_row_chunks(t, b, D))


def test_chunk_swizzle_breaks_at_d64():
    """Read on row chunks, the documented chunk swizzles are not
    permutations at D = 64 (a 64-B row has 4 chunks; XOR keys up to 7 move
    chunks into the neighbouring token's row).

    Why it matters: D = 64 is a supported head size. The kernel
    must use the line form (`k_offset`/`v_offset`), which is a permutation
    for all three D (test above).
    """
    D = 64
    t, b = np.meshgrid(np.arange(N_TOKENS), np.arange(D), indexing="ij")
    assert not _is_permutation(k_offset_row_chunks(t, b, D), N_TOKENS * D)
    assert not _is_permutation(v_offset_row_chunks(t, b, D), N_TOKENS * D)


@pytest.mark.parametrize("D", DS)
def test_swizzle_conflict_free(D):
    """Every shared-memory access of the stage is 1-way with the swizzles.

    Covered: K ldmatrix (every n-tile x chunk), Q ldmatrix (every
    8-row group x chunk), V LDS.32 (every k16 step x 32-d group x token
    offset), ks/vs LDS.64, and the cp.async destination writes of K at 8 B,
    V at 4 B, the paired ks/vs instruction at 4 B (today's widths) plus K/V
    at 16 B (#56164) and the other legal widths.

    Why it matters: a 2-way conflict halves shared-memory bandwidth on that
    instruction; the K and V reads are in the inner loop.
    """
    pats = {
        "K ldmatrix": k_ldmatrix(D),
        "Q ldmatrix": q_ldmatrix(D),
        "V LDS.32": v_lds32(D),
        "ks LDS.64": scale_lds64(D, "ks"),
        "vs LDS.64": scale_lds64(D, "vs"),
    }
    for w in (4, 8, 16):
        pats[f"K cp.async {w}"] = cp_async_writes(D, "K", w)
        pats[f"V cp.async {w}"] = cp_async_writes(D, "V", w)
    pats["ks/vs paired cp.async 4"] = cp_async_writes(D, "scales", 4)
    for name, accesses in pats.items():
        rep = worst(accesses)
        assert rep.ways == 1.0, f"{name}: {rep}"


@pytest.mark.parametrize("D,k_ways,v_ways", [(64, 4, 4), (128, 8, 4), (256, 8, 4)])
def test_unswizzled_layout_conflicts(D, k_ways, v_ways):
    """Negative control: without the XOR the same reads conflict.

    Why it matters: shows the swizzle is doing the work and that the counter
    can see conflicts. D >= 128: the 8 rows of an ldmatrix matrix start at
    the same bank (row pitch is a multiple of 128 B) -> 8-way; the 4 tokens
    2q of one LDS.32 likewise -> 4-way. D = 64: pitch 64 B, so rows t and
    t+2 alias -> 4-way. Matches the documented 8 and 4 for D = 256.
    """
    assert worst(k_ldmatrix(D, swizzled=False)).worst_phase_ways == k_ways
    assert worst(v_lds32(D, swizzled=False)).worst_phase_ways == v_ways


@pytest.mark.parametrize("D", DS)
def test_phase_model_insensitive(D):
    """Treating each LSU instruction as one whole-warp phase does not change
    any verdict (conflict-free stays conflict-free, conflicting stays so).

    Why it matters: the half/quarter-warp phase split for 8/16-B accesses is
    an assumption; the conclusions must not hinge on it.
    """
    for sw in (True, False):
        for w in (4, 8, 16):
            for s in ("K", "V"):
                a = list(cp_async_writes(D, s, w, swizzled=sw))
                assert (worst(a).ways == 1.0) == (
                    worst(a, single_phase=True).ways == 1.0
                )
        for gen in (v_lds32(D, swizzled=sw), scale_lds64(D)):
            a = list(gen)
            assert (worst(a).ways == 1.0) == (worst(a, single_phase=True).ways == 1.0)


def _paged_cache(D, Hkv=2, block_size=16, num_blocks=12, seed=RNG_SEED):
    """NHD int8_per_token_head cache bytes: cell [K D | ks f32 | V D | vs f32].

    Layout and offsets follow vLLM's Triton cache spec (`customize_spec`):
    per (block, token, head) a cell of 2D+8 bytes; K at 0, ks at D, V at
    D+4, vs at 2D+4.
    """
    rng = np.random.default_rng(seed)
    cell = 2 * D + 8
    s_head, s_tok = cell, Hkv * cell
    s_block = block_size * s_tok
    gmem = rng.integers(0, 256, size=num_blocks * s_block, dtype=np.uint8)
    block_table = rng.permutation(num_blocks)
    return gmem, block_table, (s_block, s_tok, s_head), block_size


@pytest.mark.parametrize("D", DS)
def test_loader_fills_stage_exactly_once_with_swizzled_image(D):
    """Simulated cp.async of one 64-token tile from a paged cache writes every
    stage byte exactly once and produces the same image as the swizzle maps.

    Why it matters: this is the destination mapping of the loader at today's
    widths (K 8 B, V 4 B, scales 4 B, derived by `stream_width` from the real
    cell layout). The tile starts mid-page so it straddles page boundaries.
    """
    gmem, bt, (s_block, s_tok, s_head), bs = _paged_cache(D)
    kvh, pos0 = 1, 7  # second KV head, tile not page-aligned
    field = {"K": 0, "ks": D, "V": D + 4, "vs": 2 * D + 4}
    rows = {"K": D, "V": D, "ks": 4, "vs": 4}
    widths = {s: stream_width(field[s], s_block, s_tok, s_head, rows[s]) for s in field}
    assert widths == {"K": 8, "V": 4, "ks": 4, "vs": 4}

    def src(t, b, s):
        pos = pos0 + t
        page = int(bt[pos // bs])
        return page * s_block + (pos % bs) * s_tok + kvh * s_head + field[s] + b

    offs = stage_offsets(D)
    smem = np.zeros(offs["bytes"], dtype=np.uint8)
    written = np.zeros(offs["bytes"], dtype=np.int64)
    for s in ("K", "V"):
        for ins in loader_instructions(rows[s], widths[s]):
            fn = k_offset if s == "K" else v_offset
            dst = [offs[s] + int(fn(t, b, D)) for (_, _, t, b) in ins]
            srcs = [src(t, b, s) for (_, _, t, b) in ins]
            cp_async(smem, gmem, np.array(dst), np.array(srcs), widths[s], written)
    for ins in scale_pair_instructions():
        dst = [offs[s] + 4 * t + b for (_, _, s, t, b) in ins]
        srcs = [src(t, b, s) for (_, _, s, t, b) in ins]
        cp_async(smem, gmem, np.array(dst), np.array(srcs), 4, written)
    assert np.all(written == 1)

    def field_tile(s, nbytes):
        return np.stack(
            [gmem[src(t, 0, s) : src(t, 0, s) + nbytes] for t in range(N_TOKENS)]
        )

    k = field_tile("K", D).view(np.int8)
    v = field_tile("V", D).view(np.int8)
    ks = field_tile("ks", 4).copy().view(np.float32).ravel()
    vs = field_tile("vs", 4).copy().view(np.float32).ravel()
    assert np.array_equal(smem, build_stage(k, v, ks, vs))


def test_stream_widths_for_decoupled_scales_layout():
    """With #56164's decoupled layout (512-B K|V slots, separate f32 scale
    caches) the same rule gives K/V width 16 and the loader is unchanged.

    Why it matters: one loader serves both layouts; only the
    host-chosen widths differ, and the 16-B destination writes are 1-way
    (covered in test_swizzle_conflict_free).
    """
    D, Hkv, bs = 256, 2, 16
    slot = 2 * D
    s_tok = Hkv * slot
    assert stream_width(0, bs * s_tok, s_tok, slot, D) == 16  # K
    assert stream_width(D, bs * s_tok, s_tok, slot, D) == 16  # V at +256
    assert stream_width(0, bs * Hkv * 4, Hkv * 4, 4, 4) == 4  # scales


@pytest.mark.parametrize("D", DS)
def test_padded_pitch_alternative_also_conflict_free(D):
    """Rows padded to D + 16 bytes are also 1-way for the K and V reads, so
    padding is not used because of its cost (2 x 64 x 16 B per stage; at D = 256,
    WM = 4 that exceeds the 101,376 B limit), not because of conflicts.
    """
    pitch = D + 16

    def padded(t, b):
        return t * pitch + b

    assert worst(k_ldmatrix(D, offset_fn=padded)).ways == 1.0
    assert worst(v_lds32(D, offset_fn=padded)).ways == 1.0


@pytest.mark.parametrize("D", DS)
def test_scale_pairing_with_partner_warp_is_conflict_free(D):
    """The paired ks/vs cp.async is 1-way only because lanes 16..31 take vs
    of warp w ^ 1; pairing ks and vs of the same tokens is 2-way.

    Why it matters: ks and vs regions are 256 B apart, a
    multiple of the 128-B bank sweep, so ks[t] and vs[t] share a bank; the
    partner warp's tokens sit in the other half of the banks. One
    instruction then does what two would, at no conflict cost.
    """
    assert worst(cp_async_writes(D, "scales", 4)).ways == 1.0
    assert worst(scale_pair_same_token_writes(D)).worst_phase_ways == 2


@pytest.mark.parametrize("D", DS)
def test_scale_pairing_covers_every_scale_once(D):
    """Across the 4 warps the paired instruction writes each ks and vs slot
    exactly once."""
    seen = {(s, t) for ins in scale_pair_instructions() for (_, _, s, t, _) in ins}
    assert seen == {(s, t) for s in ("ks", "vs") for t in range(N_TOKENS)}
    assert sum(len(ins) for ins in scale_pair_instructions()) == 2 * N_TOKENS


@pytest.mark.parametrize("D,width", [(64, 8), (128, 8), (256, 8), (256, 4)])
def test_cell_loader_writes_every_cell_byte_once(D, width):
    """Whole-cell challenger: the chunk stream writes bytes [0, 2D+8) of each
    of the 64 cells exactly once at pitch 2D+16, and nothing else.

    Its destination writes are reported, not gated (they can
    straddle two cells inside one 128-B pass).
    """
    pitch, cb = cell_pitch(D), cell_bytes(D)
    written = np.zeros(N_TOKENS * pitch, dtype=np.int64)
    for ins in cell_loader_instructions(D, width):
        for _, _, t, b in ins:
            written[t * pitch + b : t * pitch + b + width] += 1
    w2 = written.reshape(N_TOKENS, pitch)
    assert np.all(w2[:, :cb] == 1) and np.all(w2[:, cb:] == 0)
