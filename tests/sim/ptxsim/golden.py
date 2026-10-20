# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Golden tables for checking the CUDA headers against the simulator.

`csrc/pth_attn/dequant.cuh` and `smem_layout.cuh` keep their index and bit
math `__host__ __device__`, so `tests/cpp/test_primitives.cu` compiled with
g++ evaluates it for every input and prints the same keys as JSON;
`tests/cpu/test_primitives_golden.py` compares the two key by key. The
loader lane maps (`per_d.<D>.loader`) are compared with `loader.cuh` by
`tests/cpu/test_loader_host.py`. The device-only instructions (ldmatrix,
mma, cp.async) are compared on the GPU.

Every table is computed by the same simulator functions the sim tests prove
(`paths`, `layouts`, `dequant`, `prescale`), never re-derived here.

Export: `python tests/sim/export_golden.py OUT.json`. Everything is
deterministic (no random inputs), so the file is reproducible bit for bit.

Format version 2 added: cvt4 on whole words, the V fragment words, the lane
address tables of the three fragment loads, the Q element offsets and the Q
prescale tables. Version 3: the loader maps follow the cp.async loader (one
chunk enumeration for every width, the paired ks/vs instruction, the
whole-cell challenger).
"""

from __future__ import annotations

import numpy as np

from .dequant import (
    LUT_XOR_AB,
    MAGIC_BIAS_HALF2,
    MAGIC_EXP_BYTES,
    MAGIC_SIGN_FLIP,
    SEL_BYTES_01,
    SEL_BYTES_23,
    V_SEL_D01,
    V_SEL_D23,
    cvt4_s8_to_f16x2,
)
from .fp import (
    all_bf16_bits,
    all_f16_bits,
    bf16_bits_to_value,
    f16_bits_to_value,
    f16_value_to_bits,
    unpack_half2,
)
from .layouts import (
    N_TOKENS,
    PI_PERMUTATION,
    cell_bytes,
    cell_loader_instructions,
    k_offset,
    loader_instructions,
    pi_inverse,
    q_offset,
    scale_pair_instructions,
    stage_offsets,
    v_ntile_col_to_d,
    v_offset,
)
from .paths import (
    epilogue_columns,
    k_ldsm_tokens,
    q_ldsm_rows_k,
    v_lds32_token_byte,
    v_words_to_b_fragments,
)
from .patterns import Q_ROWS_MAX, V_TOKEN_OFFSETS
from .prescale import prescale_exponent, prescale_to_f16

DS = (64, 128, 256)
WIDTHS = (4, 8, 16)
FORMAT_VERSION = 3
SCALE_STREAM_ID = {"ks": 2, "vs": 3}  # cache_geometry.cuh StreamId
# Fixed prescale exponents swept besides each value's own: the clamp, both
# exactness edges' neighbourhoods, 0/1, and the largest e a finite bf16 row
# can get (floor(log2 max bf16) - 14 = 127 - 14).
PRESCALE_FIXED_E = (-112, -60, -17, 0, 1, 20, 60, 113)
ROWS_PER_WARP = 16  # one m16 MMA slice


def _words_from_bytes(b0, b1, b2, b3) -> np.ndarray:
    u = [
        np.asarray(b, dtype=np.int64).astype(np.uint8).astype(np.uint32)
        for b in (b0, b1, b2, b3)
    ]
    return u[0] | (u[1] << 8) | (u[2] << 16) | (u[3] << 24)


def _cvt4_words() -> np.ndarray:
    """Every int8 value in every byte position (replicated words), then every
    value once in a word of 4 consecutive values (cross-talk check)."""
    x = np.arange(-128, 128, dtype=np.int64)
    rep = _words_from_bytes(x, x, x, x)
    seq = _words_from_bytes(x[0::4], x[1::4], x[2::4], x[3::4])
    return np.concatenate([rep, seq])


def _cvt4_table() -> dict:
    x = np.arange(-128, 128, dtype=np.int64)
    w = _words_from_bytes(x, x, x, x)
    lo, _ = cvt4_s8_to_f16x2(w)
    l0, _ = unpack_half2(lo)
    words = _cvt4_words()
    wlo, whi = cvt4_s8_to_f16x2(words)
    return {
        "int8": x.tolist(),
        "f16_bits": l0.astype(int).tolist(),
        "words": words.astype(int).tolist(),
        "lo": wlo.astype(int).tolist(),
        "hi": whi.astype(int).tolist(),
    }


def _v_fragment_table() -> dict:
    """Quadruples of LDS.32 words whose 16 bytes are distinct, so every output
    byte position identifies its source; 16 quadruples cover all 256 values."""
    x = np.arange(-128, 128, dtype=np.int64).reshape(16, 4, 4)  # [quad][w][byte]
    words = _words_from_bytes(x[..., 0], x[..., 1], x[..., 2], x[..., 3])  # [16, 4]
    frags = v_words_to_b_fragments([words[:, i] for i in range(4)])  # 4 x [16, 2]
    regs = np.stack(frags, axis=1)  # [16, 4 n-tiles, 2 regs]
    return {"words": words.astype(int).tolist(), "regs": regs.astype(int).tolist()}


def _hex16(bits: np.ndarray) -> str:
    return "".join(f"{int(b):04x}" for b in np.asarray(bits).ravel())


def _prescale_table() -> dict:
    """fp16 bits of the prescaled value for every finite bf16 / fp16 input x,
    with e from |x| as the row max ("own", clamped) and with fixed e."""
    out: dict = {"fixed_e": list(PRESCALE_FIXED_E)}
    for name, x in (
        ("bf16", bf16_bits_to_value(all_bf16_bits())),
        ("fp16", f16_bits_to_value(all_f16_bits())),
    ):
        e_own = prescale_exponent(np.abs(x), clamp=True)
        tab = {
            "e_own": e_own.astype(int).tolist(),
            "own": _hex16(f16_value_to_bits(prescale_to_f16(x, e_own))),
        }
        for e in PRESCALE_FIXED_E:
            y = prescale_to_f16(x, np.full(x.shape, e))
            tab[str(e)] = _hex16(f16_value_to_bits(y))
        out[name] = tab
    return out


def _per_d(D: int) -> dict:
    t, c = np.meshgrid(np.arange(N_TOKENS), np.arange(D // 16), indexing="ij")
    r, qc = np.meshgrid(np.arange(Q_ROWS_MAX), np.arange(2 * D // 16), indexing="ij")
    rr, dd = np.meshgrid(np.arange(Q_ROWS_MAX), np.arange(D), indexing="ij")
    inv = np.asarray(pi_inverse(PI_PERMUTATION))
    # Loader lane maps in instruction order (warp, then instruction, then
    # lane), one entry per active lane: [warp, lane, token, byte]; the paired
    # scale instruction adds the stream id (2 = k_scale, 3 = v_scale, as
    # cache_geometry.cuh StreamId): [warp, lane, stream, token, byte].
    loader = {}
    for stream in ("K", "V"):
        for w in WIDTHS:
            loader[f"{stream}_w{w}"] = [
                [warp, lane, tok, byte]
                for ins in loader_instructions(D, w)
                for (warp, lane, tok, byte) in ins
            ]
    loader["scales_paired_w4"] = [
        [warp, lane, SCALE_STREAM_ID[s], tok, byte]
        for ins in scale_pair_instructions()
        for (warp, lane, s, tok, byte) in ins
    ]
    for w in WIDTHS:
        if cell_bytes(D) % w == 0:
            loader[f"cell_w{w}"] = [
                [warp, lane, tok, byte]
                for ins in cell_loader_instructions(D, w)
                for (warp, lane, tok, byte) in ins
            ]
    k_ldsm = [
        [k_offset(k_ldsm_tokens(tok0), 16 * s, D).tolist() for s in range(D // 16)]
        for tok0 in range(0, N_TOKENS, 32)
    ]
    q_ldsm = [
        [q_offset(*q_ldsm_rows_k(row0, s), D).tolist() for s in range(D // 16)]
        for row0 in range(0, Q_ROWS_MAX, ROWS_PER_WARP)
    ]
    v_lds32 = [
        [
            [
                v_offset(*v_lds32_token_byte(16 * kt, a, i), D).tolist()
                for i in range(len(V_TOKEN_OFFSETS))
            ]
            for a in range(D // 32)
        ]
        for kt in range(N_TOKENS // 16)
    ]
    return {
        "stage_offsets": stage_offsets(D),
        # Physical 16-B chunk index (within the region) of logical chunk c of
        # token t; byte b of the chunk keeps its offset b.
        "k_chunk": (k_offset(t, 16 * c, D) // 16).tolist(),
        "v_chunk": (v_offset(t, 16 * c, D) // 16).tolist(),
        # Q smem (fp16, pitch 2D): physical chunk of logical chunk qc of row r.
        "q_chunk": (q_offset(r, 8 * qc, D) // 16).tolist(),
        # Byte offset in Q smem where the prologue stores Q[row][d] (pi + swizzle).
        "q_elem_offset": q_offset(rr, 16 * (dd // 16) + inv[dd % 16], D).tolist(),
        # Byte offsets (within the region) each lane gives the fragment loads:
        # K ldmatrix.x4 [tok0/32][s][lane], Q ldmatrix.x4 [row0/16][s][lane],
        # V LDS.32 [kt][a][i][lane].
        "k_ldsm": k_ldsm,
        "q_ldsm": q_ldsm,
        "v_lds32": v_lds32,
        # PV n-tile j, fragment column n -> physical d.
        "v_ntile_col_to_d": [
            [int(v_ntile_col_to_d(j, n)) for n in range(8)] for j in range(D // 8)
        ],
        "epilogue_columns": [
            [epilogue_columns(lane, a) for a in range(D // 32)] for lane in range(32)
        ],
        "loader": loader,
    }


def golden_tables() -> dict:
    lanes = np.arange(32)
    return {
        "format_version": FORMAT_VERSION,
        "pi": list(PI_PERMUTATION),
        "pi_inverse": list(pi_inverse(PI_PERMUTATION)),
        "magic": {
            "sign_flip": MAGIC_SIGN_FLIP,
            "exp_bytes": MAGIC_EXP_BYTES,
            "bias_half2": MAGIC_BIAS_HALF2,
            "sel_bytes_01": SEL_BYTES_01,
            "sel_bytes_23": SEL_BYTES_23,
            "lop3_lut_xor": LUT_XOR_AB,
            "cvt4": _cvt4_table(),
        },
        "v_selectors": {"d01": V_SEL_D01, "d23": V_SEL_D23},
        "v_b_fragments": _v_fragment_table(),
        "prescale": _prescale_table(),
        # ldmatrix.x4 row address source per lane:
        # K: n-tile offset (tokens) and chunk offset for the 4-n-tile form.
        "k_ldmatrix_x4_lane": [[8 * (ln // 8) + ln % 8, 0] for ln in range(32)],
        # Q (A operand): row l % 16, 8-element chunk 2s + l // 16.
        "q_ldmatrix_x4_lane": [[int(ln % 16), int(ln // 16)] for ln in lanes],
        # V LDS.32: lane (g, q) reads token 2q + off, byte 32a + 4g.
        "v_lds32_lane": [
            [[2 * (ln & 3) + off for off in V_TOKEN_OFFSETS], 4 * (ln >> 2)]
            for ln in range(32)
        ],
        "per_d": {str(D): _per_d(D) for D in DS},
    }
