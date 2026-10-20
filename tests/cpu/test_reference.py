# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teaching tests for ``pth_testing.reference`` (cache builder and fp64 reference).

Every kernel test compares against ``ref_attention`` on a cache from
``make_int8_cache``. If either is wrong, every later verdict is wrong, so these
tests pin down the cache bytes, the views, the writer semantics and the
reference itself, each with a negative control showing the check can fail.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from pth_testing.reference import (
    POISON_SCALE,
    SCALE_FLOOR,
    DenseKV,
    ast_host_sync_scan,
    forbid_host_sync,
    make_int8_cache,
    make_queries,
    naive_dense_attention,
    quantize_rows,
    ref_attention,
    ref_attention_dense,
    visible_bounds,
    visible_mask,
)


# --------------------------------------------------------------------------
# Writer semantics
# --------------------------------------------------------------------------
def test_round_half_away_from_zero():
    """The writer rounds ties away from zero (+0.5 / -0.5 then truncate).

    Why: a mirror that used round-half-to-even (``torch.round``) would differ
    from vLLM on exact ties, and every dequantised value of such a row would be
    off by one scale step. Rows with absmax exactly 127 have scale 1.0, so
    ``k + 0.5`` values are exact ties. Negative control: ``torch.round`` gives a
    different answer on the same row.
    """
    x = torch.tensor([[127.0, 2.5, -2.5, 0.5, -0.5, 3.5, -126.5]])
    q, s = quantize_rows(x)
    assert s.item() == 1.0
    assert q.tolist() == [[127, 3, -3, 1, -1, 4, -127]]
    rne = torch.round(x / s).to(torch.int8)
    assert not torch.equal(rne, q), "negative control: RNE must differ on ties"


def test_scale_floor_and_extremes():
    """Scales never go below 1e-6 and full-scale rows store exactly +/-127.

    Why: the kernel's V-scale exponent and Q·K scaling assume
    ``scale >= 1e-6`` (it bounds 2^-c to 2^20), and a writer bug that let -128
    appear would break the symmetric dequantisation the reference assumes. The
    floor rows (|x| <= 1.2e-4) and the all-zero rows must give scale exactly
    fp32(1e-6); the extreme rows must give only +/-127, never -128.
    """
    c = make_int8_cache(
        [300, 200], Hkv=2, D=64, block_size=16, scale_mode=["floor", "extreme"], seed=4
    )
    fl, ex = c.dense(0), c.dense(1)
    assert torch.all(fl.k_scale == torch.tensor(SCALE_FLOOR, dtype=torch.float32))
    assert torch.all(fl.v_scale == torch.tensor(SCALE_FLOOR, dtype=torch.float32))
    assert torch.all(ex.k_int.abs() == 127) and torch.all(ex.v_int.abs() == 127)
    # No stored int anywhere is -128 (|x / scale| <= 127 (1 + 2^-23) < 127.5).
    mixed = make_int8_cache(
        [2000], Hkv=2, D=128, block_size=64, scale_mode="mixed", seed=5
    ).dense(0)
    assert int(mixed.k_int.min()) >= -127 and int(mixed.v_int.min()) >= -127
    # Negative control: removing the floor changes the floor rows' scales.
    nofloor = c.k_src[0].float().abs().amax(-1) / 127.0
    assert not torch.equal(nofloor, fl.k_scale)


# --------------------------------------------------------------------------
# Cache bytes and views
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_v_offset_is_d_plus_4(layout):
    """In a (token, head) cell K is at byte 0, ks at D, V at D+4, vs at 2D+4.

    Why: the cell is ``[K: D | ks: f32 | V: D | vs: f32]`` (vLLM's
    ``customize_spec`` adds one f32 per half). A kernel that put V at byte D
    (forgetting the K scale) would read the scale's 4 bytes as V's first four
    values and shift every V element. Checked on raw pointers for both
    layouts. Negative control: a view at byte D dequantises to the wrong V.
    """
    D = 64
    c = make_int8_cache(
        [40], Hkv=2, D=D, block_size=16, layout=layout, pad_page_bytes=8, seed=1
    )
    kp = c.k_cache.data_ptr()
    assert c.k_scale.data_ptr() - kp == D
    assert c.v_cache.data_ptr() - kp == D + 4
    assert c.v_scale.data_ptr() - kp == 2 * D + 4
    assert c.cell_bytes == 2 * (D + 4)
    blk = int(c.block_table[0, 0])
    good = c.v_cache[blk, :8, 0].double() * c.v_scale[blk, :8, 0].double()[:, None]
    want = quantize_rows(c.v_src[0][:8, 0])
    assert torch.equal(c.v_cache[blk, :8, 0], want[0])
    wrong = torch.as_strided(
        c.kv_packed,
        (8, D),
        (c.kv_packed.stride(2), 1),
        c.kv_packed[blk, 0, 0].storage_offset() + D,
    )
    wrong_deq = wrong.double() * c.v_scale[blk, :8, 0].double()[:, None]
    assert not torch.equal(wrong_deq, good), (
        "negative control: V at byte D must be wrong"
    )


@pytest.mark.parametrize("block_size", [16, 64, 528, 816])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("pad", [0, 72])
def test_paged_roundtrip(block_size, layout, pad):
    """Reading requests back through block_table + views gives the writer's data.

    Why: this is the address math the kernel must reproduce: token
    t of request r is at slot ``t % BS`` of block ``block_table[r, t // BS]``,
    with byte strides taken from the views (padded pages change only the block
    stride; HND/NHD change slot and head strides). Block sizes 528 and 816 are
    not multiples of the 64-token tile, so pages straddle tiles. Requests
    include an empty one and lengths that end mid-block.
    """
    lens = [1, 0, block_size + 3, 2 * block_size - 1]
    c = make_int8_cache(
        lens,
        Hkv=2,
        D=64,
        block_size=block_size,
        layout=layout,
        pad_page_bytes=pad,
        storage_offset_bytes=4,
        seed=7,
    )
    assert c.page_bytes == 2 * block_size * c.cell_bytes + pad
    for r, s in enumerate(lens):
        d = c.dense(r)
        kq, ks = quantize_rows(c.k_src[r])
        vq, vs = quantize_rows(c.v_src[r])
        assert d.seq_len == s
        assert torch.equal(d.k_int, kq) and torch.equal(d.k_scale, ks)
        assert torch.equal(d.v_int, vq) and torch.equal(d.v_scale, vs)
    # Unused slots of a used block hold poison (a kernel reading past the end
    # of a request gets scale 3e4, not plausible data).
    r = 3
    last = int(c.block_table[r, (lens[r] - 1) // block_size])
    tail = lens[r] % block_size
    if tail:
        assert torch.all(c.k_scale[last, tail:] == POISON_SCALE)


def test_high_block_ids_need_int64():
    """Block ids above 2^31 / stride are read correctly; int32 offsets would not be.

    Why: production pools have block offsets past 2 GiB (vLLM #49716 was an
    int32 overflow of exactly this kind). The cache places every block above
    2^31 / stride; reading through the views (int64 inside torch) must
    round-trip. Negative control: the same byte offset computed in int32
    wraps negative.
    """
    c = make_int8_cache(
        [100, 37], Hkv=1, D=64, block_size=16, high_block_ids=True, seed=3
    )
    off = c.max_block_id * c.page_bytes
    assert off > 2**31
    for r in range(2):
        d = c.dense(r)
        assert torch.equal(d.k_int, quantize_rows(c.k_src[r])[0])
        assert torch.equal(d.v_scale, quantize_rows(c.v_src[r])[1])
    wrapped = torch.tensor(c.max_block_id, dtype=torch.int32) * torch.tensor(
        c.page_bytes, dtype=torch.int32
    )
    assert int(wrapped) != off, "negative control: int32 offset must overflow"


# --------------------------------------------------------------------------
# fp64 reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("q_lens", "lens", "Hkv", "G"),
    [([1, 3, 5], [1, 17, 40], 2, 6), ([4, 2], [4, 64], 1, 1), ([2], [33], 4, 3)],
)
def test_reference_matches_naive_dense(q_lens, lens, Hkv, G):
    """``ref_attention`` (batched, chunked, masked with -inf) equals a per-row loop.

    Why: the reference is the oracle for every kernel test, so it is itself
    checked against an implementation written the dumbest possible way
    (``naive_dense_attention``: slices the visible keys, ``torch.softmax`` per
    (token, head)). Covers GQA (G = 1, 3, 6), q_len = L (prefill from empty),
    q_len = 1 and several requests in one batch.
    """
    D = 32
    c = make_int8_cache(lens, Hkv=Hkv, D=D, block_size=16, seed=11)
    q, qsl = make_queries(q_lens, Hkv * G, D, seed=12)
    kv = c.dense_all()
    ref = ref_attention_dense(q, kv, qsl, 0.3)
    for r, x in enumerate(kv):
        a, b = int(qsl[r]), int(qsl[r + 1])
        want = naive_dense_attention(q[a:b], x.k_deq(), x.v_deq(), 0.3)
        torch.testing.assert_close(ref.out[a:b], want, rtol=0, atol=1e-12)
    # Negative control: a K shifted by one token changes the answer.
    bad = ref_attention(
        q,
        [torch.roll(x.k_deq(), 1, 0) for x in kv],
        [x.v_deq() for x in kv],
        qsl,
        lens,
        0.3,
    )
    assert (bad.out - ref.out).abs().max() > 1e-3


def test_bottom_right_causal_mask():
    """Query i (q_len queries, L keys) sees exactly keys 0..L-q_len+i.

    Why: bottom-right alignment is what makes decode (q_len = 1
    sees everything) and chunked prefill consistent. With Q = 0 every visible
    key has the same weight, and V = one-hot(token), so output column t is
    1/(number visible) exactly when key t is visible: the visible set can be
    read off. Negative control: top-left alignment (key t <= i) gives a
    different set.
    """
    L, ql, D = 7, 3, 8
    k = torch.zeros(L, 1, D, dtype=torch.float64)
    v = torch.eye(L, D, dtype=torch.float64)[:, None, :]
    q = torch.zeros(ql, 1, D)
    out = ref_attention(q, [k], [v], [0, ql], [L], 1.0).out[:, 0]
    for i in range(ql):
        visible = L - ql + i + 1
        want = torch.zeros(D, dtype=torch.float64)
        want[:visible] = 1.0 / visible
        assert torch.equal(out[i], want)
        top_left = torch.zeros(D, dtype=torch.float64)
        top_left[: i + 1] = 1.0 / (i + 1)
        assert not torch.equal(out[i], top_left), "negative control"


@pytest.mark.parametrize(
    ("window", "causal"),
    [(0, False), (3, True), (3, False), (1, True), (1, False), (20, False)],
)
def test_window_visible_set(window, causal):
    """Query i at position p = L - q_len + i sees exactly keys [lo, hi].

    Why: the sliding window and the non-causal drafter (DFlash2: window 2048,
    every key of the request visible inside it, the query block included) must
    be vLLM's Triton definition: causal ``k <= p``, non-causal ``k < L``, and
    ``|p - k| < window``. Read off as in the causal test (Q = 0, one-hot V);
    the closed form :func:`visible_bounds` (the kernel's kv_window_lo/hi) gives
    the same set. ``window = 20 > L`` is no window. Negative control: the
    window one key wider (``lo = p - window``) differs whenever lo > 0.
    """
    L, ql, D = 9, 4, 16
    k = torch.zeros(L, 1, D, dtype=torch.float64)
    v = torch.eye(L, D, dtype=torch.float64)[:, None, :]
    q = torch.zeros(ql, 1, D)
    out = ref_attention(
        q, [k], [v], [0, ql], [L], 1.0, window=window, causal=causal
    ).out[:, 0]
    qpos = [L - ql + i for i in range(ql)]
    lo, hi = visible_bounds(qpos, L, window, causal)
    mask = visible_mask(torch.arange(L), torch.tensor(qpos), L, window, causal)
    wider_differs = False
    for i, p in enumerate(qpos):
        want = torch.zeros(D, dtype=torch.float64)
        want[lo[i] : hi[i] + 1] = 1.0 / (hi[i] - lo[i] + 1)
        assert torch.equal(out[i], want)
        assert mask[i].nonzero().flatten().tolist() == list(range(lo[i], hi[i] + 1))
        if window > 0 and p - window >= 0:
            wider = torch.zeros(D, dtype=torch.float64)
            wider[p - window : hi[i] + 1] = 1.0 / (hi[i] - p + window + 1)
            wider_differs |= not torch.equal(out[i], wider)
    if window in (1, 3):
        assert wider_differs, "negative control: a one-key-wider window differs"
    if causal:
        assert hi == qpos
    else:
        assert hi == [min(L - 1, p + window - 1) if window else L - 1 for p in qpos]


def test_empty_requests_and_padding_rows():
    """Empty-request rows and padding rows are 0 with lse = -inf and marked invalid.

    Why: padded CUDA-graph batches carry rows that belong to no request; the
    kernel must not write them, and a q_len = 0 request has no rows. The
    reference marks them invalid so tests skip them, and uses lse = -inf (the
    kernel's empty-split convention, so a combine never weights them).
    """
    c = make_int8_cache([5, 0, 9], Hkv=1, D=16, block_size=16, seed=2)
    q, _ = make_queries([2, 0, 3], 2, 16, seed=3)
    q = torch.cat([q, torch.randn(4, 2, 16).to(q.dtype)])  # 4 padding rows
    res = ref_attention_dense(q, c.dense_all(), [0, 2, 2, 5], 0.25)
    assert res.valid.tolist() == [True] * 5 + [False] * 4
    assert torch.all(res.out[5:] == 0) and torch.all(torch.isneginf(res.lse[5:]))
    assert torch.all(torch.isfinite(res.lse[:5]))


def test_reference_lse():
    """The reference lse is the natural-log log-sum-exp of the scaled visible scores."""
    k = torch.randn(6, 1, 4, dtype=torch.float64)
    v = torch.randn(6, 1, 4, dtype=torch.float64)
    q = torch.randn(2, 1, 4)
    res = ref_attention(q, [k], [v], [0, 2], [6], 0.5)
    for i in range(2):
        s = (k[: 5 + i, 0] @ q[i, 0].double()) * 0.5
        assert math.isclose(
            float(res.lse[i, 0]), float(torch.logsumexp(s, 0)), rel_tol=1e-14
        )


def test_dense_kv_dequant_is_exact():
    """``scale * int`` in fp64 is the exact real product (no rounding).

    Why: the reference claims to have no quantisation error of its own. An
    int8 has <= 8 significant bits and an fp32 scale 24, so the product needs
    <= 32 bits and fits fp64's 53 exactly. Checked against exact rational
    arithmetic. Negative control: the same product in fp32 is not exact.
    """
    from fractions import Fraction

    x = DenseKV(
        *quantize_rows(torch.randn(50, 2, 32)), *quantize_rows(torch.randn(50, 2, 32))
    )
    a = x.k_deq()
    inexact32 = 0
    for t in range(0, 50, 7):
        for d in range(0, 32, 5):
            exact = Fraction(int(x.k_int[t, 0, d])) * Fraction(float(x.k_scale[t, 0]))
            assert Fraction(float(a[t, 0, d])) == exact
            f32 = Fraction(float(torch.tensor(float(a[t, 0, d]), dtype=torch.float32)))
            inexact32 += f32 != exact
    assert inexact32 > 0, "negative control: fp32 products round"


# --------------------------------------------------------------------------
# Host-sync guards
# --------------------------------------------------------------------------
def test_host_sync_guards(tmp_path):
    """``forbid_host_sync`` and ``ast_host_sync_scan`` flag ``.item()`` in forward code.

    Why: host syncs in forward are forbidden (they break CUDA graphs
    and stall the pipeline). The runtime guard must raise for a call made from
    the guarded file and stay silent for other callers; the static scan must
    find the call. The clean function is the negative control.
    """
    src = tmp_path / "fwd_mod.py"
    src.write_text(
        textwrap.dedent(
            """
            def forward(t):
                return t.item()

            def clean(t):
                return t + 1
            """
        )
    )
    ns: dict = {}
    exec(compile(src.read_text(), str(src), "exec"), ns)
    t = torch.ones(())
    with forbid_host_sync("fwd_mod.py") as g:
        with pytest.raises(AssertionError):
            ns["forward"](t)
        assert ns["clean"](t) == 2  # negative control: no sync, no error
    assert g.hits and g.hits[0][0] == "item"
    hits = ast_host_sync_scan(str(src), ["forward"])
    assert len(hits) == 1 and ".item()" in hits[0]
    assert ast_host_sync_scan(str(src), ["clean"]) == []


# --------------------------------------------------------------------------
# Writer mirror == vLLM's writer (slow: Triton CPU interpreter)
# --------------------------------------------------------------------------
_EQUIV_SCRIPT = textwrap.dedent(
    """
    import torch
    import pth_testing.reference as R

    def build(writer, **kw):
        return R.make_int8_cache(writer=writer, **kw)

    FIELDS = ("k_int", "k_scale", "v_int", "v_scale")

    def same_pages(a, b):
        # every byte of every used page: data, scales, poison and padding
        pb = a.page_bytes
        for r, s in enumerate(a.seq_lens.tolist()):
            for blk in a.block_table[r, : R.cdiv(s, a.block_size)].tolist():
                lo = a.storage_offset_bytes + blk * pb
                if not torch.equal(a.storage[lo : lo + pb], b.storage[lo : lo + pb]):
                    return False
        return True

    def same_view(x, y):
        return (x.shape, x.stride(), x.storage_offset()) == (
            y.shape, y.stride(), y.storage_offset())

    def views(a, b):
        return all(same_view(getattr(a, f), getattr(b, f))
                   for f in ("key_cache", "value_cache", "k_scale", "v_scale"))

    def same_dense(a, b, r):
        da, db = a.dense(r), b.dense(r)
        return all(torch.equal(getattr(da, f), getattr(db, f)) for f in FIELDS)

    n = 0
    for layout in ("NHD", "HND"):
        for dt in (torch.bfloat16, torch.float16, torch.float32):
            for bs, pad in ((16, 0), (64, 8), (816, 0)):
                kw = dict(
                    seq_lens=[20, 0, 37], Hkv=2, D=64, block_size=bs,
                    layout=layout, pad_page_bytes=pad, seed=5,
                    storage_offset_bytes=4, src_dtype=dt,
                    scale_mode=["mixed", "normal", "extreme"],
                )
                a = build("mirror", **kw)
                b = build("vllm", **kw)
                assert views(a, b), ("views", layout, dt, bs)
                assert same_pages(a, b), ("pages", layout, dt, bs)
                for r in range(3):
                    assert same_dense(a, b, r), (layout, dt, bs, r)
                n += 1
    # D = 256 (production) once, bf16, both layouts.
    for layout in ("NHD", "HND"):
        kw = dict(seq_lens=[50], Hkv=2, D=256, block_size=16, layout=layout, seed=6)
        a, b = build("mirror", **kw), build("vllm", **kw)
        assert same_dense(a, b, 0), (layout, 256)
        n += 1
    # Negative controls: two plausible-but-wrong mirrors must differ from vLLM.
    kw = dict(seq_lens=[64], Hkv=2, D=64, block_size=16, seed=8, scale_mode="mixed")
    ref = build("vllm", **kw).dense(0)
    src = build("mirror", **kw).k_src[0].float()
    no_floor = src.abs().amax(-1) / 127.0
    assert not torch.equal(no_floor, ref.k_scale), "no-floor mutant must differ"
    inv = (1.0 / ref.k_scale).unsqueeze(-1)
    trunc = (src * inv).clamp(-128, 127).to(torch.int8)
    assert not torch.equal(trunc, ref.k_int), "truncating mutant must differ"
    print("EQUIV_OK", n)
    """
)


@pytest.mark.slow
def test_writer_mirror_equals_vllm():
    """The torch writer mirror and our views are bit-identical to vLLM v0.30.0's.

    Why: the reference is only meaningful if the cache it builds is the cache
    vLLM writes. vLLM's Triton writer ``_reshape_cache_per_token_head`` and its
    view helpers ``_pth_key_value_caches``/``_ensure_scale_caches`` run here
    through the Triton CPU interpreter (``TRITON_INTERPRET=1``, set only in the
    child process; the library reads no environment variable). Covered: NHD
    and HND, bf16/fp16/fp32 sources, block sizes 16/64/816, padded pages, a
    4-byte storage offset, mixed/extreme/floor/zero rows, D = 64 and 256.
    Negative controls: a mirror without the 1e-6 floor and one that truncates
    instead of rounding must both differ from vLLM.

    Limit: this proves equality with the interpreter's IEEE fp32 arithmetic;
    whether the GPU build of the writer uses exact division is checked on the
    GPU.
    """
    pytest.importorskip("vllm")
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    py_path = os.path.join(os.path.dirname(__file__), "..", "..", "python")
    env["PYTHONPATH"] = (
        os.path.abspath(py_path) + os.pathsep + env.get("PYTHONPATH", "")
    )
    res = subprocess.run(
        [sys.executable, "-c", _EQUIV_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert "EQUIV_OK 20" in res.stdout, res.stdout[-2000:]
