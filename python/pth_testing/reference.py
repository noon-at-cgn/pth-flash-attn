# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Exact reference for paged ``int8_per_token_head`` attention.

This module answers one question for every test of the kernel: *what is the
right answer?* It has three parts.

1. **Cache builder** (:func:`make_int8_cache`). It builds the real vLLM v0.30.0
   cache: one int8 storage with a logical ``[num_blocks, Hkv, block_size, C]``
   view, ``C = 2 * (D + 4)`` bytes per (token, head) cell laid out as
   ``[K: D int8 | k_scale: f32 | V: D int8 | v_scale: f32]``, physically NHD or
   HND, optionally with padded pages. The K/V/scale views are carved out exactly
   as ``TritonAttentionImpl._pth_key_value_caches`` and ``_ensure_scale_caches``
   do, and the data is quantised by a mirror of vLLM's writer
   ``_reshape_cache_per_token_head`` (``triton_reshape_and_cache_flash.py``).
   A slow test proves the mirror and the views are bit-identical to vLLM's own
   code run through the Triton CPU interpreter.
2. **Reference attention** (:func:`ref_attention`): fp64 GQA attention on the
   *dequantised* cache ``k_hat = k_scale * k_int``, bottom-right causal or
   with a sliding window and/or non-causal. These are exact real
   numbers (an int8 times an fp32 is exact in fp64), so the reference has no
   quantisation error of its own; any difference between a kernel and the
   reference is the kernel's arithmetic (or a bug). It is checked against an
   independent dense implementation (:func:`naive_dense_attention`).
3. **Host-sync guards** (:class:`forbid_host_sync`, :func:`ast_host_sync_scan`)
   for the "no ``.item()``/``.cpu()`` in forward" rule.

Nothing here needs a GPU or reads an environment variable. vLLM is imported
only for the optional equivalence check (``writer="vllm"`` and
:func:`vllm_packed_views`), lazily inside those functions, so this module
imports and the CPU tests run without vLLM.
"""

from __future__ import annotations

import ast
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

# --------------------------------------------------------------------------
# Writer constants (vLLM v0.30.0 triton_reshape_and_cache_flash.py,
# _PER_TOKEN_HEAD_QUANT_PARAMS[torch.int8] and the 1e-6 floor in the kernel).
# --------------------------------------------------------------------------
QUANT_MAX = 127.0  # int8 scale divisor and upper clamp
QUANT_MIN = -128.0  # lower clamp (never reached in practice, see quantize_rows)
SCALE_FLOOR = 1e-6  # scale = max(absmax / 127, 1e-6)
ROUND_HALF = 0.5  # round-half-away-from-zero offset before truncation
SCALE_BYTES = 4  # one f32 scale per (token, head) per K/V half
# Poison for bytes no kernel may read: finite but huge, so reading one gives a
# visible error instead of a silent NaN that a NaN-masking bug could hide.
POISON_SCALE = 3.0e4
INT8_LO, INT8_HI = -128, 127  # torch.randint bounds for poison bytes (hi+1)
# The "high block ids" mode must put byte offsets beyond what int32 holds.
INT32_LIMIT = 2**31
SPARE_BLOCKS = 3  # unused-but-poisoned blocks next to the real ones
NAT_LOG2E = math.log2(math.e)

LAYOUTS = ("NHD", "HND")
SCALE_MODES = ("normal", "floor", "extreme", "mixed")


def cdiv(a: int, b: int) -> int:
    """Ceiling division for non-negative ints."""
    return -(-a // b)


# --------------------------------------------------------------------------
# Writer mirror
# --------------------------------------------------------------------------
def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``x [..., D]`` per row exactly like vLLM's int8 writer.

    The writer (``_reshape_cache_per_token_head`` with ``IS_INT_QUANT``) does,
    per (token, head), all in fp32:

    1. ``scale = max(max|x| / 127, 1e-6)`` (fp32 divide, then fp32 max),
    2. ``q = x * (1 / scale)`` (fp32 reciprocal, then fp32 multiply),
    3. ``q += 0.5`` if ``q >= 0`` else ``q -= 0.5`` (round half away from zero),
    4. ``clamp(q, -128, 127)``, then the int8 store truncates toward zero.

    Every step here is the same IEEE fp32 operation in the same order, so the
    result is bit-identical (proved by ``test_writer_mirror_equals_vllm``).
    ``|x * (1/scale)| <= 127 * (1 + 2^-23)`` so ``-128`` is unreachable: the
    stored ints lie in ``[-127, 127]``.

    Args:
        x: Source rows (bf16/fp16/fp32), last dim is the head dim.

    Returns:
        ``(q_int8, scale_f32)`` with shapes ``x.shape`` and ``x.shape[:-1]``.
    """
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / QUANT_MAX, min=SCALE_FLOOR)
    q = xf * (1.0 / scale).unsqueeze(-1)
    q = torch.where(q >= 0, q + ROUND_HALF, q - ROUND_HALF)
    q = q.clamp(QUANT_MIN, QUANT_MAX)
    return q.to(torch.int8), scale


@dataclass(frozen=True)
class DenseKV:
    """One request's K/V as stored in the cache, token-major and contiguous.

    ``k_int``/``v_int`` are the stored int8 values ``[S, Hkv, D]`` and
    ``k_scale``/``v_scale`` the stored fp32 scales ``[S, Hkv]``. The
    dequantised (exact) values are ``scale * int``.
    """

    k_int: torch.Tensor
    k_scale: torch.Tensor
    v_int: torch.Tensor
    v_scale: torch.Tensor

    @property
    def seq_len(self) -> int:
        """Number of tokens."""
        return int(self.k_int.shape[0])

    def k_deq(self) -> torch.Tensor:
        """``k_scale * k_int`` in fp64 ``[S, Hkv, D]`` (exact)."""
        return self.k_int.double() * self.k_scale.double().unsqueeze(-1)

    def v_deq(self) -> torch.Tensor:
        """``v_scale * v_int`` in fp64 ``[S, Hkv, D]`` (exact)."""
        return self.v_int.double() * self.v_scale.double().unsqueeze(-1)


def quantize_kv(k_src: torch.Tensor, v_src: torch.Tensor) -> DenseKV:
    """Quantise one request's ``[S, Hkv, D]`` K/V sources with the writer mirror."""
    kq, ks = quantize_rows(k_src)
    vq, vs = quantize_rows(v_src)
    return DenseKV(kq, ks, vq, vs)


# --------------------------------------------------------------------------
# Synthetic K/V sources
# --------------------------------------------------------------------------
# "normal": per-row magnitude log-uniform in [0.3, 3] plus one outlier channel
# (real K has a few large channels); "floor": every |x| <= 1.2e-4 so every
# absmax/127 < 1e-6 and every scale sits on the floor, with ~5% all-zero rows;
# "extreme": every element is +/-amp so every int is exactly +/-127; "mixed":
# a per-row mixture of all three plus zero rows.
_MAG_LO, _MAG_HI = 0.3, 3.0
_OUTLIER_CHANNEL, _OUTLIER_GAIN = 7, 6.0
_FLOOR_STD, _FLOOR_CLIP, _ZERO_ROW_P = 3e-5, 1.2e-4, 0.05
_EXT_LO, _EXT_HI = 0.5, 6.0
_MIX_NORMAL, _MIX_FLOOR, _MIX_EXTREME = 0.5, 0.7, 0.9  # cumulative thresholds


def gen_rows(
    g: torch.Generator, n: int, Hkv: int, D: int, mode: str = "normal"
) -> torch.Tensor:
    """Synthetic ``[n, Hkv, D]`` fp32 source rows for one of :data:`SCALE_MODES`."""
    base = torch.randn(n, Hkv, D, generator=g)
    mag = torch.exp(
        torch.empty(n, Hkv, 1).uniform_(
            math.log(_MAG_LO), math.log(_MAG_HI), generator=g
        )
    )
    normal = base * mag
    if D > _OUTLIER_CHANNEL:
        normal[..., _OUTLIER_CHANNEL] *= _OUTLIER_GAIN
    if mode == "normal":
        return normal
    if mode == "floor":
        x = (base * _FLOOR_STD).clamp(-_FLOOR_CLIP, _FLOOR_CLIP)
        zero = torch.rand(n, Hkv, 1, generator=g) < _ZERO_ROW_P
        return torch.where(zero, torch.zeros_like(x), x)
    if mode == "extreme":
        amp = torch.exp(
            torch.empty(n, Hkv, 1).uniform_(
                math.log(_EXT_LO), math.log(_EXT_HI), generator=g
            )
        )
        sgn = torch.where(torch.rand(n, Hkv, D, generator=g) < ROUND_HALF, -1.0, 1.0)
        return sgn * amp
    if mode == "mixed":
        sel = torch.rand(n, Hkv, 1, generator=g)
        fl = gen_rows(g, n, Hkv, D, "floor")
        ex = gen_rows(g, n, Hkv, D, "extreme")
        z = torch.zeros_like(normal)
        return torch.where(
            sel < _MIX_NORMAL,
            normal,
            torch.where(sel < _MIX_FLOOR, fl, torch.where(sel < _MIX_EXTREME, ex, z)),
        )
    raise ValueError(f"unknown scale mode {mode!r}; expected one of {SCALE_MODES}")


# --------------------------------------------------------------------------
# The paged cache
# --------------------------------------------------------------------------
@dataclass
class Int8Cache:
    """A real ``int8_per_token_head`` paged cache plus the facts tests need.

    The views are those the kernel receives (docs/api.md): ``k_cache``/``v_cache``
    are int8 ``[NB, BS, Hkv, D]`` and ``k_scale``/``v_scale`` f32
    ``[NB, BS, Hkv]``, all with arbitrary strides into one storage.
    ``key_cache``/``value_cache`` are vLLM's ``[NB, BS, Hkv, D + 4]`` views
    (data plus the scale bytes), as ``_pth_key_value_caches`` returns them.
    """

    kv_packed: torch.Tensor  # logical [NB, Hkv, BS, C] int8
    key_cache: torch.Tensor  # [NB, BS, Hkv, D + 4] int8 (vLLM view)
    value_cache: torch.Tensor
    k_cache: torch.Tensor  # [NB, BS, Hkv, D] int8 (op input)
    v_cache: torch.Tensor
    k_scale: torch.Tensor  # [NB, BS, Hkv] f32
    v_scale: torch.Tensor
    block_table: torch.Tensor  # [num_reqs, max_blocks] int32 kernel-block ids
    seq_lens: torch.Tensor  # [num_reqs] int32
    storage: torch.Tensor  # flat int8 backing storage
    k_src: list[torch.Tensor] = field(default_factory=list)  # per request
    v_src: list[torch.Tensor] = field(default_factory=list)
    Hkv: int = 0
    D: int = 0
    block_size: int = 0
    layout: str = "NHD"
    page_bytes: int = 0  # block stride in bytes (dense page + padding)
    storage_offset_bytes: int = 0
    max_block_id: int = 0
    writer: str = "mirror"

    @property
    def num_reqs(self) -> int:
        """Number of requests."""
        return int(self.seq_lens.shape[0])

    @property
    def cell_bytes(self) -> int:
        """Bytes per (token, head) cell: ``2 * (D + 4)``."""
        return 2 * (self.D + SCALE_BYTES)

    def dense(self, r: int) -> DenseKV:
        """Gather request ``r`` through the paged views (the kernel's read path).

        Token ``t`` lives in block ``block_table[r, t // BS]`` at slot
        ``t % BS``; indexing the strided views turns that into int64 byte
        offsets inside torch, so this also exercises padded pages, HND/NHD and
        block ids whose byte offsets exceed 2^31.
        """
        s = int(self.seq_lens[r])
        pos = torch.arange(s, dtype=torch.int64)
        blk = self.block_table[r].long()[pos // self.block_size]
        sl = pos % self.block_size
        return DenseKV(
            self.k_cache[blk, sl].clone(),
            self.k_scale[blk, sl].clone(),
            self.v_cache[blk, sl].clone(),
            self.v_scale[blk, sl].clone(),
        )

    def dense_all(self) -> list[DenseKV]:
        """:meth:`dense` for every request."""
        return [self.dense(r) for r in range(self.num_reqs)]


def packed_views(
    kv_packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carve vLLM's K/V and scale views out of the packed cache.

    Mirrors ``TritonAttentionImpl._pth_key_value_caches`` (transpose to
    ``[NB, BS, Hkv, C]``, split the content dim in two halves of ``D + 4``) and
    ``_ensure_scale_caches`` (f32 views at byte ``D`` of each half, strides taken
    from the packed tensor, never from its shape).

    Returns:
        ``(key_cache, value_cache, k_scale, v_scale)``.
    """
    num_blocks, nkv, block_size, content = kv_packed.shape
    padded_hs = content // 2
    hs = padded_hs - SCALE_BYTES
    key_cache, value_cache = kv_packed.transpose(1, 2).split(padded_hs, dim=-1)
    base_f32 = torch.tensor([], dtype=torch.float32, device=kv_packed.device).set_(
        kv_packed.untyped_storage()
    )

    def f32_units(nbytes: int) -> int:
        if nbytes % SCALE_BYTES:
            raise ValueError(f"{nbytes} bytes is not a whole number of f32 words")
        return nbytes // SCALE_BYTES

    st = kv_packed.stride()  # element (= byte) strides of the int8 view
    # int() because torch types strides as int | SymInt
    stride = (f32_units(int(st[0])), f32_units(int(st[2])), f32_units(int(st[1])))
    base = f32_units(int(kv_packed.storage_offset()))
    size = (num_blocks, block_size, nkv)
    k_scale = torch.as_strided(base_f32, size, stride, base + f32_units(hs))
    v_scale = torch.as_strided(base_f32, size, stride, base + f32_units(padded_hs + hs))
    return key_cache, value_cache, k_scale, v_scale


def _vllm_writer() -> Any:
    """Import vLLM's real writer lazily (only for ``writer="vllm"``)."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    return triton_reshape_and_cache_flash_per_token_head_quant


def vllm_packed_views(
    kv_packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run vLLM's own view helpers on ``kv_packed`` (for the equivalence test).

    ``_ensure_scale_caches`` and ``_pth_key_value_caches`` are unbound methods
    of ``TritonAttentionImpl`` that only touch ``self._k_scale_cache``,
    ``self._v_scale_cache``, ``self._kv_quant_mode`` and ``self.fp8_dtype``; a
    small stand-in object carries exactly those. Note that vLLM's helper fills
    every scale slot with 1.0 (it touches all pages).
    """
    import vllm.v1.attention.backends.triton_attn as ta
    from vllm.v1.kv_cache_interface import KVQuantMode

    class _Impl:
        _k_scale_cache = None
        _v_scale_cache = None
        _kv_quant_mode = KVQuantMode.INT8_PER_TOKEN_HEAD
        fp8_dtype = torch.float8_e4m3fn

    impl: Any = _Impl()
    cls = ta.TritonAttentionImpl
    # _pth_key_value_caches calls self._ensure_scale_caches, so bind both.
    impl._ensure_scale_caches = cls._ensure_scale_caches.__get__(impl)
    impl._pth_key_value_caches = cls._pth_key_value_caches.__get__(impl)
    key_cache, value_cache = impl._pth_key_value_caches(kv_packed)
    return key_cache, value_cache, impl._k_scale_cache, impl._v_scale_cache


def make_int8_cache(
    seq_lens: Sequence[int],
    Hkv: int = 2,
    D: int = 256,
    block_size: int = 816,
    layout: str = "NHD",
    pad_page_bytes: int = 0,
    seed: int = 0,
    *,
    scale_mode: str | Sequence[str] = "normal",
    high_block_ids: bool = False,
    storage_offset_bytes: int = 0,
    src_dtype: torch.dtype = torch.bfloat16,
    sources: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    writer: str = "mirror",
) -> Int8Cache:
    """Build a paged int8 cache holding ``len(seq_lens)`` requests.

    Args:
        seq_lens: KV length of each request (0 allowed).
        Hkv: KV heads.
        D: Head size.
        block_size: Kernel block size in tokens; any value >= 1 (16, 528, 816 …).
        layout: ``"NHD"`` (physical ``[blk, slot, head, C]``, vLLM's default
            here) or ``"HND"`` (physical ``[blk, head, slot, C]``).
        pad_page_bytes: Extra bytes after every page (hybrid-model page
            equalisation): block stride = ``Hkv * BS * C + pad``. Multiple of 4
            (vLLM's scale views require whole f32 words).
        seed: Seeds block placement, poison and sources.
        scale_mode: One of :data:`SCALE_MODES` or one per request.
        high_block_ids: Place every block above ``2^31 / block_stride`` so the
            K/V byte offsets of every token need int64. The storage
            is then > 2 GiB of *virtual* memory; only touched pages are
            committed.
        storage_offset_bytes: Offset of the cache inside its storage (multiple
            of 4), so base pointers are not 16-B aligned.
        src_dtype: dtype of the K/V the model hands the writer.
        sources: Optional explicit ``(k_src, v_src)`` per request,
            ``[S, Hkv, D]``; overrides ``scale_mode``.
        writer: ``"mirror"`` (:func:`quantize_rows`) or ``"vllm"`` (vLLM's
            Triton writer and view helpers; needs ``TRITON_INTERPRET=1`` set by
            the caller on a CPU-only machine).

    Unused slots of touched blocks, spare blocks and page padding hold finite
    poison (random bytes, scale 3e4), so a kernel that reads outside a
    request's tokens produces a large, visible error.
    """
    lens = [int(s) for s in seq_lens]
    if any(s < 0 for s in lens):
        raise ValueError("seq_lens must be >= 0")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if pad_page_bytes % SCALE_BYTES or storage_offset_bytes % SCALE_BYTES:
        raise ValueError(
            "pad_page_bytes and storage_offset_bytes must be multiples of 4"
        )
    if writer not in ("mirror", "vllm"):
        raise ValueError("writer must be 'mirror' or 'vllm'")
    nreq = len(lens)
    C = 2 * (D + SCALE_BYTES)
    page = Hkv * block_size * C + pad_page_bytes
    nblk = [cdiv(s, block_size) for s in lens]
    need = sum(nblk)
    first = (INT32_LIMIT // page + 1) if high_block_ids else 0
    num_blocks = first + need + SPARE_BLOCKS

    g = torch.Generator().manual_seed(seed)
    # Shuffled ids so block tables are not monotone (a kernel that assumes
    # consecutive pages fails).
    ids = (first + torch.randperm(need + SPARE_BLOCKS, generator=g)).tolist()
    req_blocks, k = [], 0
    bt = torch.zeros(nreq, max(max(nblk, default=0), 1), dtype=torch.int32)
    for r, nb in enumerate(nblk):
        req_blocks.append(ids[k : k + nb])
        if nb:
            bt[r, :nb] = torch.tensor(ids[k : k + nb], dtype=torch.int32)
        k += nb
    touched = sorted(ids)

    storage = torch.empty(storage_offset_bytes + num_blocks * page, dtype=torch.int8)
    # Physical order: HND = [blk, head, slot, C], NHD = [blk, slot, head, C];
    # the logical view is always [blk, head, slot, C] (as vLLM allocates it).
    nhd = (page, C, Hkv * C, 1)
    strides = (page, block_size * C, C, 1) if layout == "HND" else nhd
    kv_packed = torch.as_strided(
        storage, (num_blocks, Hkv, block_size, C), strides, storage_offset_bytes
    )
    if writer == "vllm":
        key_cache, value_cache, k_scale, v_scale = vllm_packed_views(kv_packed)
    else:
        key_cache, value_cache, k_scale, v_scale = packed_views(kv_packed)

    for b in touched:  # poison every byte of every touched page, padding included
        lo = storage_offset_bytes + b * page
        storage[lo : lo + page].random_(INT8_LO, INT8_HI + 1, generator=g)
    tb = torch.tensor(touched, dtype=torch.int64)
    k_scale[tb] = POISON_SCALE
    v_scale[tb] = POISON_SCALE

    modes = [scale_mode] * nreq if isinstance(scale_mode, str) else list(scale_mode)
    if len(modes) != nreq:
        raise ValueError("one scale mode per request")
    k_src, v_src, slots = [], [], []
    for r, s in enumerate(lens):
        if sources is not None:
            kr, vr = (t.to(src_dtype) for t in sources[r])
        else:
            kr = gen_rows(g, s, Hkv, D, modes[r]).to(src_dtype)
            vr = gen_rows(g, s, Hkv, D, modes[r]).to(src_dtype)
        if kr.shape != (s, Hkv, D) or vr.shape != (s, Hkv, D):
            raise ValueError(f"request {r}: sources must be [{s}, {Hkv}, {D}]")
        k_src.append(kr)
        v_src.append(vr)
        pos = torch.arange(s, dtype=torch.int64)
        blk = torch.tensor(req_blocks[r] or [0], dtype=torch.int64)[pos // block_size]
        slots.append(blk * block_size + pos % block_size)
    if sum(lens):
        key = torch.cat(k_src).contiguous()
        val = torch.cat(v_src).contiguous()
        slot_mapping = torch.cat(slots)
        if writer == "vllm":
            from vllm.v1.kv_cache_interface import KVQuantMode

            _vllm_writer()(
                key,
                val,
                key_cache,
                value_cache,
                k_scale,
                v_scale,
                slot_mapping,
                kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
            )
        else:
            kq, ks = quantize_rows(key)
            vq, vs = quantize_rows(val)
            blk, sl = slot_mapping // block_size, slot_mapping % block_size
            key_cache[blk, sl, :, :D] = kq
            value_cache[blk, sl, :, :D] = vq
            k_scale[blk, sl] = ks
            v_scale[blk, sl] = vs

    return Int8Cache(
        kv_packed=kv_packed,
        key_cache=key_cache,
        value_cache=value_cache,
        k_cache=key_cache[..., :D],
        v_cache=value_cache[..., :D],
        k_scale=k_scale,
        v_scale=v_scale,
        block_table=bt,
        seq_lens=torch.tensor(lens, dtype=torch.int32),
        storage=storage,
        k_src=k_src,
        v_src=v_src,
        Hkv=Hkv,
        D=D,
        block_size=block_size,
        layout=layout,
        page_bytes=page,
        storage_offset_bytes=storage_offset_bytes,
        max_block_id=max(ids[:need], default=0),
        writer=writer,
    )


def make_queries(
    q_lens: Sequence[int],
    Hq: int,
    D: int,
    seed: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random queries ``[T, Hq, D]`` and ``query_start_loc`` int32 ``[R + 1]``."""
    g = torch.Generator().manual_seed(seed)
    T = int(sum(q_lens))
    q = (torch.randn(T, Hq, D, generator=g) * std).to(dtype)
    qsl = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(list(q_lens), dtype=torch.int64).cumsum(0).to(torch.int32)
    return q, qsl


# --------------------------------------------------------------------------
# fp64 reference
# --------------------------------------------------------------------------
# Rows of fp64 scores computed at once are limited so one block stays below
# ~1.2 GB (8 bytes x 1.5e8).
_SCORE_BLOCK_ELEMS = 150_000_000
_MIN_ROW_BLOCK = 4


@dataclass
class RefResult:
    """Output of :func:`ref_attention`.

    ``out`` is fp64 ``[T, Hq, D]``; ``lse`` the natural-log log-sum-exp of the
    scaled scores ``[T, Hq]``; ``valid`` marks rows that belong to a request
    (``T`` may exceed ``query_start_loc[-1]`` in a padded graph batch). Invalid
    rows and rows of empty requests have ``out = 0`` and ``lse = -inf`` (the
    kernel's empty-split convention).
    """

    out: torch.Tensor
    lse: torch.Tensor
    valid: torch.Tensor


def _as_list(x: torch.Tensor | Sequence[int]) -> list[int]:
    return [int(v) for v in (x.tolist() if isinstance(x, torch.Tensor) else x)]


def visible_mask(
    kpos: torch.Tensor,
    qpos: torch.Tensor,
    seq_len: int,
    window: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """``[len(qpos), len(kpos)]`` bool: may the query at position p see key k.

    vLLM's Triton definition (``triton_attention_helpers.py``):
    causal ``k <= p``, non-causal ``k < seq_len``; with ``window > 0`` also
    ``|p - k| < window`` (``window_size = (window - 1, 0)``). ``window = 0``
    and ``causal`` is bottom-right causal attention.
    """
    k, p = kpos[None, :], qpos[:, None]
    allowed = k <= p if causal else (k < seq_len).expand(p.shape[0], -1)
    if window > 0:
        allowed = allowed & ((p - k).abs() < window)
    return allowed


def visible_bounds(
    qpos: Sequence[int] | torch.Tensor,
    seq_len: int,
    window: int = 0,
    causal: bool = True,
) -> tuple[list[int], list[int]]:
    """Inclusive ``[lo, hi]`` per query position: :func:`visible_mask` as a range.

    Closed form (``cache_geometry.cuh kv_window_lo/hi``): ``lo = max(0, p -
    window + 1)`` (0 without a window), ``hi = p`` causal, else ``min(seq_len -
    1, p + window - 1)`` (``seq_len - 1`` without a window).
    """
    lo, hi = [], []
    for p in _as_list(qpos):
        lo.append(max(0, p - window + 1) if window > 0 else 0)
        if causal:
            hi.append(p)
        else:
            hi.append(min(seq_len - 1, p + window - 1) if window > 0 else seq_len - 1)
    return lo, hi


def ref_attention(
    q: torch.Tensor,
    k_deq: Sequence[torch.Tensor],
    v_deq: Sequence[torch.Tensor],
    query_start_loc: torch.Tensor | Sequence[int],
    seq_lens: torch.Tensor | Sequence[int],
    softmax_scale: float,
    *,
    window: int = 0,
    causal: bool = True,
) -> RefResult:
    """fp64 GQA attention over exact (dequantised) K/V.

    Query token ``i`` of request ``r`` (``q_len`` tokens, ``L`` KV tokens
    including the new ones) sits at KV position ``p = L - q_len + i``
    (bottom-right alignment) and attends to the positions
    :func:`visible_mask` allows: ``<= p`` by default, or the sliding window
    and/or non-causal range. Query head ``h`` reads KV head
    ``h // G`` with ``G = Hq / Hkv``.

    Args:
        q: ``[T, Hq, D]`` queries (any float dtype; used exactly).
        k_deq: Per request ``[L_r, Hkv, D]`` dequantised K (fp64 recommended;
            :meth:`DenseKV.k_deq`).
        v_deq: Same for V.
        query_start_loc: ``[R + 1]`` cumulative query counts.
        seq_lens: ``[R]`` KV lengths ``L_r`` (must be ``>= q_len_r``).
        softmax_scale: Multiplies ``q . k`` before the softmax.
        window: Sliding window in tokens (0 = none).
        causal: Bottom-right causal (True) or non-causal (False).
    """
    qsl = _as_list(query_start_loc)
    lens = _as_list(seq_lens)
    qd = q.detach().double()
    T, Hq, D = qd.shape
    out = torch.zeros(T, Hq, D, dtype=torch.float64)
    lse = torch.full((T, Hq), -math.inf, dtype=torch.float64)
    valid = torch.zeros(T, dtype=torch.bool)
    for r, L in enumerate(lens):
        q0, q1 = qsl[r], qsl[r + 1]
        ql = q1 - q0
        if ql == 0:
            continue
        if ql > L:
            raise ValueError(f"request {r}: q_len {ql} > seq_len {L}")
        valid[q0:q1] = True
        K = k_deq[r].double()
        V = v_deq[r].double()
        if K.shape[0] != L:
            raise ValueError(f"request {r}: K has {K.shape[0]} tokens, seq_len {L}")
        Hkv = K.shape[1]
        G = Hq // Hkv
        kt = K.permute(1, 2, 0)  # [Hkv, D, L]
        vt = V.permute(1, 0, 2)  # [Hkv, L, D]
        kpos = torch.arange(L)
        rb = max(_MIN_ROW_BLOCK, _SCORE_BLOCK_ELEMS // (Hq * max(L, 1)))
        for a in range(0, ql, rb):
            b = min(a + rb, ql)
            n = b - a
            # rows ordered (g, i) inside each KV head: [Hkv, G * n, D]
            qm = qd[q0 + a : q0 + b].reshape(n, Hkv, G, D).permute(1, 2, 0, 3)
            s = torch.matmul(qm.reshape(Hkv, G * n, D), kt) * softmax_scale
            qpos = L - ql + a + torch.arange(n)
            allowed = visible_mask(kpos, qpos, L, window, causal).repeat(G, 1)
            s = s.masked_fill(~allowed[None], -math.inf)
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            den = p.sum(-1, keepdim=True)
            o = torch.matmul(p, vt) / den
            out[q0 + a : q0 + b] = (
                o.reshape(Hkv, G, n, D).permute(2, 0, 1, 3).reshape(n, Hq, D)
            )
            lse[q0 + a : q0 + b] = (
                (m + torch.log(den)).reshape(Hkv, G, n).permute(2, 0, 1).reshape(n, Hq)
            )
    return RefResult(out, lse, valid)


def ref_attention_dense(
    q: torch.Tensor,
    kv: Sequence[DenseKV],
    query_start_loc: torch.Tensor | Sequence[int],
    softmax_scale: float,
    *,
    window: int = 0,
    causal: bool = True,
) -> RefResult:
    """:func:`ref_attention` on a list of :class:`DenseKV` (one per request)."""
    return ref_attention(
        q,
        [x.k_deq() for x in kv],
        [x.v_deq() for x in kv],
        query_start_loc,
        [x.seq_len for x in kv],
        softmax_scale,
        window=window,
        causal=causal,
    )


def naive_dense_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Deliberately simple check of :func:`ref_attention` for one request.

    Loops over every (query token, head) with ``torch.softmax``; shares no code
    with :func:`ref_attention` (no batching, no masking by ``-inf``: the visible
    keys are sliced). ``q [q_len, Hq, D]``, ``k/v [L, Hkv, D]``; returns fp64
    ``[q_len, Hq, D]``.
    """
    qd, kd, vd = q.double(), k.double(), v.double()
    ql, Hq, _ = qd.shape
    L, Hkv, _ = kd.shape
    G = Hq // Hkv
    out = torch.zeros(qd.shape, dtype=torch.float64)
    for h in range(Hq):
        kh, vh = kd[:, h // G], vd[:, h // G]
        for i in range(ql):
            n = L - ql + i + 1
            p = torch.softmax((kh[:n] @ qd[i, h]) * softmax_scale, 0)
            out[i, h] = p @ vh[:n]
    return out


# --------------------------------------------------------------------------
# Host-sync guards (no .item()/.cpu() in forward)
# --------------------------------------------------------------------------
class forbid_host_sync:
    """Context manager: host-sync Tensor calls made directly from given files raise.

    While active, ``Tensor.item/tolist/cpu/numpy/__bool__/__int__/__float__/
    __index__`` called from a frame whose file basename is in ``basenames``
    raise ``AssertionError``. Only the direct caller is checked, so torch's own
    internals are not blamed.
    """

    NAMES = (
        "item",
        "tolist",
        "cpu",
        "numpy",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
    )

    def __init__(self, *basenames: str) -> None:
        """Remember which caller files are forbidden."""
        self.basenames = set(basenames)
        self.saved: dict[str, Any] = {}
        self.hits: list[tuple[str, str, int]] = []

    def __enter__(self) -> forbid_host_sync:
        """Patch the host-sync methods."""
        guard = self
        for n in self.NAMES:
            orig = getattr(torch.Tensor, n)
            self.saved[n] = orig

            def wrapper(*a: Any, _orig: Any = orig, _n: str = n, **k: Any) -> Any:
                fr = sys._getframe(1)
                fn = os.path.basename(fr.f_code.co_filename)
                if fn in guard.basenames:
                    guard.hits.append((_n, fn, fr.f_lineno))
                    raise AssertionError(
                        f"host sync Tensor.{_n}() called from {fn}:{fr.f_lineno}"
                    )
                return _orig(*a, **k)

            setattr(torch.Tensor, n, wrapper)
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the original methods."""
        for n, orig in self.saved.items():
            setattr(torch.Tensor, n, orig)


_SYNC_ATTRS = ("item", "cpu", "tolist", "numpy", "synchronize")


def ast_host_sync_scan(path: str, func_names: Sequence[str] | None = None) -> list[str]:
    """Statically list ``.item() .cpu() .tolist() .numpy() .synchronize()`` calls.

    Args:
        path: Python file to scan.
        func_names: Restrict to calls inside functions/methods with these
            names (``None`` scans the whole module).

    Returns:
        ``"file:line .attr()"`` strings (``bool(t)``/``int(t)`` cannot be seen
        statically; :class:`forbid_host_sync` catches those at run time).
    """
    with open(path) as f:
        tree = ast.parse(f.read())
    bad: list[str] = []
    names = set(func_names) if func_names is not None else None

    def visit(node: ast.AST, inside: bool) -> None:
        for ch in ast.iter_child_nodes(node):
            ins = inside
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ins = inside or names is None or ch.name in names
            if (
                (ins or names is None)
                and isinstance(ch, ast.Call)
                and isinstance(ch.func, ast.Attribute)
                and ch.func.attr in _SYNC_ATTRS
            ):
                bad.append(f"{os.path.basename(path)}:{ch.lineno} .{ch.func.attr}()")
            visit(ch, ins)

    visit(tree, False)
    return bad
