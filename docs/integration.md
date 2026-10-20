<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Integrating the op into another engine

The attention op does not depend on vLLM. This page states what any engine must
provide to call `pth_flash_attn_fwd` correctly, and what it can leave to this
project. It is the contract that the vLLM backend ([`vllm.md`](vllm.md)) meets;
everything the backend does beyond it is vLLM-specific glue. Argument-level
detail (shapes, dtypes, errors) is in [`api.md`](api.md), which this page does
not repeat.

## 1. Who does what

| Responsibility | vLLM (with the backend) | Other engine | This project |
|---|---|---|---|
| Cache allocation, paging, block ids, prefix caching | vLLM | the engine | none |
| Cache layout and cell format | vLLM's `int8_per_token_head` cell | any layout expressible as strides (section 2) | reads whatever strides it is given |
| Quantising writer (K and V of new tokens) | vLLM's Triton writer, unchanged | the engine, matching section 3 | none: the op never writes the cache; a mirror of vLLM's writer exists for tests only |
| Batch metadata (`block_table`, `seq_lens`, `query_start_loc`) | vLLM's runner; the backend's builder slices it | the engine | none |
| Launch plan and workspace sizing | `planner` called by the backend's builder and Impl | the engine calls `planner` | `planner.py` |
| Attention read, split-KV combine | the op | the op | `pth_flash_attn_fwd` |
| Selection, refusal of unsupported features | the backend | the engine | the op refuses invalid arguments |

## 2. The cache the op reads

**Values.** `k_cache` and `v_cache` are int8 tensors (or views) of shape
`[blocks, block_size, kv_heads, head_dim]`. Any outer strides are allowed (NHD,
HND, padded pages, block-outer layouts, vLLM's interleaved cell), as long as the
head dimension is contiguous, every stride is non-negative and the base pointer
and byte strides meet the alignment of [`api.md`](api.md). The two tensors need
not share strides. Supported `head_dim`: 64, 128 or 256. `block_size` may be any
value of at least 1; the vLLM backend additionally requires a multiple of 16.

**Scales.** `k_scale` and `v_scale` are float32 tensors of shape
`[blocks, block_size, kv_heads]`: one scale for each cached token and KV head,
separately for K and V. The dequantised value of element `d` is

```
k[token, head, d] ~ k_scale[block, slot, head] * k_cache[block, slot, head, d]
```

and likewise for V. The scales may live anywhere the strides can express: in
vLLM they are float32 views into the same cell as the int8 data.

**vLLM's cell, for reference.** Per (block, token, KV head) vLLM stores
`K (D int8) | k_scale (fp32) | V (D int8) | v_scale (fp32)`, which is `2 D + 8`
bytes; the K scale sits at byte `D`, V at byte `D + 4`, the V scale at byte
`2 D + 4`. The backend builds the op's four views from that cell without
copying. An engine that keeps
K, V and the scales in four separate arrays works equally well.

**Which tokens are cached.** `seq_lens[r]` counts every key the request sees,
including the query tokens of the current call. K and V of the new tokens must
be in the cache before the op runs; the op does not take K and V as arguments.

**The block table.** `block_table[r][j]` is the id, along the first dimension of
the cache views, of the block that holds tokens `[j * block_size,
(j + 1) * block_size)` of request `r`. Ids are int32; byte offsets are computed
in int64, so caches larger than 2 GiB are fine. If the engine allocates storage
blocks larger than the kernel block size, the table must hold kernel-block ids
and the views must describe kernel blocks (vLLM splits its blocks the same way).

## 3. The quantiser the writer must match

The op itself only assumes that `scale * int8` approximates the true value, and
that scales are positive normal floating-point numbers (the V-scale
normalisation takes the exponent of the scale). It places no requirement on how
the scale was chosen. Everything the tests establish, however, was established
with the convention vLLM's writer uses, and an engine that keeps vLLM's cache
format should use it too. Per (token, KV head), over the `D` elements `x` of the
row, all in float32:

1. `scale = max(max|x| / 127, 1e-6)`
2. `q = x * (1 / scale)`
3. round half away from zero: `q = q + 0.5` if `q >= 0` else `q - 0.5`
4. clamp to `[-128, 127]`, then store as int8 (the conversion truncates toward
   zero)

The floor of `1e-6` keeps every scale positive and normal, including for all-zero
rows. Stored integers lie in `[-127, 127]` in practice, because
`|x * (1 / scale)|` is at most `127 (1 + 2^-23)`; the kernel nevertheless
converts all 256 int8 values exactly. A scale that is not the row's absmax over
127 is allowed by the kernel, but the accuracy bound used in the tests was
derived and validated for scales of this form (including the `1e-6` floor).

This is the Python (PyTorch) form, equal bit for bit to vLLM's writer according
to `tests/cpu/test_reference.py::test_writer_mirror_equals_vllm`
(`pth_testing.reference.quantize_rows` is the tested copy), together with a
writer that stores new tokens at their slots:

```python
# CPU or GPU: plain PyTorch.
import torch


def quantize_rows(x):
    """Per (token, head) int8 quantisation with vLLM's writer convention."""
    xf = x.float()
    s = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    q = xf * (1.0 / s).unsqueeze(-1)
    q = torch.where(q >= 0, q + 0.5, q - 0.5).clamp(-128.0, 127.0)
    return q.to(torch.int8), s


def write_kv(k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping, block_size):
    """Quantise [T, Hkv, D] K and V and store them at their slots.

    slot_mapping[t] = block_id * block_size + offset_in_block, int64 [T].
    """
    kq, ks = quantize_rows(k)
    vq, vs = quantize_rows(v)
    blk, off = slot_mapping // block_size, slot_mapping % block_size
    k_cache[blk, off] = kq
    v_cache[blk, off] = vq
    k_scale[blk, off] = ks
    v_scale[blk, off] = vs
```

A production engine needs a fused kernel for this (vLLM uses a Triton kernel,
one program per token and head); the Python form is the specification and a
test oracle. The sketch above writes through the four views, so it works for any
strides. It was checked against the cache builder of `pth_testing` on CPU by
overwriting a request's cache and comparing the stored integers and scales for
equality.

## 4. Batch metadata

All three are int32 tensors on the device of `q`, with a contiguous last
dimension:

- `block_table`: `[rows >= num_reqs, max_blocks]`; the row stride is free.
- `seq_lens`: `[num_reqs]`; KV length including the new tokens; each entry must be
  at least the request's query length.
- `query_start_loc`: `[num_reqs + 1]`; cumulative query-token counts of this
  call's slice, starting at 0 and ending at `T`, the number of rows of `q`.

`max_query_len` is a host integer upper bound of any request's query length in
the slice. It only selects the row-tile height (see [`api.md`](api.md)); the
kernels read the true lengths on the device. Contents of the three tensors are
not validated on the host (that would need a device synchronisation); the engine
guarantees block ids in range and consistent lengths.

**Splitting a batch.** The planner has two policies: decode-like (every request
has a short query: one token, or a few for speculative verification) and
prefill. A batch that mixes both should be cut into two slices and run as two op
calls, as the vLLM backend does: the decode-like slice first, with
`is_prefill=False`, then the prefill slice with `is_prefill=True`. Each slice
gets its own views: `q` and `out` sliced by token range, `seq_lens` and
`block_table` sliced by request range, and a `query_start_loc` rebased to 0 for
the slice. A batch in another order is still correct (the prefill launch accepts
any query length), only slower. The decode-like threshold used by the backend is
`max(64 // G, 1 + num_speculative_tokens)` query tokens, where `64 // G` is
`planner.max_tokens_per_tile(G)`.

## 5. Launch plan and workspaces

1. At start-up, reserve two float32 buffers of the shapes
   `planner.workspace_shapes(...)` returns, sized for the engine's maximum batch
   and speculative verification length. Do this before the engine sizes its KV
   cache, so the bytes are counted as model memory (vLLM does it while the model
   loads).
2. For each slice, call `planner.plan_launch(...)` with host integers of the
   slice only. It returns `wm` and `num_splits` and the shapes of the workspace
   views for this launch.
3. Pass views of those shapes, cut from the reservation, as `ws_o` and `ws_lse`
   (or empty tensors when `num_splits == 1`).

The planner has no sequence-length parameter by design, so a plan made for a
padded batch descriptor stays valid under CUDA-graph replay. Sizing formulas and
a worked example are in [`api.md`](api.md).

## 6. Properties the engine can rely on

- The op allocates no memory, never synchronises with the host, reads no device
  data on the host, reads no environment variables and runs on the current
  stream of the device of `q`. It can be captured in a CUDA graph
  ([`api.md`](api.md), section 6).
- A `torch.compile` fake kernel is registered when `pth_flash_attn.ops` is
  imported.
- The GQA group size `Hq / Hkv` is at most 64.
- Window and causal semantics: bottom-right aligned causal masking, an optional
  sliding window and a non-causal mode, the last two at head size 128 only
  ([`api.md`](api.md), section 2).
- Results are deterministic for a given batch, not batch-invariant.

## 7. Validating an integration

An engine can reuse the oracles that validate the vLLM backend. They live in
`python/pth_testing/` of a source checkout (they are not installed with the
package) and run on CPU:

- `pth_testing.reference.make_int8_cache(...)` builds a paged int8 cache in
  vLLM's format (NHD or HND, padded pages, shuffled block ids, poisoned unused
  bytes, block ids above 2^31 bytes) with the views the op receives;
- `pth_testing.reference.ref_attention_dense(...)` computes fp64 attention on the
  dequantised cache, with the semantics of [`api.md`](api.md);
- `pth_testing.bound.bound(...)` returns a per-element error bound computed from
  the data; a correct integration has no element of `|out - reference|` above it.

```python
# CPU only. Run with PYTHONPATH=python from a source checkout.
import torch

from pth_testing.reference import make_int8_cache, make_queries, ref_attention_dense

D, Hkv, G, BS = 128, 2, 4, 16
cache = make_int8_cache([37, 90], Hkv=Hkv, D=D, block_size=BS, seed=0)
q, query_start_loc = make_queries([1, 3], Hq=Hkv * G, D=D)
ref = ref_attention_dense(q, cache.dense_all(), query_start_loc, D**-0.5)
print(tuple(cache.k_cache.shape), tuple(cache.k_scale.shape), tuple(cache.block_table.shape))
# (12, 16, 2, 128) (12, 16, 2) (2, 6)
print(cache.seq_lens.tolist(), tuple(ref.out.shape), tuple(ref.lse.shape))
# [37, 90] (4, 8, 128) (4, 8)
```

`cache.k_cache`, `cache.v_cache`, `cache.k_scale`, `cache.v_scale`,
`cache.block_table` and `cache.seq_lens` are exactly the op's cache and metadata
arguments. The GPU tests show complete comparisons:
[`tests/gpu/test_backend_op.py`](../tests/gpu/test_backend_op.py) runs the op
and the tested kernel library on the same inputs and requires equal bytes,
including after a CUDA-graph replay with changed lengths;
[`tests/gpu/test_backend_impl.py`](../tests/gpu/test_backend_impl.py) compares a
real mixed batch with the fp64 reference; and
[`tests/gpu/test_attn_grid.py`](../tests/gpu/test_attn_grid.py) applies the
per-element bound to the whole correctness grid. The printed shapes are the
output of this snippet on CPU (PyTorch 2.13.0, run in the development
container).

**Checklist.**

- [ ] Head size is 64, 128 or 256; `Hq` is a multiple of `Hkv`; `Hq / Hkv <= 64`.
- [ ] Windowed or non-causal layers are used only at head size 128.
- [ ] The GPU is sm80, sm86 or sm89 and the library is built for it.
- [ ] The writer stores K and V of the new tokens, with the section 3 convention,
  before the op runs.
- [ ] `seq_lens` includes the new tokens; `query_start_loc` starts at 0 and ends
  at `T`.
- [ ] Block ids are in range for every touched block.
- [ ] `wm` and `num_splits` come from `planner.plan_launch` for the same slice,
  and the workspace views have the planned shapes.
- [ ] Workspaces are reserved once, before the cache is sized, and not freed
  while a captured graph may replay.
- [ ] An fp64 comparison against `ref_attention_dense` passes within `bound(...)`
  for decode, verification, prefill with a cached prefix and a mixed batch.
- [ ] Graph replay with changed `seq_lens` equals eager execution.

## 8. Limits of portability

Beyond the op, the vLLM backend imports vLLM internals (the attention-backend
base classes, the Triton backend, the workspace manager and the cache
specification). An engine that is not vLLM does not use `vllm_backend.py` and
`plugin.py`; it calls the op and the planner, which import only PyTorch and the
standard library. The op is forward-only and targets sm80, sm86 and sm89.
