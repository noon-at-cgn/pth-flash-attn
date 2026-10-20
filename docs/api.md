<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# API reference

The op `pth_flash_attn_fwd`, its argument contract, the launch planner and the
errors both raise. For calling the op from an engine other than vLLM, read
[`integration.md`](integration.md) after this page. Source of truth:
`csrc/pth_attn/ops.h` and `launch.cu` (C++), `python/pth_flash_attn/ops.py` and
`planner.py` (Python).

## 1. The op

The op is registered in the `pth_flash_attn` namespace through the libtorch
stable ABI, as an out-variant: it writes `out` (and, when the KV range is
split, the two workspaces) in place and returns nothing. This is the schema,
identical to the registration in `torch_bindings.cpp` and `ops.OP_SCHEMA`:

```text
pth_flash_attn_fwd(Tensor! out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor k_scale, Tensor v_scale, Tensor block_table, Tensor seq_lens, Tensor query_start_loc, int max_query_len, int wm, int num_splits, float softmax_scale, Tensor! ws_o, Tensor! ws_lse, int window=0, bool causal=True) -> ()
```

Call it as `torch.ops.pth_flash_attn.pth_flash_attn_fwd` after the library has
been loaded, or through the typed wrapper
`pth_flash_attn.ops.pth_flash_attn_fwd`, which takes the same arguments in the
same order (keyword arguments work) and raises a clear error if the library has
not been built. Importing `pth_flash_attn.ops` loads `_C.abi3.so` (an op
library without a Python init function, loaded with `torch.ops.load_library`),
registers the fake kernel for `torch.compile`, and defines `ops.is_available()`.
Importing the package alone (`import pth_flash_attn`) loads nothing.

The op is registered for CUDA and for CPU. On a CPU tensor it validates every
argument and then raises, so the checks can be tested without a GPU (section 5).

## 2. Arguments

`T` is the number of query tokens of the launch, `Hq` the query heads, `Hkv` the
KV heads, `D` the head size, `NB` the number of cache blocks, `BS` the tokens per
block, `R` the number of requests.

| Argument | Shape | Dtype | Notes |
|---|---|---|---|
| `out` | `[T, Hq, D]` | bfloat16 or float16, same as `q` | written in place; last-dimension stride 1 |
| `q` | `[T, Hq, D]` | bfloat16 or float16 | last-dimension stride 1; `D` is 64, 128 or 256 |
| `k_cache` | `[NB, BS, Hkv, D]` | int8 | arbitrary non-negative outer strides, last-dimension stride 1 |
| `v_cache` | `[NB, BS, Hkv, D]` | int8 | same sizes as `k_cache`; its strides may differ |
| `k_scale` | `[NB, BS, Hkv]` | float32 | one scale per token and KV head; dequantised K is `k_scale * k_cache` |
| `v_scale` | `[NB, BS, Hkv]` | float32 | same for V |
| `block_table` | `[>= R, max_blocks]` | int32 | kernel-block ids of each request, last-dimension stride 1; extra rows are ignored |
| `seq_lens` | `[R]` | int32 | KV length of each request **including** the query tokens of this call |
| `query_start_loc` | `[R + 1]` | int32 | cumulative query-token counts, starting at 0 and ending at `T` (rebased to this slice) |
| `max_query_len` | host int | | upper bound of any request's query length, in `[0, T]`; only chooses the row tile (section 3) |
| `wm` | host int | | warps along M: 1, 2 or 4; take it from the planner |
| `num_splits` | host int | | KV splits `S >= 1`; take it from the planner |
| `softmax_scale` | host float | | multiplies `q . k` before the softmax; finite and positive |
| `ws_o` | `[S', T', Hq, D]`, `S' >= S`, `T' >= T` | float32 | split partials; read and written only when `num_splits > 1` |
| `ws_lse` | `[S', T', Hq]` | float32 | log2-sum-exp of each partial; same rule |
| `window` | host int | | sliding window in tokens; 0 means none; default 0 |
| `causal` | host bool | | bottom-right causal (default) or non-causal |

**Shapes and strides.** `k_cache` and `v_cache` must have identical sizes.
`Hq` must be a positive multiple of `Hkv`; the group size `G = Hq / Hkv` is at
most 64 and at most `16 * wm`. All tensors must be on the device of `q`. Every
outer stride is free (NHD, HND, padded pages and block-outer layouts all work);
only the last dimension must be contiguous. Strides are taken from the tensors,
never derived from shapes. Strides and sizes must be non-negative and the byte
extent of every view must fit in int64.

**Alignment.** Base pointers and byte strides of all but the last dimension must
be multiples of 4 bytes for the int8 caches and the scales, and of 16 bytes for
`q`, `out` and `ws_o`. Tensors that come from PyTorch's allocator and whose
leading dimensions are whole rows of `D` elements satisfy this.

**Semantics.** Query token `i` of request `r` (`q_len = query_start_loc[r+1] -
query_start_loc[r]` tokens over `seq_len = seq_lens[r]` keys) sits at key
position `p = seq_len - q_len + i` (bottom-right alignment). Query head `h`
reads KV head `h // G`. The keys it sees are `[lo, hi]`, with

- `lo = max(0, p - window + 1)` if `window > 0`, else 0;
- `hi = p` if `causal`, else `min(seq_len - 1, p + window - 1)` if
  `window > 0`, else `seq_len - 1`.

The result is `softmax(softmax_scale * q . k_deq) v_deq` over those keys, with
`k_deq = k_scale * k_cache` and `v_deq = v_scale * v_cache`, written to `out` in
the dtype of `q`. A token whose visible set is empty does not occur for valid
inputs (`seq_len >= q_len`).

**Preconditions on device data (not checked).** The op reads `block_table`,
`seq_lens` and `query_start_loc` only on the device and cannot validate them on
the host without a synchronisation. The caller guarantees that block ids are in
`[0, NB)` for every block the request touches, `seq_lens[r] <= BS * max_blocks`,
`seq_lens[r] >= q_len` of the request, and that `query_start_loc` is
non-decreasing from 0 to `T`. Violations can read outside the cache; see
[`SECURITY.md`](../SECURITY.md). Scales should be positive normal floats (the
V-scale normalisation assumes it); vLLM's writer floors them at `1e-6`
([`integration.md`](integration.md)).

**Windowed and non-causal attention.** `window > 0` or `causal=False` selects
the windowed kernels, which exist for `D = 128` only; any other head size is an
error. With `window = 0` and `causal=False` every key is visible.

**Defaults.** `window` and `causal` default in the schema, so a call with the 15
earlier arguments resolves to the causal kernel without a window.

## 3. Launch semantics

- **Stream and device.** The launch goes to `q`'s device and to that device's
  current CUDA stream. Nothing else is touched.
- **Allocation and synchronisation.** The op allocates nothing, reads no device
  data on the host and does not synchronise.
- **Empty slices.** If `T == 0` or `R == 0` the op validates its arguments and
  returns without launching.
- **Two kernels at most.** One attention launch and, if `num_splits > 1`, one
  combine launch.
- **`max_query_len` and the tile.** The row tile is `min(floor(16 * wm / G),
  max_query_len)` query tokens when that keeps the planned `wm` (so a decode
  step with `G = 6` uses one 6-row tile rather than two), otherwise
  `floor(16 * wm / G)`. The kernels read each request's true length from
  `query_start_loc` on the device, and the grid has `T / bq + R` row tiles, which
  covers every request for any tile height, so a wrong `max_query_len` costs
  speed and never correctness. It must still lie in `[0, T]`.
- **`wm` and the tile must agree.** The op re-derives the warp count from the
  tile height and `G` and rejects a hand-made `wm` that disagrees. Planner
  output always agrees.
- **No environment variables** are read at run time.

## 4. The planner

`pth_flash_attn.planner` is the single source of launch geometry. It is pure
Python, imports only the standard library, runs on a CPU-only machine, and by
construction cannot receive a sequence length: every parameter is a keyword-only
host `int` or `bool` (a 0-d tensor is rejected), so the plan is a function of
the padded batch descriptor and stays valid for a CUDA graph however long the
sequences grow.

| Function | Returns |
|---|---|
| `plan_launch(*, num_tokens, num_reqs, max_query_len, G, Hkv, D, num_sms, is_prefill, ws_token_splits, window=0)` | a frozen `LaunchPlan` |
| `workspace_token_splits(*, max_num_seqs, q_dec_max, Hkv, D, num_sms)` | `W_ts = min(max_num_seqs, S_alone) * q_dec_max * S_alone` token-splits |
| `workspace_shapes(*, ws_token_splits, Hq, D)` | the two flat reservation shapes `(W_ts, Hq, D)` and `(W_ts, Hq)` |
| `workspace_bytes(*, ws_token_splits, Hq, D)` | `W_ts * Hq * (D + 1) * 4`, the bytes of both float32 buffers |
| `s_alone(*, Hkv, D, num_sms)` | splits that let one row tile fill the GPU: `ceil(C_target / Hkv)` |
| `c_target(*, D, num_sms)` | CTAs that fill one wave: `num_sms * CTAS_PER_SM[D]` (1, 2, 3 at D = 256, 128, 64) |
| `max_tokens_per_tile(G)` | `64 // G`, the query tokens of one request in a 64-row tile; the base of vLLM's decode threshold |
| `wm_for_rows(rows)` | smallest `wm` in {1, 2, 4} with `16 * wm >= rows` |
| `split_tile_range(*, split, num_splits, num_tiles)` | the KV-tile range `[start, end)` of one split, as the kernel computes it |

`LaunchPlan` fields: `wm`, `bq` (query tokens per row tile, `floor(16 wm / G)`),
`q_tiles_upper` (upper bound of the grid's x extent), `num_splits`, and the
descriptor it was made for (`num_tokens`, `G`, `Hkv`, `D`, `is_prefill`).
Properties: `Hq`, `rows_per_cta`, `grid` (`(q_tiles_upper, Hkv, num_splits)`),
`ws_token_splits_used`, and `ws_o_shape` / `ws_lse_shape`, the shapes of the
views to pass for this launch (`[S, T, Hq, D]` and `[S, T, Hq]`, with `S = 0`
when there is no split).

`is_prefill=False` plans a decode-like launch (every request has a short query:
decode or speculative verification); `is_prefill=True` plans a prefill launch
with `wm = 4`. The split count is clamped so that `S * T <= ws_token_splits`, and
for `window > 0` it is limited to the KV tiles one row tile can reach.

```python
# CPU only: runs without a GPU.
from pth_flash_attn.planner import (
    plan_launch,
    s_alone,
    workspace_bytes,
    workspace_shapes,
    workspace_token_splits,
)

# One RTX 3080 (68 SMs), two KV heads per rank, group size 6, head size 256.
num_sms, Hkv, G, D = 68, 2, 6, 256
Hq = Hkv * G

w_ts = workspace_token_splits(max_num_seqs=2, q_dec_max=4, Hkv=Hkv, D=D, num_sms=num_sms)
o_shape, lse_shape = workspace_shapes(ws_token_splits=w_ts, Hq=Hq, D=D)
print(s_alone(Hkv=Hkv, D=D, num_sms=num_sms), w_ts, o_shape, lse_shape)
# 34 272 (272, 12, 256) (272, 12)
print(workspace_bytes(ws_token_splits=w_ts, Hq=Hq, D=D) / 2**20)
# 3.199951171875

# A verification step: 1 request, 4 query tokens.
plan = plan_launch(
    num_tokens=4, num_reqs=1, max_query_len=4, G=G, Hkv=Hkv, D=D,
    num_sms=num_sms, is_prefill=False, ws_token_splits=w_ts,
)
print(plan)
# LaunchPlan(wm=2, bq=5, q_tiles_upper=1, num_splits=34, num_tokens=4, G=6, Hkv=2, D=256, is_prefill=False)
print(plan.grid, plan.ws_o_shape, plan.ws_lse_shape)
# (1, 2, 34) (34, 4, 12, 256) (34, 4, 12)

# A prefill chunk: 2048 tokens of one request.
plan = plan_launch(
    num_tokens=2048, num_reqs=1, max_query_len=2048, G=G, Hkv=Hkv, D=D,
    num_sms=num_sms, is_prefill=True, ws_token_splits=w_ts,
)
print(plan.wm, plan.bq, plan.num_splits, plan.ws_o_shape)
# 4 10 1 (0, 2048, 12, 256)
```

The printed values are the output of this snippet (planner of this repository,
pure arithmetic; no hardware is involved other than the SM count passed in).

**Sizing the workspace.** Reserve `workspace_shapes(...)` once, when the model
loads, with `max_num_seqs` the scheduler's maximum batch and `q_dec_max` the
speculative verification length (`1 + num_speculative_tokens`, or 1). For each
call, pass views of exactly the plan's `ws_o_shape` and `ws_lse_shape` (slices of
the flat reservation) and not the whole reservation: the op requires
`ws_o.size(1) >= T`. When `plan.num_splits == 1` the workspace is not read and an
empty float32 tensor on the right device may be passed. A launch whose
`S * T` exceeds the reservation cannot occur for planner output, because the
planner clamps `S`.

**Planner errors.** `TypeError` for a non-`int` or `bool` argument
(`num_tokens must be a host int, got float`; `is_prefill must be a bool`) and for
an unknown keyword (`plan_launch() got an unexpected keyword argument
'seq_len'`); `ValueError` for out-of-range values (`D must be one of (64, 128,
256), got 96`; `G must be <= 64, got N`; `max_query_len=N > num_tokens=M`;
`num_tokens=N cannot come from R requests of at most Q tokens`).

## 5. Errors raised by the op

Every message contains the prefix `pth_flash_attn_fwd: ` (the C++ checks may
also prepend their source location). A failed check raises `RuntimeError` in
Python. The checks run in the
order below; the Python fake runs the metadata checks (marked **M**) with the
same text, so `torch.compile` fails at trace time for the same reason. Checks
marked **C** need data pointers or a device and exist only in C++.

| Message (after the prefix) | Meaning |
|---|---|
| **M** `<name> must be <n>-D, got <m>-D` | wrong rank (`q`, `out`, caches 3-D/4-D as in section 2) |
| **M** `q must be bfloat16 or float16, got <dtype>` | query dtype |
| **M** `head size D = q.size(2) must be 64, 128 or 256, got <D>` | unsupported head size |
| **M** `<name> must be <dtype>, got <dtype>` | dtype of `out`, the caches (int8), scales (float32), metadata (int32), workspaces (float32) |
| **M** `<name>.size(<dim>) must equal <what> = <value>, got <value>` | a size that must match another tensor |
| **M** `<name> must have stride 1 in its last dimension, got <s>` | last dimension not contiguous |
| **M** `Hq = <a> must be a positive multiple of Hkv = <b>` | query heads not divisible by KV heads |
| **M** `wm must be 1, 2 or 4, got <n>` | invalid warp count |
| **M** `G = Hq/Hkv = <g> exceeds the supported maximum 64` | group too large |
| **M** `G = Hq/Hkv = <g> must be <= 16*wm = <n>` | `wm` too small for the group |
| **M** `block_table has <n> rows, fewer than num_reqs = <r>` | block table too short |
| **M** `max_query_len must be in [0, T = <t>], got <n>` | |
| **M** `num_splits must be >= 1, got <n>` | |
| **M** `softmax_scale must be finite and > 0, got <x>` | |
| **M** `window must be >= 0, got <n>` | |
| **M** `sliding-window and non-causal attention need head size D = 128, got <D>` | windowed kernels exist at D = 128 only |
| **M** `<name> must be on the same device as q` | mixed devices |
| **M** `ws_o must have size(0) >= num_splits = <s> and size(1) >= T = <t>, got [<a>, <b>, ...]` (and the same for `ws_lse`) | workspace view too small (only checked when `num_splits > 1`) |
| **C** `<name> has negative strides or a byte extent that overflows int64` | |
| **C** `<name> data pointer must be <n>-byte aligned` | base pointer alignment (4 or 16 bytes) |
| **C** `<name> byte stride of dim <i> (<b>) must be a multiple of <n>` | stride alignment |
| **C** `workspace (ws_o, ws_lse) must be allocated when num_splits > 1` | null workspace pointers |
| **C** `arguments are valid, but the op computes on CUDA tensors only (the CPU registration exists to test the argument checks)` | a valid call on CPU tensors |
| **C** `needs compute capability 8.x (sm_80/86/89), device has major <n>` | sm90 and newer, or older than sm80 |
| **C** `wm = <wm> is not the tile height of G = <g> (rows <r> need wm = <w>)` | hand-made `wm` inconsistent with the row tile |
| **C** `attention launch: <reason>` or `combine launch: <reason>` | the CUDA launch failed |

From the Python wrapper only: `op library <path> is not built; ...` when
`_C.abi3.so` is missing ([`troubleshooting.md`](troubleshooting.md)).

## 6. CUDA graphs and `torch.compile`

**Graph capture.** A launch can be captured. The op allocates nothing, never
synchronises and takes everything that depends on data from device tensors, so
one captured graph replays correctly for any sequence length:

- Capture with persistent tensors and replay with the same tensors. The
  contents of `q`, `seq_lens`, `query_start_loc`, `block_table` and the cache
  may change between replays; their addresses, sizes and strides may not.
- The host arguments are baked into the graph: `T` (the size of `q`), `R`,
  `max_query_len`, `wm`, `num_splits`, `softmax_scale`, `window` and `causal`.
  Capture one graph per padded batch descriptor, and obtain `wm` and
  `num_splits` from `plan_launch` for that descriptor, not from any length.
- The `seq_lens` and `query_start_loc` read at replay must still satisfy the
  preconditions of section 2. A capture or warm-up run walks the KV range implied
  by `seq_lens`, so a short `seq_lens` keeps it quick; the vLLM backend does
  that.
- Allocate the workspaces before capture and keep them alive. A captured graph
  stores the raw device address of every tensor argument, so none of them may
  be freed or reallocated before the last replay.
- The capture stream must be the current stream of `q`'s device.

**`torch.compile`.** Importing `pth_flash_attn.ops` registers a fake kernel that
runs the metadata checks of section 5 and returns `None`; the in-place writes
are declared by `Tensor!` in the schema. A bad call therefore fails during
tracing with the message it would give at run time.

## 7. Numerics

Results for one batch are deterministic (the combine step uses a fixed order)
but not batch-invariant: the split count depends on the batch shape. Accuracy is
checked against attention computed in fp64 on the dequantised cache, with a
per-element bound derived from the data and the kernel's tiling
([`architecture.md`](architecture.md#numerics)).
