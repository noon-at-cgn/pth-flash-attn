<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Architecture

An overview of the components of pth-flash-attn and how one attention call
travels through them. The header comments of the source files carry the detail
and are the authority where this page is shorter. The op's argument contract is
in [`api.md`](api.md); what another engine has to supply is in
[`integration.md`](integration.md).

## Components and data flow

```
vLLM engine (--attention-backend CUSTOM)
 |  vllm.general_plugins entry point
 v
plugin.py                register(): class path under AttentionBackendEnum.CUSTOM
 v
vllm_backend.py          PthFlashAttnBackend / Impl (subclass vLLM's Triton classes)
 |  metadata builder: decode-like slice + prefill slice
 |  planner.plan_launch(): wm, bq, q_tiles_upper, num_splits (pure Python)
 |  Impl: workspace reserved at model load, views taken per forward
 v
ops.pth_flash_attn_fwd   torch op (stable ABI), fake kernel for torch.compile
 v
launch.cu                argument validation; tensors -> launch structs
 v
attn_launch.cu           kernel parameters, copy widths, loader choice, grid
 v
launch_attn_d{64,128,256}  (one translation unit per head size)
 |-- D = 256, up to 32 rows:  attn_dec_kernel   (decode and short verification)
 |-- otherwise:               attn_fwd_kernel   (16-row tile per warp)
 v
num_splits > 1:          attn_combine_kernel: merge the partial results
```

Everything below the backend is independent of vLLM: another engine calls
`ops.pth_flash_attn_fwd` directly and supplies what the backend supplies.

**Plugin.** `plugin.register()` runs in every vLLM process before backend
selection and records only a class path under `AttentionBackendEnum.CUSTOM`, the
enum member vLLM reserves for plugins, so a server that does not select the
backend never imports the CUDA library. If another plugin has already registered
a different class there, this one leaves it alone and logs a warning.

**Backend.** `PthFlashAttnBackend` and `PthFlashAttnImpl` subclass vLLM's Triton
classes: the cache specification, page layout, scale views and cache writer are
vLLM's own, so switching backends needs no cache migration; only the read side
is new. The metadata builder is this project's own. The Impl reserves the
split-KV workspace while the model loads, inside the memory window that sizes the
KV cache. Unsupported features are refused at selection or load with a reason,
and no environment variables are read.

**Planner** (`python/pth_flash_attn/planner.py`). The single source of every
launch-geometry number. Its inputs are host integers from the batch descriptor
and never a sequence length, because a CUDA graph replays one grid for every later
step with that descriptor. It returns the number of warps along M (`wm`), the
query tokens per row tile (`bq`), the q-tile upper bound and the split count
(`num_splits`) that lets one request fill the GPU; the workspace size follows
from the same formulas. The module imports only the standard library, so it runs
on a machine without CUDA. See [`api.md`](api.md).

**Op** (`python/pth_flash_attn/ops.py`, `csrc/pth_attn/torch_bindings.cpp`,
`launch.cu`). Registered through the libtorch stable ABI. `launch.cu` checks every
argument against the schema in [`api.md`](api.md) (the Python fake kernel runs
the metadata subset with the same messages, so `torch.compile` fails at trace
time for the same reason), then fills the launch structs and calls the launcher.
It allocates nothing, reads no device data on the host and never synchronises,
which makes a launch capturable in a CUDA graph. On CPU tensors it stops after
validation with "computes on CUDA tensors only".

**Launcher** (`attn_launch.cu`, `attn_combine.cu`). Takes byte strides as given
(never derived from shapes), picks each cache stream's `cp.async` width from
alignment (16, 8 or 4 bytes), and chooses how KV tiles are loaded: a
register-staged copy of the whole inline cell for vLLM's standard layout at
D = 256, separate K, V and scale streams otherwise. The grid is (q tiles, KV
heads, splits).

**Kernels.** One CTA computes one (row tile, KV head, split):

- `attn_fwd_kernel` puts 16 query rows per warp on the MMA's M dimension, with a
  64-token load tile and a two-stage pipeline. At D = 256 four producer warps
  copy KV tiles while four compute warps work; at D = 64 and 128 the compute warps
  issue their own `cp.async`. A sliding-window and non-causal variant exists at
  D = 128.
- `attn_dec_kernel` serves row tiles of at most 32 rows at D = 256 (decode and
  short verification). It swaps the MMA operands so that tokens fill M and the few
  query rows sit on N.
- With more than one split, each split writes an fp32 partial and its log2
  sum-exp; `attn_combine_kernel` merges them in a fixed split order and skips
  empty splits.

## Numerics

int8 K and V are converted to fp16 exactly in registers, Q is fp16 with a per-row
power-of-two prescale, every MMA is fp16 x fp16 with fp32 accumulation, softmax
runs in log2 units, and split partials carry their log2 sum-exp. Correctness is
checked against an fp64 reference on the dequantised cache with a per-element
bound derived from the data and the kernel's tiling (`python/pth_testing/`);
the bound is a pass or fail criterion, not a tuned tolerance. Results are not
batch invariant: the split count depends on the batch shape.

## References

The algorithms are re-implemented from the published work; no third-party source
is included.

- Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867):
  online softmax.
- Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135), and Dao,
  [arXiv:2307.08691](https://arxiv.org/abs/2307.08691): FlashAttention and
  FlashAttention-2 (tiling, rescaled running sums).
- Dao, Haziza, Massa & Sizov,
  [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
  (2023): split-KV with a log-sum-exp combine.
- Kim et al., [arXiv:2211.10017](https://arxiv.org/abs/2211.10017): the
  int8-to-fp16 magic-number conversion.
- Granlund & Montgomery, "Division by invariant integers using multiplication"
  (PLDI 1994), and the FastDivmod idea of NVIDIA CUTLASS (BSD-3-Clause; no code
  copied): division by a run-time divisor.
- Higham, rounding-error bounds for sums and products; Fasi et al. (2021) and
  Khattak & Mikaitis (2025), the accumulation behaviour of tensor-core
  fp16 x fp16 to fp32 products: the basis of the error bound in
  `python/pth_testing/bound.py` and `emulate.py`, whose comments give the details.

## Source layout

| Layer | Files | Role |
|---|---|---|
| Instructions | `ptx.cuh` | one thin wrapper per inline-PTX instruction |
| Data formats | `dequant.cuh`, `smem_layout.cuh`, `tile_traits.cuh`, `cache_geometry.cuh` | int8 to fp16 conversion, shared-memory layouts, tile constants, paging and int64 addresses |
| Building blocks | `tiling.cuh`, `softmax.cuh`, `mask.cuh`, `loader.cuh`, `combine_kernel.cuh` | QK and PV loops, online softmax, causal and window masks, KV tile copy, combine row math |
| Kernels | `attn_kernel.cuh`, `attn_decode.cuh`, `attn_combine.cu`, `instantiate_d{64,128,256}.cu` | the two attention kernels and the combine step |

Test-only sources live beside the shipped ones but are not in the extension
(`tools/build_flags.py`, `EXTENSION_SOURCES`): the test library's C ABI, the
fragment unit tests, the scalar reference kernel and the loader test kernels.
`tools/export_upstream.py --list` prints the shipped selection.

## Tests

| Tier | Where | What |
|---|---|---|
| CPU | `tests/cpu` | reference, bound, emulator, planner, op checks and fake kernel, backend selection and builder, tools |
| Simulator | `tests/sim` | PTX fragment, swizzle, conversion and V-path models |
| Host C++ | `tests/cpp` | g++ tests of `cache_geometry.cuh`, `tile_traits.cuh` and the primitives |
| Build gates | `tools/run_ci_local.sh attn` | kernel library build with 0 spills, register limit (255), SASS lint |
| GPU | `tests/gpu` | the correctness grid, sanitizers, CUDA-graph replay, backend op and Impl tests; needs a CUDA device |

How the tiers run, and what has been run on a GPU, is in [`build.md`](build.md).
Timing is a separate discipline: [`benchmark-methodology.md`](benchmark-methodology.md)
describes how a measurement becomes a claim; results are in
[`benchmarks.md`](benchmarks.md).
