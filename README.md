<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# pth-flash-attn

[![CI](https://github.com/noon-at-cgn/pth-flash-attn/actions/workflows/ci.yml/badge.svg)](https://github.com/noon-at-cgn/pth-flash-attn/actions/workflows/ci.yml)
[![Licence: Apache-2.0](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10-3.14 declared, 3.12 tested](https://img.shields.io/badge/python-3.10--3.14%20declared%2C%203.12%20tested-blue.svg)](pyproject.toml)
[![NVIDIA sm80 / sm86 / sm89](https://img.shields.io/badge/NVIDIA-sm80%20%7C%20sm86%20%7C%20sm89-76b900.svg)](docs/build.md)

## What it is

A FlashAttention-style CUDA kernel that reads vLLM's `int8_per_token_head`
paged KV cache in place on NVIDIA Ampere and Ada GPUs (sm80, sm86, sm89: A100,
GeForce RTX 30xx and 40xx, L4, L40S). It reads the int8 bytes and the inline fp32
scales with tensor cores and never builds a dequantised copy of the cache. It is
a PyTorch stable-ABI op (`torch.ops.pth_flash_attn.pth_flash_attn_fwd`) with an
optional vLLM attention-backend plugin on top.

The kernel is forward-only. It serves decode, speculative verification (MTP and
DFlash-style drafters), chunked prefill and mixed batches, with split-KV so that
a single long request fills the GPU.

**Status.** Version 0.1.0, alpha. Every measurement comes from sm86 (two
RTX 3080 20 GB boards; one RTX 3090 in independent runs by the HyperQwen
maintainer) and one model family (see [Results](#results) and
[Requirements and limits](#requirements-and-limits)).

## The problem it solves

In vLLM 0.30 and 0.31 the `int8_per_token_head` KV cache needs roughly half the
memory of a 16-bit cache (2D + 8 bytes per token and KV head, against 4D bytes
for K and V), but the only attention backend that reads it is the Triton
backend. This project adds a hand-written reader and changes nothing else: the
cache layout, the quantising writer and the cache specification stay vLLM's own
(the Triton path writes the cache; see
[`docs/integration.md`](docs/integration.md) for the split of responsibilities).
Switching between `TRITON_ATTN` and this backend therefore needs no cache
migration.

## Results

Every measurement quoted here carries its hardware, power limit, software
versions and source. The labels are those of the benchmarks page: K1 kernel
level, E1 and E2 end to end on two RTX 3080 20 GB, A1 accuracy, M1 an
independent run on one RTX 3090. The full tables, the caveats that belong with
them and the method are in [`docs/benchmarks.md`](docs/benchmarks.md) and
[`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).

"int8 + pth" is an int8 KV cache read by this kernel; the comparison arms are
other attention paths, as built and configured by
[HyperQwen](https://github.com/syv-ai/HyperQwen) (a vLLM-based serving stack
that vendors this kernel): Triton unified attention on an int8 cache, and
FlashAttention-2 or FlashInfer on an fp8 cache.

<!-- HEADLINE_TABLE:start -->
| Label | Measurement | Result | Source |
|---|---|---|---|
| K1 | Windowed drafter attention, D=128, one RTX 3080 20 GB, 220 W power limit, median of 200 CUDA-graph samples | 6.40–9.28× faster than vLLM's Triton `unified_attention` over four shapes (18.69 µs against 170.89 µs per layer at the TP=2 shape, cold cache) | [#338](https://github.com/syv-ai/HyperQwen/pull/338) |
| E2 | `SPEC=mtp`, two RTX 3080 20 GB, TP=2, 220 W power limit, vLLM 0.31.0, one sitting, N=1 | Time per decode pass is 0.449 / 0.393 / 0.375 of fp8 + FlashInfer's at 128k / 32k / 8k (decode 110.4 against 46.5 tok/s at 128k). Kernel and CUDA-graph mode are not separated. | [#345](https://github.com/syv-ai/HyperQwen/pull/345) |
| E1 | `SPEC=dflash2`, two RTX 3080 20 GB, TP=2, 220 W power limit, vLLM 0.31.0, one sitting, N=1 | Time per decode pass is 0.770 / 0.832 / 0.913 of fp8 + FA2's at 128k / 32k / 8k (decode 115.1 against 81.5 tok/s at 128k). | [#337](https://github.com/syv-ai/HyperQwen/pull/337), [#338](https://github.com/syv-ai/HyperQwen/pull/338) |
| A1 | Teacher-forced prompt logprobs against a bf16-KV reference, 32k / 128k documents | Mean \|Δ logprob\| 0.0207 / 0.0193 (int8 + pth) against 0.0360 / 0.0338 (fp8 + FA2); 0.0214 / 0.0193 against 0.0356 / 0.0333 (fp8 + FlashInfer) on the MTP checkpoint | [#338](https://github.com/syv-ai/HyperQwen/pull/338), [#345](https://github.com/syv-ai/HyperQwen/pull/345) |
| M1 | Independent runs by the HyperQwen maintainer, one RTX 3090, TP=1, 250 W, vLLM 0.31.0, one boot per arm (M1a, M1b) or two (M1c) | `SPEC=mtp` decode at ~66k 84.5 against 77.5 tok/s (+9 %); `SPEC=dflash2` 62.4 against 55.6 tok/s (+12.2 %) at ~66k and 53.2 against 48.0 tok/s (+10.8 %) at ~100k; merged `SPEC=mtp` default, two boots per arm: 84.8 / 84.7 against 75.6 / 75.5 tok/s at ~66k (+12.2 %) and 78.5 / 78.5 against 68.9 / 68.8 tok/s at ~100k (+14.0 %). Kernel and CUDA-graph mode are not separated in M1c | [#325 comment](https://github.com/syv-ai/HyperQwen/pull/325#issuecomment-6250055160), [#339 comment](https://github.com/syv-ai/HyperQwen/pull/339#issuecomment-6262787058), [#345 comment](https://github.com/syv-ai/HyperQwen/pull/345#issuecomment-6290008416) |

The comparison arms are ahead in places: with `SPEC=dflash2` the fp8 + FA2 build prefills a cold 128k prompt faster (114.1 s against 122.1 s, one request per arm, [#338](https://github.com/syv-ai/HyperQwen/pull/338)) and holds a larger KV pool (517,656 against 504,770 tokens). Every row is a single machine, a single model family and, for M1, one or two boots per arm; the section "Read before the numbers" of the benchmarks page lists the limits.
<!-- HEADLINE_TABLE:end -->

Where the numbers come from: the two-RTX 3080 20 GB rows (K1, E1, E2, A1) were
measured on memory-modified 20 GB boards, not a retail SKU, at a 220 W power
limit. The end-to-end runs used pre-release builds proved to give identical SASS
to this release. K2a, K2b and K3 are the kernel-level tables of
[the benchmarks page](docs/benchmarks.md): K2b and K3 come from pre-release
builds whose commits are not recorded, and K2a is a repeat of the K2 shapes with
`bench/op_vs_triton.py` on a pre-publication tree whose kernel and package
sources differ from this release only in comments, docstrings and the text of two
error messages (compiled SASS identical). Four K2b rows did not reproduce in K2a;
the benchmarks page quotes both sets.

## Quick start

### With vLLM

Build and install from a checkout, in an environment that already has PyTorch
2.13 (CUDA 13), nvcc 13 and g++ (`setup.py` imports torch, hence
`--no-build-isolation`):

```bash
git clone https://github.com/noon-at-cgn/pth-flash-attn
cd pth-flash-attn
pip install --no-build-isolation .
python -c "from pth_flash_attn import ops; print(ops.is_available())"   # True
```

The package's `vllm.general_plugins` entry point registers the backend as
`AttentionBackendEnum.CUSTOM` in every vLLM process, importing no CUDA code
until the backend is selected. Select it explicitly together with the int8 cache:

```bash
vllm serve <model> \
  --kv-cache-dtype int8_per_token_head \
  --attention-backend CUSTOM
```

A speculative drafter takes its attention backend from the speculative
configuration, not from `--attention-backend`. To run the drafter's attention
layers (the MTP layer, or a DFlash-style drafter) on this kernel as well, add
`"attention_backend": "CUSTOM"` to that configuration:

```bash
vllm serve <model> \
  --kv-cache-dtype int8_per_token_head \
  --attention-backend CUSTOM \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "CUSTOM"}'
```

Dropping the key from the speculative configuration silently puts the drafter on
another backend. Nothing in vLLM selects `CUSTOM` on its own. Plugin mechanics,
version support, the behaviour when another plugin owns `CUSTOM`, and a public
deployment example ([HyperQwen](https://github.com/syv-ai/HyperQwen), which
vendors this kernel and documents its use with MTP and DFlash drafters in its
[`pthfa/README.md`](https://github.com/syv-ai/HyperQwen/blob/main/pthfa/README.md))
are in [`docs/vllm.md`](docs/vllm.md).

### From PyTorch

The op needs CUDA tensors, a plan from the pure-Python planner, and two float32
workspaces when the plan splits the KV range. The example below is a GPU example
(it cannot run on CPU); the full argument contract is in
[`docs/api.md`](docs/api.md).

```python
import math

import torch

from pth_flash_attn import ops
from pth_flash_attn.planner import (
    plan_launch,
    workspace_shapes,
    workspace_token_splits,
)

dev = torch.device("cuda")
D, Hkv, G, BS = 128, 2, 4, 16  # head size, KV heads, GQA group, block size
Hq = Hkv * G
seq_len, q_len = 1000, 1  # KV length including the new token; decode step
num_blocks = math.ceil(seq_len / BS)
scale = D**-0.5


def quantize_rows(x):
    """Per (token, head) int8 quantisation with vLLM's writer convention."""
    xf = x.float()
    s = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    q = xf * (1.0 / s).unsqueeze(-1)
    q = torch.where(q >= 0, q + 0.5, q - 0.5).clamp(-128.0, 127.0)
    return q.to(torch.int8), s  # the int8 conversion truncates toward zero


# A contiguous paged cache: K, V int8 [NB, BS, Hkv, D]; scales f32 [NB, BS, Hkv].
k_src = torch.randn(num_blocks * BS, Hkv, D, device=dev, dtype=torch.bfloat16)
v_src = torch.randn(num_blocks * BS, Hkv, D, device=dev, dtype=torch.bfloat16)
k_q, k_s = quantize_rows(k_src)
v_q, v_s = quantize_rows(v_src)
k_cache, v_cache = (t.view(num_blocks, BS, Hkv, D) for t in (k_q, v_q))
k_scale, v_scale = (t.view(num_blocks, BS, Hkv) for t in (k_s, v_s))

# Batch metadata: one request whose blocks are 0 .. num_blocks - 1.
block_table = torch.tensor([list(range(num_blocks))], dtype=torch.int32, device=dev)
seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
query_start_loc = torch.tensor([0, q_len], dtype=torch.int32, device=dev)

# Plan the launch from host integers, and reserve the workspace once.
num_sms = torch.cuda.get_device_properties(dev).multi_processor_count
w_ts = workspace_token_splits(
    max_num_seqs=8, q_dec_max=4, Hkv=Hkv, D=D, num_sms=num_sms
)
o_shape, lse_shape = workspace_shapes(ws_token_splits=w_ts, Hq=Hq, D=D)
ws_o_buf = torch.empty(o_shape, dtype=torch.float32, device=dev)
ws_lse_buf = torch.empty(lse_shape, dtype=torch.float32, device=dev)

plan = plan_launch(
    num_tokens=q_len,
    num_reqs=1,
    max_query_len=q_len,
    G=G,
    Hkv=Hkv,
    D=D,
    num_sms=num_sms,
    is_prefill=False,
    ws_token_splits=w_ts,
)
if plan.num_splits > 1:  # views of exactly this launch's size
    ws_o = ws_o_buf.view(-1)[: math.prod(plan.ws_o_shape)].view(plan.ws_o_shape)
    ws_lse = ws_lse_buf.view(-1)[: math.prod(plan.ws_lse_shape)].view(plan.ws_lse_shape)
else:  # no combine step: the workspace is not read
    ws_o = ws_lse = torch.empty(0, dtype=torch.float32, device=dev)

q = torch.randn(q_len, Hq, D, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(q)
ops.pth_flash_attn_fwd(
    out, q, k_cache, v_cache, k_scale, v_scale,
    block_table, seq_lens, query_start_loc,
    q_len, plan.wm, plan.num_splits, scale, ws_o, ws_lse,
)

# Check against attention over the dequantised cache (value = scale * int8).
# With one query token every key is visible, so no mask is needed.
k_deq = (k_cache.float() * k_scale.unsqueeze(-1)).reshape(-1, Hkv, D)[:seq_len]
v_deq = (v_cache.float() * v_scale.unsqueeze(-1)).reshape(-1, Hkv, D)[:seq_len]
k_deq, v_deq = (t.repeat_interleave(G, dim=1) for t in (k_deq, v_deq))
scores = torch.einsum("qhd,shd->hqs", q.float(), k_deq) * scale
ref = torch.einsum("hqs,shd->qhd", scores.softmax(-1), v_deq)
print("max abs difference:", (out.float() - ref).abs().max().item())
```

The cache can equally be a strided view into a larger allocation (vLLM's packed
cell is one): only the last dimension must be contiguous. The planner is pure
Python and runs on CPU.

### From another engine

The op is not tied to vLLM. Any engine that keeps a paged int8 KV cache can call
it, provided that it supplies:

1. K and V caches as int8 views `[blocks, block_size, kv_heads, head_dim]`
   (arbitrary outer strides, contiguous head dimension);
2. float32 scales per token and KV head as `[blocks, block_size, kv_heads]` for
   K and V, with dequantised value `scale * int8`;
3. an int32 block table, int32 `seq_lens` (including the new tokens) and int32
   `query_start_loc`;
4. two float32 workspaces sized by the planner, allocated by the caller;
5. a writer that quantises new tokens with the same scale convention.

The contract, the quantiser, the preconditions and a validation checklist are in
[`docs/integration.md`](docs/integration.md).

## Architecture

A call travels from the vLLM backend (or directly from an engine) through the
pure-Python planner, which chooses the launch geometry from host integers only,
to the torch op. The op validates its arguments and launches one attention kernel
and, when the KV range is split, one combine kernel. A row-tile kernel serves
prefill, mixed batches and verification; a swap-AB decode kernel serves short
queries at head size 256; the combine kernel merges split partials in a fixed
order. Nothing is allocated or synchronised by the op, so a launch can be
captured in a CUDA graph. [`docs/architecture.md`](docs/architecture.md) has the
component diagram and the source layout.

## API

`pth_flash_attn_fwd(out, q, k_cache, v_cache, k_scale, v_scale, block_table,
seq_lens, query_start_loc, max_query_len, wm, num_splits, softmax_scale, ws_o,
ws_lse, window=0, causal=True)` writes `out` in place; `pth_flash_attn.planner`
provides `plan_launch`, `workspace_shapes` and the other launch-geometry
functions. Shapes, dtypes, preconditions, error messages and CUDA-graph rules are
in [`docs/api.md`](docs/api.md).

## Requirements and limits

Accepted by the op and the backend (everything else is refused with a reason at
selection or load):

| Item | Accepted | Refused |
|---|---|---|
| GPU | NVIDIA compute capability 8.0, 8.6, 8.9 (SASS only, no PTX) | sm75 and older; sm90 and newer (not JIT-compiled) |
| Head size D | 64, 128, 256 | any other |
| Sliding window, non-causal | D = 128 only | D = 64 and 256; R-SWA; chunk lookback |
| Query dtype | bfloat16, float16 | any other |
| KV cache | `int8_per_token_head` only | fp8, int4, 16-bit caches |
| GQA group Hq / Hkv | 1 to 64 | more than 64 |
| Block size | any multiple of 16 (the op accepts any size of at least 1) | other sizes in vLLM |
| Attention kind | decoder self-attention, forward pass | encoder or cross attention, backward pass |
| Features | CUDA graphs, `torch.compile` (fake kernel), prefix caching, hybrid KV groups | ALiBi, attention sinks, logits soft cap, bidirectional multimodal prefix, DCP and PCP, cascade attention, fused output quantisation, dual-batch overlap, batch invariance |

Exercised on a GPU. Only sm86 has run, on an RTX 3080 20 GB (the test tiers) and,
through HyperQwen, on RTX 3080 20 GB and RTX 3090 (the end-to-end runs of the
[benchmarks page](docs/benchmarks.md)):

| Item | Shapes and settings that ran | Where |
|---|---|---|
| Head size D | 64, 128 and 256 in the correctness grid; 256 (target) and 128 (windowed drafter) end to end | `tests/gpu`, E1, E2 |
| Heads per rank (Hq / Hkv) | grid: 12/2, 16/2, 32/2, 8/2, 2/2, 8/1, 4/1 at D = 256; 8/2, 12/2, 16/2, 16/4, 24/4, 32/2, 32/4, 32/8 and others at D = 128; 12/2 for the target and 16/4 for the drafter on the 3080 pair (TP=2); 24/4 for the target and 32/8 for the drafter at TP=1 | `tests/gpu`, E1, E2, K1, M1 |
| GQA group | 1, 2, 4, 6, 8, 16 | correctness grid |
| Sliding window, non-causal | D = 128, windows 1 to 6,000 tokens (64, 65, 300, 2,047, 2,048, 4,096, 6,000 among them), causal and non-causal, 70 cases; 2,048 with 16/4 and 32/8 heads for the drafter | window grid, K1, E1 |
| Block size | 16, 32, 64, 528, 816 (the D = 256 page) and 864 (the D = 128 page) | correctness grid, E1, E2 |
| Query dtype | bfloat16 (313 of 328 grid cases) and float16 (15) | correctness grid |
| Context and query length | KV lengths up to 262,144 in the grid; end to end up to 128,000-token prompts; query chunks up to 8,192 in the prefill tests | grid, prefill tests, E1, E2 |
| Layout | NHD and HND views, padded pages, storage offsets, high block ids | correctness grid |
| Features | CUDA-graph replay equal to eager execution; full and piecewise graphs, prefix caching and a hybrid model in E1 and E2 | graph tests, E1, E2 |

Not exercised on a GPU: sm80 and sm89, and, end to end, any model other than the
one Qwen3.8-27B family, any engine other than vLLM 0.31.0 and tensor-parallel
sizes other than 1 and 2. The sm80 and sm89 builds are compile-checked only: the
compile, register, spill and SASS gates and the CPU suites pass, and no GPU run
exists on those two architectures.

**Validation.** The GPU test tier (the correctness grid, `compute-sanitizer`,
CUDA-graph replay, the backend and production-shape tests) ran on sm86 on a
pre-publication tree whose kernel and package sources differ from this release
only in comments, docstrings and the text of two error messages (compiled SASS
identical); the record is in
[`docs/build.md`](docs/build.md#7-what-has-been-run-on-a-gpu).

**Software.** Linux; Python 3.10 to 3.14 declared (tested with 3.12); PyTorch
2.11 or newer built for CUDA 13 (declared; tested with 2.13.0+cu130, and the
library links `libcudart.so.13`); nvcc 13 and g++ to build. The vLLM backend is
tested against unmodified vLLM 0.31.0 (the development image, which CI uses) and
was written against the vLLM 0.30.0 sources, which this repository's CI does not
run; other versions and vLLM-derived forks are untested and may need small
adaptations, because the backend imports vLLM internals. Details:
[`docs/build.md`](docs/build.md), [`docs/vllm.md`](docs/vllm.md).

**Other limits.**

- Forward only; no backward pass, no training use.
- The cache must already be `int8_per_token_head`: one float32 scale per token
  and KV head for K and for V. The op does not quantise; the engine writes the
  cache.
- Results are not batch invariant: the split count depends on the batch shape,
  so a request's output can differ in the last bits between batch compositions.
- Only sm86 has been measured. See [`docs/benchmarks.md`](docs/benchmarks.md)
  for what was and was not measured.

## Build and install

[`docs/build.md`](docs/build.md) covers requirements, installation, the dev
container, the local CI script and the test tiers. The build needs no GPU:
`TORCH_CUDA_ARCH_LIST` narrows the architectures (for example `8.6`) and
`MAX_JOBS` bounds the compile parallelism. A wheel or PyPI release is not
provided.

| Path | What |
|---|---|
| `csrc/pth_attn/` | CUDA and C++ sources: the op, the launcher, the kernel headers and per-head-size instantiations, and test-only libraries |
| `python/pth_flash_attn/` | The installed package: op wrapper and fake kernel, launch planner, vLLM backend, plugin entry point |
| `python/pth_testing/` | Test oracles (not installed): fp64 reference, error bound, bit-level emulator |
| `tests/` | `cpu`, `sim`, `cpp` (CI) and `gpu` (needs a CUDA device) |
| `tools/` | Build and CI tools; `build_flags.py` is the single source of compiler flags |
| `bench/` | Benchmark harness, hardware-ceiling kernels, result tables |
| `docker/` | Development and sanitizer images |
| `docs/` | Architecture, API, integration, vLLM, build and test, benchmarks ([index](docs/README.md)) |

Questions and answers are in [`docs/faq.md`](docs/faq.md) and
[`docs/troubleshooting.md`](docs/troubleshooting.md). See also
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md),
[`CHANGELOG.md`](CHANGELOG.md) and [`CITATION.cff`](CITATION.cff). Support is
best effort.

## Licence

Apache-2.0 ([`LICENSE`](LICENSE)). No third-party source is included in the built
or installed code ([`NOTICE`](NOTICE)).
