<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Using the kernel with vLLM

This package ships a vLLM attention backend as an out-of-tree plugin. It serves
the `int8_per_token_head` KV cache with the kernel of this repository and changes
nothing else in vLLM: no vLLM file is patched, and the cache specification,
layout and writer stay vLLM's own. This page covers installation, the flags,
supported versions, what is refused, and conflicts.

## 1. Install and select

Install the package into the environment that runs vLLM (see
[`build.md`](build.md); a CUDA 13 toolkit and g++ are needed to build, no GPU):

```bash
pip install --no-build-isolation .
```

The package declares a `vllm.general_plugins` entry point,
`pth_flash_attn = pth_flash_attn.plugin:register`. vLLM calls it in every process
it starts (the API server's engine core and every tensor-parallel worker) before
it selects an attention backend. `register` records the class path
`pth_flash_attn.vllm_backend.PthFlashAttnBackend` under
`AttentionBackendEnum.CUSTOM` and nothing else: the CUDA library and vLLM's
Triton modules are imported only when the backend is selected. A server that does
not select `CUSTOM` pays nothing and cannot be broken by this package; `CUSTOM`
is in none of vLLM's automatic selection lists.

Select the backend explicitly, together with the int8 cache:

```bash
vllm serve <model> \
  --kv-cache-dtype int8_per_token_head \
  --attention-backend CUSTOM
```

If the environment variable `VLLM_PLUGINS` is set, vLLM loads only the plugins it
names, so it must include `pth_flash_attn`.

To check the registration without starting a server (CPU; needs vLLM and this
package installed):

```python
from vllm.plugins import load_general_plugins
from vllm.v1.attention.backends.registry import AttentionBackendEnum

load_general_plugins()
print(AttentionBackendEnum.CUSTOM.is_overridden(), AttentionBackendEnum.CUSTOM.get_path())
# True pth_flash_attn.vllm_backend.PthFlashAttnBackend
```

`tools/check_installed_package.py` runs a fuller version of this check against an
installed copy (package location, op library, entry point, backend class).

## 2. Speculative decoding: the drafter's backend

A speculative drafter takes its attention backend from the speculative
configuration, not from `--attention-backend`. To run the drafter's attention
layers (an MTP layer, or a DFlash-style drafter) on this kernel as well, add
`"attention_backend": "CUSTOM"` to the JSON passed to `--speculative-config`:

```bash
vllm serve <model> \
  --kv-cache-dtype int8_per_token_head \
  --attention-backend CUSTOM \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "CUSTOM"}'
```

Notes:

- Leaving the key out silently runs the drafter on another backend. Check the
  speculative configuration whenever the drafter's speed looks wrong.
- Do not set `kv_cache_dtype` in the speculative configuration unless the
  drafter needs a different cache: when the key is absent the drafter's cache
  keeps the target's `int8_per_token_head` (this is vLLM 0.31's behaviour as
  recorded by the project that deploys this kernel; see section 8).
- A drafter with sliding-window or non-causal attention layers (the DFlash-style
  drafter measured with this kernel has window 2048, non-causal, head size 128)
  needs head size 128: the windowed kernels exist for D = 128 only.
- Speculative verification (`1 + num_speculative_tokens` query tokens per
  request) is treated as decode: the builder's decode threshold is
  `max(64 // G, 1 + k)` and the workspace is sized for `1 + k` tokens.
- `"method"` and `num_speculative_tokens` above are an example; use the method
  and token count of the model being served.

## 3. Supported vLLM versions

The backend is tested against unmodified vLLM 0.31.0, the version of the
development image (`pip install .[vllm]` installs it), whose CPU backend tests
run in CI. It was written against the vLLM 0.30.0 sources; this
repository's CI does not run 0.30.0. The plugin was also installed (`pip install --no-build-isolation .`), from the pre-publication export `bd4b3c8` (see [`build.md`](build.md#7-what-has-been-run-on-a-gpu)), next to unmodified vLLM 0.31.0 on an RTX 3080 20 GB (sm86), registered `CUSTOM` through its entry point and served a 1B Llama model (head size 64) with `--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head`: token-identical to vLLM's Triton int8 path on 5 of 6 prompts, the same KV pool as that path and 1.88 times the pool of vLLM's default bf16 cache. End-to-end serving at head sizes 128 and 256 is shown only by the HyperQwen runs of the [benchmarks page](benchmarks.md). Other vLLM versions and vLLM-derived forks
are untested and may need small adaptations: the backend is a subclass of vLLM's
Triton backend and imports vLLM internals, namely

- `vllm.v1.attention.backend` (`AttentionMetadataBuilder`,
  `CommonAttentionMetadata`, `AttentionCGSupport`, `AttentionType`,
  `MultipleOf`) and `vllm.v1.attention.backends.registry`;
- `vllm.v1.attention.backends.triton_attn` (`TritonAttentionBackend` and
  `TritonAttentionImpl`, including its cache-view helpers) and
  `vllm.v1.attention.backends.utils` (`split_decodes_and_prefills`,
  `get_num_attention_heads_from_layers`);
- `vllm.v1.kv_cache_interface` (`AttentionSpec`, `KVQuantMode`),
  `vllm.v1.worker.workspace` (the workspace manager) and
  `vllm.v1.worker.ubatch_utils`;
- `vllm.config`, `vllm.platforms` and `vllm.logger`.

A vLLM release that renames or moves any of these breaks the import of
`vllm_backend.py` at selection time, which is the only time it is imported. The
op and the planner do not import vLLM at all
([`integration.md`](integration.md)).

The cell layout of the `int8_per_token_head` cache is the same in 0.30.0 and
0.31.0. Python 3.10 to 3.14 and
PyTorch 2.11 or newer are required by the package; the tested stack is PyTorch
2.13.0 with CUDA 13.0.

## 4. What is served and what is refused

Served: decoder self-attention; decode, speculative verification, chunked
prefill and mixed batches; prefix caching; preemption; hybrid KV-cache groups
(padded pages); NHD, HND and strided layouts; block sizes that are a multiple of
16; bf16 and fp16 models; head size 64, 128 and 256; GQA groups of 1 to 64;
sliding window and non-causal attention at head size 128; CUDA graphs (full
graphs for uniform decode batches, piecewise for batches with a prefill).

An explicit `--attention-backend CUSTOM` that vLLM cannot satisfy fails with the
reasons (there is no silent fallback to another backend), at one of three
points:

| Where | Condition | Reason text |
|---|---|---|
| Backend selection | compute capability not 8.x, or the op library not built | reported as unsupported capability |
| | KV cache dtype other than `int8_per_token_head`; model dtype other than bf16 or fp16; head size not 64, 128 or 256; block size not a multiple of 16; not decoder attention | vLLM's standard reasons |
| | sinks, bidirectional multimodal prefix, ALiBi (sqrt), R-SWA | the backend's `supports_*` flag is false |
| | sliding window or non-causal attention at head size 64 or 256 | `sliding window at head size <D> (the windowed kernels exist for D = 128 only)` (or `non-causal attention ...`) |
| Model load (Impl) | ALiBi, a window below 1 or at head size other than 128, logits soft cap, attention sinks, chunk lookback, attention type other than decoder, KV dtype, head size | `pth_flash_attn does not support: <list>` |
| | context parallelism (DCP or PCP) | `pth_flash_attn: DCP/PCP are not supported` |
| | `Hq` not a multiple of `Hkv`, or group above 64 | `pth_flash_attn: Hq = <a> must be a positive multiple of Hkv = <b>`, `... G = Hq/Hkv = <g> exceeds 64` |
| Metadata build | dual-batch overlap | `pth_flash_attn: dual-batch overlap (DBO) is not supported: ...` |
| | R-SWA, cascade attention, per-request causal masks, bidirectional multimodal prefix ranges, non-causal at head size other than 128 | `pth_flash_attn: <feature> ...`; the message names the feature |
| Forward | fused output quantisation | `pth_flash_attn: fused output quantisation is not supported` |

Each message and its remedy is in [`troubleshooting.md`](troubleshooting.md).
Results are not batch invariant, so the backend does not claim vLLM's batch
invariance.

## 5. What happens at run time

- **Model load.** Each attention layer builds its Impl, which asks vLLM's
  workspace manager for the split-KV workspace and logs one line:
  `pth_flash_attn: reserved <n> token-splits of workspace (Hq=<h>, D=<d>, <m> MiB)
  inside the model-load memory window`. The reservation happens inside the
  window that vLLM's memory profile counts, so the KV cache is sized after it.
- **Per step.** The metadata builder cuts the batch into a decode-like slice
  (every request with at most the decode threshold of query tokens, speculative
  verification included) and a prefill slice, plans each with
  `planner.plan_launch`, and the Impl issues at most two op calls per layer. The
  cache write is vLLM's Triton writer and runs before the read.
- **CUDA graphs.** The builder declares uniform-batch graph support: decode
  batches of one query length replay as full graphs, batches with a prefill run
  piecewise with eager attention.
- **Device lengths.** The kernel reads every query and KV length from device
  memory, so adaptive verification (fewer verified tokens on the device than the
  host scheduled) and graph replay with changed lengths both work.
- **Environment variables.** None are read by this package.

## 6. When another plugin owns `CUSTOM`

`CUSTOM` is the only `AttentionBackendEnum` member reserved for plugins, and vLLM
resolves a backend back to the enum with `AttentionBackendEnum[get_name()]`, so
this package cannot register under a name of its own. If another plugin has
already registered a different class under `CUSTOM`, `register()` leaves it
unchanged and logs a warning:

```
pth_flash_attn: AttentionBackendEnum.CUSTOM is already registered as <path>; leaving it unchanged
```

`--attention-backend CUSTOM` then selects the other plugin's backend, not this
one. Only one of the two plugins can own `CUSTOM` in a process; remove the other
(or this) entry point from the environment, for instance by naming only one of
them in `VLLM_PLUGINS`. If the other plugin registered the very same class path,
nothing changes.

## 7. KV pool size

An int8 cell needs `2 D + 8` bytes per token and KV head (K and V data plus two
float32 scales), against `2 D` for an fp8 cache. For the same memory the int8
pool therefore holds `2 D / (2 D + 8)` as many tokens: about 1.5 % fewer tokens at
D = 256 and about 3 % fewer at D = 128 (derived from the byte counts; the same
effect read as bytes is 1.6 % more bytes per token at D = 256). Measured
pool sizes are in [`benchmarks.md`](benchmarks.md).

## 8. A public deployment: HyperQwen

[HyperQwen](https://github.com/syv-ai/HyperQwen) is a separate open-source
project that serves Qwen models with vLLM. It vendors a pinned
export of this source (its `pthfa/` directory), builds the op library into its container image,
and selects this backend for the target model and for the speculative drafters
(an MTP layer, or a DFlash-style drafter) through its launch scripts: they pass
`--attention-backend CUSTOM --kv-cache-dtype int8_per_token_head` and add
`"attention_backend":"CUSTOM"` to the speculative configuration; its launcher
knob for this is `PTH_FA=1`, which is the default of its `SPEC=dflash2` and
`SPEC=mtp` `CTX=long` launches on compute capability 8.6 GPUs when the package is
installed (HyperQwen pull requests [#339](https://github.com/syv-ai/HyperQwen/pull/339)
and [#345](https://github.com/syv-ai/HyperQwen/pull/345), both merged; `PTH_FA=0`
opts out). Its `bench/t4_battery.py` has a `pth` arm
([#347](https://github.com/syv-ai/HyperQwen/pull/347), merged). It refuses
launch options that would override those settings. Its documentation of the
integration, the limits it observes and the measured configurations:

- [`pthfa/README.md`](https://github.com/syv-ai/HyperQwen/blob/main/pthfa/README.md)
- [`docs/gotchas.md`](https://github.com/syv-ai/HyperQwen/blob/main/docs/gotchas.md),
  the item on the pth-flash-attn backend.

HyperQwen's launcher, container image and measurements are its own; the
benchmark tables of this repository label which results were taken with it
([`benchmarks.md`](benchmarks.md)).
