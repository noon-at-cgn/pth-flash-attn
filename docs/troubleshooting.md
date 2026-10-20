<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Troubleshooting

Each entry gives the symptom as the software reports it, its cause, and the
fix. Messages marked "(verbatim)" are the strings in the source; `<n>` stands for
a value. For a bug report, attach the output of `tools/collect_env.sh` (it only
reads the environment).

## 1. Building and installing

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` while running `pip install .` or `setup.py` | `setup.py` imports PyTorch to find its headers, so it needs PyTorch in the build environment | install PyTorch first and use `pip install --no-build-isolation .`; an isolated build downloads its own PyTorch ([`build.md`](build.md)) |
| `The detected CUDA version (<x>) mismatches the version that was used to compile PyTorch (<y>)` (PyTorch's own message, raised by its extension builder) | `nvcc` and PyTorch have different CUDA major versions | use nvcc 13 with a PyTorch built for CUDA 13 (`cu130`), or the matching pair: put the right toolkit first on `PATH` or set `CUDA_HOME` |
| PyTorch's extension builder cannot find a CUDA installation | no `nvcc` on `PATH` and no `CUDA_HOME` | install a CUDA 13 toolkit and point `CUDA_HOME` at it |
| the build is very slow and uses one core | `ninja` is not installed, so PyTorch falls back to a serial backend | `pip install ninja` |
| the build is killed or the machine swaps | too many parallel compile jobs for the available memory | set `MAX_JOBS` to a smaller number; build one architecture with `TORCH_CUDA_ARCH_LIST=8.6` |
| a launch fails with a CUDA `no kernel image is available` error | the library was built for other architectures than the GPU's | rebuild with the GPU's compute capability in `TORCH_CUDA_ARCH_LIST` (the default is `8.0;8.6;8.9`) |

## 2. Importing and loading the library

| Symptom | Cause | Fix |
|---|---|---|
| `ops.is_available()` is `False` | `_C.abi3.so` is not next to `pth_flash_attn/ops.py` | install the package, or run `python setup.py build_ext --inplace` in a source checkout |
| `RuntimeError: pth_flash_attn_fwd: op library <path> is not built; ...` (verbatim start) | the wrapper was called while the library is missing | same as above. Also check you are importing the package you built: a source checkout on `PYTHONPATH` shadows the installed copy that holds the `.so` |
| loading `_C.abi3.so` fails (for example a missing `libcudart.so.13` or an undefined symbol) | a PyTorch older than 2.11, which the stable-ABI library does not support, or a PyTorch built for another CUDA major than the library | use PyTorch 2.11 or newer, built for the same CUDA major as nvcc; the tested pair is PyTorch 2.13.0 with CUDA 13.0 |
| an `sm_90` or newer GPU: the op raises `needs compute capability 8.x (sm_80/86/89), device has major <n>` | the library carries SASS for sm80, sm86 and sm89 and no PTX, so newer GPUs are refused rather than JIT-compiled into kernels nobody tested | use another attention backend on that GPU; see [`faq.md`](faq.md) |

## 3. Errors from the op

All messages begin with `pth_flash_attn_fwd: ` (verbatim). The full list with
meanings is in [`api.md`](api.md), section 5. The ones users meet most:

| Message | Cause | Fix |
|---|---|---|
| `arguments are valid, but the op computes on CUDA tensors only (the CPU registration exists to test the argument checks)` | a CPU tensor was passed | move every tensor to the GPU; on CPU the op only validates |
| `<name> must be on the same device as q` | tensors on different devices | put every tensor on `q`'s device |
| `<name> must be <dtype>, got <dtype>` | wrong dtype, for example an int64 `block_table` or a float16 scale | `block_table`, `seq_lens` and `query_start_loc` are int32; the scales and workspaces are float32; the caches are int8 |
| `<name> must have stride 1 in its last dimension, got <s>` | a transposed or sliced view | pass a view whose last dimension is contiguous (`.contiguous()` on `q`; the cache views must keep the head dimension contiguous) |
| `<name> data pointer must be <n>-byte aligned`, `<name> byte stride of dim <i> (<b>) must be a multiple of <n>` | a view that starts at an odd byte offset, or strides that are not multiples of 4 bytes (caches, scales) or 16 bytes (`q`, `out`, `ws_o`) | allocate with PyTorch and keep whole rows; do not slice at odd offsets |
| `ws_o must have size(0) >= num_splits = <s> and size(1) >= T = <t>, got [<a>, <b>, ...]` | the whole workspace reservation, or a view cut for another launch, was passed | pass views of the plan's `ws_o_shape` and `ws_lse_shape` ([`api.md`](api.md), section 4) |
| `wm = <wm> is not the tile height of G = <g> (rows <r> need wm = <w>)` | a hand-chosen `wm` | take `wm` from `plan_launch` for this slice |
| `G = Hq/Hkv = <g> exceeds the supported maximum 64`, `Hq = <a> must be a positive multiple of Hkv = <b>` | unsupported GQA shape | the group size must be 1 to 64 |
| `sliding-window and non-causal attention need head size D = 128, got <D>` | `window > 0` or `causal=False` at head size 64 or 256 | those variants exist at D = 128 only |
| `head size D = q.size(2) must be 64, 128 or 256, got <D>` | unsupported head size | only 64, 128 and 256 are built |
| `attention launch: <reason>` or `combine launch: <reason>` | the CUDA launch itself failed | the reason is the launcher's own text; if it says `invalid argument` and the process also loads the test library `libpth_attn.so`, see below |

**`invalid argument` when the test library is loaded too.** Function-local
statics in the launchers that set the shared-memory limit are bound across all
libraries of a process by the dynamic loader unless the library is built with
`-fno-gnu-unique`. `setup.py` passes the flag; a custom build that omits it, in
a process that also loads `libpth_attn.so`, skips the attribute for its own copy
of the kernel and the launch fails. Build with the shipped `setup.py`.

**No error, wrong numbers.** The op cannot see these mistakes. Check, in order:

1. `seq_lens` includes the new query tokens, and K and V of those tokens are
   already in the cache (the op never writes the cache).
2. The cache was written with the convention of
   [`integration.md`](integration.md), section 3: `scale * int8` is the value.
3. `query_start_loc` starts at 0, ends at `T` and is rebased to the slice that
   `q` and `out` were cut to.
4. Block ids are the ids of the views you pass (kernel blocks, not larger storage
   blocks), and every touched block id is below the number of blocks.
5. Query heads map to KV heads as `h // (Hq / Hkv)` (consecutive heads share a KV
   head).
6. In a CUDA graph: no tensor was freed or reallocated between capture and
   replay, and `wm` and `num_splits` came from the plan for the captured
   descriptor ([`api.md`](api.md), section 6).
7. Compare against the fp64 reference of [`integration.md`](integration.md),
   section 7 on a small case.

## 4. vLLM

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Backend CUSTOM must be registered before use. ...` (vLLM 0.30.0's message) | the plugin was not loaded in that process: the package is not installed in the environment of the vLLM workers, or `VLLM_PLUGINS` is set and does not name `pth_flash_attn` | install the package into the same environment as vLLM; add `pth_flash_attn` to `VLLM_PLUGINS` if it is set. Run the registration check of [`vllm.md`](vllm.md), section 1 |
| `--attention-backend CUSTOM` selects another backend's behaviour; log line `pth_flash_attn: AttentionBackendEnum.CUSTOM is already registered as <path>; leaving it unchanged` (verbatim) | another plugin registered `CUSTOM` first | run with one `CUSTOM` plugin; see [`vllm.md`](vllm.md), section 6 |
| `ValueError` at start-up listing reasons why `CUSTOM` is not valid | an explicit selection that the backend cannot serve: compute capability, KV cache dtype, model dtype, head size, block size, attention type, or a feature in [`vllm.md`](vllm.md) section 4 | change the configuration to match, or choose another backend. There is no automatic fallback |
| `sliding window at head size <D> (the windowed kernels exist for D = 128 only)` or `non-causal attention at head size <D> (...)` (verbatim) | a sliding-window or non-causal layer at head size 64 or 256 | those variants exist at D = 128 only; use another backend for that model |
| `NotImplementedError: pth_flash_attn does not support: <list>` at model load | the Impl found a feature it refuses: ALiBi, sliding window, logits soft cap, attention sinks, chunk lookback, a non-decoder attention type, another KV cache dtype or an unsupported head size | remove the feature, or use `TRITON_ATTN` |
| `pth_flash_attn: int8_per_token_head only` | the KV cache dtype is not `int8_per_token_head` | start vLLM with `--kv-cache-dtype int8_per_token_head` |
| `pth_flash_attn: head size <D> is not one of (64, 128, 256)` | the model's head size has no kernel (raised by the metadata builder) | use another backend for that model |
| `pth_flash_attn: DCP/PCP are not supported` | context parallelism is enabled | disable it |
| `pth_flash_attn: dual-batch overlap (DBO) is not supported: ...` | DBO is enabled | disable DBO; the workspace is one buffer per micro-batch lane |
| `pth_flash_attn: R-SWA is not supported`, `... cascade attention`, `... per-request causal masks are not supported ...`, `... bidirectional multimodal prefix ranges are not supported ...` (verbatim starts) | a feature the backend refuses | disable the feature or use another backend |
| `pth_flash_attn: fused output quantisation is not supported` | a graph-level fusion passes an output scale to attention | disable the fusion |
| `pth_flash_attn: launch needs <a> token-splits, <b> were reserved` | an internal inconsistency between the planner's clamp and the reservation | report it with `tools/collect_env.sh` output; it should not occur |
| `pth_flash_attn: <n> prefill requests exceed max_num_seqs = <m>` | a batch with more requests than the scheduler maximum the builder was sized for | keep the batch within `max_num_seqs` |
| the drafter of a speculative-decoding setup is slow or on the wrong backend | the speculative configuration lacks `"attention_backend": "CUSTOM"`, so the drafter uses another backend | add the key ([`vllm.md`](vllm.md), section 2) |
| the KV pool is a little smaller than with an fp8 cache | an int8 cell holds two float32 scales: `2 D + 8` bytes per token and KV head against `2 D` | expected: about 1.5 % fewer tokens at D = 256 and about 3 % at D = 128 for the same memory ([`vllm.md`](vllm.md), section 7) |
| an `ImportError` from `pth_flash_attn.vllm_backend` after a vLLM upgrade | the backend imports vLLM internals, and the installed vLLM renamed or moved one | the tested version is 0.31.0 (0.30.0 is the version the backend was written against); see [`vllm.md`](vllm.md), section 3 |

At INFO level a successful model load logs a line like
`pth_flash_attn: reserved <n> token-splits of workspace (Hq=<h>, D=<d>, <m> MiB)
inside the model-load memory window`; if it is missing from the log, a different
backend is running.

## 5. Where to ask

Open an issue on the repository with the `tools/collect_env.sh` output, the
command line, the first error message and, for wrong numbers, the smallest case
that reproduces them. Support is best effort; see
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Security reports go through the channel
in [`SECURITY.md`](../SECURITY.md).
