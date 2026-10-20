<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the op schema, the Python API and the supported
configurations may change in a minor release.

Each release entry states the validation behind it: which checks ran, with their
counts, and on which hardware. A change that is not validated says so.

## [Unreleased]

## [0.1.0] - 2026-10-20

First public release.

### Added

- **The op.** `torch.ops.pth_flash_attn.pth_flash_attn_fwd`, a forward-only
  FlashAttention-style attention kernel that reads vLLM's
  `int8_per_token_head` paged KV cache in place: int8 keys and values and
  their inline fp32 scales are read with tensor cores, and no dequantised copy
  of the cache is made. It is a PyTorch stable-ABI extension
  (`_C.abi3.so`) that allocates nothing, reads no environment variables, runs
  on the current stream and can be captured in a CUDA graph. A fake kernel
  supports `torch.compile`; the CPU registration validates arguments only.
  Argument contract: [`docs/api.md`](docs/api.md).
- **Kernels.** A row-tile kernel for prefill, mixed batches and speculative
  verification, a swap-AB decode kernel, split-KV with a log-sum-exp combine,
  and sliding-window and non-causal variants at head size 128. Head sizes 64,
  128 and 256; GQA groups of 1 to 64 query heads per KV head; bfloat16 and
  float16 queries (fp16 tensor-core arithmetic for both, checked against a
  per-element error bound). Overview: [`docs/architecture.md`](docs/architecture.md).
- **Planner.** A pure-Python launch planner (`plan_launch`, `LaunchPlan`,
  `workspace_shapes`, `workspace_bytes`, `workspace_token_splits`) that
  computes the warp count of the row tile, the split count and the two fp32
  workspaces from host integers only.
- **vLLM plugin.** A `vllm.general_plugins` entry point that registers the
  backend as `CUSTOM` without importing CUDA code until it is selected, with
  its own attention-metadata builder and explicit refusals (with reasons) for
  configurations it does not support. See [`docs/vllm.md`](docs/vllm.md).
- **Engine integration contract** for other inference engines:
  [`docs/integration.md`](docs/integration.md).
- **Tests.** CPU suites (reference, error bound, bit-level emulator, planner,
  op registration, tooling), a PTX and layout simulator, C++ geometry and
  primitive tests, and a GPU tier with a correctness grid, loader byte-exactness
  tests, sanitizer and CUDA-graph tests. See [`docs/build.md`](docs/build.md).
- **Compile gates.** Register, spill and stack-frame limits per kernel and
  architecture, and a SASS lint, in `tools/`.
- **Benchmark harness** in `bench/`: hardware ceilings, a kernel benchmark gate
  with A/A noise measurement and mirrored interleaving, an op-versus-Triton
  comparison, end-to-end scripts and curated result tables with their sources.
  Results and caveats: [`docs/benchmarks.md`](docs/benchmarks.md) and
  [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md).
- **Documentation**: architecture, API, integration, vLLM, build and testing,
  benchmarks and their methodology, troubleshooting and FAQ.
- **Development tooling**: dev and sanitizer images (`docker/`), one-command
  local CI (`tools/run_ci_local.sh`), pre-commit configuration, and the export
  and proof scripts for the vendored subset. The containers of these scripts are limited to `min(4, nproc)` CPUs
  (`PTH_CI_CPUS` overrides), so that they also start on a 2-CPU machine.

### Validation

- **CPU checks.** On the release tree, `tools/run_ci_local.sh` passes in all
  six groups (host, lint, build, test, cost, attn; 21 steps, no
  network): 1,186 tests passed on `tests/cpu` and `tests/sim` (35 skipped: 34
  need a CUDA device or `libcuda`, 1 needs a pip-installed package; 1
  deselected), and the lint, build, cost and attention-library
  groups. The sm80, sm86 and sm89 builds pass the register, spill and SASS
  gates.
- **Machine code.** The shipped kernels compile to the same SASS, in 123 of 123
  kernels on sm80, sm86 and sm89, as those of the pre-release commit that
  HyperQwen vendors. The vendored files differ from it in comments, docstrings,
  packaging metadata, two error message texts and one sentence of `NOTICE`
  only (`tools/compare_vendor_trees.py`).
- **Benchmark harness.** `bench/op_vs_triton.py` passes vLLM's split-KV (3D)
  buffers to its Triton arm, as vLLM's own backend does, so that its q=1 ratios
  match vLLM's decode path. The K2 table of
  [`docs/benchmarks.md`](docs/benchmarks.md) is generated from this harness
  (K2a) next to an earlier development measurement (K2b), four rows of which
  did not reproduce.
- **GPU.** On a pre-publication tree whose kernel and package sources differ
  from the release only in comments, docstrings and the text of two error
  messages (compiled SASS identical), one RTX
  3080 20 GB (sm86): the correctness grid 328/328 on both kernels (window
  family 70/70), 140 quick and 752 full tests, four sanitizer tools with 0
  errors, 15 graph tests, and a plugin smoke test on vLLM 0.31.0
  ([`docs/build.md`](docs/build.md#7-what-has-been-run-on-a-gpu)). The end-to-end runs of
  [`docs/benchmarks.md`](docs/benchmarks.md) were taken on pre-release builds
  proved to give identical SASS; the kernel-level rows K2b and K3 come from
  pre-release builds whose commits are not recorded.

### Supported

- GPUs: NVIDIA compute capability 8.0, 8.6 and 8.9, built as SASS for those
  architectures only (no PTX; sm90 and newer are refused, not JIT-compiled).
- Software: Linux; Python 3.10 to 3.14 declared (tested with 3.12); PyTorch 2.11
  or newer built for CUDA 13 (declared; tested with 2.13.0+cu130); nvcc 13 and
  g++ to build; vLLM 0.31.0 (tested) and 0.30.0 (written against its sources;
  not run in CI) for the plugin.
- Install from source with `pip install --no-build-isolation .`; see
  [`docs/build.md`](docs/build.md).
- Downstream: HyperQwen vendors a pinned export and, with its merged pull
  requests #339 and #345, selects this backend by default for `SPEC=dflash2` and
  `SPEC=mtp` `CTX=long` launches on compute capability 8.6 (`PTH_FA=0` opts
  out); see [`docs/vllm.md`](docs/vllm.md).

### Limits

- Only sm86 has been measured and run on a GPU (RTX 3080 20 GB for the kernel and
  the GPU tests, an RTX 3090 for independent end-to-end runs). The sm80 and
  sm89 builds pass the compile, register and SASS gates and the CPU suites; no
  GPU test run or timing on them is recorded.
- Forward pass only; decoder self-attention only; `int8_per_token_head` cache
  only. Not supported: other cache dtypes, ALiBi, attention sinks, logit
  soft-capping, multimodal-prefix attention, R-SWA, chunk lookback, decode and
  prefill context parallelism, batch invariance.
- Sliding-window and non-causal attention are available at head size 128 only.
- No wheel and no PyPI release; the package is built from source.
- vLLM versions other than those listed, and forks of vLLM, are untested.
- Downstream projects may pin the pre-release commits `07e5063` or `a31d8ab`.
  Those commits are from a history that is not published; the shipped subset
  differs from them in comments, docstrings, packaging metadata, two error
  message texts and one sentence of `NOTICE` only, and compiles to the same SASS (123 of 123 kernels on
  sm80, sm86 and sm89).

[Unreleased]: https://github.com/noon-at-cgn/pth-flash-attn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/noon-at-cgn/pth-flash-attn/releases/tag/v0.1.0
