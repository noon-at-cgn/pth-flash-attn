<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
-->
# bench/

Benchmark programs, drivers and curated results of pth-flash-attn. Everything here needs an NVIDIA GPU of compute capability 8.x
except the statistics, the table rendering and the CPU tests. The published numbers are in
[`docs/benchmarks.md`](../docs/benchmarks.md); how they are produced is in
[`docs/benchmark-methodology.md`](../docs/benchmark-methodology.md).

## Hardware ceilings

| File | Purpose |
|---|---|
| [`sol_copy.cu`](sol_copy.cu) | copy kernel with the address stream and byte volume of the attention loader, data discarded: the memory ceiling (η_mem denominator) |
| [`mma_ceiling.cu`](mma_ceiling.cu) | sustained `mma.sync` rate at the occupancy of the attention kernel, and the graph-node overhead: the tensor-core ceiling (η_mma denominator) |
| [`bench_harness.cuh`](bench_harness.cuh), [`cupti_timer.cu`](cupti_timer.cu) | CUPTI kernel timing, CUDA-graph blocks, cold-cache rotation, shared by the programs above and the kernel benchmark |
| [`ceilings_build.sh`](ceilings_build.sh), [`ceilings_run.sh`](ceilings_run.sh) | build the ceiling programs and run a ceilings session (3 processes, paired blocks, NVML trace) |
| [`bench_report.py`](bench_report.py), [`ceilings.py`](ceilings.py) | turn raw ceiling output into schema-valid records; the ceilings of a session as JSON |

## Kernel timing and gates

| File | Purpose |
|---|---|
| [`kernel_bench.py`](kernel_bench.py) | times the attention kernel library on the production-geometry grid (decode, verify, prefill, mixed) with the same design as the ceilings |
| [`kernel_gate.py`](kernel_gate.py), [`kernel_gate.schema.json`](kernel_gate.schema.json) | turns the records into pass or fail results at fixed reference levels against the ceilings of the same session (decode η_mem ≥ 0.90 at L ≥ 64k, verify and prefill η_mma ≥ 0.60); see [Efficiency](../docs/benchmark-methodology.md#efficiency). The K3 q = 4 decode rows do not all reach 0.90 |
| [`k_build.sh`](k_build.sh), [`k_run.sh`](k_run.sh) | build the kernel library and run a gate session |
| [`k_sweep.py`](k_sweep.py), [`k_sweep.sh`](k_sweep.sh), [`sweep_table.py`](sweep_table.py) | sweeps around the planner's launch (split count, row tile) and their tables |
| [`k_timeline.py`](k_timeline.py) | per-kernel timeline of decode launches: where the fixed per-launch cost goes |
| [`op_vs_triton.py`](op_vs_triton.py) | the op against vLLM's Triton unified attention on the same cache tensor, CUDA-graph timing, cold and warm; needs vLLM. It is the stand-alone counterpart of the Triton comparisons K1 and K2 of the benchmarks page (the K2 table itself was timed with vLLM's own micro-benchmark, see [K2](../docs/benchmarks.md#k2-causal-decode-verify-and-prefill-at-d256-against-triton)) |
| [`stats.py`](stats.py) | percentile bootstrap, paired blocks, A/A minimum detectable effect, rejection rules, gate evaluation (CPU only) |
| [`nvml_log.py`](nvml_log.py) | 100 ms NVML sampler for a whole session (clocks, power, temperature, throttle reasons, compute processes) |
| [`results.schema.json`](results.schema.json) | schema of every kernel-level result record |
| [`lib.sh`](lib.sh) | environment capture (GPU, driver, power limit, git SHA) and an optional lock shared by the shell drivers |

## End to end

[`e2e/`](e2e/README.md) holds the helpers that measure the kernel inside a running vLLM server with HyperQwen's harness: the pinned-salt
decode ladder ([`ladder.sh`](e2e/ladder.sh), [`ladder_salt.py`](e2e/ladder_salt.py)) and the prompt-logprob tool for accuracy against a bf16-KV reference ([`prompt_logprob.py`](e2e/prompt_logprob.py)).

## Curated results

[`results/`](results/README.md) holds one CSV per group of tables of [`docs/benchmarks.md`](../docs/benchmarks.md) and [`render_tables.py`](results/render_tables.py), which regenerates those tables
and the headline table embedded in the repository [README](../README.md);
[`tests/cpu/test_benchmark_tables.py`](../tests/cpu/test_benchmark_tables.py) asserts that the page equals the rendered output and that the README's headline table is a verbatim copy of the page's.

## CPU-only checks

```bash
python3 -m pytest tests/cpu/test_benchmark_tables.py tests/cpu/test_ladder_salt.py tests/cpu/test_stats.py tests/cpu/test_kernel_bench.py -q
```

The statistics, the gate logic, the table rendering and the end-to-end helpers have CPU tests; the programs that time a kernel need a GPU, and the
GPU correctness tests are in [`tests/gpu/`](../tests/gpu/).
