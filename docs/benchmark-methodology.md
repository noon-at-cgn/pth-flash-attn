<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
-->
# Benchmark methodology: how a number becomes a claim

This page explains how every number of [`benchmarks.md`](benchmarks.md) is produced, so that it can be read, repeated and
challenged. Two kinds of measurement exist:

- **Kernel level** (K1-K3): the kernel is timed with the GPU's own timestamps and compared with hardware ceilings measured
  in the same session on the same card, or with vLLM's Triton path on the same cache bytes.
- **End to end** (E1, E2, A1, T1): a running vLLM server is driven by the benchmark scripts of
  [HyperQwen](https://github.com/syv-ai/HyperQwen), arm against arm, and the arms are compared by ratios of medians with
  bootstrap confidence intervals.

The numbers behind the tables are in [`bench/results/`](../bench/results/README.md).

## Hardware and software labels

Every published number states its hardware, power limit, software, configuration, date and source. The measured system is:

| Item | Value |
|---|---|
| GPUs | two NVIDIA GeForce RTX 3080 20 GB (memory-modified boards, GA102, compute capability 8.6, 68 SMs, 20,480 MiB each) |
| Interconnect | PCIe Gen3 x16, no peer-to-peer (`nvidia-smi topo` reports CNS, chipset not supported); tensor parallelism at TP=2 uses `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce` |
| Power limit | 220 W per card (the card default is 320 W), set on the host outside the container |
| Clock | SM clock limited to 1905 MHz with a +120 MHz graphics-clock offset, set on the host; below that limit the power governor chooses the clock |
| Host | Intel Xeon E5-2696 v4 (22 cores), 125 GB RAM; the container that runs vLLM sees 24 cores and 90 GB |
| Software | Ubuntu 24.04 container, driver 615.71.09, CUDA 13.0, torch 2.13.0+cu130, vLLM 0.31.0 |

Source: [HyperQwen #325](https://github.com/syv-ai/HyperQwen/pull/325) ("Numbers, and whose hardware"). A run on a different
machine states its own labels: the 3090 runs of M1 are one RTX 3090 24 GB (GA102, 82 SMs) at TP=1, 250 W, as reported
by the HyperQwen maintainer.

## Power limit and clocks

The cards run at the 220 W power limit and the clock is chosen by the governor below the 1905 MHz limit. That is the condition a user of these
cards has, so it is the condition measured. Consequences:

- The SM clock under load is set by the power governor and depends on the workload: a tensor-core loop and a memory copy
  run at different clocks. In a 2-minute copy soak at the limit the clock moved between plateaus from 1155 to 1905 MHz on
  a fixed 0.5 s grid while the copy time did not follow it (below). The copy ceiling and the tensor-core ceiling are each
  measured in their own steady state, which is also how a kernel runs.
- **Ratios measured in the same session are the claims; absolute numbers are indicative.** A different day, room
  temperature or driver can move absolute GB/s and TFLOPS by more than the margins used here, so every efficiency is a ratio to
  a ceiling measured in the same session.
- `mma_ceiling` records the effective SM clock (clock cycles over global-timer nanoseconds) and the cycles per
  `mma` per sub-partition, so "slow because the clock was low" can be told from "slow because the pipe was not full". Its
  operands are random with alternating sign, because zero or constant data draws less power, clocks higher under the limit
  and overstates the ceiling.
- A 320 W limit was measured once for the ceilings (2026-10-12): the copy ceilings were about 0.5 % higher (DRAM-bound
  rather than clock-bound; 702.6 GB/s at 128k on the first card) and the tensor-core ceiling followed the SM clock
  (67.90 TFLOPS at 1965 MHz on the first card against 66.18 TFLOPS at 1905 MHz). Results at the two limits are never mixed in one comparison.

## Hardware ceilings

Efficiencies are fractions of what the card measurably does, never ratios to other kernels. Two ceilings are measured with
programs of this repository.

- **Copy ceiling** ([`bench/sol_copy.cu`](../bench/sol_copy.cu)): a kernel that reads exactly the bytes of the attention kernel in the same way (the int8
  cache cells, K, V and two fp32 scales per token and head, 520 B at D=256, through `cp.async` with the same per-stream widths
  and strides, 64-token tiles, 4 warps, a two-stage pipeline, the planner's split count and the same shared-memory
  footprint) and discards the data. The decode efficiency is η_mem = T_copy / T_kernel at the same shape.
- **Tensor-core ceiling** ([`bench/mma_ceiling.cu`](../bench/mma_ceiling.cu)): the sustained rate of `mma.sync.m16n8k16` with fp16 inputs and fp32
  accumulation at the occupancy of the attention kernel (4 warps, 1 CTA per SM). η_mma is executed flops over (ceiling rate × time).
- **Graph-node overhead**: the extra cost of one more kernel node in a CUDA graph.

| Ceiling (RTX 3080 20 GB, 220 W, 2026-10-10 and 2026-10-11, first card full run, second card quick run) | Value |
|---|---|
| copy, separate K/V/scale streams, NHD, block 816, L=4k | 494.9 GB/s (launch-bound) |
| copy, same, L=64k / 128k / 256k | 688.4 / 698.8 / 699.6 GB/s (128k on the second card: 698.3) |
| copy, whole-cell `cp.async`, L=64k / 128k / 256k | 695.9 / 706.6 / 710.9 GB/s |
| plain streaming read, L=64k / 128k / 256k | 711.6 / 717.6 / 720.6 GB/s (94-95 % of the nominal 760 GB/s) |
| tensor core, f16 in, f32 accumulate, chain 8 | 66.2 TFLOPS (64.5 at chain 1) |
| tensor core, bf16 in / f16 accumulate / s8 | 63.1 / 125.9-129.5 / 243-255 TFLOPS or TOPS (noisy, indicative) |
| CUDA-graph node | 0.60 µs (0.59 on the second card) |
| A/A minimum detectable effect, copy | 0.59 % (first card), 0.93 % (second card) |
| A/A minimum detectable effect, f16 mma per unit of work | 0.0068 %, 0.018 % |

The two cards agree within about 1 %. The session of the K3 table measured its own ceilings on 2026-10-13 (copy 686.0 / 697.9 / 701.6 GB/s at 64k / 128k / 256k for one request, tensor core 66.26 TFLOPS at 1905 MHz). The ridge point against the loader
ceiling is 66.2e12 / 698.9e9 = 94.7 flop per byte. At q=4 and G=6 (D=256) the tensor time over the copy time is 0.665 on the two-16-row-slice formula
(32768·BW / (520·F)); the swap-AB decode kernel executes 24 rows, which gives 24576·BW / (520·F) = 0.499 (both derived from the ceilings above).

Triton, FlashAttention-2 and FlashInfer are measured as comparison rows only; the efficiency references are the ceilings above.

## Kernel timing

- **Timer.** Kernel durations are the GPU's start and end timestamps from CUPTI, via [`bench/bench_harness.cuh`](../bench/bench_harness.cuh); CUDA events are
  recorded as a cross-check. FlashInfer's `bench_gpu_time_with_cupti` falls back silently to CUDA events when the `cupti-python`
  package is missing, so it is not used for ceilings. CUDA events come in 1,024 ns steps; a 2 µs replay is two steps,
  so the per-node cost of a CUDA graph is the slope over 65, 129 and 257 nodes.
- **Block.** 20 launches captured in one CUDA graph and replayed. The block's value is the median of its 20 kernel durations,
  which ignores one disturbed launch without discarding the block. A launch of a shape is every kernel it needs (attention,
  the combine kernel when the split count exceeds 1, two slices for a mixed batch); its time runs from the first kernel's
  start to the last kernel's end.
- **Cold cache.** When one launch reads less than four times the 5 MB L2, launches rotate over enough separate copies of the data
  that each copy is evicted before it is reused.
- **Warm-up.** 2 s of all arms before the first timed block, so the card has reached its power-limit steady state. A 2 minute
  soak showed no ramp left after 2 s (the mean deviation of a block from its process median by time since the first block:
  +0.031 %, −0.011 %, +0.013 %, +0.005 % for 0-0.25, 0.25-0.5, 0.5-1 and 1-2 s on the copy arm).
- **Telemetry.** [`bench/nvml_log.py`](../bench/nvml_log.py) samples SM and memory clock, power, temperature, throttle reasons and the list of compute
  processes every 100 ms for the whole session.
- **Launches as vLLM issues them.** The split count and the row-tile multiplier `wm` come from the planner with the device's own
  SM count; the row tile is the balanced `min(planner row tile, q)`; inputs rotate so that four times the L2 passes between
  reuses; launches replay from a CUDA graph.
- **Work credited per launch.** KV bytes read once, (2D + 8) per token per KV head; useful flops, 4D per query row and visible
  key; executed flops, which depend on the kernel that ran (the 16-row-tile kernel multiplies 16·wm rows per 64-token KV tile,
  the swap-AB decode kernel 8·⌈rows/8⌉ rows).

## Comparing two kernel arms

A single "A took 180 µs, B took 198 µs" says little: the clock moves with temperature and power, other processes interfere and
memory placement differs from process to process. So:

1. **Interleave.** Blocks of the arms alternate in mirrored rounds, A B | B A | A B. A slow drift then raises both arms equally.
   In a simulation with 3 % linear drift, running all A then all B reported a fake 1.4 % difference; the mirrored order reported 0.1 % with a CI containing 0.
2. **Pair.** Each round gives one log-ratio, log(median B / median A).
3. **Repeat in fresh processes.** 3 processes with the arm order reversed in the second (6 processes for K3), at
   least 30 accepted rounds each (36 scheduled; 216 blocks per row in K3).
4. **Bootstrap.** The 95 % CI of the mean log-ratio is a percentile bootstrap of 10,000 resamples within each process,
   seeded (seed 0), so a re-analysis gives identical numbers. At this design size the CI covers the truth 94 % of the time in simulation (nominal 95 %).
5. **Per-process offsets.** Effects that differ between processes, for example memory placement, are invisible to a CI computed
   from rounds inside the processes. Identical copy arms differ per process by +0.36 / −0.04 / +0.55 % (first card) and +0.46 / −0.25 / +0.96 % (second card).
   Efficiency intervals therefore add a cluster term over the process medians (a t interval with P−1 degrees of freedom), which is why those runs use 6 processes.

A CI that merely excludes 1 is not enough. Every claiming experiment includes an **A/A** pair, the same arm twice in separate allocations; its spread, including the
spread between processes, gives the **minimum detectable effect** (MDE), the smallest difference the design detects 80 % of the time. A *difference* is claimed only if the CI excludes
1 ± max(2 %, MDE); an *optimisation* is adopted only if its gain clears max(2 %, 2 × MDE); everything else is reported as "no claim". With the MDE rule the false-claim rate
in a simulation where each process carries a 1.5 % offset fell from 85 % (CI excludes 1) to 8.5 % without the 2 % floor, and lower still with it.

## Rejection rules

A block is rejected, and with it the paired block of its round, when an NVML sample within one sampling period of it shows:

| Rule | Why |
|---|---|
| another compute process on the GPU | the card was shared; a foreign process that holds at most 64 MiB (a bare tool context) costs the block only, any other voids the run |
| a watched clock more than 3 % from the arm's typical value | the card changed state (heat soak, memory P-state, governor) |
| temperature above 83 °C | thermal throttling territory |
| thermal, hardware-slowdown or power-brake throttle | the card slowed itself down |
| within-block IQR/median above 5 % and the IQR more than 2 timer steps | launches inside the block disagree |
| no NVML sample, or a field NVML could not read | the block cannot be vouched for (fail closed) |

The software power-limit throttle bit is not a reason: at 220 W the card runs at its limit whenever it is busy, which is the condition measured.
The "typical clock" is per arm, because a tensor-core arm draws more power than a copy arm and runs at a lower clock under the same limit.

Copy arms watch only the memory clock: under copy load at the limit the SM clock jumps between plateaus (1155-1905 MHz) while the copy time moves by a median −0.02 % between low and high plateaus (98 groups on the first card), whereas a compute-bound arm follows the SM clock and keeps the SM-clock rule. Arms are compared per unit of work, so arms that do slightly different amounts of work are compared fairly. Tensor-core kinds run one per experiment, because the governor's response to the previous kind lands inside the next block.

A process that keeps fewer than 30 blocks of any arm is rerun, at most 3 times; after that the experiment is reported as "noisy". Rejection counts by reason are part of every result. NVML samples every 100 ms
but a block lasts a few milliseconds, so the samples around one block also cover its neighbours: the rules catch events in time windows, not differences between arms.

## Efficiency

The efficiency η_mem is the time of a copy kernel that reads exactly the bytes of the attention kernel divided by the kernel's time at the same shape; η_mma is the executed tensor-core flops over the time and the measured tensor-core rate. Both ceilings are measured in the same session at the same shape. A reported η carries the lower bound of its CI; a CI that straddles a level is called inconclusive and fewer than 30 surviving rounds is called noisy, so noise can lower a reported efficiency but not raise it.

At q=4 the tensor and memory terms are close, so the K3 table also prints the clock-adjusted value η_adj = (max(T_mem, T_mma(f)) + T_ovh) / T, with the tensor term rescaled linearly from the ceiling's clock to the kernel's median clock (linearity was checked at a second clock once, within 0.5 %). On every q=4 row the adjusted value differs from the raw η_mem by at most 0.009.

`bench/kernel_gate.py` checks fixed reference levels: decode η_mem ≥ 0.90 at L ≥ 64k, and η_mma ≥ 0.60 for verification (q=10) and prefill. The K3 q=1 decode rows at L ≥ 64k satisfy 0.90 (0.909 to 0.990); four of the six q=4 rows at L ≥ 64k fall below it (0.818 to 0.859; the other two are 0.917 and 0.902).

## Kernel comparison against the Triton path

- **K1**: [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) is the counterpart of this measurement in the repository, with the same design: one request, q=8, the sliding-window shape of the DFlash2 drafter, an int8_per_token_head cache in
  the production page layout, the *same cache tensor and block table* given to this kernel and to vLLM's `unified_attention`; 20 launches per CUDA-graph replay, one
  sample is one replay divided by 20, 200 samples per row, the median is reported. *Cold* rotates 20 sets of blocks so that every launch misses the L2; *warm* repeats one set. The
  run was on one card at its 1905 MHz maximum clock; at the 32/8-head shape the kernel reaches the 220 W limit (SM clock 1800-1830 MHz).
- **K2**: per-layer times from vLLM's attention micro-benchmark, which builds a layer's metadata and cache for a batch spec and times the selected backend's forward with
  `do_bench_cudagraph` (10 CUDA-graph replays per row, each of `int(20 ms / estimate)` unrolled calls; the reported value is the mean of the 10). The Triton columns are vLLM's Triton backend.
  The op column was timed through an attention backend of the vLLM 0.31.1rc1.dev355 development snapshot that calls `pth_flash_attn_fwd`; it is not the plugin backend of this
  repository ([`vllm_backend.py`](../python/pth_flash_attn/vllm_backend.py)), which calls the same op. The kernel and the op are therefore the ones the plugin uses, and only the harness around the op
  differs; the time of the plugin's own code outside the op was not measured. The int8 Triton and op columns are the medians of the means of three back-to-back runs; the bf16-cache Triton column is
  a single run. Across the three runs the run means of Triton int8 at q1s32k range from 373.6 to 406.6 µs (9 %) and of the op at q1s8k from 25.8 to 27.3 µs (6 %); every other row varies by at most 1.1 %. NVML
  during the first run: median SM clock 1770 MHz, median power 219.1 W (maximum 220.1 W), 53 °C maximum, no errors. No script of this repository produced this development measurement (K2b);
  [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) produced K2a: it times the same two kernels on a shared int8 cache with its own harness (CUDA-event timing of graph replays, cold and warm). For a single-token decode it passes vLLM's split-KV (3D) buffers to Triton, as vLLM's own backend does (without them Triton runs its 2D kernel at q=1, about ten times slower). Four K2b rows did not reproduce in K2a (+12.6 to +15.1 %, [`benchmarks.md`](benchmarks.md#k2-causal-decode-verify-and-prefill-at-d256-against-triton)).

## End-to-end harness

Servers are launched with HyperQwen's `single-user/start_qwen.sh`: `SPEC=dflash2 CTX=long` or `SPEC=mtp CTX=long`, with `PTH_FA=1` for the int8 kernel,
`PTH_FA=0` for the fp8 default of the arm, `FA2_FP8KV=0` for the int8 Triton arm. One arm is booted at a time on both cards; each boot is checked from its log (the attention backend selected for target and
drafter on both ranks, the KV dtype, full or piecewise CUDA graphs, the library mapped in both workers). The arms of one sitting use the same prompts and are repeated (E1: int8 Triton and FA2 again at the end; E2: the order
control, kernel, control, kernel).

Per boot, with the scripts of [`bench/e2e/`](../bench/e2e/README.md) and HyperQwen's `bench/`:

| Measurement | Tool | Definition |
|---|---|---|
| decode ladder | [`bench/e2e/ladder.sh`](../bench/e2e/ladder.sh) around `conc_ladder.py --n 1,2 --out 256 --reps 1` | N=1 and N=2 concurrent streams, 256 output tokens, greedy; prompts of 8,192, 32,000 and 128,000 tokens; 5, 3 and 3 repetitions at 8k, 32k and 128k, each its own invocation with its own label |
| time per decode pass | `conc_ladder.py` | wall time of the decode-only window divided by the engine's forward passes in it; the window opens when the last of the N streams has its first token and closes when the first of them finishes |
| decode tokens per second | `conc_ladder.py` | each stream's own first-token-to-last-token rate, averaged over the streams |
| tokens per step | vLLM metrics | 1 + accepted draft tokens / drafts over the run |
| TTFT | `conc_ladder.py`, `seat_ttft.py` | the ladder's mean first-token latency of its N requests; `seat_ttft.py` sends single requests of 128,016, 31,977 and 8,217 prompt tokens and reports the first as the cold TTFT; the "two concurrent prompts" value of E1e is the mean first-token latency of the two streams of the N=2 ladder |
| GSM8K | `quality_battery.py --gsm-only --gsm-n 200` | 200 questions, greedy, thinking off |
| prefix-cache residues | `residue_sweep.py <tag> 0 128` | 128 prompts checked for text copied verbatim from a cached prefix ("0 of 128 broken" means none) |
| needles, determinism | scripts that are not published | needle at 128k, depths 0.5 and 0.9, 64-token reply; at 1k, 8k and 32k one cold request then three identical greedy 256-token requests, which must be byte-identical |
| KV pool | boot log | the engine's reported KV capacity in tokens |
| preemptions | vLLM metrics | the counter over the GSM8K window |

**Prompts.** `conc_ladder.py` gives every stream, repetition and N its own salted prompt, so no request is served from the prefix cache; its default salt carries the wall clock, so two runs never share a prompt, and
acceptance in speculative decoding depends on the prompt. [`bench/e2e/ladder_salt.py`](../bench/e2e/ladder_salt.py) pins the clock the salt reads to an integer (1791800000 for all published runs), so that every arm receives byte-identical prompts; each arm boots a fresh server,
so identical salts never share a cache. Newer versions of `conc_ladder.py` offer `--salt` for the same purpose with different prompt text, so tokens per step can differ between the two ways.

## End-to-end statistics

- **Per-repetition values** (decode tokens per second, time per pass, tokens per step, TTFT) are summarised by their **median**. The medians of the tables are those of one boot per arm, except where a table says that it pools.
- **Ratios** are new ÷ old of two medians. The 95 % CI is a percentile bootstrap: 10,000 resamples, seed 0, the two samples resampled independently with replacement, the 2.5th and 97.5th percentile of the ratios; fewer than 3 values on a side give no interval.
- **Time per pass is the comparison that does not depend on the prompt**; tokens per second also carry the acceptance of the prompt (a ratio of tokens per step is reported beside it, and its CI includes 1 wherever the text says the gain comes
  from the step and not the acceptance).
- **Repeat boots and pooling.** Each arm is booted at least once; the arms with a second boot give the boot-to-boot spread (E1: int8 Triton repeated last within 1.9 % per pass, the second FA2 boot within 1.2 %; E2: the control's first boot 4-6 % slower per
  step than its repeat). The E2 ratios pool the repetitions of both boots of each arm (8 requests per arm at 128k and 32k, 10 at 8k; the first boot ran 5, 3 and 3 repetitions at 8k, 32k and 128k, and the repetitions of the repeat boot are not recorded here, so these totals cannot be re-derived from this repository). Repetitions are not independent of the boot they ran in, so a pooled CI is narrower than the boot-to-boot spread alone
  would justify; the tables state which columns are first-boot medians and which ratios are pooled.
- **Bold** marks a better value only where its ratio against every other arm has a CI that excludes 1, a paired difference has a CI that excludes 0, or the quantity is deterministic (a KV pool). Ties, single-run cells (n=1) and cells inside the noise
  are not bold, whoever leads.
- **GSM8K at n=200** has a standard error of about 1.4 points (one question is 0.5 point), so differences between arms are not claims.

## Accuracy against a bf16 reference

[`bench/e2e/prompt_logprob.py`](../bench/e2e/prompt_logprob.py) builds a document, sends its token ids to a server with `prompt_logprobs` (teacher forcing, one token generated, greedy) and stores, per position, the actual token's logprob and the server's top-1 token and logprob (with `--top K` also the top-K list).
The reference is the same checkpoint served with a bf16 KV cache on the FLASH_ATTN backend; it needs a shorter `MAX_LEN` (135,168) and a lower `GPU_UTIL` (0.88; 0.87 for the fp8 FlashInfer arm, which ran out of memory at 0.88) than the arms, because a bf16 cache does
not fit the arms' length. vLLM 0.31.0 also fails with a CUDA out-of-memory error in the prompt-logprob path of its model runner for a 32k prompt at `GPU_UTIL` 0.94 with `prompt_logprobs=10`, with any KV format; the same request succeeds at 0.90.

- **Metrics**, per document and over all positions (and over the last 2,048, which attend over the whole document): mean, p99 and maximum |Δ logprob| of the actual token against the reference; **top-1 agreement**, tie-aware (the sets of maximal tokens are compared, because bf16 logits tie often; the strict id rate is printed beside it);
  a **KL lower bound** from the top-10 lists coarse-grained onto the partition {each token in either top-10 set} + {the rest} (coarse-graining can only shrink KL).
- **Paired difference and non-inferiority.** For a NEW and an OLD arm the per-position difference |new − ref| − |old − ref| is averaged and given a 95 % CI by a non-overlapping block bootstrap (blocks of 256 and of 1,024 positions, 4,000 resamples, seed 0; positions of one document share context, so blocks are resampled whole).
  NEW is **non-inferior** when its mean error is at most OLD's plus max(10 % of OLD's mean error, 0.002 nats), judged on the point estimates over all positions and over the last 2,048; whether the CI's upper end stays below the margin is reported beside the verdict. The tables show the blocks-of-256 interval.
- **Absolute defaults of the tool**: top-1 agreement at least 99.5 % and mean |Δ| at most 0.02 nats, on both position sets.
- **Limits.** Two documents (30,616 and 127,937 scored positions), one run per arm, prompt logprobs rather than generation quality. The documents are not published; the tool builds an equivalent from the Markdown of a repository.

## T4 battery

HyperQwen's `bench/t4_battery.py` boots each arm and runs a restart gate, a quality cell (GSM8K n=200 and perplexity over three corpora), a prompt-residue sweep, decode at about 60k and 90k tokens, and a soak. Every arm other than the int8 control is judged against it with fixed bars:
GSM8K at most 1.0 point lower, perplexity over all corpora at most 2 % higher, decode at about 90k at least the control's. The results are single runs per arm without intervals. The runs used a stand-in corpus built from the repository's Markdown (calibrated to 60,015 and 90,021 prompt tokens) because
the battery's own corpus was not available, and 60-minute soaks where noted.

## Reproducing

1. Kernel ceilings and efficiencies: [`bench/ceilings_run.sh`](../bench/ceilings_run.sh), then [`bench/k_run.sh`](../bench/k_run.sh); the Triton comparison with [`bench/op_vs_triton.py`](../bench/op_vs_triton.py). Each records git SHA, driver, power limit and an NVML trace with its hash.
2. End to end: launch the arms as above, run [`bench/e2e/ladder.sh`](../bench/e2e/ladder.sh) per arm with the same `SALT_EPOCH`, HyperQwen's `seat_ttft.py` and `quality_battery.py`, and [`bench/e2e/prompt_logprob.py`](../bench/e2e/prompt_logprob.py) for accuracy ([`bench/e2e/README.md`](../bench/e2e/README.md)).
3. Compare with the bootstrap above; the page's tables are rebuilt from [`bench/results/`](../bench/results/README.md) with `python bench/results/render_tables.py --write` (the same command rebuilds the headline table embedded in the repository [README](../README.md)).

What cannot be reproduced exactly: the prompts and documents of the original runs (the salt text and the document sources differ between versions of the tools), the T4 corpus, and the absolute numbers on other cards or power limits; ratios from the same sitting are the portable result.

## Result records

Every kernel-level result produced by the scripts of this repository is a JSON record validated by [`bench/results.schema.json`](../bench/results.schema.json): git SHA and dirty flag, image digest, driver, nvcc, power limit and the environment captures with their hashes; the arm (ceiling, subject, comparison, A/A control)
and the shape; every block (per-launch durations, median, rejection and reason, NVML summary); median, IQR, IQR/median, accepted blocks of scheduled, the CI; rejection counts by reason and reruns; the verdicts of gates and claims; and the NVML trace with its hash. A kernel-level number of these scripts without a record behind it is not a result. The records are not published (the CSV files of [`bench/results/`](../bench/results/README.md) hold the numbers), and K2 was produced by vLLM's own micro-benchmark, so it has no such record.
