<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
-->
# Benchmarks

Every benchmark table published for pth-flash-attn is on this page, with the hardware, power limit, software versions,
configuration, date and source that produced it. The numbers live in the CSV files of
[`bench/results/`](../bench/results/README.md); the tables below are rendered from those files by
[`bench/results/render_tables.py`](../bench/results/render_tables.py), and [`tests/cpu/test_benchmark_tables.py`](../tests/cpu/test_benchmark_tables.py) fails when a table differs from its rows, so a
number cannot drift from its source. How each number is produced is described in
[`benchmark-methodology.md`](benchmark-methodology.md).

Rows whose source is a HyperQwen
pull request take their numbers from the text of that pull request (or, for M1, from a comment of its maintainer). Rows with
no public page are K2, K3, the three ladder TTFT rows of E1e and the whole of E1f; their source is the methodology page or
the script named in the caption.

Labels: **K1**-**K3** kernel level; **E1**, **E2** end to end on two RTX 3080 20 GB; **A1** accuracy against a bf16-KV
reference; **T1** the pass-or-fail results of HyperQwen's T4 battery (restart gate, quality cell, residue sweep, decode and
soak), an acceptance test rather than a throughput benchmark; **M1** runs on a native RTX 3090 reported by the HyperQwen maintainer. The
comparison arms are other attention paths as built and configured by [HyperQwen](https://github.com/syv-ai/HyperQwen)
at those commits (Triton unified attention on an int8 cache, FlashAttention-2 and FlashInfer on an fp8 cache), untuned.
"int8 + pth" is an int8 KV cache read by this repository's kernel (pth-flash-attn); "fp8" arms use an fp8 KV cache.

## Summary

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

## Read before the numbers

Figures in this section that are not table cells are quoted from the source pull requests of the tables
([#337](https://github.com/syv-ai/HyperQwen/pull/337), [#338](https://github.com/syv-ai/HyperQwen/pull/338),
[#345](https://github.com/syv-ai/HyperQwen/pull/345)). They have no row in the CSV files and were not re-derived for this page.

- **MTP (E2): kernel and graph mode are not separated.** On this stack vLLM prints that the full-and-piecewise CUDA-graph
  mode is not supported with speculative decoding on FlashInfer and falls back, so the fp8 control ran piecewise graphs only,
  while the kernel arm captured full and piecewise graphs. The step-time ratio is the kernel and the graph mode together. The
  kernel arm was not run forced to piecewise graphs ([#345](https://github.com/syv-ai/HyperQwen/pull/345), "Not measured on
  this arm").
- **DFlash2 (E1): the FA2 fp8 build is ahead in three places.** It prefills a cold 128k prompt faster (the N=1 ladder TTFT of
  target + drafter is 1.072 [1.071–1.074] times FA2's; one request per arm: 114.14 s against 122.13–122.28 s, 6.5–6.7 % shorter), it holds a larger KV
  pool (517,656 against 504,770 tokens, 2.6 % more) and its largest single-token logprob deviation at 128k is smaller (3.1
  against 6.0 nats; one token each).
- **KV pool.** The int8 pools are 1.6–3.2 % smaller than the fp8 pools in the same-sitting comparisons (E1 −2.5 %, E2
  −1.6 %, M1 −1.9 % and −3.2 %). The scales alone cost 1.6 % more bytes per token and head at D=256 (520 against 512 bytes),
  which is 1.5 % fewer tokens in a pool of fixed size;
  the remainder was not analysed. The sign depends on checkpoint and setup: an earlier community checkpoint on the same
  cards gave the opposite order ([#345](https://github.com/syv-ai/HyperQwen/pull/345), Limits).
- **Single samples.** The cells of M1a and M1b are one boot per arm (M1c: two boots per arm, one for the quality and soak rows). Cold TTFT cells of E1 and E2 are one request per arm; the T4
  battery is one run per arm with no intervals. A second FA2 boot of the E1 sitting gave a single-request cold TTFT of
  127.4 s, against 114.14 s for the first, while the three-request ladder medians of the two boots agreed at 113.5 s and
  113.9 s (these figures are not in the tables; the ladder rows of E1e are the better measure of prefill).
- **T4 battery conditions.** The corpus is a stand-in built from the repository's Markdown, the soaks ran 60 minutes instead
  of 240 (T1a: the FA2 arm 60, the kernel arm 240), and in the dflash2 battery the kernel arm ran from another image than the other two.
  The battery's arms in these runs came from local copies of the script; [#347](https://github.com/syv-ai/HyperQwen/pull/347), merged afterwards, adds a `pth` arm to `bench/t4_battery.py` itself.
- **GPU memory utilisation.** The MTP sitting ran at 0.93 on every arm because the fp8 control ran out of memory at 0.94;
  the kernel arm was not run at 0.94.
- **Statistical noise.** GSM8K at n=200 has a standard error of about 1.4 points; every GSM8K difference in the tables is
  inside it.
- **Mixed configurations.** The four E1 arms ran from four separately built images (the Triton control from HyperQwen main
  before the FA2 change was merged, FA2 from the head of that change, the two pth arms from earlier branches that were
  later rebased onto main), and the 3090 rows mix a same-sitting decode rerun with quality rows from an earlier battery of
  the same card (M1b).
- **3080 and 3090 runs are not compared.** The cards differ in memory, SM count and TP, the context depths differ
  (8k, 32k and 128k against about 66k and 100k), and the 3090 rows are one or two boots per arm without intervals.
- **Accuracy.** The numbers are teacher-forced prompt logprobs of two documents, one run per arm, not generation quality.
  The tool's default absolute thresholds (top-1 agreement at least 99.5 %, mean |Δ| at most 0.02 nats) are met by the int8 + pth arms on the 128k
  document and missed on the 32k document (mean over all positions 0.0207 and 0.0214 nats); both fp8 arms miss the mean on both documents.
- **Sizes of the kernel-level comparisons.** K1 comes from one card at a 1905 MHz maximum clock and, at the 32/8-head shape, runs
  against the power limit; K2b was timed through a development attention backend rather than the vLLM plugin of this
  repository (see the note under the K2b table), with a single run for the bf16 column, and four of its rows did not reproduce in K2a; K3 is
  from a pre-release build of the same kernel source. See the notes under each table.
- **Image provenance.** The 3080 images vendored pre-release commits of this repository: E1 target + drafter `07e5063`, E2 `a31d8ab`, E1 target only
  the commit HyperQwen pinned before sliding-window support (its causal kernels are SASS-identical, [#338](https://github.com/syv-ai/HyperQwen/pull/338)).
  The shipped files of the two named commits differ from v0.1.0 in comments, docstrings, packaging metadata, the text of two
  error messages and one sentence of `NOTICE` only, and give the same SASS (123 of 123 kernels on sm80, sm86 and sm89, checked with `tools/compare_vendor_trees.py` and `tools/prove_sass_equal.sh`). The HyperQwen commit behind each
  image (pull-request heads and image bases) is in the software line of the table captions. No benchmark of the exact v0.1.0 tree is part of these tables; its GPU test run is in [`build.md`](build.md#7-what-has-been-run-on-a-gpu).

  What a reader can check. The pre-release commits and the tree of the GPU test run (`bd4b3c8`) are not published, so the
  statement above cannot be re-derived from them. HyperQwen's `pthfa/` directory holds a vendored copy of the same 32 files
  (`pthfa/SHA256SUMS`); this repository can be compared with it, from a checkout of this repository and one of HyperQwen
  (`$HYPERQWEN`). The first command expects the differences listed above and nothing else (it exits 0 when only comments,
  docstrings, the two error messages, `NOTICE` and the named packaging keys differ):

  ```bash
  python3 tools/compare_vendor_trees.py "$HYPERQWEN/pthfa" . --new-export \
      --exclude README.md --exclude PIN --exclude SHA256SUMS \
      --exclude check_vendor.sh --exclude install.sh --exclude tests \
      --allow-str-changes python/pth_flash_attn/vllm_backend.py 2 \
      --allow-text-diff NOTICE \
      --allow-toml project.version --allow-toml project.description \
      --allow-toml project.keywords --allow-toml project.classifiers \
      --allow-toml project.urls --allow-toml project.optional-dependencies.vllm \
      --allow-toml tool.pytest.ini_options.markers \
      --allow-toml tool.ruff.extend-exclude \
      --allow-toml tool.typos.files.extend-exclude
  ```

  The second compiles the kernels of both trees with the same `nvcc` in the development image
  ([`build.md`](build.md#3-development-container)) and compares them instruction by instruction for sm80, sm86 and sm89
  (it needs no GPU and reports `123 identical` when the machine code is the same):

  ```bash
  docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=void \
      -v "$PWD":/repo:ro -v "$HYPERQWEN"/pthfa:/vendored:ro -v /tmp/sass:/out \
      --entrypoint nice pth-flash-attn-dev -n 19 \
      /bin/bash /repo/tools/prove_sass_equal.sh --right-export --extension \
      --out /out /vendored /repo
  ```

  This proves the equivalence of the published tree to HyperQwen's vendored copy, not to the unpublished commits, and the GPU
  test run of build.md was made on the unpublished `bd4b3c8`.
- **Documents.** The logprob documents are not published; the tool that builds and compares them is
  [`bench/e2e/prompt_logprob.py`](../bench/e2e/prompt_logprob.py).

## How to read the tables

- Every table carries its hardware, power limit, software, date, configuration and source in the paragraph above it
  (built from the CSV rows). "220 W power limit" is the per-card limit set on the host; the maximum SM clock was 1905 MHz.
- The RTX 3080 20 GB rows (K1–K3, E1, E2, A1, T1) were measured on memory-modified 20 GB GA102 boards, not a retail SKU
  (retail RTX 3080 cards carry 10 or 12 GB), at a 220 W power limit with a +120 MHz graphics-clock offset set on the host
  ([hardware labels](benchmark-methodology.md#hardware-and-software-labels)). Absolute rates belong to that machine.
- Time per decode pass is the wall time of the decode-only window divided by the number of forward passes in it, as
  reported by HyperQwen's `conc_ladder.py`; decode tokens per second are per stream. "N=1" and "N=2" are the numbers of
  concurrent streams. Contexts are prompt lengths in tokens (128k = 128,000, 32k = 32,000, 8k = 8,192).
- Medians are over repetitions (3, or 5 at 8k) of one boot per arm. Ratios are ratios of medians with a 95 % bootstrap CI
  (10,000 resamples, seed 0, both samples resampled independently) and are written new ÷ old: below 1 for a time means the
  first arm is faster, above 1 for a rate means the first arm is faster.
- **Bold marks the better value of a row.** It is applied when the difference is outside the stated noise: the ratio of
  the row's best arm to every other arm has a 95 % CI that excludes 1; a paired difference whose CI excludes 0; or a
  deterministic quantity (the KV pool). Rows whose leading arms are tied, rows whose differences sit inside the noise, and
  single-run cells (n=1) are not bold, whoever leads. Where an interval exists only for the mean |Δ logprob|, the other
  accuracy metrics are not bold.
- A dash means the arm was not measured in that row.
- Figures quoted in the prose outside the tables (acceptance on the 128k ladder prompt, boot times, the second KV pool of the T4
  battery, control repeats, profile shares) come from the pull request cited beside them. They have no row in the CSV files and
  were not re-derived for this page.

## Kernel-level results

### K1: sliding-window drafter attention against Triton

The DFlash2 drafter runs five sliding-window (2,048), non-causal attention layers at D=128. The kernel timing is
independent of the context length because pages below the window are never read.

<!-- TABLE:k1:start -->
**K1. Sliding-window drafter attention against Triton.** Hardware: one RTX 3080 20 GB (GA102, sm86, 68 SMs), single card, containerised Linux, 220 W power limit. Software: vLLM 0.31.0 (Triton unified_attention), torch 2.13.0+cu130, driver 615.71.09; source pull request #338 at head `4ba0672`. Date: 2026-10-17. Configuration: D=128, q=8 query tokens, one request, sliding window 2048, non-causal, int8_per_token_head cache in the production page layout (block 864), same cache tensor and block table for both kernels; median of 200 CUDA-graph samples (20 launches per replay); maximum SM clock 1905 MHz. Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| shape, cache state | this kernel, L=2,056 (µs) | this kernel, L=131,080 (µs) | Triton, L=2,056 (µs) | Triton, L=131,080 (µs) | Triton ÷ kernel, L=2,056 | Triton ÷ kernel, L=131,080 |
|---|---:|---:|---:|---:|---:|---:|
| 16 q / 4 kv heads per rank (TP=2 shape), cold L2 | **18.69** | **18.69** | 170.89 | 171.66 | 9.14× | 9.19× |
| 16 q / 4 kv heads per rank (TP=2 shape), warm L2 | **16.74** | **16.74** | 155.33 | 154.78 | 9.28× | 9.25× |
| 32 q / 8 kv heads per rank (TP=1 shape), cold L2 | **24.88** | **24.82** | 172.03 | 171.67 | 6.91× | 6.92× |
| 32 q / 8 kv heads per rank (TP=1 shape), warm L2 | **24.16** | **24.17** | 155.03 | 154.73 | 6.42× | 6.40× |
<!-- TABLE:k1:end -->

At the 16/4-head shape the kernel is within 0.6 % of its causal variant at L=2,056 ([#338](https://github.com/syv-ai/HyperQwen/pull/338));
at the 32/8-head shape it reaches the 220 W power limit (SM clock 1800–1830 MHz), so that ratio is measured partly in a
power-limited regime. In a profile of the DFlash2 step the drafter's attention is 3.2–3.8 % of the step ([#338](https://github.com/syv-ai/HyperQwen/pull/338)),
which bounds what this kernel can add end to end (E1).

### K2: causal decode, verify and prefill at D=256 against Triton

The shapes follow the Qwen3.8-27B layout seen by one rank at TP=2 (12 query and 2 KV heads, D=256), with 16-token blocks, not the
816-token pages that vLLM uses for this head size (K3 uses those). Times are per layer. Two sets are shown. **K2a is the primary
set**: it was produced by [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) of this repository on the v0.1.0 pre-publication
export, so that anyone can repeat it. K2b is the earlier development measurement through vLLM's own micro-benchmark and a development
attention backend, kept for comparison. **Four K2b rows did not reproduce within noise**: the op is 13.6 % and 15.1 % slower in K2a at
q=4 with 128k KV (one and two requests) and 12.6 % and 14.3 % slower at the two q=8192 rows (the q=4 32k row is +2.4 %).
At q=1 the op rows agree within +1.4 to +2.5 % (8k: −8.3 %). The cause of the difference was not isolated: the SM clock of
the four non-reproducing op rows in K2a was 1110–1515 MHz at 220 W (the q=1 rows ran at 1215–1890 MHz) where K2b recorded
1770 MHz for its first run, the op was reached through a development
backend in K2b and directly in K2a, and K2a ran on the v0.1.0 pre-publication export. The ratios of K2b are therefore not quoted
elsewhere as results.

<!-- TABLE:k2p:start -->
**K2a. Causal decode, verify and prefill at D=256 against Triton, public harness.** Hardware: one RTX 3080 20 GB (GA102, sm86, 68 SMs), single card, containerised Linux, 220 W power limit. Software: pth-flash-attn: pre-publication export of v0.1.0 (commit `bd4b3c8`) plus the 3D-buffer fix of bench/op_vs_triton.py; vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09. Date: 2026-10-20. Configuration: D=256, 12 q / 2 kv heads, block size 16, int8_per_token_head cache shared by both kernels, op at the planner's split count; median of 200 CUDA-graph samples of 20 launches (the two q=8192 rows: 20 samples of 4 launches), cold L2 and warm L2; Triton: unified_attention with the split-KV (3D) buffers of vLLM's TritonAttentionImpl at q=1 (fixed harness), as shipped at q>1; command: python3 bench/op_vs_triton.py with the arguments of each row, see its --help. Source: [`bench/op_vs_triton.py`](../bench/op_vs_triton.py), [`docs/benchmark-methodology.md#kernel-comparison-against-the-triton-path`](benchmark-methodology.md#kernel-comparison-against-the-triton-path).

| batch (per layer) | this op, cold L2 (µs) | this op, warm L2 (µs) | Triton int8, cold L2 (µs) | Triton ÷ op | this op in K2b (µs) | this op against K2b |
|---|---:|---:|---:|---:|---:|---:|
| 1 request, q=1, KV 8k (µs) | 25.0 | 25.1 | 97.6 | 3.90× | 27.3 | -8.3 % |
| 1 request, q=1, KV 32k (µs) | 63.5 | 63.5 | 373.9 | 5.89× | 62.6 | +1.4 % |
| 1 request, q=1, KV 128k (µs) | 209.8 | 211.1 | 1464 | 6.98× | 206.7 | +1.5 % |
| 2 requests, q=1, KV 128k each (µs) | 412.1 | 410.0 | 1736 | 4.21× | 402.1 | +2.5 % |
| 1 request, q=4, KV 32k (µs) | 69.1 | 68.9 | 3886 | 56.24× | 67.5 | +2.4 % |
| 1 request, q=4, KV 128k (µs) | 247.2 | 247.4 | 15561 | 62.94× | 217.6 | +13.6 % |
| 2 requests, q=4, KV 128k each (µs) | 501.5 | 498.1 | 15590 | 31.08× | 435.9 | +15.1 % |
| 1 request, q=8192 prefill, no prefix (µs) | 10463 | 10731 | 41170 | 3.93× | 9289 | +12.6 % |
| 1 request, q=8192, KV 32k (µs) | 71973 | 72039 | 292834 | 4.07× | 62980 | +14.3 % |
<!-- TABLE:k2p:end -->

At q=1, K2a is the run of the harness that passes vLLM's split-KV (3D) buffers to Triton, as vLLM's own backend does; the op and
Triton columns of those four rows come from that one run. Without the buffers Triton runs its 2D kernel at q=1 and is about ten
times slower (38–75× the op; the 3.9–7.0× shown are the ratios of the 3D kernel vLLM actually runs). For q>1 and prefill
the harness needs no such buffers. One RTX 3080 20 GB, 220 W, 2026-10-20; each cell is the median of 200 samples
(20 for the two q=8192 rows), cold and warm L2.

<!-- TABLE:k2:start -->
**K2b. Causal decode, verify and prefill at D=256 against Triton, development measurement.** Hardware: one RTX 3080 20 GB (GA102, sm86, 68 SMs), single card, containerised Linux (second card of the pair, run alone), 220 W power limit. Software: vLLM 0.31.1rc1.dev355 development tree (Triton 3.7.1), torch 2.13.0+cu132, driver 615.71.09. Date: 2026-10-13. Configuration: D=256, 12 q / 2 kv heads (TP=2 share of a 24/4-head model), block size 16, 8 layers, per-layer time via CUDA-graph replays (10 replays per run, mean of the replays); int8 cache: int8_per_token_head; the op reached through a development attention backend of the same tree. Source: [`docs/benchmark-methodology.md#kernel-comparison-against-the-triton-path`](benchmark-methodology.md#kernel-comparison-against-the-triton-path).

| batch (per layer) | Triton, bf16 cache | Triton, int8 cache | this op, int8 cache | Triton int8 ÷ op |
|---|---:|---:|---:|---:|
| 1 request, q=1, KV 8k (µs) | 56.0 | 106.4 | **27.3** | 3.90× |
| 1 request, q=1, KV 32k (µs) | 202.2 | 402.2 | **62.6** | 6.42× |
| 1 request, q=1, KV 128k (µs) | 717.2 | 1464.8 | **206.7** | 7.09× |
| 2 requests, q=1, KV 128k each (µs) | 907.0 | 1726.6 | **402.1** | 4.29× |
| 1 request, q=4, KV 32k (µs) | 1980.4 | 3882.3 | **67.5** | 57.5× |
| 1 request, q=4, KV 128k (µs) | 7910.7 | 15540 | **217.6** | 71.4× |
| 2 requests, q=4, KV 128k each (µs) | 7916.2 | 15577 | **435.9** | 35.7× |
| 1 request, q=8192 prefill, no prefix (ms) | 21.42 | 38.55 | **9.289** | 4.15× |
| 1 request, q=8192, KV 32k (ms) | 165.6 | 274.6 | **62.98** | 4.36× |
<!-- TABLE:k2:end -->

**How the K2b numbers were taken.** Each cell is the time of one attention layer's forward call, measured
with vLLM's attention micro-benchmark (batch-spec rows, CUDA-graph replays timed with `do_bench_cudagraph`) in a development
snapshot of vLLM (0.31.1rc1.dev355). The batch-spec grammar writes the shapes as `q<N>` query tokens and `s<L>` KV tokens,
with a leading count for the number of requests and `k` for 1,024 (`q1s32k` is one request, one query token, 32k KV tokens). That
benchmark selects an attention backend by name. The Triton columns come from vLLM's own Triton backend. The "this op" column comes
from a backend that belongs to that development snapshot, not from the vLLM plugin shipped in this repository
([`vllm_backend.py`](../python/pth_flash_attn/vllm_backend.py)); it called the same op, `pth_flash_attn_fwd`. No script of this
repository produced K2b; [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) produced K2a with its own harness (CUDA-event timing of
graph replays on one shared int8 cache tensor, cold and warm L2). Neither table shows end-to-end speed (E1 and E2 do), the time of
the plugin's own code around the op, or results for other shapes or cards.

In K2b the Triton int8 time at four query tokens is about ten times its single-token time at the same context (15,540 µs against 1,464.8 µs at 128k); the reason was not profiled. The int8 cache costs the Triton path 1.66–2.04× its own bf16 cache time at these nine shapes (the int8 cache buys capacity, not speed, on that path). Across the three
back-to-back runs the run means of Triton int8 at q1s32k range from 373.6 to 406.6 µs (9 %) and of the op at q1s8k from 25.8 to 27.3 µs (6 %); every other row varies by at most 1.1 % (see the [methodology](benchmark-methodology.md#kernel-comparison-against-the-triton-path)).

### K3: efficiency against the measured ceilings

The efficiency η_mem is the time of a copy kernel that reads exactly the bytes of the attention kernel (same strides and
widths, data discarded) divided by the kernel's time at the same shape; η_mma is executed tensor-core flops over the time
and the measured tensor-core rate. Both ceilings are measured in the same session at the same shape.

<!-- TABLE:k3_decode:start -->
**K3a. Decode efficiency against the copy ceiling (η_mem).** Hardware: one RTX 3080 20 GB (GA102, sm86, 68 SMs), single card, containerised Linux (GPU 1 of the pair, run alone), 220 W power limit. Software: pth-flash-attn pre-release kernel library (decode kernel), CUPTI timing, nvcc 13.0.88, driver 615.71.09. Date: 2026-10-13. Configuration: D=256, Hq/Hkv=12/2, int8 cache with fp32 scales (520 B per token and head), block size 816, NHD, bf16 queries; 6 processes x 36 blocks (216), 20 CUDA-graph launches per block; efficiency against the same session's copy ceiling at the same batch and length. Source: [`bench/k_run.sh`](../bench/k_run.sh).

| shape | median (µs) | η_mem | η_mem, CI lower bound | blocks kept | SM clock (MHz) |
|---|---:|---:|---:|---:|---:|
| q=1, B=1, L=32k | 62.3 | 0.830 | 0.822 | 216/216 | 1905 |
| q=1, B=1, L=64k | 109.3 | 0.909 | 0.905 | 216/216 | 1905 |
| q=1, B=1, L=128k | 207.1 | 0.943 | 0.936 | 216/216 | 1470 |
| q=1, B=1, L=256k | 398.6 | 0.975 | 0.972 | 216/216 | 1402 |
| q=1, B=2, L=32k | 112.8 | 0.890 | 0.884 | 216/216 | 1905 |
| q=1, B=2, L=64k | 210.9 | 0.927 | 0.923 | 216/216 | 1440 |
| q=1, B=2, L=128k | 401.3 | 0.970 | 0.967 | 216/216 | 1500 |
| q=1, B=2, L=256k | 782.2 | 0.990 | 0.985 | 104/216 | 1372 |
| q=4, B=1, L=32k | 66.5 | 0.777 | 0.769 | 216/216 | 1598 |
| q=4, B=1, L=64k | 116.4 | 0.853 | 0.850 | 215/216 | 1380 |
| q=4, B=1, L=128k | 212.9 | 0.917 | 0.909 | 216/216 | 1320 |
| q=4, B=1, L=256k | 452.2 | 0.859 | 0.808 | 82/216 | 1170 |
| q=4, B=2, L=32k | 129.3 | 0.776 | 0.771 | 216/216 | 1485 |
| q=4, B=2, L=64k | 232.7 | 0.841 | 0.837 | 216/216 | 1365 |
| q=4, B=2, L=128k | 431.4 | 0.902 | 0.881 | 56/216 | 1290 |
| q=4, B=2, L=256k | 946.2 | 0.818 | 0.810 | 181/216 | 1140 |
<!-- TABLE:k3_decode:end -->

At q=1 every row from 64k on reaches η_mem 0.90 (lower bounds 0.905–0.985). At q=4 the efficiency is lower: at 220 W the SM clock
falls to 1.1–1.6 GHz when the kernel streams at full bandwidth, and only B=1 at 128k reaches 0.90 with a full set of blocks
(lower bound 0.909). The other q=4 rows that kept all or all but one of their blocks are between 0.78 and 0.85; three rows kept only 56–181 of 216 blocks (status noisy, see
the `blocks kept` column) and read 0.82–0.90 with lower bounds 0.81–0.88. The copy ceilings measured in the same run for one request were 686.0 / 697.9 / 701.6 GB/s at 64k / 128k / 256k
([methodology](benchmark-methodology.md#hardware-ceilings)).

<!-- TABLE:k3_mma:start -->
**K3b. Verify and prefill efficiency against the tensor-core ceiling (η_mma).** Hardware: one RTX 3080 20 GB (GA102, sm86, 68 SMs), single card, containerised Linux (decode: GPU 1 of the pair; prefill: GPU 0, each run alone), 220 W power limit. Software: pth-flash-attn pre-release kernel library, CUPTI timing, nvcc 13.0.88, driver 615.71.09. Date: 2026-10-13. Configuration: D=256, Hq/Hkv=12/2, int8 cache, block size 816, NHD, bf16 queries; verify rows: q=10 query tokens; prefill rows: chunk of 8192 query tokens after a prefix of P tokens; 6 processes x 36 blocks (216); efficiency on executed tensor-core flops against the same session's tensor-core ceiling (mma f16 inputs, f32 accumulate). Source: [`bench/k_run.sh`](../bench/k_run.sh).

| shape | median (µs) | η_mma | η_mma, CI lower bound | blocks kept | SM clock (MHz) |
|---|---:|---:|---:|---:|---:|
| verify q=10, B=1, L=64k | 211.0 | 0.614 | 0.611 | 162/216 | 1575 |
| verify q=10, B=1, L=128k | 401.5 | 0.646 | 0.637 | 195/216 | 1590 |
| verify q=10, B=1, L=256k | 800.7 | 0.648 | 0.643 | 216/216 | 1515 |
| verify q=10, B=2, L=64k | 424.0 | 0.611 | 0.604 | 168/216 | 1605 |
| verify q=10, B=2, L=128k | 824.2 | 0.629 | 0.626 | 216/216 | 1530 |
| verify q=10, B=2, L=256k | 1621.3 | 0.640 | 0.636 | 213/216 | 1500 |
| prefill chunk 8192, prefix 0k | 10625.7 | 0.631 | 0.626 | 216/216 | 1515 |
| prefill chunk 8192, prefix 32k | 91765.3 | 0.652 | 0.651 | 205/216 | 1508 |
| prefill chunk 8192, prefix 128k | 338572.7 | 0.648 | 0.643 | 209/216 | 1485 |
| prefill chunk 8192, prefix 200k | 525669.4 | 0.645 | 0.640 | 207/216 | 1485 |
<!-- TABLE:k3_mma:end -->

The verify rows (q=10) and the prefill rows read η_mma 0.611–0.652 at 220 W (every lower bound is at least 0.604).

## End-to-end results on two RTX 3080 20 GB

All rows below use vLLM 0.31.0 with the repository-default checkpoint family (a W4A16 27B model with 16 full-attention
layers at D=256, head layout 24 query / 4 KV heads, 12 / 2 per rank at TP=2), prefix caching on, greedy decoding.

### E1: `SPEC=dflash2 CTX=long`, same sitting

Four arms booted back to back in one sitting (int8 Triton, FA2 fp8, int8 pth for the target only, int8 pth for target and
drafter; Triton and FA2 booted again at the end). The FA2 fp8 build was the default of this launch at the time. Time per
pass is the comparison that does not depend on the prompt: acceptance depends on it (on the 128k ladder prompt FA2 accepts
3.012 tokens per step in both of its boots and the pth target + drafter arm 3.200; on the GSM8K window all four arms accept
4.79–4.84 of 7 drafts).

The target + drafter arm is also the end-to-end run of the sliding-window, non-causal kernel: the drafter's five attention
layers ran on the `CUSTOM` backend next to the target ([#338](https://github.com/syv-ai/HyperQwen/pull/338)).

<!-- TABLE:e1_ms:start -->
**E1a. DFlash2: time per decode pass (ms, median).** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; pre-release pth-flash-attn builds in HyperQwen images on base `95ee3b6`: the two pth arms from the branches that became #337 (head `7c94122`) and #338 (head `4ba0672`), the measured images predating the rebase onto those heads; FA2 built unmodified from the head of #319 as of 2026-10-14 (`71253a4`, rebased since); int8 Triton from HyperQwen main before the FA2 merge. Date: 2026-10-17 20:11 to 2026-10-18 03:08 (one sitting, one boot per arm, int8 Triton and FA2 repeated once at the end). Configuration: SPEC=dflash2 CTX=long, DFlash2 k=7 with lookup drafting, text-only, PREFIX_CACHE=1, MAX_LEN=262144, MAX_SEQS=2, GPU_UTIL=0.94, 8192 batched tokens, --disable-custom-all-reduce; ladders: 256 output tokens, greedy, medians of 3 repetitions (8k: 5); target: HyperQwen's repository-default checkpoint Qwen3.8-27B-W4A16-AutoRound-fast (assembled by its prepare/fetch_fast_variant.py, which docker/prepare.sh runs, from the base checkpoint plus the lm_head and MTP tensors of syvai/qwen3.8-27b-3090-fast-variant), drafter syvai/Qwen3.8-27B-DFlash2-W4A16; arms: int8 on TRITON_ATTN, fp8 on FlashAttention-2 (the then-default of this launch), int8 on pth-flash-attn for the target only, int8 on pth-flash-attn for target and drafter. Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| cell | int8 + Triton | fp8 + FA2 | int8 + pth, target only | int8 + pth, target + drafter |
|---|---:|---:|---:|---:|
| N=1, 128k | 55.0 | 36.8 | 29.0 | **28.4** |
| N=1, 32k | 31.0 | 28.8 | 24.6 | **24.0** |
| N=1, 8k | 25.1 | 25.0 | 23.4 | **22.8** |
| N=2, 128k | 63.3 | 54.1 | 39.0 | **38.2** |
| N=2, 32k | 35.9 | 38.6 | 30.3 | **29.3** |
| N=2, 8k | 29.3 | 30.7 | 28.0 | **27.2** |
<!-- TABLE:e1_ms:end -->

<!-- TABLE:e1_ms_ratio:start -->
**E1b. DFlash2: ratios of time per decode pass [95 % CI].** Setup as E1a (hardware, power limit, software, date, configuration). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338), [#337](https://github.com/syv-ai/HyperQwen/pull/337).

| cell | target+drafter ÷ FA2 | target+drafter ÷ target only | target only ÷ FA2 | FA2 ÷ Triton | target only ÷ Triton |
|---|---:|---:|---:|---:|---:|
| N=1, 128k | 0.770 [0.769–0.779] | 0.976 [0.975–0.983] | 0.789 [0.788–0.793] | 0.669 [0.664–0.678] | 0.528 [0.526–0.534] |
| N=1, 32k | 0.832 [0.818–0.837] | 0.973 [0.959–0.987] | 0.854 [0.844–0.857] | 0.931 [0.921–0.938] | 0.795 [0.781–0.800] |
| N=1, 8k | 0.913 [0.908–0.920] | 0.975 [0.961–0.983] | 0.936 [0.932–0.950] | 0.995 [0.989–1.004] | 0.931 [0.925–0.948] |
| N=2, 128k | 0.707 [0.702–0.716] | 0.981 [0.969–0.993] | 0.720 [0.717–0.729] | 0.855 [0.842–0.863] | 0.616 [0.607–0.626] |
| N=2, 32k | 0.759 [0.754–0.769] | 0.966 [0.959–0.987] | 0.785 [0.777–0.788] | 1.073 [1.064–1.074] | 0.843 [0.830–0.844] |
| N=2, 8k | 0.887 [0.883–0.897] | 0.973 [0.969–0.983] | 0.912 [0.908–0.916] | 1.048 [0.988–1.058] | 0.956 [0.901–0.965] |
<!-- TABLE:e1_ms_ratio:end -->

- Target + drafter needs 9–23 % less time per pass than FA2 with one stream and 11–29 % less with two, every CI excluding 1,
  and 1.9–3.4 % less than the target-only configuration (the drafter's attention is a small part of a step, K1).
- FA2 against int8 Triton: 0.669 per pass at 128k, no difference at 8k with one stream (0.995 [0.989–1.004]), slower per
  pass at 32k with two streams (1.073 [1.064–1.074]).
- Control repeats inside the sitting: int8 Triton repeated last stays within 1.9 % per pass and the second FA2 boot within
  1.2 % ([#338](https://github.com/syv-ai/HyperQwen/pull/338)). The same arms measured in an earlier sitting of 2026-10-17
  ([#325](https://github.com/syv-ai/HyperQwen/pull/325), DFlash2 section; target only) agree within 1.6 % at every decode cell
  (int8 Triton 58.3 / 101.0 / 136.1, FA2 81.4 / 111.5 / 134.1, target only 106.7 / 125.8 / 147.1 tok/s at 128k / 32k / 8k).

<!-- TABLE:e1_tps:start -->
**E1c. DFlash2: decode tokens per second (median, per stream).** Setup as E1a (hardware, power limit, software, date, configuration). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| cell | int8 + Triton | fp8 + FA2 | int8 + pth, target only | int8 + pth, target + drafter |
|---|---:|---:|---:|---:|
| N=1, 128k | 58.6 | 81.5 | 108.4 | 115.1 |
| N=1, 32k | 102.0 | 110.8 | 127.5 | 129.5 |
| N=1, 8k | 137.1 | 134.2 | 148.5 | 151.6 |
| N=2, 128k | 32.7 | 39.3 | 54.6 | 55.7 |
| N=2, 32k | 50.5 | 54.4 | 69.9 | 67.1 |
| N=2, 8k | 78.9 | 77.0 | 82.0 | 85.3 |
<!-- TABLE:e1_tps:end -->

<!-- TABLE:e1_tps_ratio:start -->
**E1d. DFlash2: ratios of decode tokens per second [95 % CI].** Setup as E1a (hardware, power limit, software, date, configuration). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338), [#337](https://github.com/syv-ai/HyperQwen/pull/337).

| cell | target+drafter ÷ FA2 | target only ÷ FA2 | FA2 ÷ Triton | target only ÷ Triton |
|---|---:|---:|---:|---:|
| N=1, 128k | 1.412 [1.227–1.448] | 1.330 [1.177–1.414] | 1.392 [1.325–1.596] | 1.851 [1.786–1.970] |
| N=1, 32k | 1.169 [1.131–1.287] | 1.151 [1.094–1.253] | 1.085 [0.888–1.121] | 1.250 [1.059–1.289] |
| N=1, 8k | 1.130 [1.002–1.191] | 1.107 [0.988–1.143] | 0.979 [0.923–1.033] | 1.084 [0.961–1.119] |
| N=2, 128k | 1.419 [1.315–1.495] | 1.390 [1.316–1.413] | 1.202 [1.156–1.233] | 1.671 [1.560–1.698] |
| N=2, 32k | 1.233 [1.190–1.391] | 1.285 [1.170–1.377] | 1.076 [0.896–1.133] | 1.384 [1.146–1.427] |
| N=2, 8k | 1.107 [1.050–1.176] | 1.065 [1.010–1.150] | 0.976 [0.919–1.029] | 1.040 [1.009–1.132] |
<!-- TABLE:e1_tps_ratio:end -->

Target + drafter against target only on tokens per second is not established (the CIs include 1;
[#339](https://github.com/syv-ai/HyperQwen/pull/339)), so no tok/s row is bold.

<!-- TABLE:e1_other:start -->
**E1e. DFlash2: prefill, KV pool, accuracy, acceptance.** Setup as E1a (hardware, power limit, software, date, configuration). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338), [`docs/benchmark-methodology.md#end-to-end-statistics`](benchmark-methodology.md#end-to-end-statistics).

| measure | int8 + Triton | fp8 + FA2 | int8 + pth, target only | int8 + pth, target + drafter |
|---|---:|---:|---:|---:|
| cold TTFT, 128k prompt (s; one request per arm) | 249.55 | 114.14 | 122.28 | 122.13 |
| cold TTFT, 32k prompt (s; one request per arm) | 27.76 | 18.25 | 18.68 | 18.65 |
| cold TTFT, 8k prompt (s; one request per arm) | 6.22 | 5.85 | 5.45 | 5.62 |
| ladder TTFT, 128k prompt, N=1 (s; median of 3 requests) | 249.7 | **113.9** | 122.0 | 122.1 |
| ladder TTFT, 32k prompt, N=1 (s; median of 3 requests) | 31.19 | **22.90** | 23.20 | 23.15 |
| ladder TTFT, 8k prompt, N=1 (s; median of 5 requests) | 6.01 | 5.53 | 5.47 | 5.47 |
| N=2 ladder TTFT, two concurrent 128k prompts (s; mean of the two, median of 3 repetitions) | 481.6 | 283.4 | 237.9 | 237.9 |
| N=2 ladder TTFT, two concurrent 32k prompts (s; mean of the two, median of 3 repetitions) | 53.8 | 43.7 | 40.5 | 40.4 |
| N=2 ladder TTFT, two concurrent 8k prompts (s; mean of the two, median of 5 repetitions) | 9.2 | 8.5 | 8.3 | 8.3 |
| KV pool (tokens) | 477,599 | **517,656** | 504,770 | 504,770 |
| GSM8K, n=200 (accuracy) | 0.955 | 0.955 | 0.955 | 0.965 |
| accepted draft length per step, GSM8K window (of 7) | 4.794 | 4.838 | 4.811 | 4.835 |
<!-- TABLE:e1_other:end -->

<!-- TABLE:e1_ttft_ratio:start -->
**E1f. DFlash2: ratios of the N=1 ladder TTFT medians [95 % CI].** Setup as E1a (hardware, power limit, software, date, configuration). Source: [`docs/benchmark-methodology.md#end-to-end-statistics`](benchmark-methodology.md#end-to-end-statistics).

| cell | target+drafter ÷ FA2 | target only ÷ FA2 | target+drafter ÷ target only | FA2 ÷ Triton |
|---|---:|---:|---:|---:|
| ladder TTFT, 128k prompt | 1.072 [1.071–1.074] | 1.071 [1.070–1.072] | 1.001 [1.000–1.004] | 0.456 [0.456–0.457] |
| ladder TTFT, 32k prompt | 1.011 [1.011–1.014] | 1.013 [1.011–1.014] | 0.998 [0.998–1.003] | 0.734 [0.733–0.735] |
| ladder TTFT, 8k prompt | 0.990 [0.988–0.993] | 0.990 [0.988–0.995] | 1.000 [0.996–1.002] | 0.920 [0.913–0.923] |
<!-- TABLE:e1_ttft_ratio:end -->

The ladder TTFT rows are the median of three requests at 128k and 32k (five at 8k), each request with a distinct salted
prompt, so none is served from the prefix cache.

### E2: `SPEC=mtp CTX=long`, default against `PTH_FA=0`

The control is `PTH_FA=0` (fp8 KV on FlashInfer); the other arm is the launcher default on a compute-capability-8.6 card with
the package installed, which resolves to `PTH_FA=1` and is the same argv as an explicit `PTH_FA=1`. Order: control, default,
control again, default again. The columns of the table are the medians of the first boot of each arm; the ratios pool both
boots of each arm. The control's first boot was 4–6 % slower per step than its repeat (the default arm's repeat matched its
first boot within 1 %), so the first-boot ratios are slightly more favourable (0.434 / 0.387 / 0.371 per pass). Tokens per
step are equal, so the gain is per step and does not come from a different acceptance. The default arm's first boot took 324 s to become healthy (102 s on the
repeat; the control 102 s both times). Kernel and graph mode are not separated (see "Read before the numbers").

<!-- TABLE:e2_main:start -->
**E2. MTP: default (int8 + pth) against PTH_FA=0 (fp8 + FlashInfer).** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; one image built from the head of HyperQwen #345 (`72f5917`), pth-flash-attn vendored from a pre-release commit comment-equivalent to v0.1.0. Date: 2026-10-19 (one sitting; same image for both arms). Configuration: SPEC=mtp CTX=long, MTP k=3, text-only, PREFIX_CACHE=1, MAX_LEN=262144, MAX_SEQS=2, GPU_UTIL=0.93 on every arm; ladders: 256 output tokens, greedy, 5/3/3 requests at 8k/32k/128k, medians of the first boot of each arm; ratios: int8+pth over fp8+FlashInfer, 95 % bootstrap CIs, pooled over both boots of each arm (order: control, kernel, control, kernel); target: HyperQwen's repository-default checkpoint Qwen3.8-27B-W4A16-AutoRound-fast (assembled by its prepare/fetch_fast_variant.py, which docker/prepare.sh runs, from the base checkpoint plus the lm_head and MTP tensors of syvai/qwen3.8-27b-3090-fast-variant); control PTH_FA=0 (fp8 KV, FlashInfer, piecewise CUDA graphs only); kernel arm: no knob, default resolves to PTH_FA=1 (int8 KV, pth-flash-attn for target and MTP drafter, full and piecewise graphs). Source: [#345](https://github.com/syv-ai/HyperQwen/pull/345).

| cell | fp8 + FlashInfer (PTH_FA=0) | int8 + pth (default) | ratio, both boots pooled [95 % CI] |
|---|---:|---:|---:|
| decode tok/s, N=1, 128k | 46.5 | **110.4** | 2.25 [2.14–2.40] |
| decode tok/s, N=1, 32k | 51.4 | **129.3** | 2.48 [2.45–2.64] |
| decode tok/s, N=1, 8k | 52.3 | **140.3** | 2.65 [2.54–2.74] |
| time per decode pass (ms), N=1, 128k | 54.9 | **23.8** | 0.449 [0.434–0.455] |
| time per decode pass (ms), N=1, 32k | 53.6 | **20.7** | 0.393 [0.386–0.405] |
| time per decode pass (ms), N=1, 8k | 54.1 | **20.1** | 0.375 [0.370–0.395] |
| decode tok/s per stream, N=2, 128k | 28.2 | **57.5** | 1.71 [1.61–2.01] |
| decode tok/s per stream, N=2, 32k | 34.9 | **68.0** | 1.93 [1.87–2.04] |
| decode tok/s per stream, N=2, 8k | 37.7 | **80.5** | 2.09 [2.02–2.17] |
| time per decode pass (ms), N=2, 128k | 64.2 | **29.4** | 0.554 [0.458–0.572] |
| time per decode pass (ms), N=2, 32k | 53.7 | **23.9** | 0.454 [0.444–0.466] |
| time per decode pass (ms), N=2, 8k | 54.4 | **22.5** | 0.415 [0.407–0.442] |
| cold TTFT, 128k prompt (s) | 129.1 | **125.6** | 0.971 [0.931–0.972] |
| cold TTFT, 32k prompt (s) | 20.2 | **18.8** | 0.955 [0.954–0.961] |
| cold TTFT, 8k prompt (s) | 5.55 | 5.66 | 1.001 [0.995–1.007] |
| KV pool (tokens) | **589,230** | 579,517 | 1.6 % smaller |
| accepted tokens per step, GSM8K window | 3.568 | 3.570 | equal |
| GSM8K, n=200 (accuracy) | 0.965 | 0.965 | equal |
| needles, determinism, preemptions, residue sweep | retrieved; pass; 0; 0 of 128 broken | retrieved; pass; 0; 0 of 128 broken | equal |
<!-- TABLE:e2_main:end -->

### A1: accuracy against a bf16-KV reference

Teacher-forced prompt logprobs of a 32k and a 128k document, scored against the same checkpoint served with a bf16 KV cache
on FLASH_ATTN. The paired differences use a non-overlapping block bootstrap (positions of one document are not
independent). The non-inferiority rule: the new arm's mean |Δ logprob| is at most the old arm's plus max(10 % of it,
0.002 nats), judged on the point estimates and with the paired CI shown beside it.

<!-- TABLE:a1_dflash2:start -->
**A1a. DFlash2 checkpoint: prompt logprobs against a bf16-KV reference.** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; source pull request #338 at head `4ba0672` (image base `95ee3b6`). Date: 2026-10-18. Configuration: teacher-forced prompt logprobs of two documents (30,616 and 127,937 scored positions), one run per arm, against a reference run with bf16 KV on FLASH_ATTN of the same checkpoint (boots at MAX_LEN=135168, GPU_UTIL=0.88); top-1 agreement is tie-aware; non-inferiority: new arm's mean |Δ| at most the old arm's plus max(10 % of it, 0.002 nats), paired non-overlapping block bootstrap (blocks of 256), 95 % CI; checkpoints of the DFlash2 end-to-end comparison (target and drafter). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| measure | int8 + Triton | fp8 + FA2 | int8 + pth (target, and target + drafter) |
|---|---:|---:|---:|
| mean \|Δ logprob\| (nats), 32k document | 0.0219 | 0.0360 | **0.0207** |
| mean \|Δ logprob\| (nats), 128k document | 0.0194 | 0.0338 | 0.0193 |
| p99 \|Δ logprob\| (nats), 32k document | 0.156 | 0.268 | 0.150 |
| p99 \|Δ logprob\| (nats), 128k document | 0.152 | 0.274 | 0.150 |
| top-1 agreement, 32k document | 0.9992 | 0.9964 | 0.9993 |
| top-1 agreement, 128k document | 0.9993 | 0.9965 | 0.9993 |
| largest single-token deviation (nats), 32k document | 4.1 | 3.1 | 2.1 |
| largest single-token deviation (nats), 128k document | 4.1 | 3.1 | 6.0 |
<!-- TABLE:a1_dflash2:end -->

The int8 + pth column is the target-only and the target + drafter configuration: the two are bit-identical on all 158,553 positions, because the drafter
does not touch prompt logprobs. Where fp8 is better: the largest single-token deviation at 128k (3.1 nats against 6.0 here and 4.1 for
int8 Triton; one token each). Bold appears only on the mean |Δ| of the 32k document, where the paired CI excludes 0 against both other arms;
at 128k int8 + pth and int8 + Triton are tied.

<!-- TABLE:a1_dflash2_ni:start -->
**A1b. DFlash2 checkpoint: paired difference of mean |Δ logprob|, new minus old [95 % CI].** Setup as A1a (hardware, power limit, software, date, configuration). Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| comparison (new vs old) | 32k document | 128k document | non-inferiority (32k / 128k) |
|---|---:|---:|---:|
| int8+pth vs int8+Triton, all positions | -0.00116 [-0.00162, -0.00069] | -0.00011 [-0.00035, +0.00014] | pass / pass |
| int8+pth vs int8+Triton, last 2048 positions | -0.00029 [-0.00262, +0.00255] | +0.00045 [-0.00110, +0.00197] | pass / pass |
| fp8+FA2 vs int8+Triton, all positions | +0.01417 [+0.01313, +0.01520] | +0.01437 [+0.01391, +0.01484] | fail / fail |
| fp8+FA2 vs int8+Triton, last 2048 positions | +0.01566 [+0.01372, +0.01781] | +0.01347 [+0.01129, +0.01546] | fail / fail |
| fp8+FA2 vs int8+pth, all positions | +0.01533 [+0.01429, +0.01633] | +0.01448 [+0.01402, +0.01494] | fail / fail |
| fp8+FA2 vs int8+pth, last 2048 positions | +0.01595 [+0.01442, +0.01732] | +0.01302 [+0.01098, +0.01501] | fail / fail |
<!-- TABLE:a1_dflash2_ni:end -->

One CI upper bound is above its margin: int8 + pth against int8 + Triton, 32k, last 2048 positions (+0.00255 against 0.00215; +0.00116 with blocks of 1,024).

<!-- TABLE:a1_mtp:start -->
**A1c. MTP checkpoint: prompt logprobs against a bf16-KV reference.** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; source pull request #345 at head `72f5917`. Date: 2026-10-19. Configuration: teacher-forced prompt logprobs of two documents (30,616 and 127,937 scored positions), one run per arm, against a reference run with bf16 KV on FLASH_ATTN of the same checkpoint (reference at MAX_LEN=135168, GPU_UTIL=0.88; 0.87 for the fp8 arm); top-1 agreement tie-aware; top-10 KL is a lower bound from a top-10 coarse-graining; non-inferiority rule as stated for the DFlash2 comparison; SPEC=mtp k=3, checkpoint of the MTP end-to-end comparison. Source: [#345](https://github.com/syv-ai/HyperQwen/pull/345).

| measure | fp8 + FlashInfer | int8 + pth |
|---|---:|---:|
| mean \|Δ logprob\| (nats), 32k document | 0.0356 | **0.0214** |
| mean \|Δ logprob\| (nats), 128k document | 0.0333 | **0.0193** |
| p99 \|Δ logprob\| (nats), 32k document | 0.265 | 0.162 |
| p99 \|Δ logprob\| (nats), 128k document | 0.270 | 0.152 |
| top-1 agreement, 32k document | 0.9955 | 0.9992 |
| top-1 agreement, 128k document | 0.9966 | 0.9993 |
| top-10 KL lower bound (nats), 32k document | 0.00176 | 0.00073 |
| top-10 KL lower bound (nats), 128k document | 0.00160 | 0.00063 |
| largest single-token error (nats), 32k document | 2.383 | 1.793 |
| largest single-token error (nats), 128k document | 5.276 | 5.324 |
<!-- TABLE:a1_mtp:end -->

<!-- TABLE:a1_mtp_ni:start -->
**A1d. MTP checkpoint: paired difference of mean |Δ logprob|, new minus old [95 % CI].** Setup as A1c (hardware, power limit, software, date, configuration). Source: [#345](https://github.com/syv-ai/HyperQwen/pull/345).

| comparison (new vs old) | 32k document | 128k document | non-inferiority (32k / 128k) |
|---|---:|---:|---:|
| int8+pth vs fp8+FlashInfer, all positions | -0.0143 [-0.0151, -0.0134] | -0.0141 [-0.0145, -0.0136] | pass / pass |
| fp8+FlashInfer vs int8+pth, all positions | +0.0143 | +0.0141 | fail / fail |
<!-- TABLE:a1_mtp_ni:end -->

The reverse comparison (fp8 + FlashInfer as the new arm) fails the rule at both documents, with CIs that exclude 0. The two documents are 30,616 and
127,937 scored positions; the 128k single-token errors are about equal (5.276 and 5.324 nats, one token each).

### T1: the HyperQwen T4 battery

HyperQwen's `bench/t4_battery.py` boots an arm and runs a restart gate, a quality cell (GSM8K n=200 and perplexity), a residue sweep, decode at about 60k and
90k tokens and a soak, and judges every arm against the int8 control with fixed bars (GSM8K at most 1.0 point lower, perplexity at most 2 % higher, decode at ~90k not lower than the control). All bars of both runs pass
(15 bars for `SPEC=dflash2`; every bar for `SPEC=mtp`). The ~90k TTFT is not a cold prefill: the battery's prompt reuses its first 60k tokens through the prefix cache.
The int8 + pth KV pool of the dflash2 battery differed between its two boots (480,359 and 503,216 tokens; cause not investigated).

<!-- TABLE:t1_dflash2:start -->
**T1a. T4 battery, SPEC=dflash2.** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; source pull request #338 at head `4ba0672` (image base `95ee3b6`). Date: 2026-10-18. Configuration: HyperQwen's bench/t4_battery.py bars (restart gate, quality cell, residue sweep, soak, decode at ~60k and ~90k); battery defaults (MAX_SEQS=4, GPU_UTIL=0.93, 131072 max length, 2048 batched tokens), stand-in corpus built from the repository's Markdown (ctx60/ctx90 calibrated to 60,015 and 90,021 prompt tokens), one run per arm, no intervals; SPEC=dflash2 CTX=long; the int8+pth arm ran with target and drafter on the kernel, its soak ran 240 minutes and the FA2 soak 60 minutes; the int8+pth arm was run from a different image than the other two and the verdicts merged. Source: [#338](https://github.com/syv-ai/HyperQwen/pull/338).

| measure | int8 + Triton | fp8 + FA2 | int8 + pth (target + drafter) |
|---|---:|---:|---:|
| decode at ~60k (tok/s) | 55.3 | 69.5 | 81.7 |
| decode at ~90k (tok/s) | 44.2 | 62.2 | 76.1 |
| cold TTFT at ~60k (s) | 76.0 | 46.8 | 49.1 |
| TTFT at ~90k, first 60k from the prefix cache (s) | 72.5 | 32.2 | 34.2 |
| GSM8K, n=200 (accuracy) | 0.960 | 0.955 | 0.965 |
| perplexity, all corpora (32,645 tokens) | 8.4013 | 8.4020 | 8.4022 |
| residues broken (of 128) | 0 | 0 | 0 |
| soak | not run | 328 rounds, 60 min | 1907 rounds, 240 min |
| KV pool, speed boot (tokens) | 481,787 | 517,370 | 480,359 |
| bars passed | n/a | pass (all 15 bars of the run) | pass (all 15 bars of the run) |
<!-- TABLE:t1_dflash2:end -->

<!-- TABLE:t1_mtp:start -->
**T1b. T4 battery, SPEC=mtp.** Hardware: two RTX 3080 20 GB (GA102, sm86), TP=2, no P2P, PCIe Gen3 x16, containerised Linux, 220 W power limit. Software: vLLM 0.31.0, torch 2.13.0+cu130, driver 615.71.09; source pull request #345 at head `72f5917`. Date: 2026-10-19. Configuration: HyperQwen's bench/t4_battery.py bars with two arms added in a local copy; launcher defaults on both arms (MAX_SEQS=8, GPU_UTIL=0.93, MAX_LEN=150000, 2048 batched tokens), stand-in corpus built from the repository's Markdown, 60-minute soaks instead of 240, one run per arm, no intervals; SPEC=mtp CTX=long, MTP k=3, repository-default checkpoint. Source: [#345](https://github.com/syv-ai/HyperQwen/pull/345).

| measure | fp8 + FlashInfer (PTH_FA=0) | int8 + pth (default) |
|---|---:|---:|
| restart gate, quality cell, soak exit codes | pass | pass |
| residues broken (of 128) | 0 | 0 |
| GSM8K, n=200 (accuracy) | 0.955 | 0.965 |
| perplexity, all corpora | 8.4054 | 8.4015 |
| decode at ~60k (tok/s) | 46.9 | 117.7 |
| decode at ~90k (tok/s) | 46.1 | 111.0 |
| cold TTFT at ~60k (s) | 49.0 | 47.8 |
| TTFT at ~90k, first 60k from the prefix cache (s) | 32.8 | 32.1 |
| soak rounds in 60 minutes (tokens per step) | 415 (2.97) | 489 (2.89) |
| KV pool (tokens) | 596,938 | 592,462 |
<!-- TABLE:t1_mtp:end -->

## Independent runs on a native RTX 3090 (M1)

The HyperQwen maintainer ran the default configuration on a native RTX 3090 (TP=1, 250 W, vLLM 0.31.0, the stack merged with HyperQwen main)
and reported it in comments on the public pull requests. The figures below are quoted from those comments; the repository author did not run them.
M1a and M1b are one boot per arm without intervals. M1c is the maintainer's native A/B on the tree of
[#345](https://github.com/syv-ai/HyperQwen/pull/345) (head `72f5917`), two boots per arm, run before that pull request was merged as `b4b97ec` on
2026-10-20; it has no intervals either. Nothing but the KV pool is bold.

<!-- TABLE:m1_mtp:start -->
**M1a. RTX 3090, SPEC=mtp (reported by the HyperQwen maintainer).** Hardware: one RTX 3090 24 GB (GA102, sm86), TP=1, native Linux, 250 W power limit. Software: vLLM 0.31.0. Date: 2026-10-18 (comment posted 2026-10-18). Configuration: SPEC=mtp CTX=long; fp8+FlashInfer (shipped) against PTH_FA=1; one boot per arm; decode tok/s is HyperQwen's labd_bench.py total at ~66k context. Source: [#325 comment](https://github.com/syv-ai/HyperQwen/pull/325#issuecomment-6250055160).

| measure | fp8 + FlashInfer | int8 + pth | difference stated in the source |
|---|---:|---:|---:|
| decode at ~66k (tok/s) | 77.5 | 84.5 | +9 % |
| cold TTFT at ~66k (s) | 75.1 | 75.4 | – |
| GSM8K, n=200 (accuracy) | 0.960 | 0.955 | – |
| api_smoke | 12/12 | 12/12 | – |
| KV pool (tokens) | **202,040** | 198,241 | 1.9 % smaller |
<!-- TABLE:m1_mtp:end -->

<!-- TABLE:m1_dflash2:start -->
**M1b. RTX 3090, SPEC=dflash2 (reported by the HyperQwen maintainer).** Hardware: one RTX 3090 24 GB (GA102, sm86), TP=1, native Linux, 250 W power limit. Software: vLLM 0.31.0. Date: 2026-10-19 (comment posted 2026-10-19). Configuration: SPEC=dflash2 CTX=long; FA2 decode rerun in the same sitting as the int8+pth arm; the int8+Triton and FA2 quality rows (GSM8K, perplexity) and the pools are from the same card's earlier battery; one boot per arm; decode tok/s is HyperQwen's labd_bench.py total. Source: [#339 comment](https://github.com/syv-ai/HyperQwen/pull/339#issuecomment-6262787058).

| measure | int8 + Triton | fp8 + FA2 | int8 + pth (target + drafter) | int8 + pth against FA2, as stated in the source |
|---|---:|---:|---:|---:|
| decode at ~66k (tok/s) | – | 55.6 | 62.4 | +12.2 % |
| decode at ~100k (tok/s) | – | 48.0 | 53.2 | +10.8 % |
| cold TTFT at ~100k (s) | – | 54.8 | 53.2 | – |
| GSM8K, n=200 (accuracy) | 0.955 | 0.950 | 0.970 | – |
| perplexity, all corpora | 8.4017 | 8.4048 | 8.4027 | – |
| residues broken (of 128) | 0 | 0 | 0 | – |
| soak, 20 minutes | – | pass | pass (96 rounds, 2.95 tokens per step) | – |
| KV pool (tokens) | 136,429 | **140,902** | 136,429 | 3.2 % smaller than FA2 |
| restarts or engine deaths | 0 | 0 | 0 | – |
<!-- TABLE:m1_dflash2:end -->

<!-- TABLE:m1_mtp_ab:start -->
**M1c. RTX 3090, SPEC=mtp, A/B on the merged default (reported by the HyperQwen maintainer).** Hardware: one RTX 3090 24 GB (GA102, sm86), TP=1, native Linux, 250 W power limit. Software: vLLM 0.31.0, driver 580.173.02; HyperQwen #345 tree at `72f5917`. Date: 2026-10-20 (comment posted 2026-10-20). Configuration: SPEC=mtp CTX=long PREFIX_CACHE=1 with the launcher defaults; control: PTH_FA=0 (fp8 + FlashInfer), default: PTH_FA unset (resolving to PTH_FA=1, int8 + pth); boot order control, default, default, control; native Ubuntu; decode tok/s is HyperQwen's labd_bench.py total on a frozen corpus at 66,270 and 99,801 prompt tokens (--ctx 53000 / 85000); where a cell holds two numbers they are the two boots of the arm, or the two depths where the row says so; GSM8K, perplexity and api_smoke are from the first boot of each arm, the residue sweep and the soak from the second boot of the default arm. Source: [#345 comment](https://github.com/syv-ai/HyperQwen/pull/345#issuecomment-6290008416).

| measure | fp8 + FlashInfer (PTH_FA=0) | int8 + pth (default) | difference stated in the source |
|---|---:|---:|---:|
| piecewise-only fallback lines in the boot log, per boot | 1 / 1 | 0 / 0 | – |
| KV pool (tokens) | **202,040** | 198,241 | 1.9 % smaller |
| decode at ~66k (tok/s), per boot | 75.6 / 75.5 | 84.8 / 84.7 | +12.2 % |
| decode at ~100k (tok/s), per boot | 68.9 / 68.8 | 78.5 / 78.5 | +14.0 % |
| tokens per step, ~66k / ~100k | 2.49 / 2.51 | 2.52 / 2.56 | equal within 2 % |
| ms per step, ~66k / ~100k (derived) | 32.9 / 36.5 | 29.7 / 32.6 | −10 % / −11 % |
| cold TTFT at ~66k (s), per boot | 74.8 / 75.0 | 75.0 / 75.2 | equal |
| GSM8K, n=200 (accuracy) | 0.960 | 0.965 | – |
| perplexity, all corpora | 8.4049 | 8.4022 | – |
| api_smoke | 12/12 | 12/12 | – |
| residue sweep | – | 0 broken of 128 | – |
| soak, 20 minutes | – | rc 0, 131 rounds x 4, 2.86 tokens per step | – |
| engine deaths / preemptions | 0 / 0 | 0 / 0 | – |
<!-- TABLE:m1_mtp_ab:end -->

In M1c every boot of the default decoded faster than every boot of the control at both depths, and the two boots of one arm
agree to 0.1 tok/s. Tokens per step are equal within 2 %, so the gain is per step. The control logs the piecewise-only
fallback line and the default does not, so the step-time ratio is again the kernel and the graph mode together, as in E2.

The decode figures are HyperQwen's `labd_bench.py` total. The 3090 gains (+9 % to +14.0 %) and the 3080 gains of E1 and E2 are not comparable: different card,
tensor parallelism, context depths and configuration (see "Read before the numbers").

## What was not measured

- Only compute capability 8.6 was measured. The build and the CPU gates cover sm80 and sm89; no sm80 or sm89 timing exists.
- Only one model family (a W4A16 27B model with D=256 full-attention layers and a D=128 drafter) and one engine (vLLM 0.31.0).
  No other model, head size at the engine level, or engine (no SGLang, TensorRT-LLM or similar integration) was measured.
- No confidence intervals on a 3090 (M1c has two boots per arm in one sitting but no interval), no TP=1 run on the 3080 pair, no logprob comparison against a
  bf16 cache on a 3090, no run beyond two concurrent streams, no end-to-end run above 128k context.
- No run of the kernel arm forced to piecewise CUDA graphs (E2), no int8 arm at GPU utilisation 0.94 on the MTP checkpoint.
- No re-measurement of K2b and K3 with the release build: both are pre-release builds of the same kernel source (K2a ran on the v0.1.0 pre-publication export). That the release
  builds the same machine code is shown in steps, not by a direct comparison of the K2b or K3 builds with it: the windowed kernels left
  105 of 105 causal kernels identical ([#338](https://github.com/syv-ai/HyperQwen/pull/338)), and the release differs from the
  end-to-end pins in comments, docstrings, packaging metadata, two error messages and one sentence of `NOTICE` only (see "Image provenance" above). The commits of the
  K2b and K3 builds are not recorded in the result files, so the chain is not verified for them. The scripts for repeating the comparisons are
  [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) and [`bench/k_run.sh`](../bench/k_run.sh).
- Results that other users posted in comments on the same pull requests are not included: the M1 rows are the HyperQwen maintainer's runs only.

## Reproducing and regenerating

- Kernel level: [`bench/op_vs_triton.py`](../bench/op_vs_triton.py) (K1, and a K2-style comparison), [`bench/ceilings_run.sh`](../bench/ceilings_run.sh) then
  [`bench/k_run.sh`](../bench/k_run.sh) (K3).
- End to end: launch HyperQwen's `single-user/start_qwen.sh` with `PTH_FA=1` and `PTH_FA=0`, run
  [`bench/e2e/ladder.sh`](../bench/e2e/ladder.sh) against each server, HyperQwen's `seat_ttft.py`, `quality_battery.py` and `t4_battery.py`, and
  [`bench/e2e/prompt_logprob.py`](../bench/e2e/prompt_logprob.py) for A1. Details and the statistics are in the
  [methodology](benchmark-methodology.md).
- Regenerate this page's tables after editing a CSV: `python bench/results/render_tables.py --write`; check them with
  `python bench/results/render_tables.py --check` or `pytest tests/cpu/test_benchmark_tables.py`. Both commands also cover the
  headline table embedded in the repository [README](../README.md).
