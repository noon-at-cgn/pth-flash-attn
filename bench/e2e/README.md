<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
-->
# End-to-end helpers

Helpers for measuring this kernel inside a running vLLM server, using the benchmark harness of
[HyperQwen](https://github.com/syv-ai/HyperQwen). They need only the Python standard library and bash.
Launching the servers is outside this directory: HyperQwen's
[`single-user/start_qwen.sh`](https://github.com/syv-ai/HyperQwen/blob/main/single-user/start_qwen.sh)
starts one, for example `SPEC=mtp CTX=long PTH_FA=1 bash single-user/start_qwen.sh` (this kernel)
or the same line with `PTH_FA=0` (the other attention path).

| File | Purpose |
|---|---|
| [`ladder_salt.py`](ladder_salt.py) | Runs a benchmark script with `time.time()` pinned to an integer, so that two servers receive byte-identical prompts. |
| [`ladder.sh`](ladder.sh) | Runs HyperQwen's `bench/conc_ladder.py` for one server at 8k, 32k and 128k context, salt pinned through `ladder_salt.py`. |
| [`prompt_logprob.py`](prompt_logprob.py) | Teacher-forced per-position logprob dumps and the comparisons between them (`doc`, `dump`, `compare`, `refcompare`). |

## Throughput ladder

`conc_ladder.py` salts each prompt with the wall clock, so two runs never share prompts and
prompt-dependent quantities such as speculative-decoding acceptance differ between arms. `ladder.sh`
fixes the salt and runs one invocation per repetition (`--n 1,2 --out 256 --reps 1`), because the
benchmark reports only the best repetition of an invocation:

| Depth | Prompt tokens | Repetitions | Labels |
|---|---|---|---|
| 8k | 8192 | 5 | `8kr1` ... `8kr5` |
| 32k | 32000 | 3 | `32kr1` ... `32kr3` |
| 128k | 128000 | 3 | `128kr1` ... `128kr3` |

```bash
export HYPERQWEN_DIR=/path/to/HyperQwen
bench/e2e/ladder.sh --port 8000 --out-dir results/arm-new      # all depths
bench/e2e/ladder.sh --port 8000 --out-dir results/arm-old 128k   # one depth
bench/e2e/ladder.sh --dry-run                                    # print the commands only
```

Defaults: `PORT=8000`, `PYTHON=python3`, `SALT_EPOCH=1791800000` (the value of every published run),
`OUT_DIR=./ladder-results`; `REPS_8K`, `REPS_32K` and `REPS_128K` override the repetition counts; `--help` lists the flags. Per repetition the directory receives
`conc_ladder_<depth>_r<k>.json` (the raw result) and `.log` (console output). Use the same `SALT_EPOCH`
for every arm, and start each arm on a freshly launched server so that equal salts never meet a warm
prefix cache. The exit status is the number of failed repetitions (at most 125); 64 means a usage error or a missing prerequisite.

## Logprob divergence against a bf16-KV reference

The reference is the same checkpoint served with a bf16 KV cache on the FLASH_ATTN backend. The
server settings that select this depend on the vLLM version in use (see its documentation for the
KV-cache dtype and attention-backend options).
Each dump is one request that teacher-forces a long document through a server, so all servers must
use the same tokeniser, and the context length must exceed the document.

```bash
# 1. One document, built once from this repository's docs/ (or from --source PATH, repeatable)
python3 bench/e2e/prompt_logprob.py doc doc.txt --tokens 32000

# 2. One dump per server, one server at a time (--top K with K > 1 adds the KL lower bound; the published A1c rows used 10)
export VLLM_API_KEY=...        # only when the server requires a key; otherwise leave unset
python3 bench/e2e/prompt_logprob.py dump doc.txt ref.json --port 18020 --top 10   # bf16-KV reference
python3 bench/e2e/prompt_logprob.py dump doc.txt old.json --port 18020 --top 10   # incumbent arm
python3 bench/e2e/prompt_logprob.py dump doc.txt new.json --port 18020 --top 10   # arm under test

# 3. Compare
python3 bench/e2e/prompt_logprob.py compare ref.json new.json
python3 bench/e2e/prompt_logprob.py refcompare new.json old.json ref.json --raw per_position.csv.gz
```

`--tokens` is converted to characters at 4.2 characters per token, so the dump's own token count is
close to, not exactly, the requested one; the server's tokeniser decides. The key is read from
`VLLM_API_KEY`, or from the variable named by `--api-key-env`; no `Authorization` header is sent when
no key is set. The model name is omitted from requests unless `--model` or `VLLM_MODEL` is given.

Memory limits may require a shorter `MAX_LEN` and a lower `GPU_UTIL` for the reference (bf16 KV is
twice the size of fp8 or int8 KV) and for an fp8-KV arm; the document must still fit. A shorter
document is a different measurement and is compared only with dumps of the same document.

### Outputs

* `compare` prints mean, median, 99th-percentile and maximum |delta logprob| of the actual token,
  top-1 agreement and, with top-K dumps, a KL lower bound, for `all` positions (from position 1) and
  for the `tail` (last 2048 by default, the positions that attend over the whole document). The default
  thresholds are top-1 agreement >= 99.5 % and mean |delta| <= 0.02 nats in both rows; exit status 1 on
  FAIL.
* `refcompare` repeats those metrics for NEW and OLD against the reference, and reports the paired
  difference `|new - ref| - |old - ref|` with a 95 % block-bootstrap interval (block lengths 256 and
  1024, seed 0, 4000 resamples). NEW is non-inferior when mean |new - ref| <= mean |old - ref| +
  max(10 % of mean |old - ref|, 0.002 nats) in both rows; that point verdict sets the exit status, and
  whether the interval's upper end stays below the margin is printed beside it.
* `--raw` writes every position's values as CSV.gz for external analysis.

## Method and results

* [`../../docs/benchmark-methodology.md`](../../docs/benchmark-methodology.md): how the numbers are
  produced and what each caveat means.
* [`../../docs/benchmarks.md`](../../docs/benchmarks.md): the published results.
