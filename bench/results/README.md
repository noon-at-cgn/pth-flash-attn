<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
-->
# Curated benchmark results

Each file holds exactly the numbers shown in [`docs/benchmarks.md`](../../docs/benchmarks.md), one number per row. Raw
logs, traces and per-request dumps are not published; the method that produced each number is in
[`docs/benchmark-methodology.md`](../../docs/benchmark-methodology.md).

| File | Tables of the page | Content |
|---|---|---|
| [`k1_drafter_microbench.csv`](k1_drafter_microbench.csv) | K1 | sliding-window drafter attention against Triton, one card |
| [`k2_public_harness_20261020.csv`](k2_public_harness_20261020.csv) | K2a | the same shapes timed by `bench/op_vs_triton.py` on the v0.1.0 pre-publication export (primary set), with the K2b op times and the difference |
| [`k2_causal_d256_vs_triton.csv`](k2_causal_d256_vs_triton.csv) | K2b | causal decode, verify and prefill at D=256 against Triton (bf16 and int8 caches); timed through a development attention backend, see [K2](../../docs/benchmarks.md#k2-causal-decode-verify-and-prefill-at-d256-against-triton) |
| [`k3_roofline.csv`](k3_roofline.csv) | K3a, K3b | efficiency against the measured copy and tensor-core ceilings |
| [`e1_dflash2_3080_20261017.csv`](e1_dflash2_3080_20261017.csv) | E1a-E1f | `SPEC=dflash2` end to end, two RTX 3080 20 GB |
| [`e2_mtp_3080_20261019.csv`](e2_mtp_3080_20261019.csv) | E2 | `SPEC=mtp` end to end, two RTX 3080 20 GB |
| [`a1_logprob.csv`](a1_logprob.csv) | A1a-A1d | prompt-logprob accuracy against a bf16-KV reference |
| [`t1_battery.csv`](t1_battery.csv) | T1a, T1b | HyperQwen T4 battery results |
| [`m1_hq_3090.csv`](m1_hq_3090.csv) | M1a, M1b | runs on a native RTX 3090 reported by the HyperQwen maintainer |

## Columns

`table,row,metric,unit,arm,value,ci95_low,ci95_high,n,date,hardware,power_w,software,config,source_url,notes,bold`

| Column | Meaning |
|---|---|
| `table` | table identifier of the page (`k1`, `e1_ms`, ...); a file may hold several tables |
| `row`, `arm` | the row label and the column key of the cell in that table |
| `metric`, `unit` | what the number is and its unit (`ms`, `tok/s`, `tokens`, `ratio`, `nats`, `fraction`, ...) |
| `value` | the number exactly as published, or a short text for non-numeric cells |
| `ci95_low`, `ci95_high` | 95 % interval of the value where the source states one; for `eta_lo` rows the value itself is the lower bound |
| `n` | repetitions, requests, blocks or runs behind the value (`1` for a single run) |
| `date`, `hardware`, `power_w`, `software`, `config` | the label of the measurement: date, hardware and count, per-card power limit in watts, software versions, configuration |
| `source_url` | a public pull-request URL, or a path of this repository (with an optional `#anchor`) naming the method that produced a number that is not published elsewhere |
| `notes` | remarks on the cell |
| `bold` | `1` where the page marks the value as the better one of its row (rule: [how to read the tables](../../docs/benchmarks.md#how-to-read-the-tables)) |

Rows whose `source_url` is a pull request take their values from the text of that pull request; rows whose `source_url` is a
repository path (K2a, K2b, K3, the ladder TTFT rows of E1e and E1f) have no public source and can be repeated with the scripts named in
[`docs/benchmarks.md`](../../docs/benchmarks.md#reproducing-and-regenerating); the K2b rows were timed with vLLM's own attention micro-benchmark, not with those scripts, and name the methodology page that describes how.

## Regenerating the page

```bash
python bench/results/render_tables.py            # print every table block
python bench/results/render_tables.py --check    # exit 1 when docs/benchmarks.md or the README headline differs from the CSV files
python bench/results/render_tables.py --write    # rewrite the blocks of docs/benchmarks.md and the README headline in place
```

`--check` and `--write` also cover the headline table embedded in the repository [README](../../README.md): the block between
`<!-- HEADLINE_TABLE:start -->` and `<!-- HEADLINE_TABLE:end -->` there is a verbatim copy of the one in
[`docs/benchmarks.md`](../../docs/benchmarks.md).

[`render_tables.py`](render_tables.py) holds the column layout of every table (`SPECS`) and builds the caption above each table and the summary table
from the rows. [`tests/cpu/test_benchmark_tables.py`](../../tests/cpu/test_benchmark_tables.py) asserts that the page equals the rendered output, that the README headline is the page's headline block, that every row has a source,
and that the repository paths named as sources exist.
