<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Documentation index

- [`../README.md`](../README.md): what the project is, the problem it solves, headline measurements, quick start, limits.
- [`architecture.md`](architecture.md): components and data flow from vLLM to the tensor cores.
- [`api.md`](api.md): the op schema, every argument's layout and dtype, the planner API, workspace sizing, error messages, CUDA-graph rules.
- [`integration.md`](integration.md): the contract for calling the op from another engine, including the quantiser the cache writer must match.
- [`vllm.md`](vllm.md): the vLLM plugin, flags (including the speculative-config key for drafters), supported versions, conflicts, a public deployment example.
- [`build.md`](build.md): requirements, installation, the development container, the local CI script, the test tiers and what has been run on a GPU.
- [`benchmarks.md`](benchmarks.md): the benchmark tables, each labelled with hardware, power limit, software versions, date and source, and the caveats that go with them.
- [`benchmark-methodology.md`](benchmark-methodology.md): how a timing becomes a claim.
- [`troubleshooting.md`](troubleshooting.md) and [`faq.md`](faq.md): refusals, error messages and common questions.

Repository files: [`../CONTRIBUTING.md`](../CONTRIBUTING.md),
[`../SECURITY.md`](../SECURITY.md), [`../CHANGELOG.md`](../CHANGELOG.md) and
[`../CITATION.cff`](../CITATION.cff).
