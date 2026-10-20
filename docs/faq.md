<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn -->
# Frequently asked questions

Short answers. Numbers are labelled with their source; measurements are in
[`benchmarks.md`](benchmarks.md). Statements that are not backed by a
measurement say so.

## Scope and use

**1. What does this kernel do that vLLM cannot already do?**
It reads the `int8_per_token_head` KV cache with a hand-written CUDA kernel on
sm80, sm86 and sm89, using tensor cores. In vLLM 0.30 and 0.31 the only
attention backend that reads that cache is the Triton backend. The cache format
and the writer are unchanged; only the reader is new ([`vllm.md`](vllm.md)).

**2. Does it change what vLLM writes to the cache?**
No. The backend subclasses vLLM's Triton backend and implementation, so the cache
specification, the page layout, the scale views and the writer are vLLM's own.
Switching between `TRITON_ATTN` and this backend needs no cache migration.

**3. Can I use it without vLLM?**
Yes. The op `torch.ops.pth_flash_attn.pth_flash_attn_fwd` and the planner import
only PyTorch and the standard library. An engine supplies a paged int8 cache with
float32 per-token-head scales, the block table and sequence lengths, two
workspaces, and a writer with the matching scale convention
([`integration.md`](integration.md)).

**4. Which GPUs, head sizes and features are supported?**
NVIDIA sm80, sm86 and sm89; head sizes 64, 128 and 256; bf16 and fp16 queries;
int8 KV only; forward only; causal attention at every head size and sliding
window or non-causal attention at head size 128 only. The table is in the
[README](../README.md#requirements-and-limits). Only sm86 has been measured.

**5. What happens on an H100 (sm90), an RTX 50 series card or a Turing card (sm75)?**
The op reports an unsupported compute capability and the vLLM backend refuses the
selection with the reason; the user picks another backend. Nothing is silently
JIT-compiled or substituted.

**6. Why is there no PTX, only SASS?**
A JIT-compiled kernel on an untested architecture would be code nobody measured.
The library carries cubins for exactly the three architectures it was built for,
and a newer GPU is refused with a message.

**7. Is the output deterministic? Is it batch-invariant?**
Deterministic for a given batch (the combine step uses a fixed order); not
batch-invariant, because the split count depends on the batch shape, so a
request's last bits can differ between batch compositions.

**8. Which models cannot use it?**
Anything that needs attention sinks, a logits soft cap, ALiBi, a bidirectional
multimodal prefix, encoder or cross attention, head sizes other than 64, 128 or
256, a GQA group above 64, caches other than `int8_per_token_head`, context
parallelism, or sliding-window or non-causal layers at head sizes 64 and 256.

**9. Why is the int8 KV pool a little smaller than with fp8?**
An int8 cell stores two float32 scales: `2 D + 8` bytes per token and KV head
against `2 D` for fp8. For equal memory that is about 1.5 % fewer tokens at
D = 256 and about 3 % at D = 128 (derived from the byte counts).

## Integration

**10. Why is the backend called `CUSTOM`?**
vLLM resolves a backend with `AttentionBackendEnum[backend.get_name()]`, and a
plugin cannot add an enum member, so any other name raises `KeyError`. `CUSTOM`
is the one member reserved for plugins. If another plugin already owns it, this
one leaves it alone and warns ([`vllm.md`](vllm.md), section 6).

**11. Why does the drafter need its own `attention_backend` key?**
Speculative drafters take their backend from the speculative configuration, so
`--attention-backend CUSTOM` alone leaves the drafter elsewhere
([`vllm.md`](vllm.md), section 2).

**12. Does the plugin read environment variables?**
No. Selection is only through vLLM's `--attention-backend`; launch geometry is
derived by the planner from the batch shape.

**13. Does the kernel care about the block size? vLLM sometimes uses 816.**
It assumes only a block size of at least 1 (the vLLM backend requires a multiple
of 16): each token's block and offset come from a precomputed-multiplier
division, so a 64-token tile straddling a block boundary costs nothing special.
Block sizes such as 816 arise when vLLM matches the attention page to another
layer type's page in a hybrid model.

**14. Why does the kernel read the query length from the device?**
With adaptive speculative verification the device may verify fewer drafts than
the host scheduled, so the host's query lengths can be too large. The kernel
reads every length on the device, which also lets one captured CUDA graph serve
every length. `tests/gpu/test_backend_op.py` checks both cases.

## Correctness and performance

**15. How is correctness judged?**
There is no tuned tolerance. `python/pth_testing/bound.py` computes a
per-element bound from the actual data and the kernel's tiling; zero violations
are allowed, the tightness `max(err / bound)` must stay in a fixed window (so the
bound is not vacuous), and mutation tests (an off-by-one token, a wrong scale)
must fail ([`build.md`](build.md#5-test-tiers)).

**16. How is speed judged?**
Kernel-level results compare against vLLM's Triton path on the same cache (K1,
K2) and against copy and tensor-core ceilings measured on the card (K3);
end-to-end results compare launch configurations of one serving stack. All are
labelled with hardware, power limit, software and source. The method, including
noise floors and the rule that a claimed gain must exceed twice the measured
run-to-run noise, is in [`benchmark-methodology.md`](benchmark-methodology.md);
results are in [`benchmarks.md`](benchmarks.md).

**17. Why was a 220 W power limit used in the measurements?**
The measurements were taken on two RTX 3080 20 GB cards (GA102, sm86) run at a
220 W power limit (the card default is 320 W). That is the limit at which the
measured deployment runs these cards, so measuring there reproduces the condition
of a user of that deployment; it is not a tuned optimum. Every sustained
measurement is labelled with the limit. At the limit the SM clock moves with
load, which is why benchmark rows state the power limit and, where it matters, the
clocks ([`benchmark-methodology.md`](benchmark-methodology.md)). The ceilings
were measured once at 320 W: the copy ceilings were about 0.5 % higher and the
tensor-core ceiling followed the SM clock. Results at the two limits are never
mixed in one comparison.

## Project

**18. How do I report a bug, contribute or cite the project?**
See [`CONTRIBUTING.md`](../CONTRIBUTING.md) (sign-off, checks),
[`SECURITY.md`](../SECURITY.md) and [`CITATION.cff`](../CITATION.cff). Support is
best effort.

**19. What is the licence?**
Apache-2.0; no third-party source is included in the built or installed code
([`NOTICE`](../NOTICE)).
