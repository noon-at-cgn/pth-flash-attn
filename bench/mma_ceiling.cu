// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// mma_ceiling.cu: the tensor-core ceiling F and the graph-node overhead.
//
// WHAT IT MEASURES AND WHY
//   MMA-bound targets are eta_mma = executed flops / (F * T) >= 0.70 (gate
//   0.60, floor row 0.50 for prefill). F must be what *this card* sustains
//   under its power limit (220 W on the reference RTX 3080 20 GB) with this
//   kernel's instruction (mma.sync m16n8k16, fp16 inputs, fp32 accumulate) at
//   its occupancy (4 warps, 1 CTA/SM) and in the best case. Spec sheets
//   (59.5 TF at 1.71 GHz) say nothing about the clock the power limit allows
//   while the tensor cores are busy, so the ceiling is measured.
//
// MODES
//   --mode mma             arms "<kind>:<warps per CTA>:<CTAs per SM>:<chains>"
//                          kind in f16f32 (HMMA.16816.F32), f16f16
//                          (HMMA.16816.F16), bf16f32 (HMMA.16816.F32.BF16),
//                          s8s32 (IMMA.16832.S8.S8). e.g. f16f32:4:1:8 is the
//                          kernel's occupancy, f16f32:8:2:8 a best case (4
//                          warps/SMSP).
//   --mode graph-overhead  arms "nodes:<N>": a CUDA graph of N empty kernels
//                          chained like a captured vLLM step; per-node cost is
//                          the slope between two N (bench/bench_report.py).
//   --device-info          device JSON.
//
// HOW THE MMA LOOP IS BUILT
//   * CHAINS independent accumulators per warp (template, 1/2/4/8): with one
//     chain each mma waits for the previous one (latency-bound); with enough
//     chains the tensor pipe is the limit. The chain sweep at the kernel's
//     occupancy shows where latency stops mattering; 8 is the default because
//     at 1 warp per SMSP it supplies 8 independent HMMAs per dependency window,
//     which covers a latency of up to 8x the issue interval (Ampere
//     HMMA latency is assumed to be about 2x its issue interval at fp32
//     accumulate; the sweep checks).
//   * The loop body does every chain twice, with B and with -B, so the
//     accumulators oscillate around their start value: no overflow in fp16
//     accumulate however long the loop runs, and bits keep toggling, so power
//     draw (which sets the clock under the power limit) is that of real data.
//     Zero or constant operands would draw less power and overstate F.
//   * Operands are pseudo-random from a runtime seed (no constant folding);
//     the folded accumulators are stored only if they equal a runtime
//     sentinel, so nothing can be dead-code-eliminated. The loop stays a loop
//     (iteration count is a runtime argument); SASS evidence:
//     bench/ceilings_build.sh checks it.
//   * Iterations are calibrated per arm on the device so one launch lasts
//     --target-ms (2 ms): long against launch gaps and timer resolution, short
//     enough that a 20-replay block (40 ms) fits between 100 ms NVML samples.
//   * Warp 0 of each CTA records clock64 and %globaltimer around the loop:
//     cycles / ns is the effective SM clock under the power limit, and cycles
//     per mma per SMSP separates "slow clock" from "slow pipe".
//
// TIMING: CUPTI kernel durations (bench_harness.cuh), 20 launches per block in
// one CUDA graph, blocks in the mirrored order from `bench/stats.py schedule`.

#include "bench_harness.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>

namespace {

using bench::Args;
using bench::die;
using bench::Json;
using bench::KernelTimer;

enum class Kind { kF16F32, kF16F16, kBF16F32, kS8S32 };

// flop (or int op) per warp-level mma: 2*M*N*K.
constexpr int64_t kFlopM16N8K16 = 2 * 16 * 8 * 16;  // 4096
constexpr int64_t kOpsM16N8K32 = 2 * 16 * 8 * 32;   // 8192
constexpr int kMaxThreads = 1024;
constexpr int kSmspPerSm = 4;  // Ampere/Ada: 4 sub-partitions per SM
// Calibration: run this many loop iterations once, time it, scale.
constexpr int kCalibIters = 1024;

// ---------------------------------------------------------------- mma wrappers
// PTX ISA "Warp-level matrix multiply-accumulate: mma" (m16n8k16 .f16/.bf16,
// m16n8k32 .s8). A: 4 x b32, B: 2 x b32 per thread; C/D: 4 x f32/s32 or
// 2 x f16x2.
template <Kind K>
struct Mma;

template <>
struct Mma<Kind::kF16F32> {
  static constexpr int64_t kFlop = kFlopM16N8K16;
  __device__ static void run(float (&d)[4], const uint32_t (&a)[4],
                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ static uint32_t fold(const float (&d)[4]) {
    return __float_as_uint(d[0]) ^ __float_as_uint(d[1]) ^
           __float_as_uint(d[2]) ^ __float_as_uint(d[3]);
  }
  __device__ static void zero(float (&d)[4]) {
    d[0] = d[1] = d[2] = d[3] = 0.f;
  }
};

template <>
struct Mma<Kind::kBF16F32> {
  static constexpr int64_t kFlop = kFlopM16N8K16;
  __device__ static void run(float (&d)[4], const uint32_t (&a)[4],
                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ static uint32_t fold(const float (&d)[4]) {
    return Mma<Kind::kF16F32>::fold(d);
  }
  __device__ static void zero(float (&d)[4]) {
    d[0] = d[1] = d[2] = d[3] = 0.f;
  }
};

template <>
struct Mma<Kind::kF16F16> {
  static constexpr int64_t kFlop = kFlopM16N8K16;
  __device__ static void run(uint32_t (&d)[2], const uint32_t (&a)[4],
                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ static uint32_t fold(const uint32_t (&d)[2]) {
    return d[0] ^ d[1];
  }
  __device__ static void zero(uint32_t (&d)[2]) { d[0] = d[1] = 0u; }
};

template <>
struct Mma<Kind::kS8S32> {
  static constexpr int64_t kFlop = kOpsM16N8K32;
  __device__ static void run(int32_t (&d)[4], const uint32_t (&a)[4],
                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  __device__ static uint32_t fold(const int32_t (&d)[4]) {
    return static_cast<uint32_t>(d[0] ^ d[1] ^ d[2] ^ d[3]);
  }
  __device__ static void zero(int32_t (&d)[4]) {
    d[0] = d[1] = d[2] = d[3] = 0;
  }
};

template <Kind K>
struct AccOf;
template <>
struct AccOf<Kind::kF16F32> {
  using T = float[4];
};
template <>
struct AccOf<Kind::kBF16F32> {
  using T = float[4];
};
template <>
struct AccOf<Kind::kF16F16> {
  using T = uint32_t[2];
};
template <>
struct AccOf<Kind::kS8S32> {
  using T = int32_t[4];
};

// ---------------------------------------------------------------- operands
__device__ inline uint32_t mix(uint32_t x) {  // murmur3 finaliser
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

// Random magnitudes in [2^-7, 2^-6) with random signs, 10/7 random mantissa
// bits: with k = 16 products per element per mma, each mma moves an fp16
// accumulator by at most 16 * 2^-12 = 2^-8, far inside the fp16 range, and
// the +B / -B pairing returns it to the start value every iteration.
__device__ inline uint32_t rand_half2(uint32_t h) {
  const uint32_t lo = ((h & 1u) << 15) | (8u << 10) |
                      ((h >> 1) & 0x3FFu);  // exponent 8: 2^(8-15) = 2^-7
  const uint32_t hi =
      ((h >> 11) & 1u) << 15 | (8u << 10) | ((h >> 12) & 0x3FFu);
  return lo | (hi << 16);
}
__device__ inline uint32_t rand_bf162(uint32_t h) {
  const uint32_t lo = ((h & 1u) << 15) | (120u << 7) |
                      ((h >> 1) & 0x7Fu);  // exponent 120: 2^(120-127) = 2^-7
  const uint32_t hi = ((h >> 8) & 1u) << 15 | (120u << 7) | ((h >> 9) & 0x7Fu);
  return lo | (hi << 16);
}
// int8 in [-64, 63]: negation stays in range, and 32 products of <= 2^12
// per mma keep the s32 accumulator far from overflow.
__device__ inline uint32_t rand_s8x4(uint32_t h) {
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    const int v = static_cast<int>((h >> (8 * i)) & 0x7Fu) - 64;
    r |= (static_cast<uint32_t>(v) & 0xFFu) << (8 * i);
  }
  return r;
}

template <Kind K>
__device__ inline uint32_t make_operand(uint32_t h) {
  if constexpr (K == Kind::kS8S32) return rand_s8x4(h);
  if constexpr (K == Kind::kBF16F32) return rand_bf162(h);
  return rand_half2(h);
}

template <Kind K>
__device__ inline uint32_t negate(uint32_t x) {
  if constexpr (K == Kind::kS8S32) return __vneg4(x);  // per-byte negation
  return x ^ 0x80008000u;                              // flip both sign bits
}

// ---------------------------------------------------------------- the kernel
// timing[cta] = {clock64 start, clock64 end, globaltimer start, globaltimer
// end}
template <Kind K, int CHAINS>
__global__ void __launch_bounds__(kMaxThreads)
    mma_ceiling_kernel(int iters, uint32_t seed, uint32_t sentinel,
                       uint32_t* sink, unsigned long long* timing) {
  using M = Mma<K>;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4], b[2], nb[2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    a[i] = make_operand<K>(mix(seed ^ (tid * 8u + i)));
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    b[i] = make_operand<K>(mix(seed ^ (tid * 8u + 4u + i)));
    nb[i] = negate<K>(b[i]);
  }
  typename AccOf<K>::T acc[CHAINS];
#pragma unroll
  for (int c = 0; c < CHAINS; ++c) M::zero(acc[c]);

  __syncthreads();
  unsigned long long c0 = 0, g0 = 0;
  if (threadIdx.x == 0) {
    c0 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  }
  // One iteration = 2 * CHAINS mma (B then -B). Not unrolled further: the
  // loop overhead (~3 instructions) per 2*CHAINS tensor instructions issues in
  // the tensor pipe's shadow (one HMMA.16816.F32 per 32 cycles per SMSP).
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) M::run(acc[c], a, b);
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) M::run(acc[c], a, nb);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long c1 = clock64(), g1 = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    timing[blockIdx.x * 4 + 0] = c0;
    timing[blockIdx.x * 4 + 1] = c1;
    timing[blockIdx.x * 4 + 2] = g0;
    timing[blockIdx.x * 4 + 3] = g1;
  }
  uint32_t f = 0;
#pragma unroll
  for (int c = 0; c < CHAINS; ++c) f ^= M::fold(acc[c]);
  if (f == sentinel) sink[tid] = f;
}

__global__ void empty_node() {}

// ---------------------------------------------------------------- host
using KernelFn = void (*)(int, uint32_t, uint32_t, uint32_t*,
                          unsigned long long*);

template <Kind K>
KernelFn pick_chains(int chains) {
  switch (chains) {
    case 1:
      return mma_ceiling_kernel<K, 1>;
    case 2:
      return mma_ceiling_kernel<K, 2>;
    case 4:
      return mma_ceiling_kernel<K, 4>;
    case 8:
      return mma_ceiling_kernel<K, 8>;
    default:
      die("chains must be 1, 2, 4 or 8");
  }
}

struct MmaArm {
  std::string name;
  Kind kind;
  int warps, ctas_per_sm, chains;
  KernelFn fn;
  int64_t flop_per_mma;
  int iters = 0;
  int grid = 0;
  int occupancy = 0;
  cudaGraphExec_t exec = nullptr;
  unsigned long long* timing = nullptr;
};

MmaArm parse_mma_arm(const std::string& s) {
  const auto f = bench::split(s.substr(0, s.find('#')), ':');
  if (f.size() != 4)
    die("mma arm must be kind:warps:ctas_per_sm:chains, got " + s);
  MmaArm a;
  a.name = s;
  a.warps = static_cast<int>(Args::parse_i64("arms", f[1]));
  a.ctas_per_sm = static_cast<int>(Args::parse_i64("arms", f[2]));
  a.chains = static_cast<int>(Args::parse_i64("arms", f[3]));
  if (a.warps < 1 || a.warps * 32 > kMaxThreads || a.ctas_per_sm < 1)
    die("bad warps/ctas in " + s);
  if (f[0] == "f16f32") {
    a.kind = Kind::kF16F32;
    a.fn = pick_chains<Kind::kF16F32>(a.chains);
    a.flop_per_mma = Mma<Kind::kF16F32>::kFlop;
  } else if (f[0] == "f16f16") {
    a.kind = Kind::kF16F16;
    a.fn = pick_chains<Kind::kF16F16>(a.chains);
    a.flop_per_mma = Mma<Kind::kF16F16>::kFlop;
  } else if (f[0] == "bf16f32") {
    a.kind = Kind::kBF16F32;
    a.fn = pick_chains<Kind::kBF16F32>(a.chains);
    a.flop_per_mma = Mma<Kind::kBF16F32>::kFlop;
  } else if (f[0] == "s8s32") {
    a.kind = Kind::kS8S32;
    a.fn = pick_chains<Kind::kS8S32>(a.chains);
    a.flop_per_mma = Mma<Kind::kS8S32>::kFlop;
  } else {
    die("mma kind must be f16f32, f16f16, bf16f32 or s8s32, got " + f[0]);
  }
  return a;
}

void write_header(Json& j, const Args& args, const char* mode, int process,
                  int dev, const KernelTimer& timer, int replays,
                  double warmup_s, int64_t warm, size_t n_arms,
                  size_t n_blocks) {
  j.kv("tool", "mma_ceiling");
  j.kv("raw_format", 1);
  j.kv("mode", mode);
  j.begin_arr("argv");
  for (const auto& s : args.argv()) j.val(s);
  j.end_arr();
  j.kv("timer", timer.kind_name());
  j.kv("process", process);
  j.kv("t_unix_ns", bench::unix_ns());
  bench::write_device(j, dev);
  j.begin_obj("design");
  j.kv("replays_per_block", replays);
  j.kv("warmup_s", warmup_s);
  j.kv("warmup_graphs", warm);
  j.kv("n_arms", static_cast<int>(n_arms));
  j.kv("n_blocks", static_cast<int>(n_blocks));
  j.end_obj();
}

int run_mma(const Args& args, int dev, const std::string& out_path,
            const std::vector<std::string>& arm_names,
            const std::vector<int>& schedule, int replays, double warmup_s,
            double target_ms, uint32_t seed, int process, KernelTimer& timer) {
  BENCH_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t stream;
  BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  uint32_t* sink = nullptr;
  BENCH_CUDA(cudaMalloc(&sink, static_cast<size_t>(prop.multiProcessorCount) *
                                   kMaxThreads * 32 * sizeof(uint32_t)));
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));

  std::vector<MmaArm> arms;
  for (const auto& n : arm_names) {
    MmaArm a = parse_mma_arm(n);
    BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a.occupancy, a.fn,
                                                             a.warps * 32, 0));
    // The per-SM numbers assume all CTAs of a launch are co-resident.
    if (a.occupancy < a.ctas_per_sm) {
      die(n + ": only " + std::to_string(a.occupancy) +
          " CTAs/SM fit, asked for " + std::to_string(a.ctas_per_sm));
    }
    a.grid = prop.multiProcessorCount * a.ctas_per_sm;
    BENCH_CUDA(cudaMalloc(&a.timing, static_cast<size_t>(a.grid) * 4 *
                                         sizeof(unsigned long long)));
    // Calibrate: the first launch pays module load, so time the second.
    float ms = 0.f;
    for (int rep = 0; rep < 2; ++rep) {
      BENCH_CUDA(cudaEventRecord(e0, stream));
      a.fn<<<a.grid, a.warps * 32, 0, stream>>>(kCalibIters, seed, 0xFFFFFFFFu,
                                                sink, a.timing);
      BENCH_CUDA(cudaEventRecord(e1, stream));
      BENCH_CUDA(cudaStreamSynchronize(stream));
      BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    }
    a.iters = std::max(1, static_cast<int>(std::lround(kCalibIters * target_ms /
                                                       std::max(ms, 1e-3f))));
    cudaGraph_t g;
    BENCH_CUDA(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int r = 0; r < replays; ++r) {
      a.fn<<<a.grid, a.warps * 32, 0, stream>>>(a.iters,
                                                seed + static_cast<uint32_t>(r),
                                                0xFFFFFFFFu, sink, a.timing);
    }
    BENCH_CUDA(cudaStreamEndCapture(stream, &g));
    BENCH_CUDA(cudaGraphInstantiate(&a.exec, g, 0));
    BENCH_CUDA(cudaGraphDestroy(g));
    arms.push_back(a);
  }
  timer.collect("mma_ceiling_kernel");  // drop calibration records

  const uint64_t tw = bench::mono_ns();
  int64_t warm = 0;
  while (bench::mono_ns() - tw < static_cast<uint64_t>(warmup_s * 1e9)) {
    for (const auto& a : arms) {
      BENCH_CUDA(cudaGraphLaunch(a.exec, stream));
      ++warm;
    }
    BENCH_CUDA(cudaStreamSynchronize(stream));
  }
  timer.collect("mma_ceiling_kernel");

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) die("cannot open --out " + out_path);
  Json j(f);
  j.begin_obj();
  write_header(j, args, "mma", process, dev, timer, replays, warmup_s, warm,
               arms.size(), schedule.size());
  j.kv("duration_kind", timer.kind() == KernelTimer::Kind::kCupti
                            ? "kernel"
                            : "graph_replay_mean");
  j.begin_arr("arms");
  for (const auto& a : arms) {
    j.begin_obj();
    j.kv("name", a.name);
    j.kv("kind", bench::split(a.name, ':')[0]);
    j.kv("warps_per_cta", a.warps);
    j.kv("ctas_per_sm", a.ctas_per_sm);
    j.kv("occupancy_ctas_per_sm", a.occupancy);
    j.kv("chains", a.chains);
    j.kv("iters", a.iters);
    j.kv("grid", a.grid);
    const int64_t mma_per_launch =
        static_cast<int64_t>(a.grid) * a.warps * a.iters * 2 * a.chains;
    j.kv("mma_per_launch", mma_per_launch);
    j.kv("flop_per_launch", mma_per_launch * a.flop_per_mma);
    // per SMSP: all warps of the SM's CTAs spread over 4 sub-partitions
    j.kv("mma_per_smsp_per_launch", static_cast<double>(a.warps) *
                                        a.ctas_per_sm * a.iters * 2 * a.chains /
                                        kSmspPerSm);
    j.end_obj();
  }
  j.end_arr();

  j.begin_arr("blocks");
  std::vector<unsigned long long> tbuf;
  for (size_t k = 0; k < schedule.size(); ++k) {
    MmaArm& a = arms[schedule[k]];
    const uint64_t t0 = bench::mono_ns();
    bench::spin_ns<<<1, 1, 0, stream>>>(bench::kSpinBeforeTimingNs);
    BENCH_CUDA(cudaEventRecord(e0, stream));
    BENCH_CUDA(cudaGraphLaunch(a.exec, stream));
    BENCH_CUDA(cudaEventRecord(e1, stream));
    BENCH_CUDA(cudaStreamSynchronize(stream));
    const uint64_t t1 = bench::mono_ns();
    float graph_ms = 0.f;
    BENCH_CUDA(cudaEventElapsedTime(&graph_ms, e0, e1));
    // collect() keeps only the named kernel and drops everything else
    // (the spin kernel), so the next block starts with an empty buffer.
    std::vector<uint64_t> d = timer.collect("mma_ceiling_kernel");
    if (timer.kind() == KernelTimer::Kind::kCupti &&
        static_cast<int>(d.size()) != replays) {
      std::fprintf(stderr,
                   "block %zu: CUPTI returned %zu records, expected %d\n", k,
                   d.size(), replays);
      std::exit(3);
    }
    if (timer.kind() != KernelTimer::Kind::kCupti) {
      d.assign(static_cast<size_t>(replays),
               static_cast<uint64_t>(graph_ms * 1e6 / replays));
    }
    // Clock evidence from the last launch of the block (all launches write
    // the same buffer; the last one wins).
    tbuf.resize(static_cast<size_t>(a.grid) * 4);
    BENCH_CUDA(cudaMemcpy(tbuf.data(), a.timing,
                          tbuf.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    std::vector<double> mhz, cyc;
    for (int c = 0; c < a.grid; ++c) {
      const double dc = static_cast<double>(tbuf[c * 4 + 1] - tbuf[c * 4 + 0]);
      const double dg = static_cast<double>(tbuf[c * 4 + 3] - tbuf[c * 4 + 2]);
      if (dg > 0) mhz.push_back(dc / dg * 1e3);
      cyc.push_back(dc);
    }
    const double per_smsp = static_cast<double>(a.warps) * a.ctas_per_sm *
                            a.iters * 2 * a.chains / kSmspPerSm;
    j.begin_obj();
    j.kv("seq", static_cast<int>(k));
    j.kv("round", static_cast<int>(k / arms.size()));
    j.kv("arm", a.name);
    j.kv("t_start_mono_ns", t0);
    j.kv("t_end_mono_ns", t1);
    j.kv("graph_ms", static_cast<double>(graph_ms));
    j.begin_arr("durations_ns");
    for (uint64_t v : d) j.val(v);
    j.end_arr();
    j.kv("sm_clock_mhz_median", bench::median_of(mhz));
    j.kv("cycles_per_mma_per_smsp", bench::median_of(cyc) / per_smsp);
    j.end_obj();
  }
  j.end_arr();
  j.kv("t_end_unix_ns", bench::unix_ns());
  j.end_obj();
  std::fputc('\n', f);
  std::fclose(f);
  return 0;
}

// Graph-node overhead: arms "nodes:N". Each replay of a block is timed
// individually with events (CUPTI is not involved: we want the time the GPU
// spends getting from one node to the next, which kernel durations exclude).
int run_graph_overhead(const Args& args, int dev, const std::string& out_path,
                       const std::vector<std::string>& arm_names,
                       const std::vector<int>& schedule, int replays,
                       double warmup_s, int process, KernelTimer& timer) {
  BENCH_CUDA(cudaSetDevice(dev));
  cudaStream_t stream;
  BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  struct GArm {
    std::string name;
    int nodes;
    cudaGraphExec_t exec;
  };
  std::vector<GArm> arms;
  for (const auto& n : arm_names) {
    const auto f = bench::split(n.substr(0, n.find('#')), ':');
    if (f.size() != 2 || f[0] != "nodes")
      die("graph-overhead arm must be nodes:N, got " + n);
    const int nodes = static_cast<int>(Args::parse_i64("arms", f[1]));
    if (nodes < 1) die("nodes must be >= 1");
    cudaGraph_t g;
    BENCH_CUDA(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < nodes; ++i) empty_node<<<1, 32, 0, stream>>>();
    BENCH_CUDA(cudaStreamEndCapture(stream, &g));
    cudaGraphExec_t ex;
    BENCH_CUDA(cudaGraphInstantiate(&ex, g, 0));
    BENCH_CUDA(cudaGraphDestroy(g));
    arms.push_back({n, nodes, ex});
  }
  const uint64_t tw = bench::mono_ns();
  int64_t warm = 0;
  while (bench::mono_ns() - tw < static_cast<uint64_t>(warmup_s * 1e9)) {
    for (const auto& a : arms) {
      BENCH_CUDA(cudaGraphLaunch(a.exec, stream));
      ++warm;
    }
    BENCH_CUDA(cudaStreamSynchronize(stream));
  }
  std::vector<cudaEvent_t> ev(2 * static_cast<size_t>(replays));
  for (auto& e : ev) BENCH_CUDA(cudaEventCreate(&e));

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) die("cannot open --out " + out_path);
  Json j(f);
  j.begin_obj();
  write_header(j, args, "graph-overhead", process, dev, timer, replays,
               warmup_s, warm, arms.size(), schedule.size());
  j.kv("duration_kind", "graph_replay");
  j.begin_arr("arms");
  for (const auto& a : arms) {
    j.begin_obj();
    j.kv("name", a.name);
    j.kv("nodes", a.nodes);
    j.end_obj();
  }
  j.end_arr();
  j.begin_arr("blocks");
  for (size_t k = 0; k < schedule.size(); ++k) {
    const GArm& a = arms[schedule[k]];
    const uint64_t t0 = bench::mono_ns();
    // One spin covers the host-side enqueue of all replays of the block, so
    // every event pair times GPU execution only.
    bench::spin_ns<<<1, 1, 0, stream>>>(bench::kSpinBeforeTimingNs *
                                        static_cast<uint64_t>(replays));
    for (int r = 0; r < replays; ++r) {
      BENCH_CUDA(cudaEventRecord(ev[2 * r], stream));
      BENCH_CUDA(cudaGraphLaunch(a.exec, stream));
      BENCH_CUDA(cudaEventRecord(ev[2 * r + 1], stream));
    }
    BENCH_CUDA(cudaStreamSynchronize(stream));
    const uint64_t t1 = bench::mono_ns();
    j.begin_obj();
    j.kv("seq", static_cast<int>(k));
    j.kv("round", static_cast<int>(k / arms.size()));
    j.kv("arm", a.name);
    j.kv("t_start_mono_ns", t0);
    j.kv("t_end_mono_ns", t1);
    j.begin_arr("durations_ns");
    for (int r = 0; r < replays; ++r) {
      float ms = 0.f;
      BENCH_CUDA(cudaEventElapsedTime(&ms, ev[2 * r], ev[2 * r + 1]));
      j.val(static_cast<uint64_t>(std::llround(static_cast<double>(ms) * 1e6)));
    }
    j.end_arr();
    j.end_obj();
  }
  j.end_arr();
  j.kv("t_end_unix_ns", bench::unix_ns());
  j.end_obj();
  std::fputc('\n', f);
  std::fclose(f);
  return 0;
}

}  // namespace

constexpr const char kUsage[] =
    R"(mma_ceiling: tensor-core ceiling and CUDA-graph node overhead.

Needs an NVIDIA GPU (sm80, sm86 or sm89) and CUPTI (libcupti.so.13).
bench/ceilings_run.sh is the driver that runs it.

  mma_ceiling --mode mma --arms ARM[,ARM...] --schedule S --out FILE [options]
  mma_ceiling --mode graph-overhead --arms nodes:N[,nodes:N...] ... --out FILE
  mma_ceiling --device-info     print the device JSON and exit
  mma_ceiling --help            this text

mma arms: "<kind>:<warps per CTA>:<CTAs per SM>:<chains>" with kind f16f32,
f16f16, bf16f32 or s8s32 (for example f16f32:4:1:8), optionally suffixed "#tag"
(A/A control). Options (default): --schedule arm indices in launch order, from
`python3 bench/stats.py schedule`; --out raw JSON path; --replays (20),
--warmup-s (2), --target-ms (2), --seed (0), --process (0), --device (0),
--timer cupti|events (cupti for mma, events for graph-overhead). Output: one raw
JSON file (raw_format 1) that bench/bench_report.py turns into result records.
)";

int main(int argc, char** argv) {
  Args args(argc, argv);
  if (args.has("help")) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  const int dev = static_cast<int>(args.i64("device", 0));
  if (args.has("device-info")) {
    args.str("device-info", "");
    args.reject_unknown();
    Json j(stdout);
    j.begin_obj();
    bench::write_device(j, dev);
    j.end_obj();
    std::printf("\n");
    return 0;
  }
  const std::string mode = args.req("mode");
  const std::string out_path = args.req("out");
  const std::vector<std::string> arm_names =
      bench::split(args.req("arms"), ',');
  const std::vector<int> schedule = bench::parse_schedule(
      args.req("schedule"), static_cast<int>(arm_names.size()));
  const int replays = static_cast<int>(args.i64("replays", 20));
  const double warmup_s = args.f64("warmup-s", 2.0);
  const double target_ms = args.f64("target-ms", 2.0);
  const uint32_t seed = static_cast<uint32_t>(args.i64("seed", 0));
  const int process = static_cast<int>(args.i64("process", 0));
  const std::string timer_s =
      args.str("timer", mode == "mma" ? "cupti" : "events");
  args.reject_unknown();
  if (replays < 1 || target_ms <= 0)
    die("--replays and --target-ms must be positive");

  KernelTimer& timer = KernelTimer::instance();
  if (mode == "mma") {
    for (const auto& n : arm_names)
      parse_mma_arm(n);  // fail on a bad arm before touching the GPU
    timer.start(bench::parse_timer(timer_s));
    const int rc = run_mma(args, dev, out_path, arm_names, schedule, replays,
                           warmup_s, target_ms, seed, process, timer);
    timer.stop();
    return rc;
  }
  if (mode == "graph-overhead") {
    if (timer_s != "events") die("graph-overhead mode times with events only");
    timer.start(KernelTimer::Kind::kEvents);
    return run_graph_overhead(args, dev, out_path, arm_names, schedule, replays,
                              warmup_s, process, timer);
  }
  die("--mode must be mma or graph-overhead");
}
