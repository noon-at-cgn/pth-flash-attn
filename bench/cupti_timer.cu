// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// cupti_timer.cu: the CUPTI kernel timer (bench_harness.cuh KernelTimer)
// behind a C ABI, so bench/kernel_bench.py can time the attention kernel from
// Python with the same timer that measures the hardware ceilings.
//
// Why not torch.profiler or FlashInfer: the profiler adds its own start/stop
// cost and post-processing per block, and FlashInfer's CUPTI path needs the
// cupti-python package, which is not always installed (bench_harness.cuh). The
// C API ships with the CUDA 13 toolkit; reusing the class means the kernel and
// the ceilings are timed by identical code.
//
// Built by bench/k_build.sh into build/k/libpth_cupti_timer.so (no device
// code; links libcupti.so.13 from the image with an rpath).
//
// Usage (one process): pth_timer_start() once, before the timed kernels;
// after each block, synchronise the stream and call pth_timer_collect() to
// drain every kernel record since the previous call (start order); finally
// pth_timer_stop(). A CUPTI error ends the process (exit 2), as in the ceiling
// programs: a half-measured block must never be written as valid.

#include <cstdint>

#include "bench_harness.cuh"

extern "C" {

// ABI version of this wrapper (kernel_bench.py checks it at load).
int32_t pth_timer_abi_version(void) { return 1; }

// Enable CONCURRENT_KERNEL activity records. Returns 0.
int32_t pth_timer_start(void) {
  bench::KernelTimer::instance().start(bench::KernelTimer::Kind::kCupti);
  return 0;
}

// Drain the records collected so far into start_ns/end_ns (CUPTI GPU
// timestamps, ns), sorted by start. Returns the number of records, or -1 if
// there are more than `cap` (none are written then: the caller's block would
// be incomplete).
int64_t pth_timer_collect(uint64_t* start_ns, uint64_t* end_ns, int64_t cap) {
  const auto recs = bench::KernelTimer::instance().collect_records("");
  if (static_cast<int64_t>(recs.size()) > cap) return -1;
  for (size_t i = 0; i < recs.size(); ++i) {
    start_ns[i] = recs[i].start_ns;
    end_ns[i] = recs[i].end_ns;
  }
  return static_cast<int64_t>(recs.size());
}

// Flush and disable. Returns 0.
int32_t pth_timer_stop(void) {
  bench::KernelTimer::instance().stop();
  return 0;
}

}  // extern "C"
