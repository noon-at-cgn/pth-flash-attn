// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// test_hooks.cuh: compile-time test instrumentation of the warp-specialised
// stage handshake, for the test libraries only.
// tools/build_attn_lib.sh force-includes this header (nvcc -include) into
// every translation unit of a build with one of the macros below. It defines
// PTH_ATTN_TEST_HOOKS, which replaces the empty default hooks of
// attn_kernel.cuh with these; the shipped sources never include it. Driven
// by tests/gpu/attn_stress.py and the GPU correctness grid.
//
//   -DPTH_ATTN_TEST_PROTOCOL   dynamic checker of the handshake, because
//       compute-sanitizer racecheck does not model bar.arrive / bar.sync:
//       * tag[s][pw]: the tile that producer warp pw has landed in stage s.
//         A producer sets it (after its stores, before its Full arrive) and
//         resets it to -1 (after its Empty wait, before its first store of
//         the next tile); a consumer must find tag[s][0..3] == i after its
//         Full wait (tile i landed, all four warps) and again before it
//         releases the stage (no producer started rewriting it while it was
//         read);
//       * ledger[w][id - 2]: how many times warp w executed each of the four
//         stage barriers; at the end of the warp's work it must equal the
//         number of tiles of that stage parity (Full) or of tiles >= 2 of
//         that parity (Empty): every arrive is matched by a sync of the other
//         side, so no barrier is left half-complete at CTA exit;
//       * ep_a / ep_b (decode kernel): per-warp tile counters written before
//         the row-max barrier A and the P'' barrier B; all four must show the
//         current tile after the barrier.
//       A violation prints the site and traps (the launch fails). The printf
//       needs a stack frame, so this library cannot pass the ptxas gate.
//   -DPTH_ATTN_TEST_MUTATE=N   a deliberately broken handshake that the GPU
//       correctness grid must catch: 1 = producers skip the stage-empty wait, 2
//       = decode consumers skip the P'' barrier, 3 = decode consumers skip the
//       stage-full wait.
//   -DPTH_ATTN_TEST_JITTER     a pseudo-random __nanosleep (0..255 ns) at
//       every handshake point of producers and consumers; the output must
//       not change.

#pragma once

#define PTH_ATTN_TEST_HOOKS

#include <cstdint>
#include <cstdio>

#include "tile_traits.cuh"

#ifndef PTH_ATTN_TEST_MUTATE
  #define PTH_ATTN_TEST_MUTATE 0
#endif

namespace pth_attn {

constexpr int kTestMutate = PTH_ATTN_TEST_MUTATE;

#if defined(__CUDACC__)

__device__ __forceinline__ void test_jitter(int salt) {
  #ifdef PTH_ATTN_TEST_JITTER
  unsigned h = threadIdx.x * 2654435761u ^ salt * 40503u ^ clock();
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  __nanosleep(h & 255u);
  #else
  (void)salt;
  #endif
}

  #ifdef PTH_ATTN_TEST_PROTOCOL
struct ProtoState {
  int32_t tag[kStages][kWarps];
  int32_t ledger[2 * kWarps][4];
  int32_t ep_a[kWarps];
  int32_t ep_b[kWarps];
};
__device__ __forceinline__ volatile ProtoState& proto() {
  __shared__ ProtoState st;
  return st;
}
__device__ __forceinline__ bool proto_lane0() {
  return (threadIdx.x & (kWarpSize - 1)) == 0;
}
[[noreturn]] __device__ __forceinline__ void proto_trap(const char* what, int a,
                                                        int b) {
  printf(
      "PTH_ATTN protocol violation: %s: block (%d,%d,%d) warp %d: %d vs %d\n",
      what, blockIdx.x, blockIdx.y, blockIdx.z,
      static_cast<int>(threadIdx.x) / kWarpSize, a, b);
  asm volatile("trap;");
  __builtin_unreachable();
}
// Every warp zeroes its own ledger row and (producer warps) its tags.
__device__ __forceinline__ void proto_init() {
  if (!proto_lane0()) return;
  volatile ProtoState& st = proto();
  const int w = static_cast<int>(threadIdx.x) / kWarpSize;
  for (int k = 0; k < 4; ++k) st.ledger[w][k] = 0;
  if (w >= kWarps) {
    for (int s = 0; s < kStages; ++s) st.tag[s][w - kWarps] = -1;
  } else {
    st.ep_a[w] = 0;
    st.ep_b[w] = 0;
  }
}
// Counts a stage barrier (ids 2..5) executed by this warp.
__device__ __forceinline__ void proto_count(int id) {
  if (id < 2 || !proto_lane0()) return;
  volatile ProtoState& st = proto();
  st.ledger[static_cast<int>(threadIdx.x) / kWarpSize][id - 2] += 1;
}
// Producer warp pw: about to rewrite stage s / has landed tile j in s.
__device__ __forceinline__ void proto_rewrite(int s, int pw) {
  if (proto_lane0()) proto().tag[s][pw] = -1;
}
__device__ __forceinline__ void proto_landed(int s, int pw, int32_t j) {
  if (proto_lane0()) proto().tag[s][pw] = j;
}
// Consumer: tile i must be fully landed in stage s (all four producer warps).
__device__ __forceinline__ void proto_expect_landed(const char* where, int s,
                                                    int32_t i) {
  if (!proto_lane0()) return;
  volatile ProtoState& st = proto();
  for (int pw = 0; pw < kWarps; ++pw) {
    if (st.tag[s][pw] != i) proto_trap(where, st.tag[s][pw], i);
  }
}
// Barrier counts of a warp that handled the KV tiles [tb, te) (see the file
// comment).
__device__ __forceinline__ void proto_check_ledger(int32_t tb, int32_t te) {
  const int32_t tiles = te - tb;
  if (!proto_lane0()) return;
  volatile ProtoState& st = proto();
  const int w = static_cast<int>(threadIdx.x) / kWarpSize;
  const int full[2] = {(tiles + 1) / 2, tiles / 2};
  const int empty[2] = {tiles >= 1 ? (tiles - 1) / 2 : 0,
                        tiles >= 2 ? (tiles - 2) / 2 : 0};
  for (int s = 0; s < 2; ++s) {
    if (st.ledger[w][s] != full[s]) {
      proto_trap("Full barrier count at exit", st.ledger[w][s], full[s]);
    }
    if (st.ledger[w][2 + s] != empty[s]) {
      proto_trap("Empty barrier count at exit", st.ledger[w][2 + s], empty[s]);
    }
  }
}
// Decode kernel: per-warp epoch exchange around barriers A (b = false) and B.
__device__ __forceinline__ void proto_epoch_set(bool b, int warp, int32_t e) {
  if (!proto_lane0()) return;
  volatile ProtoState& st = proto();
  (b ? st.ep_b : st.ep_a)[warp] = e;
}
__device__ __forceinline__ void proto_epoch_expect(const char* where, bool b,
                                                   int32_t e) {
  if (!proto_lane0()) return;
  volatile ProtoState& st = proto();
  for (int w = 0; w < kWarps; ++w) {
    const int32_t got = (b ? st.ep_b : st.ep_a)[w];
    if (got != e) proto_trap(where, got, e);
  }
}
  #else
__device__ __forceinline__ void proto_init() {}
__device__ __forceinline__ void proto_count(int) {}
__device__ __forceinline__ void proto_rewrite(int, int) {}
__device__ __forceinline__ void proto_landed(int, int, int32_t) {}
__device__ __forceinline__ void proto_expect_landed(const char*, int, int32_t) {
}
__device__ __forceinline__ void proto_check_ledger(int32_t, int32_t) {}
__device__ __forceinline__ void proto_epoch_set(bool, int, int32_t) {}
__device__ __forceinline__ void proto_epoch_expect(const char*, bool, int32_t) {
}
  #endif  // PTH_ATTN_TEST_PROTOCOL

#endif  // __CUDACC__

}  // namespace pth_attn
