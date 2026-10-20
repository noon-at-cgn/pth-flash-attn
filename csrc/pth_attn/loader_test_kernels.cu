// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// loader_test_kernels.cu: a test-only kernel that runs the tile loader
// (loader.cuh) through the real two-stage pipeline and writes each stage
// image back to global memory, so a test can compare it byte for byte with
// the expected image (tests/sim/ptxsim/loader_ref.py). Built into
// build/loader/libpth_loader.so by tools/build_loader_lib.sh together with
// loader_capi.cu; never part of the op.

#include <cuda_runtime.h>

#include <cstdint>

#include "loader.cuh"
#include "loader_test_kernels.h"
#include "smem_layout.cuh"
#include "tile_traits.cuh"

namespace pth_attn {
namespace loader_test {
namespace {

constexpr int kVecBytes = 16;  // uint4 copies for zeroing and dumping

template <int D, bool kCell>
struct StageBytes {
  static constexpr int kValue =
      kCell ? CellStageLayout<D, kLoadTileTokens>::kBytes
            : StageLayout<D, kLoadTileTokens>::kBytes;
};

template <int D, bool kCell>
__global__ void __launch_bounds__(kThreads)
    loader_test_kernel(KernelParams p, int32_t cell_width, int32_t req,
                       int32_t kv_head, int32_t tile_begin, int32_t tile_end,
                       int32_t tok_begin, int32_t tok_end,
                       uint8_t* __restrict__ out) {
  constexpr int kStage = StageBytes<D, kCell>::kValue;
  static_assert(kStage % kVecBytes == 0, "stages are whole 16-B vectors");
  extern __shared__ __align__(kSmemLineBytes) uint8_t smem[];
  // Zero both stages once: the dump then shows 0 for every byte the loader
  // never writes (the cell pitch padding), as the expected image does.
  uint4* sv = reinterpret_cast<uint4*>(smem);
  for (int i = static_cast<int>(threadIdx.x); i < kStages * kStage / kVecBytes;
       i += kThreads) {
    sv[i] = make_uint4(0u, 0u, 0u, 0u);
  }
  __syncthreads();
  const uint32_t base = smem_u32(smem);
  const auto issue = [&](int32_t tile, int stage) {
    const uint32_t s = base + static_cast<uint32_t>(stage * kStage);
    if constexpr (kCell) {
      load_tile_cell<D>(p, cell_width, s, req, kv_head, tile, tok_end);
    } else if (tok_begin > 0) {
      // The windowed kernels' loader.
      load_tile_separate<D, true>(p, s, req, kv_head, tile, tok_end, tok_begin);
    } else {
      load_tile_separate<D>(p, s, req, kv_head, tile, tok_end);
    }
  };
  // The "consumer": copy the stage out (int64 offset: many tiles may be
  // dumped).
  const auto dump = [&](int32_t tile, int stage) {
    const uint4* src = reinterpret_cast<const uint4*>(smem + stage * kStage);
    uint4* dst = reinterpret_cast<uint4*>(
        out + static_cast<int64_t>(tile - tile_begin) * kStage);
    for (int i = static_cast<int>(threadIdx.x); i < kStage / kVecBytes;
         i += kThreads) {
      dst[i] = src[i];
    }
  };
  run_two_stage_pipeline(tile_begin, tile_end, issue, dump);
}

template <int D, bool kCell>
int32_t launch_one(const LaunchArgs& a, cudaStream_t stream) {
  auto* kernel = loader_test_kernel<D, kCell>;
  constexpr int kSmem = kStages * StageBytes<D, kCell>::kValue;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kSmem) != cudaSuccess) {
    return -1;
  }
  kernel<<<1, kThreads, kSmem, stream>>>(
      a.p, a.cell_width, a.req, a.kv_head, a.tile_begin, a.tile_end,
      a.tok_begin, a.tok_end, static_cast<uint8_t*>(a.out));
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

template <bool kCell>
int32_t launch_d(const LaunchArgs& a, cudaStream_t s) {
  switch (a.head_size) {
    case 64:
      return launch_one<64, kCell>(a, s);
    case 128:
      return launch_one<128, kCell>(a, s);
    case 256:
      return launch_one<256, kCell>(a, s);
    default:
      return -1;
  }
}

}  // namespace

int32_t launch(const LaunchArgs& a, void* stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  return a.whole_cell ? launch_d<true>(a, s) : launch_d<false>(a, s);
}

}  // namespace loader_test
}  // namespace pth_attn
