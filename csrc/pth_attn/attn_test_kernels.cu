// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_test_kernels.cu: the fragment unit tests of attn_capi.h
// (pth_attn_test_warp_tile, pth_attn_test_dec_tile). They run the kernels' own
// QK and PV loops (tiling.cuh qk_subtile / pv_subtile, attn_decode.cuh dec_qk /
// dec_pv) on one staged tile, so a fragment-mapping bug shows up as a wrong
// element, not as a wrong attention output somewhere in the grid.
//
// Built into the test library only (tools/build_attn_lib.sh); the production
// extension (tools/build_flags.py EXTENSION_SOURCES) does not contain it.

#include <cuda_runtime.h>

#include <string>

#include "attn_capi.h"
#include "attn_decode.cuh"
#include "attn_launch.h"

namespace {

// Each unit test runs over one 64-token tile.
constexpr int kTileTestTokens = 64;

// One warp: stage Q (pi + swizzle), K and V (swizzled) exactly as the kernel
// does, then run the kernel's qk_subtile / pv_subtile over 64 tokens.
template <int D>
__global__ void __launch_bounds__(pth_attn::kWarpSize)
    warp_tile_test_kernel(const PthAttnTileTestArgs a) {
  using pth_attn::kMmaM;
  using pth_attn::kMmaN;
  constexpr int kT = kTileTestTokens;
  constexpr int kQBytes = kMmaM * D * pth_attn::kF16Bytes;
  constexpr int kKBytes = kT * D;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* sq = smem;
  uint8_t* sk = smem + kQBytes;
  uint8_t* sv = sk + kKBytes;
  const int lane = static_cast<int>(threadIdx.x);
  const uint16_t* q = static_cast<const uint16_t*>(a.q);
  for (int i = lane; i < kMmaM * D / 2; i += pth_attn::kWarpSize) {
    const int r = 2 * i / D, d = 2 * i % D;
    const uint32_t w = static_cast<uint32_t>(q[r * D + d]) |
                       (static_cast<uint32_t>(q[r * D + d + 1]) << 16);
    *reinterpret_cast<uint32_t*>(sq + pth_attn::q_elem_offset(r, d, D)) = w;
  }
  constexpr int kChunk = pth_attn::kSmemChunkBytes;
  const uint8_t* k = static_cast<const uint8_t*>(a.k);
  const uint8_t* v = static_cast<const uint8_t*>(a.v);
  for (int i = lane; i < kT * D / kChunk; i += pth_attn::kWarpSize) {
    const int t = i * kChunk / D, b = i * kChunk % D;
    *reinterpret_cast<uint4*>(sk + pth_attn::k_swz(t, b, D)) =
        *reinterpret_cast<const uint4*>(k + t * D + b);
    *reinterpret_cast<uint4*>(sv + pth_attn::v_swz(t, b, D)) =
        *reinterpret_cast<const uint4*>(v + t * D + b);
  }
  __syncthreads();
  const int g = lane >> 2, qq = lane & 3;
  float s[kT / kMmaN][4];
  // sk is a separate-streams stage (K region at 0, V region at 64 D).
  pth_attn::qk_subtile<D, kT>(s, pth_attn::smem_u32(sq), pth_attn::smem_u32(sk),
                              0, 0, lane);
  for (int n = 0; n < kT / kMmaN; ++n) {
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 2; ++i) {
        a.s_out[(g + 8 * h) * kT + n * kMmaN + 2 * qq + i] = s[n][2 * h + i];
      }
    }
  }
  const uint16_t* p = static_cast<const uint16_t*>(a.p);
  uint32_t pa[kT / kMmaN][2];
  for (int n = 0; n < kT / kMmaN; ++n) {
    for (int h = 0; h < 2; ++h) {
      const int idx = (g + 8 * h) * kT + n * kMmaN + 2 * qq;
      pa[n][h] = static_cast<uint32_t>(p[idx]) |
                 (static_cast<uint32_t>(p[idx + 1]) << 16);
    }
  }
  float o[D / kMmaN][4] = {};
  pth_attn::pv_subtile<D, kT>(o, pa, sk, 0, lane);
  for (int j = 0; j < D / kMmaN; ++j) {
    for (int h = 0; h < 2; ++h) {
      for (int i = 0; i < 2; ++i) {
        a.o_out[(g + 8 * h) * D + pth_attn::v_ntile_col_to_d(j, 2 * qq + i)] =
            o[j][2 * h + i];
      }
    }
  }
}

template <int D>
cudaError_t launch_tile_test(const PthAttnTileTestArgs& a, cudaStream_t s) {
  return pth_attn::launch_with_smem<warp_tile_test_kernel<D>,
                                    pth_attn::kMmaM * D * pth_attn::kF16Bytes +
                                        2 * kTileTestTokens * D,
                                    pth_attn::kWarpSize>(1, s, a);
}

// One CTA of the four compute warps (128 threads) stages Q (rows past `rows`
// are zero), one 64-token K and V stage (separate-streams layout, as
// warp_tile_test_kernel) and the P'' buffer exactly as attn_dec_body does,
// then runs the decode kernel's dec_qk (warp w: tokens 16 w ..) and dec_pv
// (warp w: its quarter of d) and writes the accumulators out by the epilogue's
// mapping: row 8 n + 2 q + j; d = 32 (kGroups warp + a) + 4 g + 2 h + {0: c[j],
// 1: c[2 + j]} for accumulator (m-tile 2a + h, n-tile n).
template <int D, int NT>
__global__ void __launch_bounds__(pth_attn::kWarps* pth_attn::kWarpSize)
    dec_tile_test_kernel(const PthAttnDecTileTestArgs a) {
  using Sm = pth_attn::DecSmem<D, NT, false>;
  using R = typename Sm::R;
  constexpr int kRows = Sm::kRows;
  constexpr int kT = kTileTestTokens;
  constexpr int kThreads = pth_attn::kWarps * pth_attn::kWarpSize;
  constexpr int kMt = pth_attn::dec_mtiles(D);
  constexpr int kGroups = D / pth_attn::kVGroupD / pth_attn::kWarps;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / pth_attn::kWarpSize;
  const int lane = tid % pth_attn::kWarpSize;
  const uint16_t* q = static_cast<const uint16_t*>(a.q);
  for (int i = tid; i < kRows * D / 2; i += kThreads) {
    const int r = 2 * i / D, d = 2 * i % D;
    const uint32_t w = r < a.rows
                           ? static_cast<uint32_t>(q[r * D + d]) |
                                 (static_cast<uint32_t>(q[r * D + d + 1]) << 16)
                           : 0u;
    *reinterpret_cast<uint32_t*>(smem + Sm::kQ +
                                 pth_attn::q_elem_offset(r, d, D)) = w;
  }
  constexpr int kChunk = pth_attn::kSmemChunkBytes;
  uint8_t* stage = smem + Sm::stage_offset(0);
  const uint8_t* k = static_cast<const uint8_t*>(a.k);
  const uint8_t* v = static_cast<const uint8_t*>(a.v);
  for (int i = tid; i < kT * D / kChunk; i += kThreads) {
    const int t = i * kChunk / D, b = i * kChunk % D;
    *reinterpret_cast<uint4*>(stage + R::k(t, b)) =
        *reinterpret_cast<const uint4*>(k + t * D + b);
    *reinterpret_cast<uint4*>(stage + R::v(t, b)) =
        *reinterpret_cast<const uint4*>(v + t * D + b);
  }
  const uint16_t* p = static_cast<const uint16_t*>(a.p);
  uint8_t* p_buf = smem + Sm::kP;
  for (int i = tid; i < kRows * kT; i += kThreads) {
    const int r = i / kT, t = i % kT;
    *reinterpret_cast<uint16_t*>(p_buf + r * pth_attn::kDecPPitch +
                                 pth_attn::kF16Bytes * t) =
        r < a.rows ? p[r * kT + t] : static_cast<uint16_t>(0);
  }
  __syncthreads();
  const int g = lane >> 2, qq = lane & 3;
  const uint32_t base = pth_attn::smem_u32(smem);
  float s[NT][4];
  pth_attn::dec_qk<D, NT, R>(s, base + Sm::kQ, base + Sm::stage_offset(0),
                             pth_attn::kDecQkTokens * warp, lane);
  for (int n = 0; n < NT; ++n) {
    for (int j = 0; j < 2; ++j) {
      const int r = 8 * n + 2 * qq + j;
      if (r >= a.rows) continue;
      const int t0 = pth_attn::kDecQkTokens * warp + g;
      a.s_out[r * kT + t0] = s[n][j];
      a.s_out[r * kT + t0 + 8] = s[n][2 + j];
    }
  }
  float o[kMt][NT][4] = {};
  pth_attn::dec_pv<D, NT, R>(o, base + Sm::kP, stage, warp, lane);
  for (int a_ = 0; a_ < kGroups; ++a_) {
    for (int h = 0; h < 2; ++h) {
      for (int n = 0; n < NT; ++n) {
        for (int j = 0; j < 2; ++j) {
          const int r = 8 * n + 2 * qq + j;
          if (r >= a.rows) continue;
          const int d =
              pth_attn::kVGroupD * (kGroups * warp + a_) + 4 * g + 2 * h;
          a.o_out[r * D + d] = o[2 * a_ + h][n][j];
          a.o_out[r * D + d + 1] = o[2 * a_ + h][n][2 + j];
        }
      }
    }
  }
}

template <int D, int NT>
cudaError_t launch_dec_tile_test(const PthAttnDecTileTestArgs& a,
                                 cudaStream_t s) {
  return pth_attn::launch_with_smem<dec_tile_test_kernel<D, NT>,
                                    pth_attn::DecSmem<D, NT, false>::kBytes,
                                    pth_attn::kWarps * pth_attn::kWarpSize>(
      1, s, a);
}

}  // namespace

extern "C" {

int32_t pth_attn_tile_test_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnTileTestArgs));
}
int32_t pth_attn_dec_tile_test_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnDecTileTestArgs));
}

int32_t pth_attn_test_warp_tile(const PthAttnTileTestArgs* a, void* stream) {
  pth_attn::clear_launch_error();
  PTH_ATTN_REQUIRE(a != nullptr, "args is NULL");
  PTH_ATTN_REQUIRE(
      a->abi_version == PTH_ATTN_ABI_VERSION &&
          a->struct_bytes == static_cast<int32_t>(sizeof(PthAttnTileTestArgs)),
      "abi_version/struct_bytes mismatch");
  PTH_ATTN_REQUIRE(pth_attn::is_head_size(a->head_size),
                   "head_size must be 64, 128 or 256");
  PTH_ATTN_REQUIRE(a->q && a->k && a->v && a->p && a->s_out && a->o_out,
                   "all pointers must be non-NULL");
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err;
  switch (a->head_size) {
    case 64:
      err = launch_tile_test<64>(*a, s);
      break;
    case 128:
      err = launch_tile_test<128>(*a, s);
      break;
    default:
      err = launch_tile_test<256>(*a, s);
      break;
  }
  if (err != cudaSuccess) {
    return pth_attn::launch_fail(
        PTH_ATTN_ECUDA,
        std::string("warp_tile_test_kernel: ") + cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}

int32_t pth_attn_test_dec_tile(const PthAttnDecTileTestArgs* a, void* stream) {
  pth_attn::clear_launch_error();
  PTH_ATTN_REQUIRE(a != nullptr, "args is NULL");
  PTH_ATTN_REQUIRE(a->abi_version == PTH_ATTN_ABI_VERSION &&
                       a->struct_bytes ==
                           static_cast<int32_t>(sizeof(PthAttnDecTileTestArgs)),
                   "abi_version/struct_bytes mismatch");
  PTH_ATTN_REQUIRE(a->head_size == 256, "the decode kernel exists for D = 256");
  PTH_ATTN_REQUIRE(a->rows >= 1 && a->rows <= pth_attn::kDecMaxRows,
                   "rows must be 1.." + std::to_string(pth_attn::kDecMaxRows));
  PTH_ATTN_REQUIRE(a->q && a->k && a->v && a->p && a->s_out && a->o_out,
                   "all pointers must be non-NULL");
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err;
  switch ((a->rows + pth_attn::kDecRowsPerNt - 1) / pth_attn::kDecRowsPerNt) {
    case 1:
      err = launch_dec_tile_test<256, 1>(*a, s);
      break;
    case 2:
      err = launch_dec_tile_test<256, 2>(*a, s);
      break;
    case 3:
      err = launch_dec_tile_test<256, 3>(*a, s);
      break;
    default:
      err = launch_dec_tile_test<256, 4>(*a, s);
      break;
  }
  if (err != cudaSuccess) {
    return pth_attn::launch_fail(
        PTH_ATTN_ECUDA,
        std::string("dec_tile_test_kernel: ") + cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}

}  // extern "C"
