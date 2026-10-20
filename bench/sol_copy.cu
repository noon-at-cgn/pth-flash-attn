// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// sol_copy.cu: the memory ceiling for the decode loader.
//
// WHAT IT MEASURES AND WHY IT MUST BE FAITHFUL
//   The decode target is T <= max(T_mem, T_mma/0.75) + T_ovh with
//   T_mem = Hkv*(2D+8)*L / BW_copy, and the decode gate is eta_mem >= 0.90.
//   BW_copy is measured here: the GPU reads exactly the bytes the
//   attention kernel will read, with its global->shared access pattern, its
//   tile, its pipeline and its occupancy, and does no math. If this program
//   were faster than any real loader could be, the gate would be unreachable;
//   if it were slower, the gate would be meaningless. Every choice below copies
//   the kernel and says so.
//
// ARMS (a process interleaves several arms in the mirrored order from
//       `python3 bench/stats.py schedule`)
//   sep-nhd / sep-hnd    separate streams K, V, ks, vs:
//                        per-stream cp.async width chosen on the host from
//                        alignment; today K 8 B, V 4 B, scales 4 B.
//   cell-nhd / cell-hnd  whole 2D+8-byte cell with one width (8 B today) into a
//                        smem pitch of 2D+16 (528 at D=256): the
//                        alternative loader, kept as the challenger.
//   ldr-sep-nhd / ldr-sep-hnd / ldr-cell-nhd / ldr-cell-hnd
//                        the same two designs, but issued by the shipped
//                        loader (csrc/pth_attn/loader.cuh: load_tile_separate,
//                        load_tile_cell, run_two_stage_pipeline, with
//                        cache_geometry.cuh's StreamGeometry, FastDivmod and
//                        int64 offsets; its StageLayout/CellStageLayout) so
//                        the ceiling and the kernel share the address stream.
//                        Same grid, tile partition,
//                        smem footprint and consumer stand-in as the arms
//                        above, which stay as the baseline.
//   plain                streaming LDG read of the same number of bytes from a
//                        contiguous (or --plain-seg/--plain-stride strided)
//                        buffer at full occupancy: the upper reference.
//   "<arm>#<tag>" makes two otherwise identical arms distinct (A/A control);
//   each arm owns its own allocation.
//
// CACHE MODEL (vLLM int8_per_token_head, triton_attn.py _ensure_scale_caches)
//   Logical tensor (num_blocks, Hkv, block_size, 2*(D+4)) of bytes; each
//   (block, head, token) cell is [K D | ks f32 | V D | vs f32]. NHD physical
//   order is page[token][head][cell], HND is page[head][token][cell]; padded
//   hybrid pages only grow the block stride. The host here plays the role of
//   the allocator: it defines the strides of the view. The kernel takes byte
//   strides and stream offsets as inputs and never derives one from a shape;
//   all address arithmetic is int64.
//
// KERNEL GEOMETRY
//   grid (B, Hkv, S): x = request (one row tile at decode), y = KV head,
//   z = split; 128 threads = 4 warps; 64-token tiles, each warp loads 16
//   tokens; CTA s takes tiles [s*T/S, (s+1)*T/S) of its request's T =
//   ceil(L/64) tiles (tile-balanced). Two smem stages; per iteration
//   `wait_group 0; __syncthreads(); issue(tile+1); commit; consume(tile)`.
//   Dynamic smem defaults to the real kernel's total
//   for --wm (Q region 16*WM rows + 2 stages: 74,752 B at D=256 WM=1, the
//   decode policy for q=1; 99,328 B at WM=4) so occupancy (1 CTA/SM at D=256)
//   matches the kernel; the Q region is reserved and unused. The JSON
//   reports ctas_per_sm from the occupancy API: where registers, not smem,
//   would decide the kernel's occupancy (D=64), pin it with --smem-bytes.
//
// LOADER ADDRESSING (a choice made here; the shipped loader follows it)
//   The obvious scheme is "all lanes compute the same (block, offset) per
//   token". That handles w=8 (one 256-B K row per warp
//   instruction) and w=4 (half a V row) but not w=16, where one instruction
//   spans two tokens. Here each lane owns one token of the warp's 16 (lane &
//   15): one FastDivmod and one block_table load per lane per tile, and the row
//   base is broadcast with __shfl_sync to whichever lanes copy that token. The
//   global and shared addresses issued per instruction are identical to the
//   scheme's for w=8/4; only the integer work differs (1 divmod per lane per
//   tile instead of 16).
//   The sep-*/cell-* arms keep this file's own copy of that addressing as the
//   baseline; the ldr-* arms run the same design through csrc/pth_attn/
//   loader.cuh, the code the attention kernel ships, so the two can be
//   compared in one run.
//
// DATA IS DISCARDED BUT NOT OPTIMISED AWAY
//   cp.async is `asm volatile` with a memory clobber, so it cannot be removed.
//   After each tile every thread folds one word of K and one of V from smem
//   into an accumulator (a stand-in for the consumer); the accumulator is
//   written to global memory only if it equals a runtime sentinel, which the
//   compiler cannot prove false. The plain arm folds every loaded word.
//
// SELF-TEST (CPU only, no GPU needed): `sol_copy --self-test` checks
//   FastDivmod against division, the width picker, the split partition and
//   the schedule parser.

#include "bench_harness.cuh"
// The shipped loader and its geometry. A relative include, so that
// bench/ceilings_build.sh and tools/ptxas_gate.py compile this file unchanged.
#include "../csrc/pth_attn/loader.cuh"

#include <cinttypes>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>

namespace {

using bench::Args;
using bench::die;
using bench::Json;
using bench::KernelTimer;

// ---------------------------------------------------------------- constants
constexpr int kThreads = 128;  // 4 warps: O = D/2 registers per thread
constexpr int kWarps = kThreads / 32;
constexpr int kTileTokens = 64;  // N: <= 65 from the smem inequality
constexpr int kTokensPerWarp = kTileTokens / kWarps;  // 16
constexpr int kStages = 2;      // 3 stages exceed 101,376 B at D=256
constexpr int kScaleBytes = 4;  // one f32 scale per (token, head) per side
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kTokensPerWarp == 16,
              "lane & 15 owns a token: 16 tokens per warp");

// ---------------------------------------------------------------- FastDivmod
// Unsigned division by a runtime constant d >= 1 as one __umulhi and one shift
// (the CUTLASS FastDivmod construction): with p = 31 + ceil(log2 d),
// mul = ceil(2^p / d), shr = p - 32, n / d == umulhi(n, mul) >> shr for all
// 0 <= n < 2^31. d == 1 is special-cased (mul would need 33 bits). Positions in
// a sequence are < 2^31, so the domain is sufficient. Verified exhaustively for
// small n and at the domain edge by --self-test.
struct FastDivmod {
  uint32_t div;
  uint32_t mul;
  uint32_t shr;
};

__host__ __device__ inline uint32_t umulhi32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

FastDivmod make_divmod(uint32_t d) {
  if (d == 0) die("divisor must be >= 1");
  FastDivmod f{d, 0, 0};
  if (d == 1) return f;
  uint32_t log2_ceil = 0;
  while ((1ull << log2_ceil) < d) ++log2_ceil;
  const uint32_t p = 31 + log2_ceil;
  f.mul = static_cast<uint32_t>(((1ull << p) + d - 1) / d);
  f.shr = p - 32;
  return f;
}

__host__ __device__ inline void fast_divmod(const FastDivmod& f, int32_t n,
                                            int32_t& q, int32_t& r) {
  const uint32_t un = static_cast<uint32_t>(n);
  const uint32_t uq = f.div == 1 ? un : (umulhi32(un, f.mul) >> f.shr);
  q = static_cast<int32_t>(uq);
  r = n - q * static_cast<int32_t>(f.div);
}

// ---------------------------------------------------------------- PTX wrappers
// cp.async (PTX ISA 7.0+, "Data Movement: cp.async"): .ca allows 4/8/16 B,
// .cg only 16 B. src_bytes < W zero-fills the remainder; 0 reads nothing and
// is how tokens past the end of the split are masked, as the real kernel must
// do so that garbage scales cannot turn p = 0 into NaN.
template <int W>
__device__ __forceinline__ void cp_async(uint32_t dst, const void* src,
                                         int src_bytes) {
  static_assert(W == 4 || W == 8 || W == 16, "cp.async widths are 4, 8, 16");
  if constexpr (W == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
                 "l"(src), "r"(src_bytes)
                 : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst),
                 "l"(src), "n"(W), "r"(src_bytes)
                 : "memory");
  }
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(addr));
  return v;
}

// ---------------------------------------------------------------- kernel
// parameters
struct PagedParams {
  const uint8_t* cache;        // pool base
  const int32_t* block_table;  // [B][bt_row_stride] kernel-block ids
  const int32_t* seq_lens;     // [B], read on the device as the kernel will
  int64_t bt_row_stride;       // elements
  int64_t s_block;             // bytes between pages (includes padding)
  int64_t s_tok;               // bytes between tokens of one head
  int64_t s_head;              // bytes between heads of one token
  int32_t off_k, off_ks, off_v,
      off_vs;                // stream offsets inside the cell (bytes)
  int32_t w_k, w_v, w_cell;  // cp.async widths, warp-uniform
  FastDivmod bs;             // block_size
  int32_t num_splits;
  int32_t q_bytes;  // size of the (unused) Q region before the stages
  uint32_t sentinel;
  uint32_t* sink;
};

// Shared-memory layout of one stage:
//   K [64][D] bytes and V [64][D] bytes stored densely (row-major offset
//   o = t*D + b), each 128-B line split into 8 chunks of 16 B, and chunk
//   (o / 16) % 8 of token t stored at chunk ^ key(t) within the same line:
//   K key = t & 7, V key = 2((t >> 1) & 3). ks[64], vs[64] f32 unswizzled.
// For D >= 128 this is bit-for-bit the row form c ^ (t & 7) on the token row;
// at D = 64 (4 chunks per row) the literal row form is not a permutation,
// the line form is (tests/sim/test_sim_swizzle.py). Every byte stays in its
// 128-B line, so the stage never overflows. tests/sim/ptxsim/layouts.py
// (k_offset, v_offset) is the reference; csrc/pth_attn/smem_layout.cuh
// implements it.
constexpr int kLineBytes = 128;  // 8 chunks of 16 B: one smem bank sweep
constexpr int kChunkBytes = 16;
constexpr int kChunksPerLine = kLineBytes / kChunkBytes;  // 8
constexpr int kSwizzleMaskK = 7;                          // t & 7
constexpr int kSwizzleMaskV = 3;                          // 2 * ((t >> 1) & 3)

__host__ __device__ __forceinline__ uint32_t line_swizzle(int t, int b, int d,
                                                          int key) {
  const int o = t * d + b;  // < 64 * 256: fits easily in int
  const int chunk = (o / kChunkBytes) % kChunksPerLine;
  return static_cast<uint32_t>((o / kLineBytes) * kLineBytes +
                               (chunk ^ key) * kChunkBytes + o % kChunkBytes);
}

template <int D>
struct SepStage {
  static_assert(D % kLineBytes == 0 || kLineBytes % D == 0,
                "rows tile 128-B lines exactly");
  static constexpr int kOffK = 0;
  static constexpr int kOffV = kTileTokens * D;
  static constexpr int kOffKs = 2 * kTileTokens * D;
  static constexpr int kOffVs = kOffKs + kTileTokens * kScaleBytes;
  static constexpr int kBytes =
      kOffVs + kTileTokens * kScaleBytes;  // 33,280 at D=256
  __device__ static uint32_t k_byte(int t, int b) {
    return line_swizzle(t, b, D, t & kSwizzleMaskK);
  }
  __device__ static uint32_t v_byte(int t, int b) {
    return line_swizzle(t, b, D, 2 * ((t >> 1) & kSwizzleMaskV));
  }
};

// Whole-cell stage: [64][pitch] with pitch = 2D+16 (528 at D=256), an odd
// multiple of 16 B, so consecutive tokens start in different bank groups
// (pitch 528 is 1-way for both K ldmatrix and V LDS.32).
template <int D>
struct CellStage {
  static constexpr int kCell = 2 * D + 2 * kScaleBytes;
  static constexpr int kPitch = 2 * D + 16;
  static constexpr int kBytes = kTileTokens * kPitch;
};

// Q region of the real kernel: 16*WM rows x D fp16. It is
// reserved and unused here; its only role is to give this program the same
// shared-memory footprint, hence the same CTAs/SM, as the kernel.
constexpr int kRowsPerWarpM = 16;  // m16 MMA rows per warp
constexpr int q_region_bytes(int d, int wm) {
  return kRowsPerWarpM * wm * d * 2;
}

// ----------------------------------------------------------------
// separate-stream loader Copies one stream (K or V) of the warp's 16 tokens
// with width W. Chunk i of the warp covers token t = i / (D/W), bytes [c*W,
// c*W+W) with c = i % (D/W); lane l takes chunks l, l+32, ... so each warp
// instruction covers 32*W contiguous bytes of one row (W = 4, 8) or two rows (W
// = 16). The row base of token t lives in lane t and is broadcast by shuffle.
template <int D, int W, bool IS_V>
__device__ __forceinline__ void copy_stream(const uint8_t* cache,
                                            int64_t lane_row,
                                            int32_t stream_off,
                                            uint32_t stage_smem, int warp,
                                            int lane, int valid_tokens) {
  constexpr int kChunksPerRow = D / W;
  constexpr int kTotal = kTokensPerWarp * kChunksPerRow;
  static_assert(kTotal % 32 == 0,
                "every lane issues the same number of copies");
  constexpr int kIters = kTotal / 32;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int i = it * 32 + lane;
    const int t = i / kChunksPerRow;  // power of two: a shift
    const int b = (i % kChunksPerRow) * W;
    const int64_t row = __shfl_sync(kFullMask, lane_row, t);
    const int tt = warp * kTokensPerWarp + t;  // token within the 64-token tile
    const uint32_t dst =
        stage_smem + (IS_V ? SepStage<D>::kOffV + SepStage<D>::v_byte(tt, b)
                           : SepStage<D>::kOffK + SepStage<D>::k_byte(tt, b));
    const bool ok = t < valid_tokens;
    cp_async<W>(dst, ok ? cache + row + stream_off + b : cache, ok ? W : 0);
  }
}

template <int D>
__device__ __forceinline__ void issue_tile_sep(const PagedParams& p,
                                               uint32_t stage_smem, int req,
                                               int kvh, int tok0, int tok_end,
                                               int warp, int lane) {
  const int warp_tok0 = tok0 + warp * kTokensPerWarp;
  const int valid = min(kTokensPerWarp, max(0, tok_end - warp_tok0));
  // Lane l owns token (l & 15): one divmod and one block_table read per lane.
  const int pos = warp_tok0 + (lane & 15);
  int64_t row = 0;
  if (pos < tok_end) {
    int32_t blk, off;
    fast_divmod(p.bs, pos, blk, off);
    const int64_t page =
        static_cast<int64_t>(p.block_table[req * p.bt_row_stride + blk]);
    row = page * p.s_block + static_cast<int64_t>(off) * p.s_tok +
          static_cast<int64_t>(kvh) * p.s_head;
  }
  // Widths are warp-uniform kernel arguments, so these
  // switches never diverge; all three paths exist in the SASS.
  switch (p.w_k) {
    case 16:
      copy_stream<D, 16, false>(p.cache, row, p.off_k, stage_smem, warp, lane,
                                valid);
      break;
    case 8:
      copy_stream<D, 8, false>(p.cache, row, p.off_k, stage_smem, warp, lane,
                               valid);
      break;
    default:
      copy_stream<D, 4, false>(p.cache, row, p.off_k, stage_smem, warp, lane,
                               valid);
      break;
  }
  switch (p.w_v) {
    case 16:
      copy_stream<D, 16, true>(p.cache, row, p.off_v, stage_smem, warp, lane,
                               valid);
      break;
    case 8:
      copy_stream<D, 8, true>(p.cache, row, p.off_v, stage_smem, warp, lane,
                              valid);
      break;
    default:
      copy_stream<D, 4, true>(p.cache, row, p.off_v, stage_smem, warp, lane,
                              valid);
      break;
  }
  // Scales: lanes 0-15 copy ks of their token, lanes 16-31 vs: one 4-B
  // instruction for both scale streams of 16 tokens (2 of the ~98 LSU
  // thread-ops per token).
  const bool is_vs = lane >= 16;
  const int tt = warp * kTokensPerWarp + (lane & 15);
  const uint32_t dst = stage_smem +
                       (is_vs ? SepStage<D>::kOffVs : SepStage<D>::kOffKs) +
                       tt * kScaleBytes;
  const bool ok = (lane & 15) < valid;
  cp_async<4>(dst, ok ? p.cache + row + (is_vs ? p.off_vs : p.off_ks) : p.cache,
              ok ? 4 : 0);
}

// ---------------------------------------------------------------- whole-cell
// loader
template <int D, int W>
__device__ __forceinline__ void copy_cells(const uint8_t* cache,
                                           int64_t lane_row,
                                           uint32_t stage_smem, int warp,
                                           int lane, int valid_tokens) {
  constexpr int kChunksPerCell = CellStage<D>::kCell / W;  // 65 at D=256, W=8
  static_assert(CellStage<D>::kCell % W == 0,
                "cell bytes must be a multiple of the width");
  constexpr int kTotal = kTokensPerWarp * kChunksPerCell;
  constexpr int kIters =
      (kTotal + 31) / 32;  // 33 at D=256, W=8: last one partial
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int i = it * 32 + lane;
    // Clamp t for the shuffle (all lanes must take part); the copy itself is
    // predicated on i < kTotal.
    const int t = min(i / kChunksPerCell, kTokensPerWarp - 1);
    const int b = (i % kChunksPerCell) * W;
    const int64_t row = __shfl_sync(kFullMask, lane_row, t);
    if (i < kTotal) {
      const int tt = warp * kTokensPerWarp + t;
      const uint32_t dst =
          stage_smem + static_cast<uint32_t>(tt * CellStage<D>::kPitch + b);
      const bool ok = t < valid_tokens;
      cp_async<W>(dst, ok ? cache + row + b : cache, ok ? W : 0);
    }
  }
}

template <int D>
__device__ __forceinline__ void issue_tile_cell(const PagedParams& p,
                                                uint32_t stage_smem, int req,
                                                int kvh, int tok0, int tok_end,
                                                int warp, int lane) {
  const int warp_tok0 = tok0 + warp * kTokensPerWarp;
  const int valid = min(kTokensPerWarp, max(0, tok_end - warp_tok0));
  const int pos = warp_tok0 + (lane & 15);
  int64_t row = 0;
  if (pos < tok_end) {
    int32_t blk, off;
    fast_divmod(p.bs, pos, blk, off);
    const int64_t page =
        static_cast<int64_t>(p.block_table[req * p.bt_row_stride + blk]);
    row = page * p.s_block + static_cast<int64_t>(off) * p.s_tok +
          static_cast<int64_t>(kvh) * p.s_head;
  }
  if (p.w_cell == 8) {
    copy_cells<D, 8>(p.cache, row, stage_smem, warp, lane, valid);
  } else {
    copy_cells<D, 4>(p.cache, row, stage_smem, warp, lane, valid);
  }
}

// ---------------------------------------------------------------- the paged
// kernels One template for both loaders so the pipeline, partition and consumer
// are literally the same code; only issue_tile differs.
template <int D, bool WHOLE_CELL>
__global__ void __launch_bounds__(kThreads) sol_copy_paged(PagedParams p) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int kStageBytes =
      WHOLE_CELL ? CellStage<D>::kBytes : SepStage<D>::kBytes;
  const int req = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split = blockIdx.z;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;

  const int L = p.seq_lens[req];
  const int64_t n_tiles =
      (static_cast<int64_t>(L) + kTileTokens - 1) / kTileTokens;
  const int t_begin = static_cast<int>(split * n_tiles / p.num_splits);
  const int t_end = static_cast<int>((split + 1) * n_tiles / p.num_splits);
  const int tok_end = min(L, t_end * kTileTokens);

  // Stages sit after the (unused) Q region, as in the real kernel.
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) +
                        static_cast<uint32_t>(p.q_bytes);
  auto issue = [&](int tile, int stage) {
    const uint32_t s = base + stage * kStageBytes;
    if constexpr (WHOLE_CELL) {
      issue_tile_cell<D>(p, s, req, kvh, tile * kTileTokens, tok_end, warp,
                         lane);
    } else {
      issue_tile_sep<D>(p, s, req, kvh, tile * kTileTokens, tok_end, warp,
                        lane);
    }
    cp_async_commit();
  };

  uint32_t acc = 0;
  if (t_begin < t_end) issue(t_begin, 0);
  for (int tile = t_begin; tile < t_end; ++tile) {
    const int stage = (tile - t_begin) & 1;
    // Tile `tile` has landed for this thread; the barrier makes every thread's
    // copies visible and guarantees nobody still reads the other stage, which
    // the next issue overwrites.
    cp_async_wait_all();
    __syncthreads();
    if (tile + 1 < t_end) issue(tile + 1, stage ^ 1);
    // Consumer stand-in: one word of the first and of the second half of the
    // stage per thread (K and V for the separate layout).
    const uint32_t s = base + stage * kStageBytes;
    acc ^= lds32(s + threadIdx.x * 4u);
    acc += lds32(s + kStageBytes / 2 + threadIdx.x * 4u);
  }
  if (acc == p.sentinel)
    p.sink[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] =
        acc;
}

// ---------------------------------------------------------------- loader.cuh
// arms The same copy-only CTA as sol_copy_paged (grid, tile partition, Q
// region, two stages, consumer stand-in), but the global->shared copies and
// the pipeline discipline are the shipped code of csrc/pth_attn/loader.cuh,
// not this file's own. Its only job is to say what the shipped loader can
// move at best: if it matches the sep-*/cell-* arms, those
// ceilings are the loader's ceilings; if it is slower, the difference is the
// cost of the shipped address math.
template <int D, bool kCell>
struct LoaderStageBytes {
  static constexpr int kValue =
      kCell ? pth_attn::CellStageLayout<D, pth_attn::kLoadTileTokens>::kBytes
            : pth_attn::StageLayout<D, pth_attn::kLoadTileTokens>::kBytes;
};
// The host sizes shared memory from the stage formulas of the sep-*/cell-*
// arms; the loader arms must need exactly the same bytes, or the occupancy
// comparison is void.
static_assert(LoaderStageBytes<256, false>::kValue == SepStage<256>::kBytes,
              "loader separate stage = sep-* arm stage");
static_assert(LoaderStageBytes<256, true>::kValue == CellStage<256>::kBytes,
              "loader cell stage = cell-* arm stage");
static_assert(LoaderStageBytes<64, false>::kValue == SepStage<64>::kBytes &&
                  LoaderStageBytes<128, false>::kValue == SepStage<128>::kBytes,
              "loader separate stage = sep-* arm stage at D = 64, 128");
static_assert(LoaderStageBytes<64, true>::kValue == CellStage<64>::kBytes &&
                  LoaderStageBytes<128, true>::kValue == CellStage<128>::kBytes,
              "loader cell stage = cell-* arm stage at D = 64, 128");
struct LoaderParams {
  pth_attn::KernelParams k;  // the four streams, block table, FastDivmod, S
  int32_t cell_width;        // cp.async width of the whole-cell variant
  int32_t q_bytes;           // size of the (unused) Q region before the stages
  uint32_t sentinel;
  uint32_t* sink;
};
template <int D, bool kCell>
__global__ void __launch_bounds__(kThreads) sol_loader_paged(LoaderParams lp) {
  static_assert(kThreads == pth_attn::kThreads, "same CTA as the kernel");
  static_assert(kTileTokens == pth_attn::kLoadTileTokens, "same tile");
  constexpr int kStageBytes = LoaderStageBytes<D, kCell>::kValue;
  extern __shared__ __align__(16) uint8_t smem[];
  const pth_attn::KernelParams& p = lp.k;
  const int32_t req = blockIdx.x;
  const int32_t kvh = blockIdx.y;
  const int32_t split = blockIdx.z;
  const int32_t L = p.seq_lens[req];
  const int32_t n_tiles = (L + kTileTokens - 1) / kTileTokens;
  int32_t t_begin, t_end;
  pth_attn::split_tile_range(split, p.num_splits, n_tiles, t_begin, t_end);
  const int32_t tok_end = min(L, t_end * kTileTokens);
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) +
                        static_cast<uint32_t>(lp.q_bytes);
  uint32_t acc = 0;
  const auto issue = [&](int32_t tile, int stage) {
    const uint32_t s = base + static_cast<uint32_t>(stage * kStageBytes);
    if constexpr (kCell) {
      pth_attn::load_tile_cell<D>(p, lp.cell_width, s, req, kvh, tile, tok_end);
    } else {
      pth_attn::load_tile_separate<D>(p, s, req, kvh, tile, tok_end);
    }
  };
  // As in sol_copy_paged: one word of each half of the stage per thread.
  const auto consume = [&](int32_t, int stage) {
    const uint32_t s = base + static_cast<uint32_t>(stage * kStageBytes);
    acc ^= lds32(s + threadIdx.x * 4u);
    acc += lds32(s + kStageBytes / 2 + threadIdx.x * 4u);
  };
  pth_attn::run_two_stage_pipeline(t_begin, t_end, issue, consume);
  if (acc == lp.sentinel)
    lp.sink[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] =
        acc;
}

// ---------------------------------------------------------------- plain
// streaming read Reads n_seg segments of seg_bytes, segment k starting at
// k*stride_bytes, with W-byte LDGs, grid-stride, 4 independent loads in flight
// per thread. Upper reference: no paging, no smem, maximum occupancy.
constexpr int kPlainThreads = 256;
constexpr int kPlainUnroll = 4;

template <int W>
struct LoadType;
template <>
struct LoadType<4> {
  using T = uint32_t;
  __device__ static uint32_t fold(T v) { return v; }
};
template <>
struct LoadType<8> {
  using T = uint2;
  __device__ static uint32_t fold(T v) { return v.x ^ v.y; }
};
template <>
struct LoadType<16> {
  using T = uint4;
  __device__ static uint32_t fold(T v) { return v.x ^ v.y ^ v.z ^ v.w; }
};

template <int W>
__global__ void __launch_bounds__(kPlainThreads)
    sol_stream_plain(const uint8_t* __restrict__ buf, int32_t n_chunks,
                     FastDivmod chunks_per_seg, int64_t stride_bytes,
                     uint32_t sentinel, uint32_t* sink) {
  using LT = LoadType<W>;
  const int32_t gstride = gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int32_t i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n_chunks;
       i0 += gstride * kPlainUnroll) {
    typename LT::T v[kPlainUnroll];
#pragma unroll
    for (int u = 0; u < kPlainUnroll; ++u) {
      const int32_t i = i0 + u * gstride;
      if (i < n_chunks) {
        int32_t seg, c;
        fast_divmod(chunks_per_seg, i, seg, c);
        v[u] = *reinterpret_cast<const typename LT::T*>(
            buf + seg * stride_bytes + static_cast<int64_t>(c) * W);
      } else {
        v[u] = typename LT::T{};
      }
    }
#pragma unroll
    for (int u = 0; u < kPlainUnroll; ++u) acc ^= LT::fold(v[u]);
  }
  if (acc == sentinel) sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

// Pseudo-random fill so DRAM traffic has realistic bit toggling.
__global__ void fill_random(uint32_t* p, int64_t n_words, uint32_t seed) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       i < n_words; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 0x9E3779B1u ^
                 seed;  // golden-ratio multiplicative hash
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;  // murmur3 finaliser constants
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    p[i] = h;
  }
}

// ---------------------------------------------------------------- host:
// geometry
enum class Layout { kNHD, kHND };
// kLdrSep / kLdrCell: the same two designs issued by csrc/pth_attn/loader.cuh.
enum class Variant { kSep, kCell, kPlain, kLdrSep, kLdrCell };
bool is_cell(Variant v) {
  return v == Variant::kCell || v == Variant::kLdrCell;
}
bool is_loader(Variant v) {
  return v == Variant::kLdrSep || v == Variant::kLdrCell;
}

struct ArmSpec {
  std::string name;
  Variant variant;
  Layout layout;
};

ArmSpec parse_arm(const std::string& s) {
  const std::string body = s.substr(0, s.find('#'));
  if (body == "sep-nhd") return {s, Variant::kSep, Layout::kNHD};
  if (body == "sep-hnd") return {s, Variant::kSep, Layout::kHND};
  if (body == "cell-nhd") return {s, Variant::kCell, Layout::kNHD};
  if (body == "cell-hnd") return {s, Variant::kCell, Layout::kHND};
  if (body == "ldr-sep-nhd") return {s, Variant::kLdrSep, Layout::kNHD};
  if (body == "ldr-sep-hnd") return {s, Variant::kLdrSep, Layout::kHND};
  if (body == "ldr-cell-nhd") return {s, Variant::kLdrCell, Layout::kNHD};
  if (body == "ldr-cell-hnd") return {s, Variant::kLdrCell, Layout::kHND};
  if (body == "plain") return {s, Variant::kPlain, Layout::kNHD};
  die("unknown arm " + s +
      " (sep-nhd sep-hnd cell-nhd cell-hnd ldr-sep-nhd ldr-sep-hnd "
      "ldr-cell-nhd ldr-cell-hnd plain, optional #tag)");
}

// The width of a stream is the largest of {16, 8, 4}
// that divides the base address, every byte stride, and the bytes per row. 0
// means "not even 4-B aligned": the op rejects such caches (the TORCH_CHECK of
// docs/api.md).
int pick_width(uint64_t base_addr, int64_t stream_off, int64_t row_bytes,
               std::initializer_list<int64_t> strides) {
  for (int w : {16, 8, 4}) {
    bool ok = (base_addr + static_cast<uint64_t>(stream_off)) % w == 0 &&
              row_bytes % w == 0;
    for (int64_t s : strides) ok = ok && s % w == 0;
    if (ok) return w;
  }
  return 0;
}

struct Geometry {
  int D, Hkv, block_size, B, L, splits;
  int64_t pad_page_bytes;
  int64_t cell, page_bytes, s_block, s_tok, s_head;
  int pages_per_req;
};

Geometry make_geometry(int D, int Hkv, int block_size, int B, int L, int splits,
                       int64_t pad, Layout layout) {
  Geometry g{};
  g.D = D;
  g.Hkv = Hkv;
  g.block_size = block_size;
  g.B = B;
  g.L = L;
  g.splits = splits;
  g.pad_page_bytes = pad;
  g.cell = 2 * D + 2 * kScaleBytes;
  g.page_bytes = static_cast<int64_t>(block_size) * Hkv * g.cell + pad;
  g.s_block = g.page_bytes;
  if (layout == Layout::kNHD) {  // page[token][head][cell]
    g.s_tok = Hkv * g.cell;
    g.s_head = g.cell;
  } else {  // page[head][token][cell]
    g.s_tok = g.cell;
    g.s_head = static_cast<int64_t>(block_size) * g.cell;
  }
  g.pages_per_req = (L + block_size - 1) / block_size;
  return g;
}

// Tile partition, identical to the device code; used by --self-test.
void split_range(int64_t n_tiles, int s, int S, int& b, int& e) {
  b = static_cast<int>(s * n_tiles / S);
  e = static_cast<int>((s + 1) * n_tiles / S);
}

// ---------------------------------------------------------------- self-test
// (CPU)
int self_test() {
  int fails = 0;
  auto check = [&](bool ok, const std::string& what) {
    if (!ok) {
      std::fprintf(stderr, "FAIL %s\n", what.c_str());
      ++fails;
    }
  };
  // FastDivmod: every divisor 1..4096 (block sizes of interest are 16..1056)
  // against every n < 70,000 (covers several pages at the largest block size)
  // and the domain edge 2^31 - 1.
  for (uint32_t d = 1; d <= 4096; ++d) {
    const FastDivmod f = make_divmod(d);
    for (int32_t n = 0; n < 70000; ++n) {
      int32_t q, r;
      fast_divmod(f, n, q, r);
      if (q != n / static_cast<int32_t>(d) ||
          r != n % static_cast<int32_t>(d)) {
        check(false,
              "divmod d=" + std::to_string(d) + " n=" + std::to_string(n));
        break;
      }
    }
    for (int32_t n : {0x7fffffff, 0x7ffffffe, 0x40000000, 0x3fffffff}) {
      int32_t q, r;
      fast_divmod(f, n, q, r);
      check(
          q == n / static_cast<int32_t>(d) && r == n % static_cast<int32_t>(d),
          "divmod edge d=" + std::to_string(d) + " n=" + std::to_string(n));
    }
  }
  // Widths on the target geometry (D=256, Hkv=2, block 816, NHD):
  // Expected: K 8, V 4, scales 4.
  {
    const Geometry g = make_geometry(256, 2, 816, 1, 4096, 1, 0, Layout::kNHD);
    const uint64_t base =
        1u << 20;  // cudaMalloc returns >= 256-B aligned pointers
    const int wk = pick_width(base, 0, 256, {g.s_block, g.s_tok, g.s_head});
    const int wv =
        pick_width(base, 256 + 4, 256, {g.s_block, g.s_tok, g.s_head});
    const int ws = pick_width(base, 256, 4, {g.s_block, g.s_tok, g.s_head});
    const int wc = pick_width(base, 0, g.cell, {g.s_block, g.s_tok, g.s_head});
    check(wk == 8, "K width 8, got " + std::to_string(wk));
    check(wv == 4, "V width 4, got " + std::to_string(wv));
    check(ws == 4, "scale width 4, got " + std::to_string(ws));
    check(wc == 8, "cell width 8, got " + std::to_string(wc));
    // Negative control: a decoupled 512-B slot layout (#56164) gives K/V 16.
    const int wk16 = pick_width(base, 0, 256, {512 * 816 * 2, 512 * 2, 512});
    check(wk16 == 16, "decoupled K width 16, got " + std::to_string(wk16));
    // A misaligned base must not get a width it cannot honour.
    check(pick_width(base + 2, 0, 256, {512}) == 0,
          "2-B misaligned base must give 0");
    // The shipped loader (cache_geometry.cuh stream_width, used by the
    // ldr-* arms) must pick the same widths from the same four views, or the
    // loader arms would not be the sep-*/cell-* arms' address stream: K 8, V 4,
    // scales 4 and, for the whole cell, 8.
    {
      const auto* pool = reinterpret_cast<const uint8_t*>(base);
      const int64_t sizes[3] = {64, g.block_size, g.Hkv};
      const int64_t st[3] = {g.s_block, g.s_tok, g.s_head};
      pth_attn::KernelParams p{};
      const uint8_t* const start[4] = {pool, pool + 256 + 4, pool + 256,
                                       pool + 2 * 256 + 4};
      for (int s = 0; s < pth_attn::kNumStreams; ++s) {
        const bool scale = s >= pth_attn::kStreamKScale;
        p.stream[s] =
            pth_attn::make_stream(start[s], sizes, st, 1, scale ? 4 : 256);
      }
      check(p.stream[pth_attn::kStreamK].width == wk, "loader K width");
      check(p.stream[pth_attn::kStreamV].width == wv, "loader V width");
      check(p.stream[pth_attn::kStreamKScale].width == ws &&
                p.stream[pth_attn::kStreamVScale].width == ws,
            "loader scale widths");
      check(pth_attn::cell_layout_width<256>(p) == wc, "loader cell width");
      // Negative control: a V view one byte off the cell is not a cell.
      p.stream[pth_attn::kStreamV].base += 1;
      check(pth_attn::cell_layout_width<256>(p) == 0,
            "a shifted V view is not an inline cell");
    }
    for (const char* arm : {"ldr-sep-nhd", "ldr-sep-hnd", "ldr-cell-nhd",
                            "ldr-cell-hnd", "ldr-sep-nhd#aa"}) {
      check(is_loader(parse_arm(arm).variant), std::string("parse ") + arm);
    }
    check(!is_loader(parse_arm("sep-nhd").variant) &&
              is_cell(parse_arm("ldr-cell-nhd").variant) &&
              !is_cell(parse_arm("ldr-sep-nhd").variant),
          "arm variants are told apart");
  }
  // HND strides.
  {
    const Geometry g = make_geometry(256, 2, 816, 1, 4096, 1, 0, Layout::kHND);
    check(g.s_tok == 520 && g.s_head == 816 * 520 && g.s_block == 816 * 2 * 520,
          "HND strides");
  }
  // Stage swizzle (line form): for every D the K and V maps are
  // permutations of the 64*D-byte region (so no copy lands outside the stage
  // or on another copy) and equal the row form c ^ key for D >= 128.
  // Negative control: the row form read literally at D = 64 is not
  // a permutation (it writes past a 64-B row), which is why the line form is
  // used.
  for (int d : {64, 128, 256}) {
    for (int v = 0; v < 2; ++v) {
      std::vector<int> hit(static_cast<size_t>(kTileTokens * d), 0);
      bool equals_row_form = true;
      for (int t = 0; t < kTileTokens; ++t) {
        const int key =
            v ? 2 * ((t >> 1) & kSwizzleMaskV) : (t & kSwizzleMaskK);
        for (int b = 0; b < d; ++b) {
          const uint32_t o = line_swizzle(t, b, d, key);
          if (o < hit.size()) hit[o]++;
          const int row_form =
              t * d + ((b / kChunkBytes) ^ key) * kChunkBytes + b % kChunkBytes;
          equals_row_form = equals_row_form && static_cast<int>(o) == row_form;
        }
      }
      check(std::all_of(hit.begin(), hit.end(), [](int h) { return h == 1; }),
            "line swizzle is a permutation, D=" + std::to_string(d));
      if (d >= 128) {
        check(equals_row_form,
              "line swizzle equals the row form, D=" + std::to_string(d));
      } else {
        check(!equals_row_form, "negative control: row form differs at D=64");
      }
    }
  }
  {
    const int d = 64;  // negative control: row form escapes the region
    int max_off = 0;
    for (int t = 0; t < kTileTokens; ++t) {
      for (int b = 0; b < d; ++b) {
        max_off = std::max(max_off,
                           t * d + ((b / kChunkBytes) ^ (t & 7)) * kChunkBytes +
                               b % kChunkBytes);
      }
    }
    check(max_off >= kTileTokens * d,
          "negative control: row form at D=64 leaves the K region");
  }
  // Split partition: covers [0, T) exactly once, balanced within one tile.
  for (int64_t T : {1, 2, 63, 64, 65, 2048, 4096}) {
    for (int S : {1, 2, 3, 34, 68}) {
      int prev = 0, mn = 1 << 30, mx = 0;
      for (int s = 0; s < S; ++s) {
        int b, e;
        split_range(T, s, S, b, e);
        check(b == prev, "split contiguous");
        prev = e;
        mn = std::min(mn, e - b);
        mx = std::max(mx, e - b);
      }
      check(prev == T, "split covers all tiles");
      check(mx - mn <= 1, "split balanced");
    }
  }
  // Schedule parser accepts a mirrored schedule (parse_schedule exits on
  // error).
  bench::parse_schedule("0,1,2,2,1,0", 3);
  std::printf("{\"self_test\": \"%s\", \"failures\": %d}\n",
              fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------- host: arms
struct Arm {
  ArmSpec spec;
  Geometry g;
  int n_rot = 1;  // rotating copies (cold L2)
  uint8_t* pool = nullptr;
  int32_t* block_tables = nullptr;  // [n_rot][B][pages_per_req]
  int32_t* seq_lens = nullptr;
  uint32_t* sink = nullptr;
  int64_t bytes_per_launch = 0;
  int smem_bytes = 0;
  int ctas_per_sm = 0;
  dim3 grid, block;
  int w_k = 0, w_v = 0, w_s = 0, w_cell = 0;
  // plain arm
  int plain_width = 16;
  int64_t plain_seg = 0, plain_stride = 0;
  cudaGraphExec_t exec = nullptr;
  const char* kernel_needle = nullptr;
};

template <int D>
void set_smem_attr(int bytes) {
  BENCH_CUDA(cudaFuncSetAttribute(sol_copy_paged<D, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes));
  BENCH_CUDA(cudaFuncSetAttribute(sol_copy_paged<D, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes));
  BENCH_CUDA(cudaFuncSetAttribute(sol_loader_paged<D, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes));
  BENCH_CUDA(cudaFuncSetAttribute(sol_loader_paged<D, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes));
}

template <int D>
void launch_paged(const Arm& a, const PagedParams& p, cudaStream_t s) {
  if (a.spec.variant == Variant::kCell) {
    sol_copy_paged<D, true><<<a.grid, a.block, a.smem_bytes, s>>>(p);
  } else {
    sol_copy_paged<D, false><<<a.grid, a.block, a.smem_bytes, s>>>(p);
  }
}

template <int D>
void launch_loader(const Arm& a, const LoaderParams& p, cudaStream_t s) {
  if (a.spec.variant == Variant::kLdrCell) {
    sol_loader_paged<D, true><<<a.grid, a.block, a.smem_bytes, s>>>(p);
  } else {
    sol_loader_paged<D, false><<<a.grid, a.block, a.smem_bytes, s>>>(p);
  }
}

template <int D>
int occupancy_paged(const Arm& a) {
  int n = 0;
  switch (a.spec.variant) {
    case Variant::kCell:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_copy_paged<D, true>, kThreads, a.smem_bytes));
      break;
    case Variant::kLdrSep:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_loader_paged<D, false>, kThreads, a.smem_bytes));
      break;
    case Variant::kLdrCell:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_loader_paged<D, true>, kThreads, a.smem_bytes));
      break;
    default:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_copy_paged<D, false>, kThreads, a.smem_bytes));
      break;
  }
  return n;
}

void launch_plain(const Arm& a, const uint8_t* buf, cudaStream_t s) {
  const int64_t n_seg = a.bytes_per_launch / a.plain_seg;
  const int64_t n_chunks64 = n_seg * (a.plain_seg / a.plain_width);
  if (n_chunks64 > INT32_MAX)
    die("plain arm: too many chunks for 32-bit indexing");
  const FastDivmod cps =
      make_divmod(static_cast<uint32_t>(a.plain_seg / a.plain_width));
  const int32_t n = static_cast<int32_t>(n_chunks64);
  switch (a.plain_width) {
    case 4:
      sol_stream_plain<4><<<a.grid, a.block, 0, s>>>(
          buf, n, cps, a.plain_stride, 0xFFFFFFFFu, a.sink);
      break;
    case 8:
      sol_stream_plain<8><<<a.grid, a.block, 0, s>>>(
          buf, n, cps, a.plain_stride, 0xFFFFFFFFu, a.sink);
      break;
    default:
      sol_stream_plain<16><<<a.grid, a.block, 0, s>>>(
          buf, n, cps, a.plain_stride, 0xFFFFFFFFu, a.sink);
      break;
  }
}

int occupancy_plain(int w) {
  int n = 0;
  switch (w) {
    case 4:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_stream_plain<4>, kPlainThreads, 0));
      break;
    case 8:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_stream_plain<8>, kPlainThreads, 0));
      break;
    default:
      BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &n, sol_stream_plain<16>, kPlainThreads, 0));
      break;
  }
  return n;
}

// The shipped KernelParams of a loader arm, built the way the op builds
// them from four views (strides come from the views, never from
// a shape): K at the cell start, ks at +D, V at +D+4, vs at +2D+4, all with the
// cell's byte strides. The widths are cache_geometry.cuh's stream_width, not
// this file's pick_width, unless one is forced down for an experiment.
template <int D>
int32_t loader_cell_width(const pth_attn::KernelParams& p) {
  return pth_attn::cell_layout_width<D>(p);
}
LoaderParams make_loader_params(const Arm& a, int rot, int num_splits,
                                int q_bytes, int force_wk, int force_wv,
                                uint32_t* sink) {
  const Geometry& g = a.g;
  const int64_t n_pages = static_cast<int64_t>(g.pages_per_req) * g.B * a.n_rot;
  const int64_t sizes[3] = {n_pages, g.block_size, g.Hkv};
  const int64_t strides[3] = {g.s_block, g.s_tok, g.s_head};  // bytes
  const uint8_t* const start[pth_attn::kNumStreams] = {
      a.pool,                 // K
      a.pool + g.D + 4,       // V
      a.pool + g.D,           // k_scale
      a.pool + 2 * g.D + 4};  // v_scale
  LoaderParams lp{};
  pth_attn::KernelParams& p = lp.k;
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    const bool scale = s >= pth_attn::kStreamKScale;
    p.stream[s] = pth_attn::make_stream(start[s], sizes, strides, 1,
                                        scale ? kScaleBytes : g.D);
    if (p.stream[s].width == 0) die("cache is not 4-B aligned");
  }
  // A forced width must not exceed the natural one (as for the sep-*/cell-*
  // arms).
  if (force_wk > p.stream[pth_attn::kStreamK].width ||
      force_wv > p.stream[pth_attn::kStreamV].width)
    die("a forced width is larger than the alignment allows");
  if (force_wk) p.stream[pth_attn::kStreamK].width = force_wk;
  if (force_wv) p.stream[pth_attn::kStreamV].width = force_wv;
  p.block_table =
      a.block_tables + static_cast<int64_t>(rot) * g.B * g.pages_per_req;
  p.bt_row_stride = g.pages_per_req;
  p.seq_lens = a.seq_lens;
  p.block_size =
      pth_attn::FastDivmod::make(static_cast<uint32_t>(g.block_size));
  p.num_reqs = g.B;
  p.num_kv_heads = g.Hkv;
  p.head_size = g.D;
  p.num_splits = num_splits;
  lp.cell_width = g.D == 64    ? loader_cell_width<64>(p)
                  : g.D == 128 ? loader_cell_width<128>(p)
                               : loader_cell_width<256>(p);
  if (a.spec.variant == Variant::kLdrCell && lp.cell_width == 0)
    die("the four views are not one inline cell");
  lp.q_bytes = q_bytes;
  lp.sentinel = 0xFFFFFFFFu;
  lp.sink = sink;
  return lp;
}

// Rotation for a cold L2: if one launch touches less than 4x the
// L2 capacity, rotate over enough disjoint copies that the bytes touched
// between two uses of the same copy are >= 4x L2. This gives >= 4 copies
// whenever one copy fits in L2.
constexpr int64_t kColdL2Factor = 4;

int rotation_count(int64_t bytes_per_launch, int64_t l2_bytes, int replays) {
  if (bytes_per_launch >= kColdL2Factor * l2_bytes) return 1;
  const int64_t n =
      (kColdL2Factor * l2_bytes + bytes_per_launch - 1) / bytes_per_launch;
  return static_cast<int>(std::min<int64_t>(n, replays));
}

}  // namespace

constexpr const char kUsage[] =
    R"(sol_copy: memory-bandwidth ceiling for the decode loader.

Reads the same KV bytes as the attention kernel, with the same global->shared
access pattern, tile, pipeline and occupancy, and does no math. Needs an NVIDIA
GPU (sm80, sm86 or sm89) and CUPTI (libcupti.so.13); --self-test needs neither.
bench/ceilings_run.sh and bench/k_run.sh are the drivers that run it.

  sol_copy --arms ARM[,ARM...] --schedule S --out FILE [options]
  sol_copy --device-info        print the device JSON and exit
  sol_copy --self-test          host checks (FastDivmod, widths, split partition)
  sol_copy --help               this text

Arms: sep-nhd sep-hnd cell-nhd cell-hnd ldr-sep-nhd ldr-sep-hnd ldr-cell-nhd
ldr-cell-hnd plain, each optionally suffixed "#tag" (A/A control).
Options (default): --schedule arm indices in launch order, from
`python3 bench/stats.py schedule`; --out raw JSON path; --batch (1), --seq-len
(131072), --block-size (816), --head-dim (256), --num-kv-heads (2), --splits
(required: the planner's split count, pth_flash_attn.planner.plan_launch), --wm (1),
--replays (20), --warmup-s (2), --seed (0), --process (0), --timer cupti|events
(cupti), --device (0), --page-order shuffle|seq (shuffle),
--smem-bytes, --pad-page-bytes, --force-width-k, --force-width-v, --plain-width,
--plain-seg-bytes, --plain-stride-bytes. Output: one raw JSON file
(raw_format 1) that bench/bench_report.py turns into result records.
)";

int main(int argc, char** argv) {
  Args args(argc, argv);
  if (args.has("help")) {
    std::fputs(kUsage, stdout);
    return 0;
  }
  if (args.has("self-test")) {
    args.str("self-test", "");
    args.reject_unknown();
    return self_test();
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

  // ---- configuration
  const std::string out_path = args.req("out");
  const std::vector<std::string> arm_names =
      bench::split(args.req("arms"), ',');
  const int D = static_cast<int>(args.i64("head-dim", 256));
  const int Hkv = static_cast<int>(args.i64("num-kv-heads", 2));
  const int block_size = static_cast<int>(args.i64("block-size", 816));
  const int B = static_cast<int>(args.i64("batch", 1));
  const int L = static_cast<int>(args.i64("seq-len", 131072));
  const int S = static_cast<int>(args.i64("splits", 0));
  const int64_t pad = args.i64("pad-page-bytes", 0);
  const int replays = static_cast<int>(args.i64("replays", 20));
  const double warmup_s = args.f64("warmup-s", 2.0);
  const uint32_t seed = static_cast<uint32_t>(args.i64("seed", 0));
  const std::string page_order = args.str("page-order", "shuffle");
  const int64_t smem_override = args.i64("smem-bytes", 0);
  const int wm = static_cast<int>(args.i64("wm", 1));
  const int process = static_cast<int>(args.i64("process", 0));
  const std::string schedule_str = args.req("schedule");
  const KernelTimer::Kind timer_kind =
      bench::parse_timer(args.str("timer", "cupti"));
  const int force_wk = static_cast<int>(args.i64("force-width-k", 0));
  const int force_wv = static_cast<int>(args.i64("force-width-v", 0));
  const int plain_width = static_cast<int>(args.i64("plain-width", 16));
  const int64_t plain_seg_arg = args.i64("plain-seg-bytes", 0);
  const int64_t plain_stride_arg = args.i64("plain-stride-bytes", 0);
  args.reject_unknown();

  if (D != 64 && D != 128 && D != 256) die("--head-dim must be 64, 128 or 256");
  if (wm != 1 && wm != 2 && wm != 4) die("--wm must be 1, 2 or 4");
  if (Hkv < 1 || block_size < 1 || B < 1 || L < 1 || replays < 1)
    die("shape arguments must be >= 1");
  if (S < 1)
    die("--splits is required (take it from "
        "pth_flash_attn.planner.plan_launch)");
  if (page_order != "shuffle" && page_order != "seq")
    die("--page-order must be shuffle or seq");
  if (plain_width != 4 && plain_width != 8 && plain_width != 16)
    die("--plain-width must be 4, 8 or 16");
  for (int w : {force_wk, force_wv}) {
    if (w != 0 && w != 4 && w != 8 && w != 16)
      die("--force-width-* must be 4, 8 or 16");
  }
  const std::vector<int> schedule =
      bench::parse_schedule(schedule_str, static_cast<int>(arm_names.size()));
  for (const auto& n : arm_names)
    parse_arm(n);  // fail on a bad arm before touching the GPU

  // CUPTI must be enabled before the context exists.
  KernelTimer& timer = KernelTimer::instance();
  timer.start(timer_kind);
  BENCH_CUDA(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, dev));
  cudaStream_t stream;
  BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  uint32_t* sink = nullptr;
  BENCH_CUDA(cudaMalloc(&sink, 1 << 20));

  // ---- arms
  std::vector<Arm> arms;
  std::mt19937_64 rng(seed);
  for (const auto& name : arm_names) {
    Arm a;
    a.spec = parse_arm(name);
    a.g = make_geometry(D, Hkv, block_size, B, L, S, pad, a.spec.layout);
    a.sink = sink;
    a.bytes_per_launch = static_cast<int64_t>(B) * Hkv * L * a.g.cell;
    a.n_rot = rotation_count(a.bytes_per_launch, prop.l2CacheSize, replays);

    if (a.spec.variant == Variant::kPlain) {
      a.plain_width = plain_width;
      a.plain_seg = plain_seg_arg > 0 ? plain_seg_arg : a.bytes_per_launch;
      a.plain_stride = plain_stride_arg > 0 ? plain_stride_arg : a.plain_seg;
      if (a.plain_seg % plain_width || a.plain_stride % plain_width ||
          a.bytes_per_launch % a.plain_seg) {
        die("plain arm: seg and stride must be multiples of the width, and seg "
            "must divide the bytes");
      }
      const int64_t n_seg = a.bytes_per_launch / a.plain_seg;
      const int64_t span = (n_seg - 1) * a.plain_stride + a.plain_seg;
      BENCH_CUDA(cudaMalloc(&a.pool, static_cast<size_t>(span) * a.n_rot));
      fill_random<<<prop.multiProcessorCount * 8, 256, 0, stream>>>(
          reinterpret_cast<uint32_t*>(a.pool), span * a.n_rot / 4,
          seed ^ 0xA5u);
      a.ctas_per_sm = occupancy_plain(plain_width);
      a.grid = dim3(prop.multiProcessorCount * a.ctas_per_sm);
      a.block = dim3(kPlainThreads);
      a.smem_bytes = 0;
      a.kernel_needle = "sol_stream_plain";
    } else {
      const bool cell = is_cell(a.spec.variant);
      const int stage =
          cell ? (2 * D + 16) * kTileTokens
               : 2 * kTileTokens * D + 2 * kTileTokens * kScaleBytes;
      const int need = q_region_bytes(D, wm) + kStages * stage;
      a.smem_bytes = smem_override > 0 ? static_cast<int>(smem_override) : need;
      if (a.smem_bytes < need)
        die("--smem-bytes is smaller than Q region + 2 stages");
      if (a.smem_bytes > static_cast<int>(prop.sharedMemPerBlockOptin))
        die("smem exceeds the per-block limit");
      if (D == 64) set_smem_attr<64>(a.smem_bytes);
      if (D == 128) set_smem_attr<128>(a.smem_bytes);
      if (D == 256) set_smem_attr<256>(a.smem_bytes);
      a.ctas_per_sm = D == 64    ? occupancy_paged<64>(a)
                      : D == 128 ? occupancy_paged<128>(a)
                                 : occupancy_paged<256>(a);

      const int64_t n_pages =
          static_cast<int64_t>(a.g.pages_per_req) * B * a.n_rot;
      if (n_pages > INT32_MAX) die("too many pages for int32 block ids");
      BENCH_CUDA(
          cudaMalloc(&a.pool, static_cast<size_t>(n_pages * a.g.page_bytes)));
      fill_random<<<prop.multiProcessorCount * 8, 256, 0, stream>>>(
          reinterpret_cast<uint32_t*>(a.pool), n_pages * a.g.page_bytes / 4,
          seed ^ 0x5Au);
      // Block tables: a permutation of page ids, split into n_rot disjoint
      // sets of B requests. Shuffled by default because a long-running
      // server's pages are scattered; --page-order seq is the comparison.
      std::vector<int32_t> ids(static_cast<size_t>(n_pages));
      std::iota(ids.begin(), ids.end(), 0);
      if (page_order == "shuffle") std::shuffle(ids.begin(), ids.end(), rng);
      BENCH_CUDA(cudaMalloc(&a.block_tables, ids.size() * sizeof(int32_t)));
      BENCH_CUDA(cudaMemcpy(a.block_tables, ids.data(),
                            ids.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      std::vector<int32_t> lens(static_cast<size_t>(B), L);
      BENCH_CUDA(cudaMalloc(&a.seq_lens, lens.size() * sizeof(int32_t)));
      BENCH_CUDA(cudaMemcpy(a.seq_lens, lens.data(),
                            lens.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice));

      const uint64_t base = reinterpret_cast<uint64_t>(a.pool);
      const auto strides = {a.g.s_block, a.g.s_tok, a.g.s_head};
      // Natural widths. A forced width is valid iff it
      // is not larger than the natural one: divisibility by 16 implies by 8
      // and 4.
      const int nat_k = pick_width(base, 0, D, strides);
      const int nat_v = pick_width(base, D + kScaleBytes, D, strides);
      a.w_s = pick_width(base, D, kScaleBytes, strides);
      a.w_cell = pick_width(base, 0, a.g.cell, strides);
      if (nat_k == 0 || nat_v == 0 || a.w_s == 0 || a.w_cell == 0)
        die("cache is not 4-B aligned");
      if (force_wk > nat_k || force_wv > nat_v)
        die("a forced width is larger than the alignment allows");
      a.w_k = force_wk ? force_wk : nat_k;
      a.w_v = force_wv ? force_wv : nat_v;
      if (is_loader(a.spec.variant)) {
        // Report the widths the production loader picked, not pick_width's.
        const LoaderParams lp = make_loader_params(
            a, 0, S, q_region_bytes(D, wm), force_wk, force_wv, sink);
        a.w_k = lp.k.stream[pth_attn::kStreamK].width;
        a.w_v = lp.k.stream[pth_attn::kStreamV].width;
        a.w_s = lp.k.stream[pth_attn::kStreamKScale].width;
        a.w_cell = lp.cell_width;
      }
      a.grid = dim3(B, Hkv, S);
      a.block = dim3(kThreads);
      a.kernel_needle =
          is_loader(a.spec.variant) ? "sol_loader_paged" : "sol_copy_paged";
    }

    // ---- graph of `replays` launches, launch r uses rotation copy r % n_rot
    cudaGraph_t graph;
    BENCH_CUDA(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int r = 0; r < replays; ++r) {
      const int rot = r % a.n_rot;
      if (a.spec.variant == Variant::kPlain) {
        const int64_t n_seg = a.bytes_per_launch / a.plain_seg;
        const int64_t span = (n_seg - 1) * a.plain_stride + a.plain_seg;
        launch_plain(a, a.pool + rot * span, stream);
      } else if (is_loader(a.spec.variant)) {
        const LoaderParams lp = make_loader_params(
            a, rot, S, q_region_bytes(D, wm), force_wk, force_wv, sink);
        if (D == 64) launch_loader<64>(a, lp, stream);
        if (D == 128) launch_loader<128>(a, lp, stream);
        if (D == 256) launch_loader<256>(a, lp, stream);
      } else {
        PagedParams p{};
        p.cache = a.pool;
        p.block_table =
            a.block_tables + static_cast<int64_t>(rot) * B * a.g.pages_per_req;
        p.seq_lens = a.seq_lens;
        p.bt_row_stride = a.g.pages_per_req;
        p.s_block = a.g.s_block;
        p.s_tok = a.g.s_tok;
        p.s_head = a.g.s_head;
        p.off_k = 0;
        p.off_ks = D;
        p.off_v = D + kScaleBytes;
        p.off_vs = 2 * D + kScaleBytes;
        p.w_k = a.w_k;
        p.w_v = a.w_v;
        p.w_cell = a.w_cell;
        p.bs = make_divmod(static_cast<uint32_t>(block_size));
        p.num_splits = S;
        p.q_bytes = q_region_bytes(D, wm);
        p.sentinel = 0xFFFFFFFFu;
        p.sink = sink;
        if (D == 64) launch_paged<64>(a, p, stream);
        if (D == 128) launch_paged<128>(a, p, stream);
        if (D == 256) launch_paged<256>(a, p, stream);
      }
    }
    BENCH_CUDA(cudaStreamEndCapture(stream, &graph));
    BENCH_CUDA(cudaGraphInstantiate(&a.exec, graph, 0));
    BENCH_CUDA(cudaGraphDestroy(graph));
    arms.push_back(a);
  }
  BENCH_CUDA(cudaStreamSynchronize(stream));
  BENCH_CUDA(cudaGetLastError());

  // ---- warm-up: round-robin over all arms until warmup_s has elapsed, so the
  // card reaches its power-limit equilibrium (220 W on the reference
  // cards) before the first block.
  const uint64_t t_warm0 = bench::mono_ns();
  int64_t warm_graphs = 0;
  while (bench::mono_ns() - t_warm0 < static_cast<uint64_t>(warmup_s * 1e9)) {
    for (const auto& a : arms) {
      BENCH_CUDA(cudaGraphLaunch(a.exec, stream));
      ++warm_graphs;
    }
    BENCH_CUDA(cudaStreamSynchronize(stream));
  }
  for (const auto& a : arms)
    timer.collect(a.kernel_needle);  // drop warm-up records

  // ---- measured blocks
  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) die("cannot open --out " + out_path);
  Json j(f);
  j.begin_obj();
  j.kv("tool", "sol_copy");
  j.kv("raw_format", 1);
  j.begin_arr("argv");
  for (const auto& s : args.argv()) j.val(s);
  j.end_arr();
  j.kv("timer", timer.kind_name());
  j.kv("duration_kind", timer.kind() == KernelTimer::Kind::kCupti
                            ? "kernel"
                            : "graph_replay_mean");
  j.kv("process", process);
  j.kv("t_unix_ns", bench::unix_ns());
  bench::write_device(j, dev);
  j.begin_obj("shape");
  j.kv("D", D);
  j.kv("Hkv", Hkv);
  j.kv("block_size", block_size);
  j.kv("batch", B);
  j.kv("seq_len", L);
  j.kv("splits", S);
  j.kv("pad_page_bytes", pad);
  j.kv("page_order", page_order);
  j.end_obj();
  j.begin_obj("design");
  j.kv("replays_per_block", replays);
  j.kv("warmup_s", warmup_s);
  j.kv("warmup_graphs", warm_graphs);
  j.kv("n_arms", static_cast<int>(arms.size()));
  j.kv("n_blocks", static_cast<int>(schedule.size()));
  j.end_obj();
  j.begin_arr("arms");
  for (const auto& a : arms) {
    j.begin_obj();
    j.kv("name", a.spec.name);
    j.kv("variant", a.spec.variant == Variant::kSep       ? "sep"
                    : a.spec.variant == Variant::kCell    ? "cell"
                    : a.spec.variant == Variant::kLdrSep  ? "ldr-sep"
                    : a.spec.variant == Variant::kLdrCell ? "ldr-cell"
                                                          : "plain");
    j.kv("layout", a.spec.layout == Layout::kNHD ? "NHD" : "HND");
    j.kv("bytes_per_launch", a.bytes_per_launch);
    j.kv("rotations", a.n_rot);
    j.kv("smem_bytes", a.smem_bytes);
    j.kv("wm", wm);
    j.kv("ctas_per_sm", a.ctas_per_sm);
    j.kv("grid_x", static_cast<int>(a.grid.x));
    j.kv("grid_y", static_cast<int>(a.grid.y));
    j.kv("grid_z", static_cast<int>(a.grid.z));
    j.kv("threads", static_cast<int>(a.block.x));
    j.kv("w_k", a.w_k);
    j.kv("w_v", a.w_v);
    j.kv("w_scale", a.w_s);
    j.kv("w_cell", a.w_cell);
    j.kv("s_block", a.g.s_block);
    j.kv("s_tok", a.g.s_tok);
    j.kv("s_head", a.g.s_head);
    j.kv("plain_width", a.plain_width);
    j.kv("plain_seg_bytes", a.plain_seg);
    j.kv("plain_stride_bytes", a.plain_stride);
    j.end_obj();
  }
  j.end_arr();

  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  j.begin_arr("blocks");
  for (size_t k = 0; k < schedule.size(); ++k) {
    const Arm& a = arms[schedule[k]];
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
    std::vector<uint64_t> d = timer.collect(a.kernel_needle);
    if (timer.kind() == KernelTimer::Kind::kCupti &&
        static_cast<int>(d.size()) != replays) {
      std::fprintf(
          stderr, "block %zu: CUPTI returned %zu kernel records, expected %d\n",
          k, d.size(), replays);
      std::exit(3);
    }
    if (timer.kind() != KernelTimer::Kind::kCupti) {
      // Event timing cannot separate launches inside one graph: every replay
      // gets the graph mean, and the JSON says so (duration_kind).
      d.assign(static_cast<size_t>(replays),
               static_cast<uint64_t>(graph_ms * 1e6 / replays));
    }
    j.begin_obj();
    j.kv("seq", static_cast<int>(k));
    j.kv("round", static_cast<int>(k / arms.size()));
    j.kv("arm", a.spec.name);
    j.kv("t_start_mono_ns", t0);
    j.kv("t_end_mono_ns", t1);
    j.kv("graph_ms", static_cast<double>(graph_ms));
    j.begin_arr("durations_ns");
    for (uint64_t v : d) j.val(v);
    j.end_arr();
    j.end_obj();
  }
  j.end_arr();
  j.kv("t_end_unix_ns", bench::unix_ns());
  j.end_obj();
  std::fputc('\n', f);
  std::fclose(f);
  timer.stop();
  BENCH_CUDA(cudaGetLastError());
  return 0;
}
