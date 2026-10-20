// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// reference_kernel.cu: the scalar fp32 reference kernel, the executable
// specification of pth_flash_attn_fwd.
//
// Its only job is to be obviously correct, so the optimised kernel can be
// checked against it on every shape, split for split. It therefore uses no
// tensor cores, no cp.async, no shared memory and no online softmax, but it
// does share the parts of the optimised kernel that ARE the specification:
//   - the grid (q-tile x, KV head, split): the planner's grid;
//   - every address: cache_geometry.cuh (paging, int64 byte strides from the
//     views, any layout, block size and padded page);
//   - which query rows a CTA owns (row_tile, BQ packing, token-major rows)
//     and which KV tiles a split takes (split_tile_range);
//   - the visible range of a row: the bottom-right causal limit, or the
//     sliding / non-causal window [lo, hi] (kv_window_lo/hi);
//   - the partials and the combine (combine_kernel.cuh).
//
// Work decomposition: a CTA of 4 warps per (q-tile, KV head, split); each
// query row (one query token x one query head of the group) is computed by ONE
// warp. Lane l owns the head-size columns
// d = l + 32 j (j < D/32 <= 8), so a lane keeps <= 8 q values and <= 8 output
// accumulators in registers (no local memory). Per row, per split:
//   pass 1: m = max of the visible scores s_t in the split's KV tiles;
//   pass 2: l = sum_t 2^(s_t - m),  acc_d = sum_t (2^(s_t - m) v_scale_t) v_td;
//   o = acc / l;  lse = m + log2(l).
// Scores are in log2 units: s_t = ((q . k_int_t) * k_scale_t) * c with
// c = fp32(fp32(softmax_scale) * fp32(log2 e)), so 2^s_t is exp of the scaled
// natural-log score. The dot product q . k_int_t is the only cross-lane
// operation: each lane sums its columns in order of j, then a fixed xor
// butterfly (masks 16, 8, 4, 2, 1) adds the 32 partial sums; every lane ends
// with the bit-identical total (a + b == b + a in IEEE arithmetic), so every
// later decision (max, mask, skip) is warp-uniform.
//
// The same source runs on the CPU: ref_capi.cu's host entry point executes
// ref_row with HostWarp, which runs the 32 lanes as 32 threads in lock step
// and implements the shuffle with a barrier. Only the shuffle primitive and
// the libm (exp2f, log2f) differ between the two builds.
//
// Arithmetic contract (every rounding the bound must cover):
//   q: bf16/fp16 -> fp32 exactly; q_d * k_int: exact (<= 19 significant
//   bits); dot: fp32 sums (per lane in order, then the butterfly);
//   s: two fp32 multiplies; p = exp2f(s - m) (CUDA: <= 2 ulp); l: fp32 sum in
//   token order; w = fp32(p * v_scale); acc_d = fmaf(w, v_int, acc_d) in token
//   order; o = acc / l (IEEE division); lse = m + log2f(l).
// Built with -fmad=false (tools/build_ref_lib.sh) so nvcc does not contract
// any other multiply-add: every rounding is the one written here.

#include <cmath>
#include <cstdint>

#include "cache_geometry.cuh"
#include "combine_kernel.cuh"
#include "ref_capi.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace pth_attn {

// Threads per reference CTA: 4 warps (any count works; 4 matches the
// optimised kernel's CTA so occupancy-free code paths look alike).
constexpr int32_t kRefWarpsPerCta = 4;
constexpr int32_t kRefThreadsPerCta = kRefWarpsPerCta * kWarpLanes;
// __launch_bounds__ minimum CTAs per SM. Measured (nvcc 13.0.88, sm_86): with
// only the thread count, ptxas limited the D = 256 kernel to 72 registers and
// spilled 12 bytes; with an explicit minimum of 1 it used 109 registers and
// no local memory (build/ref/ptxas_sm*.log). Occupancy does not matter for
// the reference; local memory would (the kernels must use no LDL/STL).
constexpr int32_t kRefMinCtasPerSm = 1;
// KV tokens per tile: the optimised kernel's load tile N; the
// split partition is defined in these tiles (planner.split_tile_range).
constexpr int32_t kRefTileTokens = 64;
// fp32(log2 e): scores are kept in log2 units (exp2 domain).
constexpr float kRefLog2e = 1.44269504088896340736f;
// The xor butterfly starts at half a warp: masks 16, 8, 4, 2, 1.
constexpr int32_t kButterflyFirstMask = kWarpLanes / 2;

// Everything a reference launch needs: the frozen KernelParams plus the
// element types (the optimised kernel has them as template parameters).
struct RefParams {
  KernelParams k;
  int32_t q_dtype;    // PTH_ATTN_F16 or PTH_ATTN_BF16
  int32_t out_dtype;  // PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16
  int32_t window;     // 0 = none (PthAttnArgs.window)
  bool causal;        // PthAttnArgs.causal != 0
};

// ------------------------------------------------------------------ warps
#if defined(__CUDACC__)
// One hardware warp; all 32 lanes are active (blockDim is a multiple of 32
// and every loop below has a warp-uniform trip count).
struct DeviceWarp {
  int32_t lane_id;
  __device__ int32_t lane() const { return lane_id; }
  __device__ float sum(float v) const {
    for (int32_t mask = kButterflyFirstMask; mask >= 1; mask >>= 1) {
      v = v + __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
  }
};
#endif

// 32 host threads standing in for one warp. sum() publishes every lane's
// value, waits for all 32, and then lets each lane evaluate the same xor
// butterfly as DeviceWarp::sum on a private copy of the 32 values (stage by
// stage, lane i adds its own value and its partner's: t[i] + t[i ^ mask]).
// That is the device's arithmetic exactly, with 2 barriers per sum instead
// of 2 per stage.
struct HostWarpShared {
  std::mutex mu;
  std::condition_variable cv;
  int32_t arrived = 0;
  uint64_t generation = 0;
  float slot[kWarpLanes] = {};

  void barrier() {
    std::unique_lock<std::mutex> lock(mu);
    const uint64_t gen = generation;
    if (++arrived == kWarpLanes) {
      arrived = 0;
      ++generation;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return generation != gen; });
    }
  }
};

struct HostWarp {
  int32_t lane_id;
  HostWarpShared* shared;
  int32_t lane() const { return lane_id; }
  float sum(float v) const {
    shared->slot[lane_id] = v;
    shared->barrier();
    float t[kWarpLanes];
    for (int32_t i = 0; i < kWarpLanes; ++i) t[i] = shared->slot[i];
    shared->barrier();  // all lanes have read the slots: they may be reused
    for (int32_t mask = kButterflyFirstMask; mask >= 1; mask >>= 1) {
      float u[kWarpLanes];
      for (int32_t i = 0; i < kWarpLanes; ++i) u[i] = t[i] + t[i ^ mask];
      for (int32_t i = 0; i < kWarpLanes; ++i) t[i] = u[i];
    }
    return t[lane_id];
  }
};

// ------------------------------------------------------------ one query row
// Score of KV position `pos` for the lane's q columns, in log2 units.
#ifdef __CUDACC__
  #pragma nv_exec_check_disable
#endif
template <int32_t kCols, class Warp>
PTH_HD float ref_score(const RefParams& rp, const Warp& w, int32_t req,
                       int32_t pos, int32_t kv_head, const float* qv) {
  const KernelParams& p = rp.k;
  uint32_t slot = 0;
  const int64_t page = token_page(p, req, static_cast<uint32_t>(pos), slot);
  const char* krow = p.stream[kStreamK].address(page, slot, kv_head);
  float part = 0.0f;
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) {
    const int8_t kint = static_cast<int8_t>(krow[w.lane() + kWarpLanes * j]);
    part = fmaf(qv[j], static_cast<float>(kint), part);  // product is exact
  }
  const float dot = w.sum(part);
  const float ks =
      load_f32(p.stream[kStreamKScale].address(page, slot, kv_head));
  return (dot * ks) * p.score_scale_log2;
}

// Computes query row (token `tok` of row tile `t`, query head `q_head`) over
// the split's KV positions [kv_lo, kv_hi) and writes out (S == 1) or the
// split's partial (S > 1). Every branch below depends only on warp-uniform
// values, so all lanes run the same iterations (required by Warp::sum).
#ifdef __CUDACC__
  #pragma nv_exec_check_disable
#endif
template <int32_t kCols, class Warp>
PTH_HD void ref_row(const RefParams& rp, const Warp& w, const RowTile& t,
                    int32_t tok, int32_t q_head, int32_t kv_head, int32_t split,
                    int32_t kv_lo, int32_t kv_hi) {
  const KernelParams& p = rp.k;
  const int32_t lane = w.lane();
  const int64_t token = static_cast<int64_t>(t.q_start) + t.tok_begin + tok;
  // Visible positions [vis_lo, vis_hi]: bottom-right causal (window 0,
  // causal) or the window; intersected with the split's range.
  const int32_t row_tok = t.tok_begin + tok;
  const int32_t vis_lo = kv_window_lo(t.q_len, t.seq_len, row_tok, rp.window);
  const int32_t vis_hi =
      kv_window_hi(t.q_len, t.seq_len, row_tok, rp.window, rp.causal);
  const int32_t lo = kv_lo > vis_lo ? kv_lo : vis_lo;
  const int32_t hi = kv_hi < vis_hi + 1 ? kv_hi : vis_hi + 1;

  float qv[kCols];
  const char* qrow = static_cast<const char*>(p.q) + q_offset(p, token, q_head);
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) {
    const int64_t d = lane + kWarpLanes * j;
    qv[j] = load_q16(qrow + d * 2, rp.q_dtype);  // 2 bytes per element
  }

  // Pass 1: the split's maximum visible score.
  float m = -INFINITY;
  for (int32_t pos = lo; pos < hi; ++pos) {
    const float s = ref_score<kCols>(rp, w, t.req, pos, kv_head, qv);
    m = s > m ? s : m;
  }

  // Pass 2: denominator and V accumulation, relative to m.
  float l = 0.0f;
  float acc[kCols] = {};
  for (int32_t pos = lo; pos < hi; ++pos) {
    const float s = ref_score<kCols>(rp, w, t.req, pos, kv_head, qv);
    const float pr = exp2f(s - m);
    l += pr;
    uint32_t slot = 0;
    const int64_t page = token_page(p, t.req, static_cast<uint32_t>(pos), slot);
    const float vs =
        load_f32(p.stream[kStreamVScale].address(page, slot, kv_head));
    const float wv = pr * vs;
    const char* vrow = p.stream[kStreamV].address(page, slot, kv_head);
    PTH_ATTN_UNROLL
    for (int32_t j = 0; j < kCols; ++j) {
      const int8_t vint = static_cast<int8_t>(vrow[lane + kWarpLanes * j]);
      acc[j] = fmaf(wv, static_cast<float>(vint), acc[j]);
    }
  }

  const bool empty = hi <= lo;  // no visible token in this split
  if (p.num_splits == 1) {
    // With one split every row sees at least its own position (lo <= p <=
    // hi), so the row is never empty here; o = acc / l.
    char* orow = static_cast<char*>(p.out) + out_offset(p, token, q_head);
    const int64_t elem = rp.out_dtype == PTH_ATTN_F32 ? 4 : 2;
    PTH_ATTN_UNROLL
    for (int32_t j = 0; j < kCols; ++j) {
      store_out(orow + (lane + kWarpLanes * j) * elem, rp.out_dtype,
                empty ? 0.0f : acc[j] / l);
    }
    return;
  }
  char* wo = static_cast<char*>(p.ws_o) + ws_o_offset(p, split, token, q_head);
  PTH_ATTN_UNROLL
  for (int32_t j = 0; j < kCols; ++j) {
    store_f32(
        wo + (lane + kWarpLanes * j) * static_cast<int64_t>(sizeof(float)),
        empty ? 0.0f : acc[j] / l);
  }
  if (lane == 0) {
    char* wl =
        static_cast<char*>(p.ws_lse) + ws_lse_offset(p, split, token, q_head);
    store_f32(wl, empty ? -INFINITY : m + log2f(l));
  }
}

// One CTA: q-tile x, KV head kv_head, split. Warp `warp` of `num_warps`
// takes rows warp, warp + num_warps, ... of the tile (row -> (token, head)
// token-major).
#ifdef __CUDACC__
  #pragma nv_exec_check_disable
#endif
template <int32_t kCols, class Warp>
PTH_HD void ref_cta(const RefParams& rp, const Warp& w, int32_t x,
                    int32_t kv_head, int32_t split, int32_t warp,
                    int32_t num_warps) {
  const KernelParams& p = rp.k;
  RowTile t{};
  if (!row_tile(p.query_start_loc, p.seq_lens, p.bq_div, p.bq, p.num_reqs, x,
                t)) {
    return;  // surplus q-tile: no tokens
  }
  int32_t tile_lo = 0, tile_hi = 0;
  window_split_tile_range(split, p.num_splits,
                          kv_window_range_begin(t, rp.window),
                          kv_window_range_end(t, rp.window, rp.causal),
                          kRefTileTokens, tile_lo, tile_hi);
  const int32_t kv_lo = tile_lo * kRefTileTokens;
  const int32_t kv_hi = tile_hi * kRefTileTokens;
  const int32_t rows = t.num_tok * p.group_size;
  for (int32_t r = warp; r < rows; r += num_warps) {
    int32_t tok = 0, g = 0;
    row_to_token_head(static_cast<uint32_t>(r), p.group, t.num_tok, tok, g);
    ref_row<kCols>(rp, w, t, tok, kv_head * p.group_size + g, kv_head, split,
                   kv_lo, kv_hi);
  }
}

#if defined(__CUDACC__)
// grid = (num_q_tiles, num_kv_heads, num_splits), block = 128 threads.
// kCols = D / 32 is a template parameter so the per-lane arrays are indexed
// with constants after unrolling and stay in registers (no local memory).
template <int32_t kCols>
__global__ void __launch_bounds__(kRefThreadsPerCta, kRefMinCtasPerSm)
    ref_attention_kernel(const RefParams rp) {
  const DeviceWarp w{static_cast<int32_t>(threadIdx.x) % kWarpLanes};
  ref_cta<kCols>(rp, w, static_cast<int32_t>(blockIdx.x),
                 static_cast<int32_t>(blockIdx.y),
                 static_cast<int32_t>(blockIdx.z),
                 static_cast<int32_t>(threadIdx.x) / kWarpLanes,
                 static_cast<int32_t>(blockDim.x) / kWarpLanes);
}

// Arguments of the combine kernel.
struct CombineParams {
  PartialsView v;
  char* out;
  int64_t s_out_token, s_out_head;
  const int32_t* query_start_loc;
  int32_t num_reqs, num_tokens, num_q_heads, head_size, num_splits, out_dtype;
};

// One warp per row (token, head); rows past the last request's last token
// (read from the device) are skipped.
template <int32_t kCols>
__global__ void __launch_bounds__(kRefThreadsPerCta, kRefMinCtasPerSm)
    ref_combine_kernel(const CombineParams c) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRefWarpsPerCta +
                      threadIdx.x / kWarpLanes;
  const int64_t token = row / c.num_q_heads;
  const int32_t head = static_cast<int32_t>(row % c.num_q_heads);
  if (token >= c.num_tokens || token >= c.query_start_loc[c.num_reqs]) return;
  combine_row_store<kCols>(c.v, c.num_splits, token, head,
                           static_cast<int32_t>(threadIdx.x) % kWarpLanes,
                           c.out, c.s_out_token, c.s_out_head, c.out_dtype);
}
#endif

#if defined(__CUDACC__)
// Explicit instantiations for D = 64, 128, 256 (D / 32 columns per lane), so
// this file compiled on its own (tools/ptxas_gate.py, tools/sass_check.py)
// contains every kernel the library launches.
template __global__ void ref_attention_kernel<64 / kWarpLanes>(RefParams);
template __global__ void ref_attention_kernel<128 / kWarpLanes>(RefParams);
template __global__ void ref_attention_kernel<256 / kWarpLanes>(RefParams);
template __global__ void ref_combine_kernel<64 / kWarpLanes>(CombineParams);
template __global__ void ref_combine_kernel<128 / kWarpLanes>(CombineParams);
template __global__ void ref_combine_kernel<256 / kWarpLanes>(CombineParams);
#endif

// Host execution of the whole grid: every CTA, every row, each row on 32
// lock-step threads (HostWarp). Slow and simple; for CPU tests only.
template <int32_t kCols>
void ref_attention_host_run(const RefParams& rp, int32_t num_q_tiles) {
  const KernelParams& p = rp.k;
  for (int32_t split = 0; split < p.num_splits; ++split) {
    for (int32_t kvh = 0; kvh < p.num_kv_heads; ++kvh) {
      for (int32_t x = 0; x < num_q_tiles; ++x) {
        // One emulated warp handles all rows of the CTA (num_warps = 1): the
        // row -> warp assignment does not change any row's arithmetic.
        HostWarpShared shared;
        std::vector<std::thread> lanes;
        lanes.reserve(kWarpLanes);
        for (int32_t lane = 0; lane < kWarpLanes; ++lane) {
          lanes.emplace_back([&, lane] {
            const HostWarp w{lane, &shared};
            ref_cta<kCols>(rp, w, x, kvh, split, 0, 1);
          });
        }
        for (auto& th : lanes) th.join();
      }
    }
  }
}

template <int32_t kCols>
void ref_combine_host_run(const PartialsView& v, char* out, int64_t s_out_token,
                          int64_t s_out_head, const int32_t* qsl,
                          int32_t num_reqs, int32_t num_tokens,
                          int32_t num_q_heads, int32_t num_splits,
                          int32_t out_dtype) {
  const int64_t valid = qsl[num_reqs] < num_tokens ? qsl[num_reqs] : num_tokens;
  for (int64_t token = 0; token < valid; ++token) {
    for (int32_t head = 0; head < num_q_heads; ++head) {
      for (int32_t lane = 0; lane < kWarpLanes; ++lane) {
        combine_row_store<kCols>(v, num_splits, token, head, lane, out,
                                 s_out_token, s_out_head, out_dtype);
      }
    }
  }
}
}  // namespace pth_attn
