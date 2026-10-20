// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// loader_capi.cu: the test-only C ABI of loader.cuh (see loader_capi.h).
// Compiled by g++ (-x c++, host emulation only) and by nvcc (adds the GPU
// launch). Argument validation and KernelParams construction are shared, so
// the GPU run and the host emulation see the same streams and widths.

#include "loader_capi.h"

#include <cstdint>
#include <cstring>

#include "cache_geometry.cuh"
#include "loader.cuh"
#include "loader_test_kernels.h"
#include "smem_layout.cuh"
#include "tile_traits.cuh"

namespace {

using pth_attn::CellStageLayout;
using pth_attn::KernelParams;
using pth_attn::StageLayout;

constexpr int kSizes = 3;                                   // [NB, BS, Hkv]
constexpr int64_t kKvElemBytes = 1;                         // int8
constexpr int64_t kScaleElemBytes = pth_attn::kScaleBytes;  // f32
constexpr int kMapFields = 5;  // warp, lane, stream, token, byte

bool valid_head_size(int32_t d) { return d == 64 || d == 128 || d == 256; }

int32_t stage_bytes(int32_t variant, int32_t d) {
  if (!valid_head_size(d)) return PTH_LOADER_ERR_ARGS;
  const bool cell = variant == PTH_LOADER_VARIANT_CELL;
  if (!cell && variant != PTH_LOADER_VARIANT_SEPARATE)
    return PTH_LOADER_ERR_ARGS;
  switch (d) {
    case 64:
      return cell ? CellStageLayout<64, pth_attn::kLoadTileTokens>::kBytes
                  : StageLayout<64, pth_attn::kLoadTileTokens>::kBytes;
    case 128:
      return cell ? CellStageLayout<128, pth_attn::kLoadTileTokens>::kBytes
                  : StageLayout<128, pth_attn::kLoadTileTokens>::kBytes;
    default:
      return cell ? CellStageLayout<256, pth_attn::kLoadTileTokens>::kBytes
                  : StageLayout<256, pth_attn::kLoadTileTokens>::kBytes;
  }
}

int32_t cell_width_for(int32_t d, const KernelParams& p) {
  switch (d) {
    case 64:
      return pth_attn::cell_layout_width<64>(p);
    case 128:
      return pth_attn::cell_layout_width<128>(p);
    default:
      return pth_attn::cell_layout_width<256>(p);
  }
}

// Fills the four StreamGeometry entries (host-picked widths) from the views.
int32_t make_streams(int32_t d, const void* k, const void* v, const void* ks,
                     const void* vs, const int64_t* sizes3,
                     const int64_t* const st[4], KernelParams& p) {
  if (k == nullptr || v == nullptr || ks == nullptr || vs == nullptr ||
      sizes3 == nullptr) {
    return PTH_LOADER_ERR_ARGS;
  }
  for (int i = 0; i < kSizes; ++i) {
    if (sizes3[i] <= 0 || st[0] == nullptr || st[1] == nullptr ||
        st[2] == nullptr || st[3] == nullptr) {
      return PTH_LOADER_ERR_ARGS;
    }
  }
  const void* data[pth_attn::kNumStreams] = {k, v, ks, vs};
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    const bool scale = s >= pth_attn::kStreamKScale;
    p.stream[s] = pth_attn::make_stream(data[s], sizes3, st[s],
                                        scale ? kScaleElemBytes : kKvElemBytes,
                                        scale ? pth_attn::kScaleBytes : d);
  }
  return PTH_LOADER_OK;
}

// Picked widths, or the checked override. For the cell variant, entry 0.
int32_t apply_widths(int32_t variant, int32_t d, const int32_t* override4,
                     KernelParams& p, int32_t& cell_width) {
  cell_width = 0;
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    if (p.stream[s].width == 0) return PTH_LOADER_ERR_WIDTH;
  }
  if (variant == PTH_LOADER_VARIANT_CELL) {
    cell_width = cell_width_for(d, p);
    if (cell_width == 0) return PTH_LOADER_ERR_NOT_CELL;
    if (override4 != nullptr) {
      const int32_t w = override4[0];
      if ((w != pth_attn::kGeomWidth4 && w != pth_attn::kGeomWidth8) ||
          w > cell_width) {
        return PTH_LOADER_ERR_WIDTH;
      }
      cell_width = w;
    }
    return PTH_LOADER_OK;
  }
  if (override4 == nullptr) return PTH_LOADER_OK;
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    const int32_t w = override4[s];
    const bool legal = w == pth_attn::kGeomWidth4 ||
                       w == pth_attn::kGeomWidth8 ||
                       w == pth_attn::kGeomWidth16;
    // Powers of two: any legal w <= the picked width divides everything the
    // picked width divides.
    if (!legal || w > p.stream[s].width) return PTH_LOADER_ERR_WIDTH;
    p.stream[s].width = w;
  }
  return PTH_LOADER_OK;
}

struct Request {
  KernelParams p;
  int32_t cell_width;
  int32_t stage;
};

int32_t prepare(int32_t variant, int32_t d, const void* k, const void* v,
                const void* ks, const void* vs, const int64_t* sizes3,
                const int64_t* k_st, const int64_t* v_st, const int64_t* ks_st,
                const int64_t* vs_st, const int32_t* override4,
                const int32_t* block_table, int64_t bt_row_stride,
                int32_t block_size, int32_t req, int32_t kv_head,
                int32_t tile_begin, int32_t tile_end, int32_t tok_begin,
                int32_t tok_end, const void* out, int64_t out_bytes,
                Request& r) {
  std::memset(&r, 0, sizeof(r));
  r.stage = stage_bytes(variant, d);
  if (r.stage < 0) return r.stage;
  const int64_t* const st[4] = {k_st, v_st, ks_st, vs_st};
  int32_t rc = make_streams(d, k, v, ks, vs, sizes3, st, r.p);
  if (rc != PTH_LOADER_OK) return rc;
  if (block_table == nullptr || out == nullptr || bt_row_stride < 0 ||
      block_size != sizes3[1] || req < 0 || kv_head < 0 ||
      kv_head >= sizes3[2] || tile_begin < 0 || tile_end < tile_begin ||
      tok_end < 0 || tok_begin < 0 || tok_begin > tok_end ||
      (tok_begin > 0 && variant != PTH_LOADER_VARIANT_SEPARATE)) {
    return PTH_LOADER_ERR_ARGS;
  }
  rc = apply_widths(variant, d, override4, r.p, r.cell_width);
  if (rc != PTH_LOADER_OK) return rc;
  if (out_bytes < static_cast<int64_t>(tile_end - tile_begin) * r.stage) {
    return PTH_LOADER_ERR_OUT_SIZE;
  }
  r.p.block_table = block_table;
  r.p.bt_row_stride = bt_row_stride;
  r.p.block_size =
      pth_attn::FastDivmod::make(static_cast<uint32_t>(block_size));
  r.p.head_size = d;
  return PTH_LOADER_OK;
}

template <int D>
void emulate_tiles(const Request& r, bool cell, int32_t req, int32_t kv_head,
                   int32_t tile_begin, int32_t tile_end, int32_t tok_begin,
                   int32_t tok_end, uint8_t* out) {
  for (int32_t tile = tile_begin; tile < tile_end; ++tile) {
    uint8_t* img = out + static_cast<int64_t>(tile - tile_begin) * r.stage;
    std::memset(img, 0,
                static_cast<size_t>(r.stage));  // as the kernel's zeroing
    pth_attn::emulate_tile<D>(r.p, cell, r.cell_width, img, req, kv_head, tile,
                              tok_end, tok_begin);
  }
}

// Lane maps, from the loader's own index functions: rows of kRow bytes
// (D for K/V, 2D + 8 for the whole cell) in chunks of W bytes.
template <int kRow, int W>
int32_t map_rows(int32_t stream, int32_t* out, int64_t cap) {
  int64_t n = 0;
  for (int warp = 0; warp < pth_attn::kWarps; ++warp) {
    for (int it = 0; it < pth_attn::stream_iters<kRow, W>(); ++it) {
      for (int lane = 0; lane < pth_attn::kWarpSize; ++lane) {
        const pth_attn::LoaderChunk c =
            pth_attn::stream_chunk<kRow, W>(it, lane);
        if (!c.active) continue;
        if (n >= cap) return PTH_LOADER_ERR_OUT_SIZE;
        const int32_t e[kMapFields] = {
            warp, lane, stream, pth_attn::kLoaderTokensPerWarp * warp + c.token,
            c.byte};
        std::memcpy(out + n * kMapFields, e, sizeof(e));
        ++n;
      }
    }
  }
  return static_cast<int32_t>(n);
}

template <int D>
int32_t map_d(int32_t which, int32_t width, int32_t* out, int64_t cap) {
  constexpr int kCellBytes =
      CellStageLayout<D, pth_attn::kLoadTileTokens>::kCellBytes;
  if (which == PTH_LOADER_MAP_CELL) {
    // 16 B never divides a 2D + 8 cell, so the cell has widths 8 and 4 only.
    switch (width) {
      case pth_attn::kGeomWidth8:
        return map_rows<kCellBytes, pth_attn::kGeomWidth8>(pth_attn::kStreamK,
                                                           out, cap);
      case pth_attn::kGeomWidth4:
        return map_rows<kCellBytes, pth_attn::kGeomWidth4>(pth_attn::kStreamK,
                                                           out, cap);
      default:
        return PTH_LOADER_ERR_ARGS;
    }
  }
  const int32_t stream =
      which == PTH_LOADER_MAP_V ? pth_attn::kStreamV : pth_attn::kStreamK;
  switch (width) {
    case pth_attn::kGeomWidth16:
      return map_rows<D, pth_attn::kGeomWidth16>(stream, out, cap);
    case pth_attn::kGeomWidth8:
      return map_rows<D, pth_attn::kGeomWidth8>(stream, out, cap);
    case pth_attn::kGeomWidth4:
      return map_rows<D, pth_attn::kGeomWidth4>(stream, out, cap);
    default:
      return PTH_LOADER_ERR_ARGS;
  }
}

}  // namespace

extern "C" {

int32_t pth_loader_abi_version(void) { return PTH_LOADER_ABI_VERSION; }

int32_t pth_loader_stage_bytes(int32_t variant, int32_t head_size) {
  return stage_bytes(variant, head_size);
}

int32_t pth_loader_widths(int32_t variant, int32_t head_size, const void* k,
                          const void* v, const void* ks, const void* vs,
                          const int64_t* sizes3, const int64_t* k_strides3,
                          const int64_t* v_strides3, const int64_t* ks_strides3,
                          const int64_t* vs_strides3, int32_t* widths_out4) {
  if (stage_bytes(variant, head_size) < 0 || widths_out4 == nullptr) {
    return PTH_LOADER_ERR_ARGS;
  }
  KernelParams p;
  std::memset(&p, 0, sizeof(p));
  const int64_t* const st[4] = {k_strides3, v_strides3, ks_strides3,
                                vs_strides3};
  int32_t rc = make_streams(head_size, k, v, ks, vs, sizes3, st, p);
  if (rc != PTH_LOADER_OK) return rc;
  int32_t cell_width = 0;
  rc = apply_widths(variant, head_size, nullptr, p, cell_width);
  if (rc != PTH_LOADER_OK) return rc;
  for (int s = 0; s < pth_attn::kNumStreams; ++s) {
    widths_out4[s] = variant == PTH_LOADER_VARIANT_CELL
                         ? (s == 0 ? cell_width : 0)
                         : p.stream[s].width;
  }
  return PTH_LOADER_OK;
}

int32_t pth_loader_lane_map(int32_t which, int32_t head_size, int32_t width,
                            int32_t* out, int64_t cap_entries) {
  if (out == nullptr || cap_entries < 0 || !valid_head_size(head_size)) {
    return PTH_LOADER_ERR_ARGS;
  }
  if (which == PTH_LOADER_MAP_SCALES) {
    int64_t n = 0;
    for (int warp = 0; warp < pth_attn::kWarps; ++warp) {
      for (int lane = 0; lane < pth_attn::kWarpSize; ++lane) {
        if (n >= cap_entries) return PTH_LOADER_ERR_OUT_SIZE;
        const int32_t e[kMapFields] = {warp, lane, pth_attn::scale_stream(lane),
                                       pth_attn::lane_token(warp, lane), 0};
        std::memcpy(out + n * kMapFields, e, sizeof(e));
        ++n;
      }
    }
    return static_cast<int32_t>(n);
  }
  if (which != PTH_LOADER_MAP_K && which != PTH_LOADER_MAP_V &&
      which != PTH_LOADER_MAP_CELL) {
    return PTH_LOADER_ERR_ARGS;
  }
  switch (head_size) {
    case 64:
      return map_d<64>(which, width, out, cap_entries);
    case 128:
      return map_d<128>(which, width, out, cap_entries);
    default:
      return map_d<256>(which, width, out, cap_entries);
  }
}

int32_t pth_loader_emulate(
    int32_t variant, int32_t head_size, const void* k, const void* v,
    const void* ks, const void* vs, const int64_t* sizes3,
    const int64_t* k_strides3, const int64_t* v_strides3,
    const int64_t* ks_strides3, const int64_t* vs_strides3,
    const int32_t* widths_override4, const int32_t* block_table,
    int64_t bt_row_stride, int32_t block_size, int32_t req, int32_t kv_head,
    int32_t tile_begin, int32_t tile_end, int32_t tok_begin, int32_t tok_end,
    void* out, int64_t out_bytes) {
  Request r;
  const int32_t rc =
      prepare(variant, head_size, k, v, ks, vs, sizes3, k_strides3, v_strides3,
              ks_strides3, vs_strides3, widths_override4, block_table,
              bt_row_stride, block_size, req, kv_head, tile_begin, tile_end,
              tok_begin, tok_end, out, out_bytes, r);
  if (rc != PTH_LOADER_OK) return rc;
  const bool cell = variant == PTH_LOADER_VARIANT_CELL;
  auto* o = static_cast<uint8_t*>(out);
  switch (head_size) {
    case 64:
      emulate_tiles<64>(r, cell, req, kv_head, tile_begin, tile_end, tok_begin,
                        tok_end, o);
      break;
    case 128:
      emulate_tiles<128>(r, cell, req, kv_head, tile_begin, tile_end, tok_begin,
                         tok_end, o);
      break;
    default:
      emulate_tiles<256>(r, cell, req, kv_head, tile_begin, tile_end, tok_begin,
                         tok_end, o);
      break;
  }
  return PTH_LOADER_OK;
}

int32_t pth_loader_run(int32_t variant, int32_t head_size, const void* k,
                       const void* v, const void* ks, const void* vs,
                       const int64_t* sizes3, const int64_t* k_strides3,
                       const int64_t* v_strides3, const int64_t* ks_strides3,
                       const int64_t* vs_strides3,
                       const int32_t* widths_override4,
                       const int32_t* block_table, int64_t bt_row_stride,
                       int32_t block_size, int32_t req, int32_t kv_head,
                       int32_t tile_begin, int32_t tile_end, int32_t tok_begin,
                       int32_t tok_end, void* out, int64_t out_bytes,
                       void* cuda_stream) {
  Request r;
  const int32_t rc =
      prepare(variant, head_size, k, v, ks, vs, sizes3, k_strides3, v_strides3,
              ks_strides3, vs_strides3, widths_override4, block_table,
              bt_row_stride, block_size, req, kv_head, tile_begin, tile_end,
              tok_begin, tok_end, out, out_bytes, r);
  if (rc != PTH_LOADER_OK) return rc;
#if defined(__CUDACC__)
  pth_attn::loader_test::LaunchArgs a;
  a.p = r.p;
  a.whole_cell = variant == PTH_LOADER_VARIANT_CELL ? 1 : 0;
  a.head_size = head_size;
  a.cell_width = r.cell_width;
  a.req = req;
  a.kv_head = kv_head;
  a.tile_begin = tile_begin;
  a.tile_end = tile_end;
  a.tok_begin = tok_begin;
  a.tok_end = tok_end;
  a.out = out;
  if (tile_end == tile_begin) return PTH_LOADER_OK;
  return pth_attn::loader_test::launch(a, cuda_stream) == 0
             ? PTH_LOADER_OK
             : PTH_LOADER_ERR_CUDA;
#else
  (void)cuda_stream;
  return PTH_LOADER_ERR_NO_CUDA;
#endif
}

}  // extern "C"
