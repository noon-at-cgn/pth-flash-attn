/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
 *
 * loader_capi.h: test-only C ABI of the tile loader (csrc/pth_attn/loader.cuh),
 * for ctypes tests. Two libraries are built from loader_capi.cu:
 *
 *   build/loader/libpth_loader.so       nvcc, sm_80/86/89, with the GPU test
 *                                       kernel (tools/build_loader_lib.sh)
 *   libpth_loader_host.so               g++ -x c++: everything except
 *                                       pth_loader_run (which then returns
 *                                       PTH_LOADER_ERR_NO_CUDA)
 *
 * Arguments shared by pth_loader_run and pth_loader_emulate:
 *   variant          PTH_LOADER_VARIANT_SEPARATE or PTH_LOADER_VARIANT_CELL
 *   head_size        D: 64, 128 or 256
 *   k, v, ks, vs     data_ptr() of the op's views k_cache, v_cache
 *                    [NB, BS, Hkv, D] int8 and k_scale, v_scale [NB, BS, Hkv]
 *                    f32 (device memory for run, host memory for emulate)
 *   sizes3           host int64[3] = [NB, BS, Hkv], shared by the four views
 *   *_strides3       host int64[3]: each view's stride()[:3] in ELEMENTS
 *                    (converted to bytes here: x1 for K/V, x4 for scales)
 *   widths_override4 host int32[4] or NULL. NULL: the host-picked widths
 *                    (K, V, ks, vs; for the cell variant entry 0 is the cell
 *                    width). Otherwise each entry must be a legal width for
 *                    its stream (in {4, 8, 16}, at most the picked width;
 *                    scales 4), else PTH_LOADER_ERR_WIDTH.
 *   block_table      int32 [num_reqs, *] kernel-block ids, row stride
 *                    bt_row_stride elements (device for run, host for
 *                    emulate); block_size must equal sizes3[1]
 *   req, kv_head     which request and KV head
 *   tile_begin/end   KV tiles [tile_begin, tile_end) of 64 tokens each
 *   tok_begin        tokens at positions < tok_begin are zero-filled too
 *                    (the windowed kernels' range begin; 0 = the
 *                    causal loader; > 0 needs the separate variant and
 *                    tok_begin <= tok_end, else PTH_LOADER_ERR_ARGS)
 *   tok_end          tokens at positions >= tok_end are zero-filled (the
 *                    CTA's KV range end; must not exceed the request's
 *                    block_table coverage)
 *   out, out_bytes   (tile_end - tile_begin) stage images back to back,
 *                    pth_loader_stage_bytes() each (device for run, host for
 *                    emulate). Separate: StageLayout<D, 64> (K, V swizzled,
 *                    then ks, vs). Cell: CellStageLayout<D, 64>
 *                    (pitch 2D + 16, padding 0). Unwritten bytes are 0.
 *   cuda_stream      run only: a cudaStream_t, or NULL for the default
 *                    stream. The launch is asynchronous; synchronise before
 *                    reading `out`.
 * Return: 0, or one of the negative PTH_LOADER_ERR_* codes.
 */
#ifndef PTH_ATTN_LOADER_CAPI_H_
#define PTH_ATTN_LOADER_CAPI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PTH_LOADER_ABI_VERSION 2

#define PTH_LOADER_VARIANT_SEPARATE 0
#define PTH_LOADER_VARIANT_CELL 1

#define PTH_LOADER_OK 0
#define PTH_LOADER_ERR_ARGS (-1)
#define PTH_LOADER_ERR_WIDTH (-2)
#define PTH_LOADER_ERR_NOT_CELL (-3)
#define PTH_LOADER_ERR_OUT_SIZE (-4)
#define PTH_LOADER_ERR_CUDA (-5)
#define PTH_LOADER_ERR_NO_CUDA (-6)

/* pth_loader_lane_map `which` */
#define PTH_LOADER_MAP_K 0
#define PTH_LOADER_MAP_V 1
#define PTH_LOADER_MAP_SCALES 2
#define PTH_LOADER_MAP_CELL 3

int32_t pth_loader_abi_version(void);

/* Bytes of one stage image, or PTH_LOADER_ERR_ARGS. */
int32_t pth_loader_stage_bytes(int32_t variant, int32_t head_size);

/* The widths the host picks for these views (host-only, reads no memory):
 * widths_out4 = [K, V, ks, vs] (separate) or [cell, 0, 0, 0] (cell). */
int32_t pth_loader_widths(int32_t variant, int32_t head_size, const void* k,
                          const void* v, const void* ks, const void* vs,
                          const int64_t* sizes3, const int64_t* k_strides3,
                          const int64_t* v_strides3, const int64_t* ks_strides3,
                          const int64_t* vs_strides3, int32_t* widths_out4);

/* The loader's lane -> (token, byte) map for one stream and width, in
 * instruction order (warp, instruction, lane); 5 int32 per active lane:
 * warp, lane, stream id (0 K, 1 V, 2 k_scale, 3 v_scale; 0 for the cell),
 * token within the 64-token tile, byte within the row (or cell). `width` is
 * ignored for SCALES (always 4). Returns the number of entries written, or
 * PTH_LOADER_ERR_ARGS / PTH_LOADER_ERR_OUT_SIZE if cap_entries is short. */
int32_t pth_loader_lane_map(int32_t which, int32_t head_size, int32_t width,
                            int32_t* out, int64_t cap_entries);

/* Host emulation of the loader (host pointers everywhere). */
int32_t pth_loader_emulate(
    int32_t variant, int32_t head_size, const void* k, const void* v,
    const void* ks, const void* vs, const int64_t* sizes3,
    const int64_t* k_strides3, const int64_t* v_strides3,
    const int64_t* ks_strides3, const int64_t* vs_strides3,
    const int32_t* widths_override4, const int32_t* block_table,
    int64_t bt_row_stride, int32_t block_size, int32_t req, int32_t kv_head,
    int32_t tile_begin, int32_t tile_end, int32_t tok_begin, int32_t tok_end,
    void* out, int64_t out_bytes);

/* The GPU loader through the real two-stage pipeline (device pointers). */
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
                       void* cuda_stream);

#ifdef __cplusplus
}
#endif

#endif /* PTH_ATTN_LOADER_CAPI_H_ */
