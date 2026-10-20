// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// C ABI of the test library build/attn/libpth_attn.so
// (tools/build_attn_lib.sh), which the kernel test ladder drives through
// ctypes. Not part of the extension: the op calls the launcher (attn_launch.h)
// directly.
//
// pth_attn_attention and pth_attn_combine check the ABI fields of the structs
// of attn_args.h, then call launch_attention / launch_combine, so a test runs
// exactly the launcher the op runs. They take the reference library's
// arguments (ref_capi.h) unchanged: any test that drives the scalar reference
// (tests/gpu/grid_cases.py) drives the tensor-core kernels by swapping the
// library, and both libraries can be compared split for split. The
// tensor-core path has no host entry points (no emulator;
// python/pth_testing/emulate.py models its arithmetic on the CPU), and its
// error bound is bound.py's with the Tiling that pth_attn_tiling() reports.
//
// The fragment unit tests (pth_attn_test_*) are attn_test_kernels.cu.

#ifndef PTH_ATTN_ATTN_CAPI_H_
#define PTH_ATTN_ATTN_CAPI_H_

#include <stdint.h>

#include "attn_args.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loader choice of pth_attn_attention_ex.
typedef struct PthAttnOptions {
  int32_t abi_version;   // = PTH_ATTN_ABI_VERSION
  int32_t struct_bytes;  // = sizeof(PthAttnOptions) as compiled by the caller
  // KV loader: 0 = the shipped choice, 1 = separate streams, 2 = whole cell
  // (D = 256 with the inline cell layout only; EINVAL otherwise).
  int32_t loader;
} PthAttnOptions;

// One-warp fragment unit test (build ladder step 12): one warp stages
// q [16][D] (fp16), k and v [64][D] (int8) into the kernel's swizzled shared
// layout and runs the kernel's QK and PV fragment loops (tiling.cuh):
//   s_out[r][t] = sum_d q[r][d] * k[t][d]           (fp32, raw MMA result)
//   o_out[r][d] = sum_t p[r][t] * v[t][d]           (fp32)
// All arrays dense row-major on the device.
typedef struct PthAttnTileTestArgs {
  int32_t abi_version;   // = PTH_ATTN_ABI_VERSION
  int32_t struct_bytes;  // = sizeof(PthAttnTileTestArgs)
  int32_t head_size;     // 64, 128 or 256
  const void* q;         // fp16 [16][D]
  const void* k;         // int8 [64][D]
  const void* v;         // int8 [64][D]
  const void* p;         // fp16 [16][64]
  float* s_out;          // [16][64]
  float* o_out;          // [16][D]
} PthAttnTileTestArgs;

// One-CTA fragment unit test of the decode kernel (swap-AB, d-split;
// attn_decode.cuh), D = 256 only: four warps stage q [rows][256] (fp16, rows
// <= 32), k and v [64][256] (int8) into the kernel's shared layout and run
// the kernel's dec_qk / dec_pv loops (warp w: tokens 16 w .. 16 w + 15 of QK,
// d in [64 w, 64 w + 64) of PV; the P'' buffer is shared as in the kernel):
//   s_out[r][t] = sum_d q[r][d] * k[t][d]           (fp32, raw MMA result)
//   o_out[r][d] = sum_t p[r][t] * v[t][d]           (fp32)
// All arrays dense row-major on the device.
typedef struct PthAttnDecTileTestArgs {
  int32_t abi_version;   // = PTH_ATTN_ABI_VERSION
  int32_t struct_bytes;  // = sizeof(PthAttnDecTileTestArgs)
  int32_t head_size;     // 256
  int32_t rows;          // 1..32 query rows (n-tiles = ceil(rows / 8))
  const void* q;         // fp16 [rows][256]
  const void* k;         // int8 [64][256]
  const void* v;         // int8 [64][256]
  const void* p;         // fp16 [rows][64]
  float* s_out;          // [rows][64]
  float* o_out;          // [rows][256]
} PthAttnDecTileTestArgs;

int32_t pth_attn_abi_version(void);
int32_t pth_attn_attn_args_bytes(void);
int32_t pth_attn_combine_args_bytes(void);
int32_t pth_attn_options_bytes(void);
int32_t pth_attn_tile_test_args_bytes(void);
int32_t pth_attn_dec_tile_test_args_bytes(void);
int32_t pth_attn_has_cuda(void);  // always 1
const char* pth_attn_last_error(void);

// The Tiling of bound.py this kernel uses for (head_size, bq, G):
// *n_tile = 64, *sub_tile, *wn (1 for the decode kernel, else 4 / wm).
// Returns PTH_ATTN_OK or EINVAL.
int32_t pth_attn_tiling(int32_t head_size, int32_t bq, int32_t group,
                        int32_t* n_tile, int32_t* sub_tile, int32_t* wn);

// Device pointers, enqueue on `stream` (a cudaStream_t), no sync.
int32_t pth_attn_attention(const PthAttnArgs* args, void* stream);
int32_t pth_attn_attention_ex(const PthAttnArgs* args,
                              const PthAttnOptions* options, void* stream);
int32_t pth_attn_combine(const PthAttnCombineArgs* args, void* stream);
int32_t pth_attn_test_warp_tile(const PthAttnTileTestArgs* args, void* stream);
int32_t pth_attn_test_dec_tile(const PthAttnDecTileTestArgs* args,
                               void* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PTH_ATTN_ATTN_CAPI_H_
