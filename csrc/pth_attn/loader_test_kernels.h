// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// loader_test_kernels.h: the test-only launch interface between the C API
// (loader_capi.cu, which validates arguments and builds KernelParams) and the
// test kernel (loader_test_kernels.cu). Not part of the op; never included by
// the attention kernel.

#pragma once

#include <cstdint>

#include "cache_geometry.cuh"

namespace pth_attn {
namespace loader_test {

// One test launch: load KV tiles [tile_begin, tile_end) of (req, kv_head)
// through the real two-stage pipeline and write every stage image to `out`
// (device memory, (tile_end - tile_begin) * stage bytes).
struct LaunchArgs {
  KernelParams p;      // streams, block_table and block size filled in
  int32_t whole_cell;  // 0: separate streams, 1: whole cell
  int32_t head_size;   // 64, 128 or 256
  int32_t cell_width;  // whole cell only: 8 or 4
  int32_t req;
  int32_t kv_head;
  int32_t tile_begin;
  int32_t tile_end;
  int32_t tok_begin;  // > 0: zero-fill below it too (separate streams only)
  int32_t tok_end;
  void* out;
};

// Launches one CTA of 128 threads on `stream` (a cudaStream_t, or nullptr
// for the default stream). Returns 0, or -1 if a CUDA call failed. Defined in
// loader_test_kernels.cu (nvcc builds only).
int32_t launch(const LaunchArgs& a, void* stream);

}  // namespace loader_test
}  // namespace pth_attn
