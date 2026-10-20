// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_capi.cu: the C ABI of attn_capi.h over the launcher of attn_launch.h
// (test library only, tools/build_attn_lib.sh). It checks what only a C
// caller can get wrong (NULL args, the ABI version and struct sizes of a
// ctypes mirror, the options struct) and forwards; every other check is the
// launcher's, so a test sees the op's messages.

#include "attn_capi.h"

#include <cuda_runtime.h>

#include <string>

#include "attn_decode.cuh"
#include "attn_launch.h"

namespace {

template <class T>
bool abi_matches(const T& a) {
  return a.abi_version == PTH_ATTN_ABI_VERSION &&
         a.struct_bytes == static_cast<int32_t>(sizeof(T));
}

}  // namespace

extern "C" {

int32_t pth_attn_abi_version(void) { return PTH_ATTN_ABI_VERSION; }
int32_t pth_attn_attn_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnArgs));
}
int32_t pth_attn_combine_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnCombineArgs));
}
int32_t pth_attn_options_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnOptions));
}
int32_t pth_attn_has_cuda(void) { return 1; }
const char* pth_attn_last_error(void) { return pth_attn::last_launch_error(); }

int32_t pth_attn_tiling(int32_t head_size, int32_t bq, int32_t group,
                        int32_t* n_tile, int32_t* sub_tile, int32_t* wn) {
  pth_attn::clear_launch_error();
  PTH_ATTN_REQUIRE(pth_attn::is_head_size(head_size),
                   "head_size must be 64, 128 or 256");
  PTH_ATTN_REQUIRE(bq >= 1 && group >= 1, "bq and group must be >= 1");
  PTH_ATTN_REQUIRE(n_tile && sub_tile && wn,
                   "output pointers must be non-NULL");
  const int32_t wm = pth_attn::attn_wm_for_rows(bq * group);
  PTH_ATTN_REQUIRE(wm != 0, "bq * G exceeds 64 rows");
  *n_tile = pth_attn::kLoadTileTokens;
  if (head_size == 256 && bq * group <= pth_attn::kDecMaxRows) {
    // The decode kernel: one softmax step and one V-scale exponent per 64-token
    // tile, one warp's state (attn_decode.cuh).
    *sub_tile = pth_attn::kLoadTileTokens;
    *wn = 1;
    return PTH_ATTN_OK;
  }
  // The 16-row-tile kernel: its warp's token slice, limited per D (Traits).
  const int32_t slice = pth_attn::kLoadTileTokens * wm / pth_attn::kWarps;
  const int32_t cap = pth_attn::sub_tile_cap(head_size);
  *sub_tile = slice < cap ? slice : cap;
  *wn = pth_attn::kWarps / wm;
  return PTH_ATTN_OK;
}

int32_t pth_attn_attention_ex(const PthAttnArgs* args,
                              const PthAttnOptions* options, void* stream) {
  pth_attn::clear_launch_error();
  PTH_ATTN_REQUIRE(args != nullptr, "args is NULL");
  PTH_ATTN_REQUIRE(args->abi_version == PTH_ATTN_ABI_VERSION,
                   "abi_version " + std::to_string(args->abi_version) +
                       " != library ABI " +
                       std::to_string(PTH_ATTN_ABI_VERSION));
  PTH_ATTN_REQUIRE(
      args->struct_bytes == static_cast<int32_t>(sizeof(PthAttnArgs)),
      "struct_bytes " + std::to_string(args->struct_bytes) +
          " != sizeof(PthAttnArgs) " + std::to_string(sizeof(PthAttnArgs)));
  int32_t loader = pth_attn::kLoaderDefault;
  if (options != nullptr) {
    PTH_ATTN_REQUIRE(abi_matches(*options),
                     "options: abi_version/struct_bytes mismatch");
    loader = options->loader;
  }
  return pth_attn::launch_attention(*args, loader,
                                    static_cast<cudaStream_t>(stream));
}

int32_t pth_attn_attention(const PthAttnArgs* args, void* stream) {
  return pth_attn_attention_ex(args, nullptr, stream);
}

int32_t pth_attn_combine(const PthAttnCombineArgs* args, void* stream) {
  pth_attn::clear_launch_error();
  PTH_ATTN_REQUIRE(args != nullptr, "args is NULL");
  PTH_ATTN_REQUIRE(abi_matches(*args), "abi_version/struct_bytes mismatch");
  return pth_attn::launch_combine(*args, static_cast<cudaStream_t>(stream));
}

}  // extern "C"
