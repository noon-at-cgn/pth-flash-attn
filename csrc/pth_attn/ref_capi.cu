// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// ref_capi.cu: the C ABI of ref_capi.h (test builds only).
//
// This translation unit includes reference_kernel.cu (a unity build), so one
// file builds both libraries of tools/build_ref_lib.sh:
//   nvcc -shared ... ref_capi.cu    -> build/ref/libpth_ref.so (GPU + host)
//   g++ -x c++ -shared ref_capi.cu  -> build/ref/libpth_ref_host.so (host)
// The host-only build has no kernels; its CUDA entry points return
// PTH_ATTN_ENOCUDA.
//
// Responsibilities: check the arguments (every check names the field), turn
// the C structs into KernelParams/RefParams (byte strides copied as given,
// divisors built here), and launch. Nothing is allocated and nothing is read
// back from the device.

#include "ref_capi.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>

#include "reference_kernel.cu"

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
#endif

namespace {

using pth_attn::FastDivmod;
using pth_attn::KernelParams;
using pth_attn::kNumStreams;
using pth_attn::RefParams;
using pth_attn::StreamGeometry;

thread_local std::string g_last_error;

int32_t fail(int32_t code, const std::string& msg) {
  g_last_error = msg;
  return code;
}

#define REF_REQUIRE(cond, msg)                        \
  do {                                                \
    if (!(cond)) return fail(PTH_ATTN_EINVAL, (msg)); \
  } while (0)

bool is_head_size(int32_t d) { return d == 64 || d == 128 || d == 256; }

// Calls f(std::integral_constant<int32_t, D/32>) for the supported D; the
// kernels take D/32 (columns per lane) as a template parameter.
template <class F>
void dispatch_cols(int32_t head_size, F&& f) {
  constexpr int32_t lanes = pth_attn::kWarpLanes;
  switch (head_size) {
    case 64:
      f(std::integral_constant<int32_t, 64 / lanes>{});
      break;
    case 128:
      f(std::integral_constant<int32_t, 128 / lanes>{});
      break;
    default:
      f(std::integral_constant<int32_t, 256 / lanes>{});
      break;
  }
}

StreamGeometry stream_from(const PthAttnStream& s, int32_t row_bytes) {
  StreamGeometry g{};
  g.base = static_cast<const char*>(s.data);
  g.s_block = s.stride_block;
  g.s_token = s.stride_slot;
  g.s_head = s.stride_head;
  g.width = 0;  // the reference does not use cp.async
  g.row_bytes = row_bytes;
  return g;
}

// Checks of the argument struct that need no data; fills `rp` and the grid's
// x extent.
int32_t build_attn(const PthAttnArgs* a, RefParams* rp, int32_t* num_q_tiles) {
  REF_REQUIRE(a != nullptr, "args is NULL");
  REF_REQUIRE(a->abi_version == PTH_ATTN_ABI_VERSION,
              "abi_version " + std::to_string(a->abi_version) +
                  " != library ABI " + std::to_string(PTH_ATTN_ABI_VERSION));
  REF_REQUIRE(a->struct_bytes == static_cast<int32_t>(sizeof(PthAttnArgs)),
              "struct_bytes " + std::to_string(a->struct_bytes) +
                  " != sizeof(PthAttnArgs) " +
                  std::to_string(sizeof(PthAttnArgs)));
  REF_REQUIRE(a->q_dtype == PTH_ATTN_F16 || a->q_dtype == PTH_ATTN_BF16,
              "q_dtype must be PTH_ATTN_F16 or PTH_ATTN_BF16");
  REF_REQUIRE(a->out_dtype == PTH_ATTN_F32 || a->out_dtype == PTH_ATTN_F16 ||
                  a->out_dtype == PTH_ATTN_BF16,
              "out_dtype must be PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16");
  REF_REQUIRE(is_head_size(a->head_size), "head_size must be 64, 128 or 256");
  REF_REQUIRE(a->num_kv_heads >= 1 && a->num_q_heads >= a->num_kv_heads &&
                  a->num_q_heads % a->num_kv_heads == 0,
              "num_q_heads must be a positive multiple of num_kv_heads");
  REF_REQUIRE(a->block_size >= 1, "block_size must be >= 1");
  REF_REQUIRE(a->num_blocks >= 1, "num_blocks must be >= 1");
  REF_REQUIRE(a->num_reqs >= 1, "num_reqs must be >= 1");
  REF_REQUIRE(a->num_tokens >= 1, "num_tokens must be >= 1");
  REF_REQUIRE(a->bq >= 1, "bq must be >= 1");
  REF_REQUIRE(a->num_splits >= 1, "num_splits must be >= 1");
  REF_REQUIRE(a->num_q_tiles >= 0, "num_q_tiles must be >= 0");
  REF_REQUIRE(std::isfinite(a->softmax_scale) && a->softmax_scale > 0.0f,
              "softmax_scale must be finite and > 0");
  REF_REQUIRE(a->window >= 0, "window must be >= 0 (0 = none)");
  REF_REQUIRE(a->causal == 0 || a->causal == 1, "causal must be 0 or 1");
  REF_REQUIRE(a->q && a->k_cache.data && a->v_cache.data && a->k_scale.data &&
                  a->v_scale.data && a->block_table && a->seq_lens &&
                  a->query_start_loc,
              "q, the four cache streams, block_table, seq_lens and "
              "query_start_loc must be non-NULL");
  if (a->num_splits == 1) {
    REF_REQUIRE(a->out != nullptr, "out must be non-NULL when num_splits == 1");
  } else {
    REF_REQUIRE(a->ws_o != nullptr && a->ws_lse != nullptr,
                "ws_o and ws_lse must be non-NULL when num_splits > 1");
  }

  KernelParams& k = rp->k;
  k = KernelParams{};
  k.q = a->q;
  k.out = a->out;
  k.s_q_token = a->q_stride_token;
  k.s_q_head = a->q_stride_head;
  k.s_out_token = a->out_stride_token;
  k.s_out_head = a->out_stride_head;
  k.ws_o = a->ws_o;
  k.ws_lse = a->ws_lse;
  k.s_wso_split = a->ws_o_stride_split;
  k.s_wso_token = a->ws_o_stride_token;
  k.s_wso_head = a->ws_o_stride_head;
  k.s_wsl_split = a->ws_lse_stride_split;
  k.s_wsl_token = a->ws_lse_stride_token;
  k.s_wsl_head = a->ws_lse_stride_head;
  k.stream[pth_attn::kStreamK] = stream_from(a->k_cache, a->head_size);
  k.stream[pth_attn::kStreamV] = stream_from(a->v_cache, a->head_size);
  k.stream[pth_attn::kStreamKScale] =
      stream_from(a->k_scale, static_cast<int32_t>(sizeof(float)));
  k.stream[pth_attn::kStreamVScale] =
      stream_from(a->v_scale, static_cast<int32_t>(sizeof(float)));
  k.block_table = a->block_table;
  k.bt_row_stride = a->block_table_row_stride;
  k.seq_lens = a->seq_lens;
  k.query_start_loc = a->query_start_loc;
  const int32_t group = a->num_q_heads / a->num_kv_heads;
  k.block_size = FastDivmod::make(static_cast<uint32_t>(a->block_size));
  k.group = FastDivmod::make(static_cast<uint32_t>(group));
  k.bq_div = FastDivmod::make(static_cast<uint32_t>(a->bq));
  k.num_reqs = a->num_reqs;
  k.num_tokens = a->num_tokens;
  k.num_q_heads = a->num_q_heads;
  k.num_kv_heads = a->num_kv_heads;
  k.group_size = group;
  k.head_size = a->head_size;
  k.wm = 0;  // not used by the reference (it takes bq directly)
  k.bq = a->bq;
  k.num_splits = a->num_splits;
  k.max_query_len = 0;  // not used: q_len is read on the device
  k.softmax_scale = a->softmax_scale;
  k.score_scale_log2 = a->softmax_scale * pth_attn::kRefLog2e;
  rp->q_dtype = a->q_dtype;
  rp->out_dtype = a->out_dtype;
  rp->window = a->window;
  rp->causal = a->causal != 0;
  *num_q_tiles =
      a->num_q_tiles > 0 ? a->num_q_tiles : a->num_tokens / a->bq + a->num_reqs;
  return PTH_ATTN_OK;
}

int32_t build_combine(const PthAttnCombineArgs* a, pth_attn::PartialsView* v) {
  REF_REQUIRE(a != nullptr, "args is NULL");
  REF_REQUIRE(a->abi_version == PTH_ATTN_ABI_VERSION,
              "abi_version " + std::to_string(a->abi_version) +
                  " != library ABI " + std::to_string(PTH_ATTN_ABI_VERSION));
  REF_REQUIRE(
      a->struct_bytes == static_cast<int32_t>(sizeof(PthAttnCombineArgs)),
      "struct_bytes " + std::to_string(a->struct_bytes) +
          " != sizeof(PthAttnCombineArgs) " +
          std::to_string(sizeof(PthAttnCombineArgs)));
  REF_REQUIRE(a->out_dtype == PTH_ATTN_F32 || a->out_dtype == PTH_ATTN_F16 ||
                  a->out_dtype == PTH_ATTN_BF16,
              "out_dtype must be PTH_ATTN_F32, PTH_ATTN_F16 or PTH_ATTN_BF16");
  REF_REQUIRE(is_head_size(a->head_size), "head_size must be 64, 128 or 256");
  REF_REQUIRE(a->num_splits >= 1 && a->num_reqs >= 1 && a->num_tokens >= 1 &&
                  a->num_q_heads >= 1,
              "num_splits, num_reqs, num_tokens, num_q_heads must be >= 1");
  REF_REQUIRE(a->ws_o && a->ws_lse && a->out && a->query_start_loc,
              "ws_o, ws_lse, out and query_start_loc must be non-NULL");
  v->ws_o = reinterpret_cast<const char*>(a->ws_o);
  v->s_o_split = a->ws_o_stride_split;
  v->s_o_token = a->ws_o_stride_token;
  v->s_o_head = a->ws_o_stride_head;
  v->ws_lse = reinterpret_cast<const char*>(a->ws_lse);
  v->s_l_split = a->ws_lse_stride_split;
  v->s_l_token = a->ws_lse_stride_token;
  v->s_l_head = a->ws_lse_stride_head;
  return PTH_ATTN_OK;
}

// Host entry only: the int32 data are host-readable, so check what the kernel
// trusts (the device entry cannot without a sync).
int32_t check_host_data(const PthAttnArgs* a) {
  const int32_t* qsl = a->query_start_loc;
  REF_REQUIRE(qsl[0] == 0, "query_start_loc[0] must be 0");
  for (int32_t r = 0; r < a->num_reqs; ++r) {
    const int32_t q_len = qsl[r + 1] - qsl[r];
    REF_REQUIRE(q_len >= 0, "query_start_loc must be non-decreasing");
    REF_REQUIRE(a->seq_lens[r] >= q_len,
                "request " + std::to_string(r) + ": seq_len " +
                    std::to_string(a->seq_lens[r]) + " < q_len " +
                    std::to_string(q_len));
    if (q_len == 0) continue;
    const int32_t blocks = (a->seq_lens[r] + a->block_size - 1) / a->block_size;
    for (int32_t b = 0; b < blocks; ++b) {
      const int32_t id =
          a->block_table[static_cast<int64_t>(r) * a->block_table_row_stride +
                         b];
      REF_REQUIRE(id >= 0 && id < a->num_blocks,
                  "request " + std::to_string(r) + ": block id " +
                      std::to_string(id) + " outside [0, num_blocks)");
    }
  }
  REF_REQUIRE(qsl[a->num_reqs] <= a->num_tokens,
              "query_start_loc[num_reqs] > num_tokens");
  return PTH_ATTN_OK;
}

}  // namespace

extern "C" {

int32_t pth_ref_abi_version(void) { return PTH_ATTN_ABI_VERSION; }
int32_t pth_ref_attn_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnArgs));
}
int32_t pth_ref_combine_args_bytes(void) {
  return static_cast<int32_t>(sizeof(PthAttnCombineArgs));
}
const char* pth_ref_last_error(void) { return g_last_error.c_str(); }

#if defined(__CUDACC__)
int32_t pth_ref_has_cuda(void) { return 1; }

int32_t pth_ref_attention(const PthAttnArgs* args, void* stream) {
  g_last_error.clear();
  RefParams rp{};
  int32_t tiles = 0;
  const int32_t rc = build_attn(args, &rp, &tiles);
  if (rc != PTH_ATTN_OK) return rc;
  const dim3 grid(static_cast<unsigned>(tiles),
                  static_cast<unsigned>(args->num_kv_heads),
                  static_cast<unsigned>(args->num_splits));
  dispatch_cols(args->head_size, [&](auto cols) {
    pth_attn::ref_attention_kernel<decltype(cols)::value>
        <<<grid, pth_attn::kRefThreadsPerCta, 0,
           static_cast<cudaStream_t>(stream)>>>(rp);
  });
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return fail(PTH_ATTN_ECUDA, std::string("ref_attention_kernel launch: ") +
                                    cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}

int32_t pth_ref_combine(const PthAttnCombineArgs* args, void* stream) {
  g_last_error.clear();
  pth_attn::CombineParams c{};
  const int32_t rc = build_combine(args, &c.v);
  if (rc != PTH_ATTN_OK) return rc;
  c.out = static_cast<char*>(args->out);
  c.s_out_token = args->out_stride_token;
  c.s_out_head = args->out_stride_head;
  c.query_start_loc = args->query_start_loc;
  c.num_reqs = args->num_reqs;
  c.num_tokens = args->num_tokens;
  c.num_q_heads = args->num_q_heads;
  c.head_size = args->head_size;
  c.num_splits = args->num_splits;
  c.out_dtype = args->out_dtype;
  const int64_t rows =
      static_cast<int64_t>(args->num_tokens) * args->num_q_heads;
  const int64_t blocks =
      (rows + pth_attn::kRefWarpsPerCta - 1) / pth_attn::kRefWarpsPerCta;
  dispatch_cols(args->head_size, [&](auto cols) {
    pth_attn::ref_combine_kernel<decltype(cols)::value>
        <<<static_cast<unsigned>(blocks), pth_attn::kRefThreadsPerCta, 0,
           static_cast<cudaStream_t>(stream)>>>(c);
  });
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return fail(PTH_ATTN_ECUDA, std::string("ref_combine_kernel launch: ") +
                                    cudaGetErrorString(err));
  }
  return PTH_ATTN_OK;
}
#else
int32_t pth_ref_has_cuda(void) { return 0; }
int32_t pth_ref_attention(const PthAttnArgs*, void*) {
  return fail(PTH_ATTN_ENOCUDA, "host-only build: no CUDA kernels");
}
int32_t pth_ref_combine(const PthAttnCombineArgs*, void*) {
  return fail(PTH_ATTN_ENOCUDA, "host-only build: no CUDA kernels");
}
#endif

int32_t pth_ref_attention_host(const PthAttnArgs* args) {
  g_last_error.clear();
  RefParams rp{};
  int32_t tiles = 0;
  int32_t rc = build_attn(args, &rp, &tiles);
  if (rc != PTH_ATTN_OK) return rc;
  rc = check_host_data(args);
  if (rc != PTH_ATTN_OK) return rc;
  dispatch_cols(args->head_size, [&](auto cols) {
    pth_attn::ref_attention_host_run<decltype(cols)::value>(rp, tiles);
  });
  return PTH_ATTN_OK;
}

int32_t pth_ref_combine_host(const PthAttnCombineArgs* args) {
  g_last_error.clear();
  pth_attn::PartialsView v{};
  const int32_t rc = build_combine(args, &v);
  if (rc != PTH_ATTN_OK) return rc;
  dispatch_cols(args->head_size, [&](auto cols) {
    pth_attn::ref_combine_host_run<decltype(cols)::value>(
        v, static_cast<char*>(args->out), args->out_stride_token,
        args->out_stride_head, args->query_start_loc, args->num_reqs,
        args->num_tokens, args->num_q_heads, args->num_splits, args->out_dtype);
  });
  return PTH_ATTN_OK;
}

}  // extern "C"
