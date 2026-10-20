// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// C ABI of the scalar reference kernel and its split combine (test builds
// only; not part of the extension).
//
// The reference (reference_kernel.cu) is the executable specification that
// the tensor-core kernels are tested against. Tests drive it through ctypes,
// on the GPU (libpth_ref.so, nvcc) and on the CPU (libpth_ref_host.so, g++,
// where the same kernel source runs under a lock-step warp emulator). Both
// libraries are built by tools/build_ref_lib.sh into build/ref/.
//
// It takes the production launch structs of attn_args.h unchanged, with
// their semantics, so any test drives the reference and the tensor-core
// kernels (attn_capi.h) with the same arguments and compares them split for
// split. Return value: PTH_ATTN_OK (0) or a negative PthAttnStatus; the
// message of the last failure on the calling thread is pth_ref_last_error().

#ifndef PTH_ATTN_REF_CAPI_H_
#define PTH_ATTN_REF_CAPI_H_

#include <stdint.h>

#include "attn_args.h"

#ifdef __cplusplus
extern "C" {
#endif

// Library identity (for ctypes mirrors).
int32_t pth_ref_abi_version(void);
int32_t pth_ref_attn_args_bytes(void);
int32_t pth_ref_combine_args_bytes(void);
// 1 if this library contains the CUDA kernels, 0 for the host-only build.
int32_t pth_ref_has_cuda(void);
// Message of the last failure on this thread ("" if none). Valid until the
// next call into this library on the same thread.
const char* pth_ref_last_error(void);

// CUDA entry points: device pointers, enqueue on `stream` (a cudaStream_t;
// NULL = legacy default stream), no sync.
int32_t pth_ref_attention(const PthAttnArgs* args, void* stream);
int32_t pth_ref_combine(const PthAttnCombineArgs* args, void* stream);

// Host entry points: host pointers; run the SAME kernel source on the CPU
// (warps emulated in lock step, see reference_kernel.cu). Synchronous.
int32_t pth_ref_attention_host(const PthAttnArgs* args);
int32_t pth_ref_combine_host(const PthAttnCombineArgs* args);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PTH_ATTN_REF_CAPI_H_
