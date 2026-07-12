// Shared internals for the CUDA backend translation units (R4 split).
//
// cuda_backend.cu was a single 2300-line .cu; it is being carved into focused TUs (cuda_runtime.cu,
// and the R4b kernel files) for navigability. They share this device-memory pool + grid math and the
// dtype-folding device loaders. The host helpers are DECLARED here and DEFINED once in cuda_runtime.cu
// (so a SINGLE DevicePool serves every TU — steady-state forwards do no per-op cudaMalloc); the
// __device__ loaders are header-inline (each kernel TU compiles its own copy, the usual CUDA pattern).
//
// nvcc-only: CUDA types appear in the signatures, so a host-compiled TU must never include this.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace ni {

// Throw a std::runtime_error on a non-success CUDA status (`what` labels the call site).
void cuda_check(cudaError_t e, const char* what);
// The Tensor's type-erased device buffer as a float* (the backend owns this knowledge).
float* dptr(const Tensor& t);
// A device tensor of `shape` (+ dtype) from the shared pool, with a deleter that returns the buffer to
// the pool (no per-op cudaMalloc/cudaFree). Default F32. See cuda_runtime.cu for the pool rationale.
Tensor device_alloc(const std::vector<int64_t>& shape, DType dt = DType::F32);
// Concatenate two [n_kv, *, head_dim] tensors along the seq axis into a fresh device buffer — the
// contiguous KV cache's per-token append.
Tensor cat_seq(const Tensor& a, const Tensor& b);
// Round a flat element count up to whole `block`-sized 1-D grid blocks.
int grid1d(int64_t total, int block);
// SM count of the current device (queried once) — Flash-Decoding sizes its split count from it.
int sm_count();
// How many KV splits Flash-Decoding should use for nq = H*sq query pairs over key length sk (1 = none).
int split_count(int64_t nq, int64_t sk);

// Post-launch error check (cheap, immediate). The per-op device sync was dropped in G5 — kernels run
// ordered on the one default stream, so correctness needs no per-op sync (results are read only through
// to_host's synchronizing D2H). Header-inline so every kernel TU shares it; `what` labels the launch.
inline void launch_check(const char* what) { cuda_check(cudaGetLastError(), what); }

// G6 (CUDA graphs): the per-step DECODE inputs, made device-resident so ONE captured graph spans all
// steps (nullptr = eager, the kernels use their host-int args — bit-identical). Defined in
// cuda_backend.cu; read by rope/embedding (cuda_elementwise.cu) and the paged attention kernels.
// g_cuda_graph_pos: the KV length / write position; g_cuda_graph_token: the decode token id the
// embedding gathers from instead of a host id-upload (a sync H2D, illegal inside a capture).
extern const int64_t* g_cuda_graph_pos;
extern const int64_t* g_cuda_graph_token;

constexpr int kBlock = 256;
// The m at/below which linear() is memory-bound and runs the warp-per-output GEMV (decode) instead of
// the compute-bound tiled GEMM (prefill). Shared by the fp32/fp16 linear() and the int8 cuda_linear_q8.
constexpr int64_t kGemvMaxM = 16;

// Dtype-folding weight loaders (G5d fp16; B1 bf16): one templated kernel serves fp32, fp16, and
// bf16 weights. ldf -> float (the fp32-accumulate GEMV/tiled paths), from_f32<ST> -> the wmma
// staging dtype (half / __nv_bfloat16 — B2 templates the tensor-core kernels on it), load4 -> a
// float4 of 4 consecutive weights (a half dtype reads 8 bytes + converts, so one float4 register
// path stages any dtype).
__device__ inline float ldf(const float* w, size_t i) { return w[i]; }
__device__ inline float ldf(const half* w, size_t i) { return __half2float(w[i]); }
__device__ inline float ldf(const __nv_bfloat16* w, size_t i) { return __bfloat162float(w[i]); }
template <typename ST>
__device__ inline ST from_f32(float v);
template <>
__device__ inline half from_f32<half>(float v) { return __float2half(v); }
template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
__device__ inline half ldh(const float* w, size_t i) { return __float2half(w[i]); }
__device__ inline half ldh(const half* w, size_t i) { return w[i]; }
__device__ inline float4 load4(const float* w, size_t i) {
    return *reinterpret_cast<const float4*>(w + i);
}
__device__ inline float4 load4(const half* w, size_t i) {
    const half2* h = reinterpret_cast<const half2*>(w + i);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}
__device__ inline float4 load4(const __nv_bfloat16* w, size_t i) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(w + i);
    const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}

}  // namespace ni
