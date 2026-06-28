// Host<->device transfer + runtime probe for the CUDA backend (G1).
//
// Pure C++ declarations (no CUDA types in the signatures), so code compiled by the
// host compiler — tests, model.cpp — can call these; the definitions live in
// cuda_backend.cu, compiled by nvcc. A Tensor returned by to_device() has
// device()==CUDA and its data in GPU memory; to_host() copies it back to a CPU Tensor.
#pragma once

#include "tensor.hpp"

namespace ni {

// True if at least one CUDA device is visible at runtime (lets GPU tests skip
// gracefully on a CPU-only machine instead of crashing).
bool cuda_available();

// Copy a CPU tensor up to the GPU (H2D). Returns a device tensor of the same shape.
Tensor to_device(const Tensor& host);

// Copy a CPU tensor to the GPU AND convert it to fp16 (a half device buffer). Used once at
// load for the layer weights when the backend's fp16_weights config is set — halves their DRAM bytes (G5d).
// The returned tensor has device()==CUDA and dtype()==F16.
Tensor to_device_f16(const Tensor& host);

// Copy a device tensor back down to the CPU (D2H). Returns a host tensor.
Tensor to_host(const Tensor& dev);

// Upload int8 codes (a QTensor's q) to a device I8 tensor of `shape` (G5d W8A8 weights). One H2D.
Tensor to_device_i8(const int8_t* host, const std::vector<int64_t>& shape);

// W8A8 GEMM on the GPU (G5d): x fp32 [m,k] on device; wq int8 [n,k] + w_scale fp32 [n] on device;
// bias fp32 [n] or null. Quantizes x per-row to int8 on device, runs the int8×int8→int32 DP4A GEMM,
// dequantizes with both scales -> fp32 y [m,n] on device. The integer core matches the CPU
// linear_w8a8 oracle exactly; only the float dequant drifts (the usual accelerator tolerance).
// Requires k % 16 == 0 (the DP4A tile's K step); m and n may be ragged.
Tensor cuda_linear_w8a8(const Tensor& x, const Tensor& wq, const Tensor& w_scale, const Tensor* bias);

// Weight-only int8 embedding gather on the GPU (G5d): codes int8 [vocab, hidden] + scale fp32 [vocab]
// on device, ids on host. out[r,:] = float(codes[ids[r],:]) * scale[ids[r]] -> fp32 [n, hidden] on
// device. The GPU mirror of the CPU embedding_q8 oracle (exact dequant; only the gather is on device).
Tensor cuda_embedding_q8(const Tensor& codes, const Tensor& scale, const std::vector<int64_t>& ids);

// Weight-only int8 linear on the GPU (G5d): x fp32 [m,k]; codes int8 [n,k] + scale fp32 [n]; bias or
// null. y[i,o] = (sum_j x[i,j]*codes[o,j]) * scale[o] + bias[o], fp32 accumulate (x is NOT quantized,
// unlike cuda_linear_w8a8). Matches the CPU linear_q8 oracle within accelerator tolerance — the int8
// codes are identical, only the fp32 reduction order drifts. The tied lm_head reads the same
// codes+scale as the gather above.
Tensor cuda_linear_q8(const Tensor& x, const Tensor& codes, const Tensor& scale, const Tensor* bias);

}  // namespace ni
