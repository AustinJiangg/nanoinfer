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
// load for the layer weights when g_cuda_fp16_weights is set — halves their DRAM bytes (G5d).
// The returned tensor has device()==CUDA and dtype()==F16.
Tensor to_device_f16(const Tensor& host);

// Copy a device tensor back down to the CPU (D2H). Returns a host tensor.
Tensor to_host(const Tensor& dev);

}  // namespace ni
