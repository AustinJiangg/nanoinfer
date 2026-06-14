// Core ops for stage C0 — the C++ counterparts of what nanoinfer does with torch.
//
// All operate on contiguous Tensors and return fresh contiguous Tensors. The
// implementations are deliberately the naive, readable version (e.g. matmul is a
// triple loop); speed comes later (SIMD/threads in stage C5). Correctness is
// pinned by parity tests against numpy / nanoinfer.
#pragma once

#include "tensor.hpp"

namespace ni {

// 2-D matrix multiply: a [m, k] x b [k, n] -> [m, n].
Tensor matmul(const Tensor& a, const Tensor& b);

// RMSNorm over the last dimension: x / sqrt(mean(x^2) + eps) * weight.
// x is [..., d] (treated as rows of length d); weight is [d]. Matches
// nanoinfer.layers.RMSNorm (computed in float32).
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over the last dimension, max-subtracted for numerical stability.
Tensor softmax(const Tensor& x);

// Elementwise add of two equally-shaped tensors.
Tensor add(const Tensor& a, const Tensor& b);

}  // namespace ni
