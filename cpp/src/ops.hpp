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

// Affine projection, nn.Linear convention: x [m, in], weight [out, in] (row per
// output feature), optional bias [out] -> [m, out], i.e. y = x @ weight^T + bias.
// This is the layout HF stores q/k/v/o/gate/up/down/lm_head in, so we never
// transpose the weight at load time.
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias = nullptr);

// Embedding lookup: table [vocab, hidden], ids -> [len(ids), hidden]. ids are a
// plain integer vector (token ids), not a float Tensor.
Tensor embedding(const Tensor& table, const std::vector<int64_t>& ids);

// SiLU / swish activation, elementwise: x * sigmoid(x).
Tensor silu(const Tensor& x);

// Elementwise multiply of two equally-shaped tensors (the SwiGLU gate).
Tensor mul(const Tensor& a, const Tensor& b);

// RMSNorm over the last dimension: x / sqrt(mean(x^2) + eps) * weight.
// x is [..., d] (treated as rows of length d); weight is [d]. Matches
// nanoinfer.layers.RMSNorm (computed in float32).
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over the last dimension, max-subtracted for numerical stability.
Tensor softmax(const Tensor& x);

// Elementwise add of two equally-shaped tensors.
Tensor add(const Tensor& a, const Tensor& b);

}  // namespace ni
