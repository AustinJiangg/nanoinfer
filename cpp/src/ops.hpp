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

// Precomputed rotary position embedding tables, each [seq_len, head_dim]. Each
// frequency is duplicated across the two halves of head_dim to match the
// neox / half-split rotation (the convention Qwen2/Llama use — NOT GPT-J
// interleave; see nanoinfer CLAUDE.md "RoPE rotation").
struct RopeCache {
    Tensor cos;
    Tensor sin;
};
RopeCache build_rope_cache(int64_t seq_len, int64_t head_dim, float theta);

// Apply RoPE to a [heads, seq, head_dim] tensor (batch-1). cos/sin are
// [>=seq, head_dim], aligned so row p is the rotation for position p. Returns a
// new tensor; q and k are rotated by separate calls.
Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin);

// RMSNorm over the last dimension: x / sqrt(mean(x^2) + eps) * weight.
// x is [..., d] (treated as rows of length d); weight is [d]. Matches
// nanoinfer.layers.RMSNorm (computed in float32).
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over the last dimension, max-subtracted for numerical stability.
Tensor softmax(const Tensor& x);

// Elementwise add of two equally-shaped tensors.
Tensor add(const Tensor& a, const Tensor& b);

}  // namespace ni
