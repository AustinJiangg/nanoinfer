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

// Optional RoPE frequency scaling resolved at build time (A2 Llama-3.2). When
// `enabled`, the base inv_freq is rescaled in three frequency bands before the
// cos/sin table is built — the rotation (apply_rope) is unchanged, the same
// "fix the frequencies at load, not the hot path" lesson as rope_theta. Kept a
// plain-double struct here so ops stays independent of config.hpp's enum; the
// model maps its Config's RopeScaling::Llama3 onto this. Defaults = disabled.
struct RopeScalingParams {
    bool enabled = false;
    double factor = 1.0;
    double low_freq_factor = 1.0;
    double high_freq_factor = 4.0;
    double orig_max_pos = 0.0;
};
RopeCache build_rope_cache(int64_t seq_len, int64_t head_dim, float theta,
                           RopeScalingParams scaling = {});

// Apply RoPE to a [heads, seq, head_dim] tensor (batch-1). cos/sin are
// [>=pos_offset+seq, head_dim]; token i in x is at absolute position
// pos_offset+i (pos_offset > 0 for cached decode, where the new token sits past
// the cached prefix). Returns a new tensor; q and k are rotated by separate calls.
Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                  int64_t pos_offset = 0);

// --- Attention building blocks (batch-1; per-head tensors are [heads, seq, head_dim]) ---

// [seq, n_heads*head_dim] -> [n_heads, seq, head_dim] (split the projection into heads).
Tensor split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim);

// [n_heads, seq, head_dim] -> [seq, n_heads*head_dim] (inverse of split_heads).
Tensor merge_heads(const Tensor& x);

// GQA: repeat each KV head n_rep times to match the query heads.
// [n_kv, seq, head_dim] -> [n_kv*n_rep, seq, head_dim]; KV head j feeds output
// heads j*n_rep .. (j+1)*n_rep-1 (see nanoinfer CLAUDE.md "GQA").
Tensor repeat_kv(const Tensor& x, int64_t n_rep);

// Scaled dot-product attention over [heads, seq, head_dim] q/k/v (k/v may be a
// different seq length than q). scale = 1/sqrt(head_dim). With causal=true, query
// i sits at absolute position query_offset+i and attends keys 0..(query_offset+i)
// — query_offset=0 is prefill; query_offset>0 is cached decode, where one new
// query attends the whole cached history.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                 int64_t query_offset = 0);

// RMSNorm over the last dimension: x / sqrt(mean(x^2) + eps) * weight.
// x is [..., d] (treated as rows of length d); weight is [d]. Matches
// nanoinfer.layers.RMSNorm (computed in float32).
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps);

// Softmax over the last dimension, max-subtracted for numerical stability.
Tensor softmax(const Tensor& x);

// Elementwise add of two equally-shaped tensors.
Tensor add(const Tensor& a, const Tensor& b);

}  // namespace ni
