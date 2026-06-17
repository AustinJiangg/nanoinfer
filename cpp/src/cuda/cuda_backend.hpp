// CudaBackend: the GPU implementation of the Backend interface (C++ stage G1+).
//
// Declared in a pure-C++ header (no CUDA types) on purpose: model.cpp is compiled by
// the host compiler, not nvcc, yet needs to construct a CudaBackend under -DNI_CUDA.
// The method bodies — and the kernels they launch — live in cuda_backend.cu.
//
// Contract: every Tensor passed in/out lives on the GPU (device()==CUDA). The caller
// (a test in G1; the Model in G2) is responsible for moving weights/activations to the
// device once and keeping them resident — H2D/D2H only at the edges, never per op.
//
// G1 implements linear() (a hand-written GEMM); the remaining ops throw until G2, so a
// premature full GPU forward fails loudly rather than silently producing garbage.
#pragma once

#include <cstdint>
#include <vector>

#include "backend.hpp"
#include "cache.hpp"

namespace ni {

class CudaBackend : public Backend {
public:
    Device device() const override;

    Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) override;

    // --- not yet on the GPU (G2) ---
    Tensor embedding(const Tensor& table, const std::vector<int64_t>& ids) override;
    Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) override;
    Tensor silu(const Tensor& x) override;
    Tensor mul(const Tensor& a, const Tensor& b) override;
    Tensor add(const Tensor& a, const Tensor& b) override;
    Tensor split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) override;
    Tensor merge_heads(const Tensor& x) override;
    Tensor repeat_kv(const Tensor& x, int64_t n_rep) override;
    Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                      int64_t pos_offset) override;
    Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                     int64_t query_offset) override;
    Tensor alloc(const std::vector<int64_t>& shape) override;
    Tensor extract_row(const Tensor& x, int64_t s, int64_t heads, int64_t dim) override;
    void place_row(Tensor& dst, int64_t s, const Tensor& row) override;
};

// Device-resident KV cache (G3). Each layer's K/V is a contiguous [n_kv, len, head_dim]
// tensor that lives on the GPU and grows by concatenation as tokens arrive; attend()
// appends the new K/V, then reuses the backend's repeat_kv + attention kernels over the
// whole history — semantically identical to the CPU KVCache's gather + attend. The CPU
// KVCache (cache.cpp) is left untouched (and bit-identical); this is its GPU sibling,
// the same KVCacheBase the forward already drives through one pointer.
class CudaKVCache : public KVCacheBase {
public:
    CudaKVCache(Backend* backend, int64_t num_layers, int64_t n_kv_heads, int64_t head_dim);
    Tensor attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                  int64_t n_rep, bool causal, int64_t query_offset) override;
    void advance(int64_t t) override;
    int64_t length() const override;

private:
    Backend* backend_;
    int64_t n_kv_heads_;
    int64_t head_dim_;
    std::vector<Tensor> k_, v_;  // per layer, growing [n_kv, len, head_dim] on the device
    int64_t length_ = 0;
};

}  // namespace ni
