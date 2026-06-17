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

#include "backend.hpp"

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
};

}  // namespace ni
