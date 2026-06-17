// Backend: the device-dispatch seam (C++ stage G0).
//
// Model::forward / forward_batch are written ONCE against this interface, so the
// model is device-agnostic. CpuBackend wraps the existing free ops in ops.cpp with
// zero math change — CPU output stays bit-identical to the pre-Backend engine, which
// is exactly why it can serve as the parity oracle for the accelerator backends.
// A CudaBackend (G1+) and MetalBackend (cross-platform) override the same methods to
// run on device; tensors carry their own Device tag (see tensor.hpp), and a backend
// operates on tensors resident on its device.
//
// Scope note (G0): only the ops Model actually calls are here. Quantized projections
// keep their own CPU path in Model::project for now (GPU quant is post-G5), and the
// KV cache keeps calling the free ops until its Backend injection in G3.
#pragma once

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace ni {

class Backend {
public:
    virtual ~Backend() = default;
    virtual Device device() const = 0;

    virtual Tensor embedding(const Tensor& table, const std::vector<int64_t>& ids) = 0;
    virtual Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) = 0;
    virtual Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) = 0;
    virtual Tensor silu(const Tensor& x) = 0;
    virtual Tensor mul(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor add(const Tensor& a, const Tensor& b) = 0;
    virtual Tensor split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) = 0;
    virtual Tensor merge_heads(const Tensor& x) = 0;
    virtual Tensor repeat_kv(const Tensor& x, int64_t n_rep) = 0;
    virtual Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                              int64_t pos_offset) = 0;
    virtual Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                             int64_t query_offset) = 0;

    // --- batched decode (F8a / G4) plumbing: forward_batch builds its per-sequence
    // attention output one row at a time, so these move a row between the [n, heads*dim]
    // batched layout and the [heads, 1, dim] per-token layout, on the backend's device. ---
    // Allocate an (uninitialized) tensor on this backend's device; the caller fills it.
    virtual Tensor alloc(const std::vector<int64_t>& shape) = 0;
    // Row s of a [n, heads*dim] projection as a [heads, 1, dim] per-head tensor.
    virtual Tensor extract_row(const Tensor& x, int64_t s, int64_t heads, int64_t dim) = 0;
    // Write a merged-head [1, width] row back into row s of dst [n, width], in place.
    virtual void place_row(Tensor& dst, int64_t s, const Tensor& row) = 0;
};

// The CPU backend: every method forwards to the corresponding free function in
// ops.cpp (which keeps the SIMD/OpenMP kernels). No math changes here, so the CPU
// path is bit-for-bit what it was before the Backend seam existed.
class CpuBackend : public Backend {
public:
    Device device() const override { return Device::CPU; }

    Tensor embedding(const Tensor& table, const std::vector<int64_t>& ids) override;
    Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) override;
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

}  // namespace ni
