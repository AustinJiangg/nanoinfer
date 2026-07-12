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
#include <memory>
#include <utility>
#include <vector>

#include "quant.hpp"  // Weight — the base DenseWeight (below) and the quant weights share
#include "tensor.hpp"

namespace ni {

class KVCacheBase;  // cache.hpp — make_kv_cache returns one (forward-declared to avoid an include cycle)

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

    // --- ragged-batch (speculative verify, S3b) plumbing: forward_spec_batch verifies N
    // sequences at once, sequence s contributing count_s query rows (k_s+1). Unlike the
    // decode case above, a sequence's attention block is MULTI-query, so these move a
    // CONTIGUOUS block of `count` rows [count, width] out of / into a [M, width] batched
    // tensor — a plain slice (rows are contiguous), NO transpose: split_heads/merge_heads
    // do the head transpose, exactly as forward() does per sequence. count==1 recovers
    // extract_row/place_row (which specialize to a single-row reshape). ---
    virtual Tensor extract_rows(const Tensor& x, int64_t row_start, int64_t count) = 0;
    virtual void place_rows(Tensor& dst, int64_t row_start, const Tensor& block) = 0;

    // --- R1: device objects the factory owns, so Model stays #ifdef-free ---
    // The KV cache native to this backend (CPU KVCache / device CudaKVCache), returned through
    // the base so the model drives either via one pointer. max_seq sizes the preallocated CPU
    // cache; the device cache grows on demand and ignores it.
    virtual std::unique_ptr<KVCacheBase> make_kv_cache(int64_t num_layers, int64_t n_kv_heads,
                                                       int64_t head_dim, int64_t max_seq) = 0;
    // Bring a forward()'s logits to the host for return. CPU: identity (already host). CUDA: D2H,
    // unless a graph driver kept them on device for its own post-replay copy. The result-landing
    // #ifdef that used to sit at the end of Model::forward now lives here.
    virtual Tensor finalize_logits(Tensor logits) { return logits; }

    // R3: make a host weight resident on this backend's device, once at load. CPU: identity (the
    // weight stays a host tensor). CUDA: H2D upload — as fp16/bf16 when half_eligible AND the
    // backend's matching half-storage mode is on (the big projections/embed), else fp32. The model
    // calls this in a plain loop over its norms/biases/embed, with no #ifdef and no device global
    // in sight — the storage decision (fp16_weights/bf16_weights config) lives in the CUDA override.
    virtual Tensor to_resident(Tensor weight, bool /*half_eligible*/) { return weight; }

    // R3c: the backend as weight factory, so the model builds quantized weights with no #ifdef.
    // make_quant_weight: a quantized projection (Q8/Q4/Q4G/W8A8) from a host fp32 tensor — CPU keeps
    // it host (make_quantized); the CUDA override uploads W8A8 as a device int8 weight (make_cuda_w8a8).
    // make_embed_weight: the tied embed/lm_head as weight-only int8 — CPU EmbedQ8Weight, CUDA device.
    // The defaults are the CPU construction; CudaBackend overrides both. The model names no cuda symbol.
    virtual std::unique_ptr<Weight> make_quant_weight(const Tensor& host, QuantMode mode) {
        return make_quantized(host, mode);
    }
    virtual std::unique_ptr<Weight> make_embed_weight(const Tensor& host) {
        return make_q8_embed(host);
    }
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
    Tensor extract_rows(const Tensor& x, int64_t row_start, int64_t count) override;
    void place_rows(Tensor& dst, int64_t row_start, const Tensor& block) override;
    std::unique_ptr<KVCacheBase> make_kv_cache(int64_t num_layers, int64_t n_kv_heads,
                                               int64_t head_dim, int64_t max_seq) override;
};

// R3: a plain fp32/fp16/bf16 weight exposed as a Weight, so Model::project drives dense and
// quantized projections through one pointer with no branch. It forwards to the backend's GEMM over
// its (already device-resident) tensor; the half-storage cases are handled inside the backend's
// linear dispatch (dtype()==F16/BF16), so DenseWeight is device- and precision-agnostic. The quant
// weights (quant.cpp, cuda W8A8) are the other Weight impls. (gather() for the embedding: R3b.)
class DenseWeight : public Weight {
public:
    DenseWeight(Backend* backend, Tensor w) : backend_(backend), w_(std::move(w)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override {
        return backend_->linear(x, w_, bias);
    }
    Tensor gather(const std::vector<int64_t>& ids) const override {  // R3b: the embed table case
        return backend_->embedding(w_, ids);
    }
    // R5: the dtype is the only thing a dense weight's format depends on (the device is orthogonal).
    Format format() const override {
        return w_.dtype() == DType::F16    ? Format::F16
               : w_.dtype() == DType::BF16 ? Format::BF16
                                           : Format::F32;
    }
    int64_t bytes() const override {
        return w_.numel() *
               ((w_.dtype() == DType::F16 || w_.dtype() == DType::BF16) ? 2 : 4);
    }
    int64_t fp32_bytes() const override { return w_.numel() * 4; }

private:
    Backend* backend_;
    Tensor w_;
};

// Construction-time backend/model configuration — the typed, per-instance home for what were
// load-time globals (the de-globalization the R1/R2 comments anticipated). fp16_weights: the CUDA
// backend uploads the big eligible weights (layer projections + the tied embed/lm_head) as fp16, half
// the DRAM bytes (G5d) — read in CudaBackend::to_resident. bf16_weights: the same upload as bf16 (B1)
// — the same byte win, but byte-exact to the bf16 the checkpoints ship (fp32's exponent range, no
// overflow risk; the trade is a coarser mantissa: 8 bits vs fp16's 11). Exclusive with fp16_weights
// (make_backend rejects both). quantize_embed: weight-only int8 for the tied token-embedding /
// lm_head (G5d), read in the Model ctor (CPU + CUDA). Threaded Model ctor -> make_backend ->
// CudaBackend. (The kernel-selection CudaPolicy folds in next; CUDA-graph state stays per-call.)
struct BackendConfig {
    bool fp16_weights = false;
    bool bf16_weights = false;
    bool quantize_embed = false;
};

// The single place that maps a Device to its concrete backend — the former #ifdef ladder in
// Model's constructor, now isolated here. Throws if that device's backend wasn't compiled in
// (e.g. Device::CUDA without -DNI_CUDA).
std::unique_ptr<Backend> make_backend(Device device, const BackendConfig& cfg = {});

}  // namespace ni
