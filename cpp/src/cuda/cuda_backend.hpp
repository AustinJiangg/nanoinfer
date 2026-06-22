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
#include <memory>
#include <vector>

#include "backend.hpp"
#include "cache.hpp"
#include "quant.hpp"

namespace ni {

// G5d: build a device-resident W8A8 weight — quantize `w` to int8 (per-channel, like Q8) and upload
// the codes + per-row scales to the GPU once. Its linear() runs cuda_linear_w8a8 (int8×int8 DP4A on
// device). The model holds these in qweights_ for the CUDA + W8A8 path, so Model::project drives the
// int8 compute through the same QuantizedWeight interface as the CPU quant modes (no forward change).
std::unique_ptr<QuantizedWeight> make_cuda_w8a8(const Tensor& w);

// Bench/diagnostic knob (G5b): force the naive one-thread-per-output GEMM even for small
// m, so run_cuda_decode_bench can A/B the warp-GEMV's decode win in one process with
// everything else held constant. Left false in all normal use; not thread-safe to flip.
extern bool g_cuda_force_naive_gemm;

// Opt-in (G5d): run the prefill GEMM on the tensor cores (fp16 inputs, fp32 accumulate).
// Default off — fp16 is lossy, so it stays opt-in until the accuracy/speed tradeoff is
// accepted; flip it to measure (test_cuda, run_cuda_bench NI_WMMA=1). Not thread-safe.
extern bool g_cuda_use_wmma;

// Opt-in (G5d): upload the big weights as fp16 (half the DRAM bytes) — the layer projections
// plus the token embedding / tied lm_head (embed_tokens, the single largest weight). Set it
// BEFORE constructing a CUDA Model — the conversion happens at the once-per-load upload; the
// linear dispatch then routes fp16 weights through the GEMV/wmma fp16 paths (so fp16 weights
// force the tensor-core kernel for prefill), and the embedding gather reads the fp16 table
// directly. Default off; not thread-safe.
extern bool g_cuda_fp16_weights;

// Bench/diagnostic knob (G5e): force the naive one-thread-per-query attention kernel instead of
// the warp-per-query kernel, so a bench can A/B the attention speedup. Default false; not thread-safe.
extern bool g_cuda_force_naive_attn;

// Opt-in knob (G5f): use the shared-memory K/V tiled attention kernel at prefill (sq>1) instead of
// the non-tiled online kernel. Default false — tiling only ties on this model (KV fits in L2); it's
// the FlashAttention structure for when K/V outgrow L2. Bit-identical output either way.
extern bool g_cuda_use_tiled_attn;

// Opt-in knob (G5g — Flash-Decoding): split the KV across multiple blocks per (head,query), so decode
// attention fills the GPU instead of launching only H*sq warps (~3% of it at sq=1, each then walking
// the whole KV serially). Each split runs the one-pass online softmax over its key chunk and writes a
// partial (m,l,acc); a combine kernel merges the partials. Default false; engages only when the shape
// warrants it (small sq, long context — see split_count in the .cu) and otherwise degrades to the
// non-split warp kernel. Reorders the per-key reduction, so NOT bit-identical to the non-split kernel,
// but within GPU tolerance of the CPU oracle. The long-context decode lever G5f left open.
extern bool g_cuda_use_split_attn;

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

// A pool of fixed-size physical KV blocks on the GPU, shared across sequences (G4b).
// Mirrors the CPU BlockPool's allocator (a free list + refcounts — pure host-side
// bookkeeping), but stores K/V in one big device buffer each (laid out
// [num_layers, num_blocks, n_kv_heads, block_size, head_dim]); the paged-attention
// kernel indexes that buffer by the block table. K/V never leave the device.
class CudaBlockPool {
public:
    CudaBlockPool(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim, int64_t block_size,
                  int64_t num_blocks);
    CudaBlockPool(const CudaBlockPool&) = delete;
    CudaBlockPool& operator=(const CudaBlockPool&) = delete;
    CudaBlockPool(CudaBlockPool&&) = default;
    CudaBlockPool& operator=(CudaBlockPool&&) = default;

    int64_t allocate();          // a free block id (refcount 1); throws when exhausted
    void incref(int64_t block);  // add a holder (prefix sharing)
    void free(int64_t block);    // drop a holder; recycle at refcount 0

    int64_t num_blocks() const { return num_blocks_; }
    int64_t block_size() const { return block_size_; }
    int64_t free_blocks() const { return static_cast<int64_t>(free_list_.size()); }
    int64_t used_blocks() const { return num_blocks_ - free_blocks(); }
    int64_t n_kv_heads() const { return n_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t refcount(int64_t block) const { return refcount_[static_cast<size_t>(block)]; }

    // Device buffers + strides for the kernels (a layer's slice is base + layer*layer_stride).
    float* k_base() const { return static_cast<float*>(dK_.get()); }
    float* v_base() const { return static_cast<float*>(dV_.get()); }
    int64_t layer_stride() const { return num_blocks_ * block_stride_; }
    int64_t block_stride() const { return block_stride_; }

private:
    int64_t num_layers_, n_kv_heads_, head_dim_, block_size_, num_blocks_, block_stride_;
    std::shared_ptr<void> dK_, dV_;  // device K/V buffers, RAII-freed (cudaFree deleter)
    std::vector<int64_t> free_list_, refcount_;
};

// One sequence's paged KV cache on the GPU (G4b): a block table into a CudaBlockPool,
// frees its blocks back on destruction. attend() writes the new K/V into the sequence's
// blocks and attends directly over them via the paged-attention kernel — no gather, no
// repeat_kv (GQA folded into the read). Bit-identical to the contiguous CudaKVCache.
class CudaPagedKVCache : public KVCacheBase {
public:
    explicit CudaPagedKVCache(CudaBlockPool* pool);
    ~CudaPagedKVCache() override;
    CudaPagedKVCache(const CudaPagedKVCache&) = delete;
    CudaPagedKVCache& operator=(const CudaPagedKVCache&) = delete;

    Tensor attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v, int64_t n_rep,
                  bool causal, int64_t query_offset) override;
    void advance(int64_t t) override { length_ += t; }
    int64_t length() const override { return length_; }
    int64_t num_blocks() const { return static_cast<int64_t>(block_table_.size()); }
    const std::vector<int64_t>& block_table() const { return block_table_; }
    // Prefix sharing (RadixAttention): seed a fresh cache with shared prefix blocks
    // (increfing each); the sequence then prefills only its suffix past `length`.
    void share_prefix(const std::vector<int64_t>& blocks, int64_t length);

private:
    void ensure_capacity(int64_t positions);  // grow the block table to cover positions
    CudaBlockPool* pool_;
    std::vector<int64_t> block_table_;  // logical block i -> physical block id
    int64_t length_ = 0;
};

}  // namespace ni
