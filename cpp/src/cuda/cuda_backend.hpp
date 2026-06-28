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
// int8 compute through the same Weight interface as the CPU quant modes (no forward change).
std::unique_ptr<Weight> make_cuda_w8a8(const Tensor& w);

// R2: the kernel-selection policy — the bench/diagnostic and opt-in knobs that pick WHICH CUDA
// kernel a dispatch launches, consolidated from seven scattered g_cuda_* globals into one cohesive,
// typed object. Reached via cuda_policy() (still a file-scope instance: the readers include the
// cuda_linear_q8 free function and CudaPagedKVCache::attend, which a per-instance backend member
// can't reach until R3 threads the policy through the weight seam). Mutable on purpose — the A/B
// harness flips a field mid-run on a shared model (run_cuda_bench toggles use_dbuf to measure it);
// R2 retypes that capability, it does not remove it. Not thread-safe to flip during a launch.
struct CudaPolicy {
    // GEMM (CudaBackend::linear). force_naive_gemm: the naive one-thread-per-output kernel, the A/B
    // baseline (G5b). use_wmma: the tensor-core prefill kernel (fp16 in, fp32 accumulate; G5d, lossy,
    // opt-in). use_dbuf: the double-buffered tiled projection kernel — bit-identical to the default
    // tiled kernel (only load timing moves), an A/B knob not a correctness one (G5 micro-gain).
    bool force_naive_gemm = false;
    bool use_wmma = false;
    bool use_dbuf = false;
    // int8 lm_head (cuda_linear_q8): force the prefill-tuned tiled kernel even at decode m, so a bench
    // can A/B the int8 decode GEMV's win on the quantized lm_head (G5d). Left false in normal use.
    bool force_tiled_q8 = false;
    // Attention (CudaBackend::attention + CudaPagedKVCache::attend). force_naive_attn: the naive
    // one-thread-per-query kernel, the A/B baseline (G5e). use_tiled_attn: the shared-mem K/V tiled
    // kernel at prefill (sq>1) — bit-identical, only ties on this model (KV fits in L2), the
    // FlashAttention structure for when K/V outgrow L2 (G5f). use_split_attn: Flash-Decoding split-KV
    // when the shape warrants it (small sq, long context) — reorders the reduction (not bit-identical
    // to non-split, within GPU tolerance), degrades to the warp kernel otherwise (G5g).
    bool force_naive_attn = false;
    bool use_tiled_attn = false;
    bool use_split_attn = false;
};

// The process-wide kernel-selection policy (R2). Returns a mutable reference: A/B harnesses set
// cuda_policy().use_dbuf = true etc. instead of the former loose g_cuda_* globals.
CudaPolicy& cuda_policy();

// Opt-in (G5d): upload the big weights as fp16 (half the DRAM bytes) — the layer projections
// plus the token embedding / tied lm_head (embed_tokens, the single largest weight). Set it
// BEFORE constructing a CUDA Model — the conversion happens at the once-per-load upload; the
// linear dispatch then routes fp16 weights through the GEMV/wmma fp16 paths (so fp16 weights
// force the tensor-core kernel for prefill), and the embedding gather reads the fp16 table
// directly. Default off; not thread-safe. (Load-config, not kernel selection — folds into the
// R3 weight seam alongside g_quantize_embed.)
extern bool g_cuda_fp16_weights;

// G6 (CUDA graphs): when true, Model::forward leaves the logits ON DEVICE (skips the final D2H) so a
// graph driver can capture the forward and do its own D2H after replay — a sync D2H can't be captured.
// Set only by CudaGraphDecoder around a capture; default false (eager D2Hs at the edge as before).
// (Per-call graph state, not policy — folds into R3's per-call context with g_cuda_graph_pos/token.)
extern bool g_cuda_keep_device_logits;

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
    // R1: the device-resident KV cache (grows by concat, so max_seq is unused) and the result
    // D2H (respecting the graph driver's keep-on-device flag), behind the Backend so the model
    // needs no #ifdef for either. Bodies in cuda_backend.cu.
    std::unique_ptr<KVCacheBase> make_kv_cache(int64_t num_layers, int64_t n_kv_heads,
                                               int64_t head_dim, int64_t max_seq) override;
    Tensor finalize_logits(Tensor logits) override;
    // R3: H2D upload at load (fp16 for the big eligible weights under g_cuda_fp16_weights, else fp32).
    Tensor to_resident(Tensor weight, bool fp16_eligible) override;
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

    // G6 (CUDA graphs): the host-side per-step bookkeeping a graph driver must run OUTSIDE a
    // capture — allocate any block the next position needs and refresh the device block table.
    // attend() also calls it (eager); it's idempotent across layers and a NO-OP (no CUDA call)
    // when nothing grew, so it's safe to leave in the captured region. `end` = length after this
    // step (length()+t). The device block table d_block_table_ keeps a STABLE address so the
    // captured paged kernels read it by pointer while its contents update between replays.
    void prepare(int64_t end);
    const int64_t* device_block_table() const { return static_cast<const int64_t*>(d_block_table_.get()); }
    // G6: undo the length advance the capture pass's forward() does — a graph capture RECORDS the
    // decode forward (host bookkeeping runs, kernels don't), so its advance() must be rolled back
    // before the real replays drive the length forward themselves.
    void set_length(int64_t l) { length_ = l; }

private:
    void ensure_capacity(int64_t positions);  // grow the block table to cover positions
    void sync_device_block_table();           // upload appended entries to d_block_table_
    CudaBlockPool* pool_;
    std::vector<int64_t> block_table_;     // logical block i -> physical block id
    std::shared_ptr<void> d_block_table_;  // device int64 copy of block_table_ (stable address)
    int64_t d_bt_count_ = 0;               // entries already uploaded to the device buffer
    int64_t length_ = 0;
};

class Model;  // full def in model.hpp; the .cu includes it, the host-compiled header only points

// G6 (CUDA graphs): drive single-sequence DECODE through a captured CUDA graph — one cudaGraphLaunch
// replaces a decode step's ~360 per-op kernel launches (the launch overhead that, on this small model,
// keeps decode ~3.5x over its memory-bandwidth floor). Requires a PAGED cache: the contiguous cat_seq
// cache can't be captured (sync memcpy + growing buffers), while paged blocks have fixed addresses and
// the cache feeds the step's length/token through DEVICE buffers the graph reads by pointer — so ONE
// capture spans every length (decode keeps sq=1, so all kernel grids are fixed; only the loop bounds /
// positions move, and those are now device-side). Prefill and the first decode step run EAGER (warming
// the device pool so the capture itself does no cudaMalloc); the step is captured once and replayed
// thereafter. Output is bit-identical to eager paged decode — golden tokens hold. Handles are void* so
// the header stays CUDA-type-free (it's compiled by the host compiler too).
class CudaGraphDecoder {
public:
    CudaGraphDecoder(const Model& model, CudaPagedKVCache& cache, int64_t vocab);
    ~CudaGraphDecoder();
    CudaGraphDecoder(const CudaGraphDecoder&) = delete;
    CudaGraphDecoder& operator=(const CudaGraphDecoder&) = delete;

    // Decode `token` at the cache's current length; returns [1, vocab] host logits, identical to
    // model.forward({token}, &cache). Warms on the 1st call, captures on the 2nd, replays after.
    Tensor decode(int64_t token);

private:
    void capture(int64_t token);
    const Model& model_;
    CudaPagedKVCache& cache_;
    int64_t vocab_;
    int64_t* d_pos_ = nullptr;         // device int64: KV length for this step
    int64_t* d_token_ = nullptr;       // device int64: decode token id
    const float* d_logits_ = nullptr;  // device logits buffer baked into the captured graph
    Tensor captured_logits_;           // keeps that pool buffer alive across replays
    void* graph_ = nullptr;            // cudaGraph_t
    void* exec_ = nullptr;             // cudaGraphExec_t
    bool warmed_ = false;
    bool captured_ = false;
};

}  // namespace ni
