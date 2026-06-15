// Paged KV cache (C++ stage F8b) — the vLLM idea, the merge point of the two tracks.
//
// K/V live in fixed-size blocks drawn from a shared BlockPool; each sequence keeps a
// block table mapping logical positions to physical blocks (logical position p is in
// block_table[p / block_size] at offset p % block_size). There is no per-sequence
// contiguous [max_seq] preallocation, so memory tracks actual lengths and a finished
// sequence returns its blocks to the pool for reuse — the high KV-utilization win.
//
// PagedKVCache implements the same KVCacheBase interface as the contiguous KVCache:
// attend() writes the new tokens into the sequence's blocks (allocating as it crosses
// a block boundary) and then attends directly over those blocks — the kernel indexes
// K/V by the block table, folding GQA into the read, so there is no gather into a
// contiguous buffer and no repeat_kv expansion. It mirrors the ops attention()
// arithmetic exactly (same dot/softmax order), so paged output is bit-identical to
// the contiguous cache (run_paged.cpp) while touching far less memory for the K/V.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "cache.hpp"
#include "tensor.hpp"

namespace ni {

// A pool of fixed-size physical KV blocks, shared across sequences. Per layer it
// owns [num_blocks, n_kv_heads, block_size, head_dim] of K and the same of V (flat,
// row-major); allocate()/free() hand block ids in and out of a free list.
class BlockPool {
public:
    BlockPool(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim,
              int64_t block_size, int64_t num_blocks);

    // Move-only (the buffers are large; a copy is never intended).
    BlockPool(const BlockPool&) = delete;
    BlockPool& operator=(const BlockPool&) = delete;
    BlockPool(BlockPool&&) = default;
    BlockPool& operator=(BlockPool&&) = default;

    int64_t allocate();        // a free physical block id; throws when exhausted
    void free(int64_t block);  // return a block to the pool

    int64_t num_blocks() const { return num_blocks_; }
    int64_t block_size() const { return block_size_; }
    int64_t free_blocks() const { return static_cast<int64_t>(free_list_.size()); }
    int64_t used_blocks() const { return num_blocks_ - free_blocks(); }
    int64_t n_kv_heads() const { return n_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }

    // Pointer to the head_dim contiguous floats for (layer, block, head, offset).
    float* k_at(int64_t layer, int64_t block, int64_t head, int64_t off);
    float* v_at(int64_t layer, int64_t block, int64_t head, int64_t off);

private:
    int64_t num_layers_, n_kv_heads_, head_dim_, block_size_, num_blocks_;
    int64_t block_stride_;               // n_kv_heads * block_size * head_dim
    std::vector<std::vector<float>> k_;  // per layer: num_blocks * block_stride_
    std::vector<std::vector<float>> v_;
    std::vector<int64_t> free_list_;     // free block ids (a stack)
};

// One sequence's view onto a BlockPool: its block table + filled length. Frees its
// blocks back to the pool on destruction (RAII — that is the utilization win).
class PagedKVCache : public KVCacheBase {
public:
    explicit PagedKVCache(BlockPool* pool);
    ~PagedKVCache() override { release(); }

    // Move-only; the moved-from cache is left empty so it frees nothing.
    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&& o) noexcept { steal(o); }
    PagedKVCache& operator=(PagedKVCache&& o) noexcept {
        if (this != &o) { release(); steal(o); }
        return *this;
    }

    // Write the new k/v into this sequence's blocks and attend directly over them —
    // no gather, no repeat_kv (GQA is folded into the read). Bit-identical to the
    // contiguous cache's gather + attention (run_paged.cpp). This is the true paged
    // attention read path: the kernel indexes blocks via the block table.
    Tensor attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                  int64_t n_rep, bool causal, int64_t query_offset) override;
    void advance(int64_t t) override { length_ += t; }
    int64_t length() const override { return length_; }
    int64_t num_blocks() const { return static_cast<int64_t>(block_table_.size()); }

private:
    void release();                            // free all blocks back to the pool
    void steal(PagedKVCache& o);               // take o's table/length, null o
    void ensure_capacity(int64_t positions);   // grow the block table to cover positions

    BlockPool* pool_;
    std::vector<int64_t> block_table_;  // logical block i -> physical block id
    int64_t length_ = 0;
};

}  // namespace ni
