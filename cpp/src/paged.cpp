#include "paged.hpp"

#include <stdexcept>
#include <string>

namespace ni {

BlockPool::BlockPool(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim,
                     int64_t block_size, int64_t num_blocks)
    : num_layers_(num_layers), n_kv_heads_(n_kv_heads), head_dim_(head_dim),
      block_size_(block_size), num_blocks_(num_blocks) {
    if (num_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || block_size <= 0 ||
        num_blocks <= 0)
        throw std::invalid_argument("BlockPool: all dimensions must be positive");
    block_stride_ = n_kv_heads_ * block_size_ * head_dim_;
    k_.resize(static_cast<size_t>(num_layers_));
    v_.resize(static_cast<size_t>(num_layers_));
    for (int64_t l = 0; l < num_layers_; ++l) {
        k_[static_cast<size_t>(l)].assign(static_cast<size_t>(num_blocks_ * block_stride_), 0.0f);
        v_[static_cast<size_t>(l)].assign(static_cast<size_t>(num_blocks_ * block_stride_), 0.0f);
    }
    // Free list as a stack; push high ids first so allocate() hands out 0,1,2,...
    free_list_.reserve(static_cast<size_t>(num_blocks_));
    for (int64_t i = num_blocks_ - 1; i >= 0; --i) free_list_.push_back(i);
}

int64_t BlockPool::allocate() {
    if (free_list_.empty())
        throw std::runtime_error("BlockPool: out of blocks (" + std::to_string(num_blocks_) +
                                 " total) — raise num_blocks or evict a sequence");
    const int64_t b = free_list_.back();
    free_list_.pop_back();
    return b;
}

void BlockPool::free(int64_t block) {
    if (block < 0 || block >= num_blocks_)
        throw std::out_of_range("BlockPool::free: block id out of range");
    free_list_.push_back(block);
}

float* BlockPool::k_at(int64_t layer, int64_t block, int64_t head, int64_t off) {
    return k_[static_cast<size_t>(layer)].data() +
           ((block * n_kv_heads_ + head) * block_size_ + off) * head_dim_;
}

float* BlockPool::v_at(int64_t layer, int64_t block, int64_t head, int64_t off) {
    return v_[static_cast<size_t>(layer)].data() +
           ((block * n_kv_heads_ + head) * block_size_ + off) * head_dim_;
}

PagedKVCache::PagedKVCache(BlockPool* pool) : pool_(pool) {
    if (pool == nullptr) throw std::invalid_argument("PagedKVCache: null pool");
}

void PagedKVCache::release() {
    if (pool_)
        for (int64_t b : block_table_) pool_->free(b);
    block_table_.clear();
    length_ = 0;
}

void PagedKVCache::steal(PagedKVCache& o) {
    pool_ = o.pool_;
    block_table_ = std::move(o.block_table_);
    length_ = o.length_;
    o.pool_ = nullptr;
    o.block_table_.clear();
    o.length_ = 0;
}

void PagedKVCache::ensure_capacity(int64_t positions) {
    const int64_t bs = pool_->block_size();
    while (static_cast<int64_t>(block_table_.size()) * bs < positions)
        block_table_.push_back(pool_->allocate());
}

std::pair<Tensor, Tensor> PagedKVCache::update(int64_t layer, const Tensor& k,
                                               const Tensor& v) {
    const int64_t t = k.size(1);
    const int64_t n_kv = pool_->n_kv_heads(), hd = pool_->head_dim(), bs = pool_->block_size();
    const int64_t start = length_, end = start + t;
    // Allocate blocks for the new positions. Idempotent across layers: only layer 0
    // grows the table (length_ is advanced once after all layers, like KVCache).
    ensure_capacity(end);

    // Write the t new tokens into their blocks for this layer.
    for (int64_t i = 0; i < t; ++i) {
        const int64_t pos = start + i;
        const int64_t blk = block_table_[static_cast<size_t>(pos / bs)];
        const int64_t off = pos % bs;
        for (int64_t h = 0; h < n_kv; ++h) {
            float* kp = pool_->k_at(layer, blk, h, off);
            float* vp = pool_->v_at(layer, blk, h, off);
            for (int64_t d = 0; d < hd; ++d) {
                kp[d] = k.at(h, i, d);
                vp[d] = v.at(h, i, d);
            }
        }
    }

    // Gather the filled prefix [n_kv, end, head_dim] from the blocks (the paged
    // read). Contiguous, so the attention kernel is unchanged and the result is
    // bit-identical to the contiguous cache's read-back.
    Tensor ko({n_kv, end, hd}), vo({n_kv, end, hd});
    for (int64_t s = 0; s < end; ++s) {
        const int64_t blk = block_table_[static_cast<size_t>(s / bs)];
        const int64_t off = s % bs;
        for (int64_t h = 0; h < n_kv; ++h) {
            const float* kp = pool_->k_at(layer, blk, h, off);
            const float* vp = pool_->v_at(layer, blk, h, off);
            for (int64_t d = 0; d < hd; ++d) {
                ko.at(h, s, d) = kp[d];
                vo.at(h, s, d) = vp[d];
            }
        }
    }
    return {std::move(ko), std::move(vo)};
}

}  // namespace ni
