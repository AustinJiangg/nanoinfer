#include "paged.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "simd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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
    refcount_.assign(static_cast<size_t>(num_blocks_), 0);  // all free
}

int64_t BlockPool::allocate() {
    if (free_list_.empty())
        throw std::runtime_error("BlockPool: out of blocks (" + std::to_string(num_blocks_) +
                                 " total) — raise num_blocks or evict a sequence");
    const int64_t b = free_list_.back();
    free_list_.pop_back();
    refcount_[static_cast<size_t>(b)] = 1;  // one holder: the allocating sequence
    return b;
}

void BlockPool::incref(int64_t block) {
    if (block < 0 || block >= num_blocks_)
        throw std::out_of_range("BlockPool::incref: block id out of range");
    if (refcount_[static_cast<size_t>(block)] <= 0)
        throw std::runtime_error("BlockPool::incref: block is free");
    ++refcount_[static_cast<size_t>(block)];
}

void BlockPool::free(int64_t block) {
    if (block < 0 || block >= num_blocks_)
        throw std::out_of_range("BlockPool::free: block id out of range");
    int64_t& rc = refcount_[static_cast<size_t>(block)];
    if (rc <= 0) throw std::runtime_error("BlockPool::free: double free");
    if (--rc == 0) free_list_.push_back(block);  // last holder gone -> recycle
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

void PagedKVCache::truncate(int64_t length) {
    if (length < 0 || length > length_)
        throw std::invalid_argument("PagedKVCache::truncate: length " + std::to_string(length) +
                                    " out of range [0, " + std::to_string(length_) + "]");
    const int64_t bs = pool_->block_size();
    // Block i covers logical positions [i*bs, (i+1)*bs), so it holds ONLY rejected positions
    // iff i*bs >= length. Keep ceil(length/bs) blocks (0 when length==0) — the last kept block
    // may straddle `length`, and its stale tail is overwritten by the next write before it's
    // read. Free the rest to the pool (a following forward re-allocates, reusing them).
    const int64_t keep = (length + bs - 1) / bs;
    for (int64_t i = keep; i < static_cast<int64_t>(block_table_.size()); ++i)
        pool_->free(block_table_[static_cast<size_t>(i)]);
    block_table_.resize(static_cast<size_t>(keep));
    length_ = length;
}

void PagedKVCache::share_prefix(const std::vector<int64_t>& blocks, int64_t length) {
    if (!block_table_.empty() || length_ != 0)
        throw std::runtime_error("share_prefix: must seed a fresh cache");
    const int64_t bs = pool_->block_size();
    if (length != static_cast<int64_t>(blocks.size()) * bs)
        throw std::invalid_argument("share_prefix: length must be a block boundary "
                                    "(blocks.size() * block_size)");
    // Become a holder of each shared block; the sequence writes its own blocks past
    // `length` (a block boundary), so the shared blocks are never mutated here.
    block_table_ = blocks;
    for (int64_t b : block_table_) pool_->incref(b);
    length_ = length;
}

Tensor PagedKVCache::attend(int64_t layer, const Tensor& q, const Tensor& k,
                            const Tensor& v, int64_t n_rep, bool causal,
                            int64_t query_offset) {
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

    // Paged attention: each query head reads K/V straight from the blocks via the
    // block table — no contiguous gather, no repeat_kv (query head h uses KV head
    // h/n_rep). The arithmetic mirrors ops::attention() exactly (simd dot product,
    // double-accumulated softmax + value sum, same key order), so the result is
    // bit-identical to the contiguous cache's gather + attention. Threaded over query
    // heads (each is one thread's complete reduction → deterministic).
    const int64_t n_heads = q.size(0), sq = q.size(1), dim = q.size(2);
    const float scale = 1.0f / std::sqrt(float(dim));
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    Tensor out({n_heads, sq, dim});
    const float* qp = q.data();
    float* op = out.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_heads >= 2)
#endif
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t kvh = h / n_rep;  // GQA: this query head's KV head
        std::vector<float> scores(static_cast<size_t>(end));
        const float* qh = qp + h * sq * dim;
        for (int64_t i = 0; i < sq; ++i) {
            const float* qi = qh + i * dim;
            // Query i is at absolute position query_offset+i; attend keys 0..that.
            const int64_t limit = causal ? (query_offset + i + 1) : end;
            float maxv = kNegInf;
            for (int64_t j = 0; j < limit; ++j) {
                const int64_t blk = block_table_[static_cast<size_t>(j / bs)];
                const float* kj = pool_->k_at(layer, blk, kvh, j % bs);
                const float s = float(simd::dot_f32(qi, kj, dim)) * scale;
                scores[static_cast<size_t>(j)] = s;
                if (s > maxv) maxv = s;
            }
            double denom = 0.0;
            for (int64_t j = 0; j < limit; ++j) {
                const float e = std::exp(scores[static_cast<size_t>(j)] - maxv);
                scores[static_cast<size_t>(j)] = e;
                denom += e;
            }
            const float inv = float(1.0 / denom);
            float* oi = op + (h * sq + i) * dim;
            for (int64_t d = 0; d < dim; ++d) {
                double acc = 0.0;
                for (int64_t j = 0; j < limit; ++j) {
                    const int64_t blk = block_table_[static_cast<size_t>(j / bs)];
                    acc += double(scores[static_cast<size_t>(j)]) * inv *
                           pool_->v_at(layer, blk, kvh, j % bs)[d];
                }
                oi[d] = float(acc);
            }
        }
    }
    return out;
}

}  // namespace ni
