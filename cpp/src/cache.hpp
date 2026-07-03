// Key/value cache for incremental decoding (C++ stage C3).
//
// Counterpart of nanoinfer/cache.py. Each layer keeps a [n_kv_heads, max_seq,
// head_dim] buffer for K and one for V, preallocated. Prefill fills them from the
// prompt; each decode step appends one token's K/V and reads the rest back. The
// stored K is post-RoPE (rotated once at insertion), pre-repeat_kv (n_kv heads,
// not n_heads) — the model expands to query heads after reading.
//
// `length` (filled positions) is shared across layers and advanced once per
// forward by the model, after every layer has written its slice — so update()
// reads it rather than taking a start argument every caller would pass alike.
//
// update() returns contiguous COPIES of the cached prefix (not torch-style views
// of the middle dimension, which wouldn't be contiguous for our ops). The copy is
// O(filled) per layer per step — negligible vs the forward; C5 can avoid it.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor.hpp"

namespace ni {

// The cache interface forward()/forward_batch() drive — so the same forward pass
// runs over either the contiguous cache (C3) or the paged cache (F8b) through one
// pointer, the same polymorphism the Weight interface uses for quant modes.
// An implementation must return, from update(), a CONTIGUOUS [n_kv, length+t,
// head_dim] view of the filled prefix, so the attention kernel stays cache-agnostic.
class KVCacheBase {
public:
    virtual ~KVCacheBase() = default;

    // Append the new `k`/`v` ([n_kv_heads, t, head_dim]) for one layer at the current
    // length, then attend: queries `q` ([n_heads, t, head_dim], post-RoPE) over the
    // whole cached history, returning [n_heads, t, head_dim]. Query i sits at absolute
    // position query_offset+i. GQA (n_rep) is folded in, so a paged cache reads each
    // KV head once without materializing repeats. The cache owns the read path: the
    // contiguous cache gathers + reuses the attention op; the paged cache indexes
    // blocks directly (true paged attention). Both give bit-identical results.
    virtual Tensor attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                          int64_t n_rep, bool causal, int64_t query_offset) = 0;

    // Mark `t` more positions filled (once per forward, after all layers).
    virtual void advance(int64_t t) = 0;

    // Roll back to `length` filled positions — the speculative-decode rollback (S1).
    // Speculative decode's verify forward writes K+1 *tentative* K/V onto the cache;
    // when the target rejects a drafted token, the tail beyond the accepted prefix is
    // dropped by discarding positions >= length. The contiguous cache just moves the
    // length pointer (stale slots are overwritten before they're read again); the paged
    // / CUDA overrides land in S1. The base default refuses, so a cache that can't yet
    // roll back fails loudly instead of silently serving stale K/V.
    virtual void truncate(int64_t length) {
        (void)length;
        throw std::runtime_error("KVCacheBase::truncate: this cache has no rollback yet (S1)");
    }

    virtual int64_t length() const = 0;  // filled positions
};

// Contiguous KV cache (C3): each layer keeps a preallocated [n_kv_heads, max_seq,
// head_dim] buffer for K and one for V.
class KVCache : public KVCacheBase {
public:
    KVCache(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim, int64_t max_seq);

    // Move-only: the per-layer buffers are large and a copy is never intended;
    // make an accidental copy a compile error rather than a silent deep clone.
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&) = default;
    KVCache& operator=(KVCache&&) = default;

    int64_t length() const override { return length_; }
    int64_t max_seq() const { return max_seq_; }

    // Append `k`/`v` and return contiguous [n_kv_heads, length+t, head_dim] copies of
    // the filled prefix. Used by attend() and by the cache unit test directly.
    std::pair<Tensor, Tensor> update(int64_t layer, const Tensor& k, const Tensor& v);
    Tensor attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                  int64_t n_rep, bool causal, int64_t query_offset) override;
    void advance(int64_t t) override;
    void truncate(int64_t length) override;

private:
    std::vector<Tensor> k_;  // per layer: [n_kv_heads, max_seq, head_dim]
    std::vector<Tensor> v_;
    int64_t n_kv_heads_;
    int64_t head_dim_;
    int64_t max_seq_;
    int64_t length_ = 0;
};

}  // namespace ni
