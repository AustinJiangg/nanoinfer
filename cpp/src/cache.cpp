#include "cache.hpp"

#include <stdexcept>
#include <utility>

#include "ops.hpp"

namespace ni {

KVCache::KVCache(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim, int64_t max_seq)
    : n_kv_heads_(n_kv_heads), head_dim_(head_dim), max_seq_(max_seq) {
    if (num_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0)
        throw std::invalid_argument("KVCache: all dimensions must be positive");
    k_.reserve(static_cast<size_t>(num_layers));
    v_.reserve(static_cast<size_t>(num_layers));
    for (int64_t i = 0; i < num_layers; ++i) {
        k_.emplace_back(Tensor({n_kv_heads, max_seq, head_dim}));
        v_.emplace_back(Tensor({n_kv_heads, max_seq, head_dim}));
    }
}

std::pair<Tensor, Tensor> KVCache::update(int64_t layer, const Tensor& k, const Tensor& v) {
    if (layer < 0 || static_cast<size_t>(layer) >= k_.size())
        throw std::out_of_range("KVCache::update: layer index out of range");
    const int64_t t = k.size(1);
    const int64_t start = length_;     // same for every layer this forward
    const int64_t end = start + t;
    if (end > max_seq_)
        throw std::runtime_error("KVCache overflow: need " + std::to_string(end) +
                                 " positions, capacity is " + std::to_string(max_seq_));

    Tensor& kbuf = k_[static_cast<size_t>(layer)];
    Tensor& vbuf = v_[static_cast<size_t>(layer)];

    // Write the new token(s) at [:, start:end, :].
    for (int64_t h = 0; h < n_kv_heads_; ++h)
        for (int64_t i = 0; i < t; ++i)
            for (int64_t d = 0; d < head_dim_; ++d) {
                kbuf.at(h, start + i, d) = k.at(h, i, d);
                vbuf.at(h, start + i, d) = v.at(h, i, d);
            }

    // Read back a contiguous [n_kv, end, head_dim] copy of the filled prefix.
    Tensor ko({n_kv_heads_, end, head_dim_});
    Tensor vo({n_kv_heads_, end, head_dim_});
    for (int64_t h = 0; h < n_kv_heads_; ++h)
        for (int64_t s = 0; s < end; ++s)
            for (int64_t d = 0; d < head_dim_; ++d) {
                ko.at(h, s, d) = kbuf.at(h, s, d);
                vo.at(h, s, d) = vbuf.at(h, s, d);
            }
    return {std::move(ko), std::move(vo)};
}

Tensor KVCache::attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                       int64_t n_rep, bool causal, int64_t query_offset) {
    // Contiguous storage: gather the prefix, expand KV heads (GQA), reuse the
    // attention op. The paged cache (paged.cpp) does this without the copies.
    auto kv = update(layer, k, v);
    Tensor kk = repeat_kv(kv.first, n_rep);
    Tensor vv = repeat_kv(kv.second, n_rep);
    return attention(q, kk, vv, causal, query_offset);
}

void KVCache::advance(int64_t t) { length_ += t; }

}  // namespace ni
