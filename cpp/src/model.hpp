// The full Qwen2.5 decoder: embedding -> N transformer blocks -> norm -> lm_head.
// Loads NIT0 weights + config.txt from a directory (the export_weights.py output)
// and reproduces nanoinfer's forward pass. Batch-1, full-recompute (no KV cache
// yet — that's C3).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cache.hpp"
#include "config.hpp"
#include "ops.hpp"
#include "tensor.hpp"

namespace ni {

class Model {
public:
    // Load config.txt and every <name>.bin in `weights_dir`.
    explicit Model(const std::string& weights_dir);

    // Token ids -> logits [seq, vocab]. Without a cache (C1) the whole sequence is
    // recomputed; with a cache (C3) `ids` is just the new token(s), placed at
    // positions cache.length()..; the cache is advanced by t after the pass.
    Tensor forward(const std::vector<int64_t>& ids, KVCache* cache = nullptr) const;

    // Allocate a KV cache sized to this model.
    KVCache make_cache(int64_t max_seq) const;

    const Config& config() const { return cfg_; }

private:
    const Tensor& W(const std::string& name) const;

    Config cfg_;
    std::unordered_map<std::string, Tensor> w_;
    RopeCache rope_;  // built once for max_position_embeddings, sliced per forward
};

}  // namespace ni
