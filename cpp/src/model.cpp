#include "model.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

#include "serialize.hpp"

namespace fs = std::filesystem;

namespace ni {

Model::Model(const std::string& weights_dir) {
    cfg_ = load_config(weights_dir + "/config.txt");
    for (const auto& entry : fs::directory_iterator(weights_dir)) {
        const fs::path& p = entry.path();
        if (p.extension() == ".bin") {
            // stem() drops the trailing ".bin" but keeps the dotted name, e.g.
            // "layers.0.self_attn.q_proj.weight".
            w_.emplace(p.stem().string(), load_bin(p.string()));
        }
    }
    if (w_.empty()) throw std::runtime_error("Model: no .bin weights in " + weights_dir);

    // Build the RoPE tables once for the full context; forward slices them.
    rope_ = build_rope_cache(cfg_.max_position_embeddings, cfg_.head_dim, cfg_.rope_theta);
}

KVCache Model::make_cache(int64_t max_seq) const {
    return KVCache(cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
}

const Tensor& Model::W(const std::string& name) const {
    auto it = w_.find(name);
    if (it == w_.end()) throw std::runtime_error("Model: missing weight " + name);
    return it->second;
}

Tensor Model::forward(const std::vector<int64_t>& ids, KVCache* cache) const {
    const int64_t seq = static_cast<int64_t>(ids.size());
    const float eps = cfg_.rms_norm_eps;

    // With a cache, the new tokens sit at positions start_pos.. (start_pos is the
    // cache length before this pass — the same for every layer; advanced once at
    // the end). Without a cache they run 0..seq-1.
    const int64_t start_pos = cache ? cache->length() : 0;
    if (start_pos + seq > rope_.cos.size(0))
        throw std::runtime_error("forward: position " + std::to_string(start_pos + seq) +
                                 " exceeds context length " + std::to_string(rope_.cos.size(0)));

    Tensor x = embedding(W("embed_tokens.weight"), ids);  // [seq, hidden]

    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string L = "layers." + std::to_string(i) + ".";

        // --- attention (pre-norm + residual) ---
        Tensor h = rmsnorm(x, W(L + "input_layernorm.weight"), eps);
        Tensor q = linear(h, W(L + "self_attn.q_proj.weight"), &W(L + "self_attn.q_proj.bias"));
        Tensor k = linear(h, W(L + "self_attn.k_proj.weight"), &W(L + "self_attn.k_proj.bias"));
        Tensor v = linear(h, W(L + "self_attn.v_proj.weight"), &W(L + "self_attn.v_proj.bias"));

        q = split_heads(q, cfg_.num_attention_heads, cfg_.head_dim);
        k = split_heads(k, cfg_.num_kv_heads, cfg_.head_dim);
        v = split_heads(v, cfg_.num_kv_heads, cfg_.head_dim);

        // RoPE at absolute positions; cached keys are stored already rotated.
        q = apply_rope(q, rope_.cos, rope_.sin, start_pos);
        k = apply_rope(k, rope_.cos, rope_.sin, start_pos);

        if (cache) {
            auto kv = cache->update(i, k, v);  // append new K/V, read back the prefix
            k = std::move(kv.first);
            v = std::move(kv.second);
        }

        k = repeat_kv(k, cfg_.n_rep());
        v = repeat_kv(v, cfg_.n_rep());

        Tensor a = attention(q, k, v, /*causal=*/true, /*query_offset=*/start_pos);
        a = merge_heads(a);                               // [seq, n_heads*head_dim]
        a = linear(a, W(L + "self_attn.o_proj.weight"));  // o_proj has no bias
        x = add(x, a);

        // --- SwiGLU MLP (pre-norm + residual) ---
        Tensor hm = rmsnorm(x, W(L + "post_attention_layernorm.weight"), eps);
        Tensor gate = silu(linear(hm, W(L + "mlp.gate_proj.weight")));
        Tensor up = linear(hm, W(L + "mlp.up_proj.weight"));
        Tensor down = linear(mul(gate, up), W(L + "mlp.down_proj.weight"));
        x = add(x, down);
    }

    if (cache) cache->advance(seq);

    x = rmsnorm(x, W("norm.weight"), eps);
    // Tied models share the embedding as the output projection, so the exporter
    // skips the duplicate lm_head.weight; fall back to embed_tokens here.
    const std::string lm = cfg_.tie_word_embeddings ? "embed_tokens.weight" : "lm_head.weight";
    return linear(x, W(lm));  // [seq, vocab]
}

}  // namespace ni
