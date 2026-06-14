// Model dimensions, read from the exported config.txt (key/value lines).
// Mirrors nanoinfer.config.ModelConfig — read from the file, never hardcoded.
#pragma once

#include <cstdint>
#include <string>

namespace ni {

struct Config {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int64_t tie_word_embeddings = 0;  // 1 == lm_head shares the embedding weight

    // GQA: how many query heads each KV head feeds.
    int64_t n_rep() const { return num_attention_heads / num_kv_heads; }
};

Config load_config(const std::string& path);

}  // namespace ni
