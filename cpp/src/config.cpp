#include "config.hpp"

#include <fstream>
#include <stdexcept>

namespace ni {

Config load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_config: cannot open " + path);

    Config c;
    std::string key;
    while (f >> key) {
        if (key == "vocab_size") f >> c.vocab_size;
        else if (key == "hidden_size") f >> c.hidden_size;
        else if (key == "intermediate_size") f >> c.intermediate_size;
        else if (key == "num_layers") f >> c.num_layers;
        else if (key == "num_attention_heads") f >> c.num_attention_heads;
        else if (key == "num_kv_heads") f >> c.num_kv_heads;
        else if (key == "head_dim") f >> c.head_dim;
        else if (key == "max_position_embeddings") f >> c.max_position_embeddings;
        else if (key == "rms_norm_eps") f >> c.rms_norm_eps;
        else if (key == "rope_theta") f >> c.rope_theta;
        else if (key == "tie_word_embeddings") f >> c.tie_word_embeddings;
        else if (key == "eos_token_id") f >> c.eos_token_id;
        else { std::string ignored; f >> ignored; }  // unknown key: skip its value
    }

    // Validate loudly: a missing/misparsed key otherwise leaves a 0 that surfaces
    // far away (n_rep div-by-zero, zero layers, wrong shapes deep in forward).
    auto need = [](int64_t v, const char* name) {
        if (v <= 0) throw std::runtime_error("load_config: missing or invalid '" + std::string(name) + "'");
    };
    need(c.vocab_size, "vocab_size");
    need(c.hidden_size, "hidden_size");
    need(c.intermediate_size, "intermediate_size");
    need(c.num_layers, "num_layers");
    need(c.num_attention_heads, "num_attention_heads");
    need(c.num_kv_heads, "num_kv_heads");
    need(c.head_dim, "head_dim");
    need(c.max_position_embeddings, "max_position_embeddings");
    if (c.num_attention_heads % c.num_kv_heads != 0)
        throw std::runtime_error("load_config: num_attention_heads must be divisible by num_kv_heads");
    return c;
}

}  // namespace ni
