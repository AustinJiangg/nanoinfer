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
        // --- A0 architecture-description fields. A v1 config omits all of these;
        // the Config defaults (= Qwen2.5) then stand, which is what "still reads v1"
        // means. String-valued keys (act_fn, rope_scaling) map onto the enums. ---
        else if (key == "nit0_version") f >> c.nit0_version;
        else if (key == "qkv_bias") f >> c.qkv_bias;
        else if (key == "qk_norm") f >> c.qk_norm;
        else if (key == "act_fn") {
            std::string v; f >> v;
            if (v == "silu") c.act_fn = ActFn::Silu;
            else if (v == "gelu") c.act_fn = ActFn::Gelu;
            else throw std::runtime_error("load_config: unknown act_fn '" + v + "'");
        }
        else if (key == "rope_scaling") {
            std::string v; f >> v;
            if (v == "none") c.rope_scaling = RopeScaling::None;
            else if (v == "llama3") c.rope_scaling = RopeScaling::Llama3;
            else throw std::runtime_error("load_config: unknown rope_scaling '" + v + "'");
        }
        else if (key == "rope_scaling_factor") f >> c.rope_scaling_factor;
        else if (key == "rope_scaling_low_freq") f >> c.rope_scaling_low_freq;
        else if (key == "rope_scaling_high_freq") f >> c.rope_scaling_high_freq;
        else if (key == "rope_scaling_orig_max_pos") f >> c.rope_scaling_orig_max_pos;
        else if (key == "embedding_multiplier") f >> c.embedding_multiplier;
        else if (key == "attention_multiplier") f >> c.attention_multiplier;
        else if (key == "residual_multiplier") f >> c.residual_multiplier;
        else if (key == "logits_scaling") f >> c.logits_scaling;
        else if (key == "n_experts") f >> c.n_experts;
        else if (key == "moe_top_k") f >> c.moe_top_k;
        else if (key == "moe_intermediate_size") f >> c.moe_intermediate_size;
        else if (key == "sliding_window") f >> c.sliding_window;
        else if (key == "sliding_pattern") f >> c.sliding_pattern;
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
