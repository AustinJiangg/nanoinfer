// Model dimensions, read from the exported config.txt (key/value lines).
// Mirrors nanoinfer.config.ModelConfig — read from the file, never hardcoded.
#pragma once

#include <cstdint>
#include <string>

namespace ni {

// Activation for the FFN gate. SwiGLU (silu) is the Llama family; GeGLU (gelu)
// arrives with Gemma-3 (A4). Stored as an int so the text parser stays uniform.
enum class ActFn : int64_t { Silu = 0, Gelu = 1 };
// RoPE frequency scaling resolved at load time (A2 Llama-3.2). None keeps the
// plain inv_freq; Llama3 rescales three frequency bands.
enum class RopeScaling : int64_t { None = 0, Llama3 = 1 };

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
    int64_t eos_token_id = -1;        // -1 == unknown / no EOS stop

    // --- A0 architecture-description fields ---
    // Export-format version. v1 (the original config.txt) had none of the fields
    // below; a v1 dump is a Qwen2.5 export, so the defaults here reproduce it
    // exactly (qkv_bias on, everything else identity/off). load_config still
    // reads a v1 file — the missing keys simply keep these defaults.
    int64_t nit0_version = 1;
    int64_t qkv_bias = 1;             // q/k/v projections carry a bias (Qwen2.5; A1 Qwen3 = 0)
    int64_t qk_norm = 0;             // per-head RMSNorm on Q,K pre-RoPE (A1/A4)
    ActFn act_fn = ActFn::Silu;      // FFN gate activation
    // RoPE scaling (A2). Only the parameters Llama3 needs; identity when None.
    RopeScaling rope_scaling = RopeScaling::None;
    float rope_scaling_factor = 1.0f;
    float rope_scaling_low_freq = 1.0f;
    float rope_scaling_high_freq = 4.0f;
    int64_t rope_scaling_orig_max_pos = 0;
    // muP-style scalars, identity for Qwen2.5 (A3 Granite).
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 1.0f;
    float residual_multiplier = 1.0f;
    float logits_scaling = 1.0f;
    // Mixture-of-experts (A3). n_experts == 0 => dense FFN.
    int64_t n_experts = 0;
    int64_t moe_top_k = 0;
    int64_t moe_intermediate_size = 0;
    // Sliding-window attention (A4 Gemma-3). 0 == full attention.
    int64_t sliding_window = 0;
    int64_t sliding_pattern = 0;

    // GQA: how many query heads each KV head feeds.
    int64_t n_rep() const { return num_attention_heads / num_kv_heads; }
};

Config load_config(const std::string& path);

}  // namespace ni
