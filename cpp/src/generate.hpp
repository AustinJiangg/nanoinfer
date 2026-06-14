// Autoregressive generation loop (C++ stage C2).
//
// Mirrors nanoinfer/generate.py: forward -> take the last-position logits ->
// sample -> append -> repeat until EOS or max_tokens. No KV cache yet (C3), so
// each step re-runs the full forward over the running sequence.
#pragma once

#include <cstdint>
#include <vector>

#include "model.hpp"
#include "sampling.hpp"

namespace ni {

struct GenerateConfig {
    int max_tokens = 20;
    SamplingParams params;       // default is greedy
    uint64_t seed = 0;           // seeds the sampler's RNG
    int64_t eos_id = -1;         // stop when this token is produced; -1 disables
};

// Continue `prompt` token ids, returning the generated ids (not including prompt).
std::vector<int64_t> generate(const Model& model, const std::vector<int64_t>& prompt,
                              const GenerateConfig& cfg);

}  // namespace ni
