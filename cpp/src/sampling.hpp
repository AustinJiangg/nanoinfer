// Next-token sampling (C++ stage C2) — the counterpart of nanoinfer/sampling.py.
//
// The warpers compose in the same order as the Python engine (and HF): repetition
// penalty on the raw logits (it shapes greedy too), then a greedy short-circuit,
// then temperature -> top-k -> top-p -> softmax -> a categorical draw. Operates on
// a 1-D logits vector for the current position.
//
// The draw uses a std::mt19937_64 the caller seeds, so sampling is reproducible —
// but the RNG differs from torch, so sampled output is NOT token-identical to the
// Python engine (greedy decoding is, and that's what the parity test checks).
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace ni {

struct SamplingParams {
    float temperature = 0.0f;          // 0 == greedy (argmax)
    int64_t top_k = 0;                 // <= 0 == off
    float top_p = 1.0f;                // >= 1 == off
    float repetition_penalty = 1.0f;   // 1 == off; > 1 discourages seen tokens

    bool greedy() const { return temperature == 0.0f; }
};

// In-place logit transforms (each a no-op at its "off" value).
void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<int64_t>& context, float penalty);
void apply_temperature(std::vector<float>& logits, float temperature);
void apply_top_k(std::vector<float>& logits, int64_t k);
void apply_top_p(std::vector<float>& logits, float p);

// Pick the next token id from a [vocab] logit row. `context` (prompt + generated)
// is only read when repetition_penalty != 1. `logits` is taken by value since the
// pipeline mutates it.
int64_t sample_next_token(std::vector<float> logits, const SamplingParams& params,
                          const std::vector<int64_t>& context, std::mt19937_64& rng);

}  // namespace ni
