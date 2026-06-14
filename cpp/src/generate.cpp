#include "generate.hpp"

#include <utility>

#include "tensor.hpp"

namespace ni {

std::vector<int64_t> generate(const Model& model, const std::vector<int64_t>& prompt,
                              const GenerateConfig& cfg) {
    std::mt19937_64 rng(cfg.seed);
    std::vector<int64_t> ids = prompt;  // running context: prompt + generated so far
    std::vector<int64_t> generated;

    for (int step = 0; step < cfg.max_tokens; ++step) {
        Tensor logits = model.forward(ids);  // [seq, vocab], full recompute (no cache yet)
        const int64_t seq = logits.size(0), vocab = logits.size(1);

        // Pull out the last position's logit row to sample from.
        std::vector<float> last(static_cast<size_t>(vocab));
        for (int64_t j = 0; j < vocab; ++j) last[static_cast<size_t>(j)] = logits.at(seq - 1, j);

        // `ids` (prompt + generated) is the repetition-penalty context.
        const int64_t next = sample_next_token(std::move(last), cfg.params, ids, rng);

        if (cfg.eos_id >= 0 && next == cfg.eos_id) break;
        generated.push_back(next);
        ids.push_back(next);
    }
    return generated;
}

}  // namespace ni
