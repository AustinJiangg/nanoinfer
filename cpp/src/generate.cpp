#include "generate.hpp"

#include <memory>
#include <utility>

#include "tensor.hpp"

namespace ni {

std::vector<int64_t> generate(const Model& model, const std::vector<int64_t>& prompt,
                              const GenerateConfig& cfg) {
    cfg.params.validate();
    if (prompt.empty()) return {};  // nothing to condition on; forward needs >=1 token

    std::mt19937_64 rng(cfg.seed);
    std::vector<int64_t> ids = prompt;  // full context (prompt + generated): rep-penalty
    std::vector<int64_t> generated;

    // One KV cache sized to hold the prompt plus everything we'll emit. `cur` is
    // what we feed each step: the whole prompt on step 0 (prefill), then a single
    // new token (decode) once the cache holds the past. make_kv_cache() picks the
    // cache for the model's device, so this loop runs unchanged on CPU or GPU.
    std::unique_ptr<KVCacheBase> cache =
        model.make_kv_cache(static_cast<int64_t>(prompt.size()) + cfg.max_tokens);
    std::vector<int64_t> cur = prompt;

    for (int step = 0; step < cfg.max_tokens; ++step) {
        Tensor logits = model.forward(cur, cache.get());
        const int64_t seq = logits.size(0), vocab = logits.size(1);

        std::vector<float> last(static_cast<size_t>(vocab));
        for (int64_t j = 0; j < vocab; ++j) last[static_cast<size_t>(j)] = logits.at(seq - 1, j);

        const int64_t next = sample_next_token(std::move(last), cfg.params, ids, rng);

        if (cfg.eos_id >= 0 && next == cfg.eos_id) break;
        generated.push_back(next);
        ids.push_back(next);
        cur = {next};
    }
    return generated;
}

}  // namespace ni
