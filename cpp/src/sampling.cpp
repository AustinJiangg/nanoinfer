#include "sampling.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace ni {

namespace {
int64_t argmax(const std::vector<float>& v) {
    int64_t best = 0;
    float bv = v[0];
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > bv) { bv = v[i]; best = static_cast<int64_t>(i); }
    return best;
}
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
}  // namespace

void apply_repetition_penalty(std::vector<float>& logits,
                              const std::vector<int64_t>& context, float penalty) {
    if (penalty == 1.0f) return;
    // Unique tokens only — penalizing a repeated context id twice would compound.
    std::unordered_set<int64_t> seen(context.begin(), context.end());
    for (int64_t id : seen) {
        float& s = logits[static_cast<size_t>(id)];
        s = (s > 0.0f) ? s / penalty : s * penalty;  // both lower the score
    }
}

void apply_temperature(std::vector<float>& logits, float temperature) {
    for (float& x : logits) x /= temperature;
}

void apply_top_k(std::vector<float>& logits, int64_t k) {
    if (k <= 0 || k >= static_cast<int64_t>(logits.size())) return;
    // Find the k-th largest logit, then drop everything strictly below it (ties
    // at the threshold are kept, matching nanoinfer / HF).
    std::vector<float> tmp(logits);
    std::nth_element(tmp.begin(), tmp.begin() + (k - 1), tmp.end(), std::greater<float>());
    const float kth = tmp[static_cast<size_t>(k - 1)];
    for (float& x : logits)
        if (x < kth) x = kNegInf;
}

void apply_top_p(std::vector<float>& logits, float p) {
    if (p >= 1.0f) return;
    const int64_t n = static_cast<int64_t>(logits.size());

    std::vector<int64_t> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });

    // Softmax over the sorted logits to get a cumulative probability.
    const float maxv = logits[idx[0]];
    std::vector<double> prob(static_cast<size_t>(n));
    double denom = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double e = std::exp(double(logits[idx[i]]) - maxv);
        prob[static_cast<size_t>(i)] = e;
        denom += e;
    }
    // Keep the smallest prefix whose cumulative prob reaches p (including the
    // token that crosses it); drop the rest. Always keeps at least the top-1.
    double cum = 0.0;
    bool crossed = false;
    for (int64_t i = 0; i < n; ++i) {
        if (crossed) {
            logits[idx[i]] = kNegInf;
            continue;
        }
        cum += prob[static_cast<size_t>(i)] / denom;
        if (cum >= p) crossed = true;  // keep this one; drop everything after
    }
}

int64_t sample_next_token(std::vector<float> logits, const SamplingParams& params,
                          const std::vector<int64_t>& context, std::mt19937_64& rng) {
    // Processor: shapes greedy and sampled decoding alike.
    apply_repetition_penalty(logits, context, params.repetition_penalty);

    if (params.greedy()) return argmax(logits);

    // Warpers: only when sampling.
    apply_temperature(logits, params.temperature);
    apply_top_k(logits, params.top_k);
    apply_top_p(logits, params.top_p);

    // softmax -> categorical draw. Masked (-inf) logits get weight 0.
    const float maxv = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) weights[i] = std::exp(double(logits[i]) - maxv);
    std::discrete_distribution<int64_t> dist(weights.begin(), weights.end());
    return dist(rng);
}

}  // namespace ni
