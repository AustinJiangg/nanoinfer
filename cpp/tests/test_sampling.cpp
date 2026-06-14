// Fast sampling tests — warper behavior + the greedy-equivalence invariants,
// mirroring nanoinfer's tests/test_sampling.py. No model needed.
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "sampling.hpp"
#include "test_util.hpp"

using ni::SamplingParams;

static int count_finite(const std::vector<float>& v) {
    int n = 0;
    for (float x : v) if (std::isfinite(x)) ++n;
    return n;
}

int main() {
    // temperature scales logits and preserves the argmax.
    {
        std::vector<float> x{1, 2, 3};
        ni::apply_temperature(x, 0.5f);
        CHECK_CLOSE(x[0], 2.0, 1e-6);
        CHECK_CLOSE(x[2], 6.0, 1e-6);
    }

    // top_k keeps exactly k finite logits (the largest), ties aside.
    {
        std::vector<float> x{1, 5, 3, 4, 2};
        ni::apply_top_k(x, 2);
        CHECK(count_finite(x) == 2);
        CHECK(std::isfinite(x[1]) && std::isfinite(x[3]));  // values 5 and 4
        CHECK(!std::isfinite(x[0]) && !std::isfinite(x[2]) && !std::isfinite(x[4]));
    }
    { std::vector<float> x{1, 2, 3}; ni::apply_top_k(x, 0); CHECK(count_finite(x) == 3); }  // off

    // top_p keeps the nucleus: probs 0.5/0.3/0.15/0.05, p=0.9 -> drop the last.
    {
        std::vector<float> probs{0.5f, 0.3f, 0.15f, 0.05f};
        std::vector<float> x;
        for (float p : probs) x.push_back(std::log(p));
        ni::apply_top_p(x, 0.9f);
        CHECK(std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]));
        CHECK(!std::isfinite(x[3]));
    }
    // top_p keeps at least the top-1 even when its own prob exceeds p.
    {
        std::vector<float> x{std::log(0.95f), std::log(0.04f), std::log(0.01f)};
        ni::apply_top_p(x, 0.9f);
        CHECK(std::isfinite(x[0]) && !std::isfinite(x[1]) && !std::isfinite(x[2]));
    }

    // top_p at an exact cumulative boundary keeps the crossing token (parity with
    // nanoinfer: drop iff cumprob-before > p, strict). probs 0.5/0.3/0.2, p=0.8 ->
    // cumprob before token2 is exactly 0.8, NOT > 0.8, so all three are kept.
    {
        std::vector<float> probs{0.5f, 0.3f, 0.2f};
        std::vector<float> x;
        for (float p : probs) x.push_back(std::log(p));
        ni::apply_top_p(x, 0.8f);
        CHECK(std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]));
    }

    // top_p with vocab order != sorted order exercises the index remap.
    {
        std::vector<float> probs{0.15f, 0.5f, 0.05f, 0.3f};  // vocab order
        std::vector<float> x;
        for (float p : probs) x.push_back(std::log(p));
        ni::apply_top_p(x, 0.9f);
        // nucleus by prob is {idx1=0.5, idx3=0.3, idx0=0.15} (cum 0.95); idx2 dropped.
        CHECK(std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[3]));
        CHECK(!std::isfinite(x[2]));
    }

    // repetition penalty pushes seen tokens down (positive /p, negative *p).
    {
        std::vector<float> x{2, -2, 1};
        ni::apply_repetition_penalty(x, {0, 1}, 2.0f);
        CHECK_CLOSE(x[0], 1.0, 1e-6);   // 2 / 2
        CHECK_CLOSE(x[1], -4.0, 1e-6);  // -2 * 2
        CHECK_CLOSE(x[2], 1.0, 1e-6);   // unseen, untouched
    }

    // temperature 0 == greedy (argmax), deterministic.
    {
        std::vector<float> x{0.1f, 9.0f, 0.2f, 3.0f};
        std::mt19937_64 rng(0);
        CHECK(ni::sample_next_token(x, SamplingParams{}, {}, rng) == 1);
    }

    // top_k == 1 collapses to greedy regardless of seed.
    {
        std::vector<float> x{0.1f, 9.0f, 0.2f, 3.0f};
        for (uint64_t seed : {0u, 1u, 2u, 1234u}) {
            SamplingParams p;
            p.temperature = 1.0f;
            p.top_k = 1;
            std::mt19937_64 rng(seed);
            CHECK(ni::sample_next_token(x, p, {}, rng) == 1);
        }
    }

    // same seed -> same token; different seeds CAN diverge; draws respect top_k.
    {
        std::vector<float> x(50);
        std::mt19937_64 fill(0);
        std::normal_distribution<float> nd;
        for (float& v : x) v = nd(fill);
        SamplingParams p;
        p.temperature = 1.0f;
        p.top_k = 10;

        auto draw = [&](uint64_t s) {
            std::mt19937_64 rng(s);
            return ni::sample_next_token(x, p, {}, rng);
        };
        CHECK(draw(7) == draw(7));  // reproducible

        std::set<int64_t> tokens;
        for (uint64_t s = 0; s < 20; ++s) tokens.insert(draw(s));
        CHECK(tokens.size() > 1);  // RNG actually varies the draw
    }

    // repetition penalty routed THROUGH sample_next_token (the path generate uses):
    // it runs before the greedy short-circuit, so it can flip the greedy argmax.
    {
        std::vector<float> x{10.0f, 9.5f};  // argmax is token 0
        SamplingParams p;                   // greedy
        p.repetition_penalty = 2.0f;
        std::mt19937_64 rng(0);
        CHECK(ni::sample_next_token(x, p, {0}, rng) == 1);  // token 0 penalized -> flips to 1
    }

    // an out-of-range context id must not write out of bounds.
    {
        std::vector<float> x{1.0f, 2.0f, 3.0f};
        SamplingParams p;
        p.repetition_penalty = 2.0f;
        std::mt19937_64 rng(0);
        ni::sample_next_token(x, p, {99, -1}, rng);  // ids outside [0,3) are ignored
        CHECK(true);  // reaching here without UB is the assertion
    }

    // the full pipeline (temp + top_k + top_p + rep_penalty) composes and returns
    // a valid token id.
    {
        std::vector<float> x(100);
        std::mt19937_64 fill(1);
        std::normal_distribution<float> nd;
        for (float& v : x) v = nd(fill);
        SamplingParams p;
        p.temperature = 0.8f;
        p.top_k = 20;
        p.top_p = 0.95f;
        p.repetition_penalty = 1.2f;
        std::mt19937_64 rng(5);
        int64_t tok = ni::sample_next_token(x, p, {3, 7, 11}, rng);
        CHECK(tok >= 0 && tok < 100);
    }

    // invalid params are rejected (mirrors the Python __post_init__ guard).
    {
        SamplingParams bad;
        bad.repetition_penalty = 0.0f;
        bool threw = false;
        try { bad.validate(); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        SamplingParams bad2;
        bad2.temperature = -1.0f;
        threw = false;
        try { bad2.validate(); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    std::printf(g_failures ? "test_sampling: %d failures\n" : "test_sampling: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
