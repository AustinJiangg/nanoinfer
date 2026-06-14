// Throughput benchmark for stage C5: time prefill and decode and report the
// active SIMD target + thread count. This is a timing tool, not a parity check —
// run_parity / run_generate guard correctness; this just measures the speedup
// that SIMD + OpenMP buy. Tokens are synthesized (cycled from ref_ids), so it
// needs only an exported model, no reference dump.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   ./build/run_bench weights/qwen2.5-0.5b [fp32|q8|q4|q4g] [prefill_len] [decode_len]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.hpp"
#include "model.hpp"
#include "parallel.hpp"
#include "parity_util.hpp"
#include "simd.hpp"

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_bench <weights_dir> [fp32|q8|q4|q4g] [prefill_len] [decode_len]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string mode_str = argc > 2 ? argv[2] : "fp32";
    ni::QuantMode mode = ni::QuantMode::None;
    if (mode_str == "q8") mode = ni::QuantMode::Q8;
    else if (mode_str == "q4") mode = ni::QuantMode::Q4;
    else if (mode_str == "q4g") mode = ni::QuantMode::Q4G;
    else if (mode_str != "fp32") {
        std::printf("unknown mode '%s' (use fp32, q8, q4, q4g)\n", mode_str.c_str());
        return 2;
    }
    const int64_t prefill_len = argc > 3 ? std::stoll(argv[3]) : 64;
    const int64_t decode_len = argc > 4 ? std::stoll(argv[4]) : 64;

    try {
        ni::Model model(dir, mode);
        const int64_t vocab = model.config().vocab_size;
        std::printf("engine: simd=%s, threads=%d (%s); mode=%s\n", ni::simd::target(),
                    ni::max_threads(), ni::threading_backend(), mode_str.c_str());
        std::printf("prefill_len=%lld decode_len=%lld\n", (long long)prefill_len,
                    (long long)decode_len);

        // Synthesize a prompt of prefill_len tokens by cycling the reference ids
        // (any valid ids work for timing). Falls back to a ramp if none exported.
        std::vector<int64_t> seed_ids;
        try {
            seed_ids = read_ids(dir + "/ref_ids.txt");
        } catch (...) {
        }
        if (seed_ids.empty()) seed_ids = {1, 2, 3, 4, 5};
        std::vector<int64_t> prompt(static_cast<size_t>(prefill_len));
        for (int64_t i = 0; i < prefill_len; ++i)
            prompt[static_cast<size_t>(i)] = seed_ids[static_cast<size_t>(i) % seed_ids.size()];

        // Warm up caches/branch predictors so the timed runs are steady-state.
        {
            ni::KVCache warm = model.make_cache(prefill_len + 1);
            std::vector<int64_t> small(prompt.begin(),
                                       prompt.begin() + std::min<int64_t>(prefill_len, 8));
            model.forward(small, &warm);
        }

        // Prefill: one forward over the whole prompt (the compute-bound regime —
        // the weights are reused across all prefill_len rows).
        ni::KVCache cache = model.make_cache(prefill_len + decode_len + 1);
        Clock::time_point t0 = Clock::now();
        ni::Tensor logits = model.forward(prompt, &cache);
        Clock::time_point t1 = Clock::now();
        const double prefill_s = secs(t0, t1);
        int64_t next = argmax_row(logits, logits.size(0) - 1, vocab);

        // Decode: one token at a time (the memory-bound regime — every weight is
        // streamed once per token). Greedy, matching real generation's work.
        Clock::time_point t2 = Clock::now();
        for (int64_t d = 0; d < decode_len; ++d) {
            ni::Tensor l = model.forward({next}, &cache);
            next = argmax_row(l, 0, vocab);
        }
        Clock::time_point t3 = Clock::now();
        const double decode_s = secs(t2, t3);

        std::printf("prefill: %lld tok in %.3f s -> %.1f tok/s (%.2f ms/tok)\n",
                    (long long)prefill_len, prefill_s, prefill_len / prefill_s,
                    1e3 * prefill_s / double(prefill_len));
        std::printf("decode : %lld tok in %.3f s -> %.1f tok/s (%.2f ms/tok)\n",
                    (long long)decode_len, decode_s, decode_len / decode_s,
                    1e3 * decode_s / double(decode_len));
        std::printf("run_bench: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("run_bench: exception: %s\n", e.what());
        return 1;
    }
}
