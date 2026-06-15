// Batched decode parity + throughput (C++ stage F8a).
//
// Parity: forward_batch over N sequences (one new token each, at each sequence's
// own cache position) must equal N standalone forward() decodes, bit-for-bit — the
// rows of a linear are independent, and attention is per-sequence, so batching must
// not change any sequence's logits. This is the seam test the Python serving layer
// relies on (run_serve.py checks the same thing end-to-end).
//
// Throughput: the point of F8a. Decode at increasing batch sizes; tokens/sec should
// climb because every weight row is streamed once and reused across the N tokens
// (decode stops being purely memory-bound). A timing tool, not a correctness gate.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   ./build/run_batch weights/qwen2.5-0.5b
#include <algorithm>
#include <chrono>
#include <cmath>
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
        std::printf("usage: run_batch <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir);
        const int64_t vocab = model.config().vocab_size;
        std::printf("engine: simd=%s, threads=%d (%s)\n", ni::simd::target(),
                    ni::max_threads(), ni::threading_backend());

        std::vector<int64_t> base = read_ids(dir + "/ref_ids.txt");
        if (base.empty()) {
            std::printf("run_batch: ref_ids.txt is empty\n");
            return 2;
        }

        // --- Parity: N sequences, distinct prompts/lengths, decoded both ways ---
        std::vector<std::vector<int64_t>> prompts = {
            base,
            std::vector<int64_t>(base.begin(), base.begin() + std::max<size_t>(1, base.size() - 2)),
            std::vector<int64_t>(base.begin(), base.begin() + std::max<size_t>(1, base.size() / 2)),
            base,
        };
        const int64_t N = static_cast<int64_t>(prompts.size());
        const int STEPS = 16;

        // Independent caches for each path; prefill seeds them identically.
        std::vector<ni::KVCache> single, batched;
        std::vector<ni::KVCache*> batched_ptrs;
        std::vector<int64_t> cur(static_cast<size_t>(N));
        for (int64_t s = 0; s < N; ++s) {
            const int64_t cap = static_cast<int64_t>(prompts[s].size()) + STEPS + 1;
            single.push_back(model.make_cache(cap));
            batched.push_back(model.make_cache(cap));
        }
        for (int64_t s = 0; s < N; ++s) {
            model.forward(prompts[s], &single[s]);
            ni::Tensor lp = model.forward(prompts[s], &batched[s]);  // identical prefill
            cur[s] = argmax_row(lp, lp.size(0) - 1, vocab);          // first decode token
            batched_ptrs.push_back(&batched[s]);
        }

        double maxd = 0.0;
        int tok_mism = 0;
        for (int step = 0; step < STEPS; ++step) {
            ni::Tensor lb = model.forward_batch(cur, batched_ptrs);  // [N, vocab]
            std::vector<int64_t> next(static_cast<size_t>(N));
            for (int64_t s = 0; s < N; ++s) {
                ni::Tensor ls = model.forward({cur[static_cast<size_t>(s)]}, &single[s]);  // [1, vocab]
                for (int64_t j = 0; j < vocab; ++j)
                    maxd = std::max(maxd, std::fabs(double(lb.at(s, j)) - double(ls.at(0, j))));
                if (argmax_row(lb, s, vocab) != argmax_row(ls, 0, vocab)) ++tok_mism;
                next[static_cast<size_t>(s)] = argmax_row(lb, s, vocab);
            }
            cur = next;
        }
        const bool parity_ok = (maxd == 0.0) && (tok_mism == 0);
        std::printf("parity: %lld seqs x %d steps  max|diff|=%g  token mismatches=%d  -> %s\n",
                    (long long)N, STEPS, maxd, tok_mism,
                    parity_ok ? "BIT-IDENTICAL" : "MISMATCH");

        // --- Throughput: decode tok/s vs batch size (the F8a win) ---
        {
            ni::KVCache warm = model.make_cache(static_cast<int64_t>(base.size()) + 2);
            std::vector<ni::KVCache*> wp{&warm};
            model.forward(base, &warm);
            model.forward_batch({base[0]}, wp);  // warm code paths before timing
        }
        std::printf("\nbatched decode throughput (greedy, %d steps each):\n", STEPS);
        std::printf("  batch   tok/s   speedup\n");
        double base_tps = 0.0;
        for (int64_t B : {1, 2, 4, 8, 16}) {
            std::vector<ni::KVCache> caches;
            std::vector<ni::KVCache*> ptrs;
            std::vector<int64_t> tok(static_cast<size_t>(B));
            const int64_t cap = static_cast<int64_t>(base.size()) + STEPS + 1;
            for (int64_t b = 0; b < B; ++b) caches.push_back(model.make_cache(cap));
            for (int64_t b = 0; b < B; ++b) {
                ni::Tensor lp = model.forward(base, &caches[static_cast<size_t>(b)]);  // prefill (untimed)
                tok[static_cast<size_t>(b)] = argmax_row(lp, lp.size(0) - 1, vocab);
                ptrs.push_back(&caches[static_cast<size_t>(b)]);
            }
            Clock::time_point t0 = Clock::now();
            for (int step = 0; step < STEPS; ++step) {
                ni::Tensor lb = model.forward_batch(tok, ptrs);
                for (int64_t b = 0; b < B; ++b) tok[static_cast<size_t>(b)] = argmax_row(lb, b, vocab);
            }
            const double dt = secs(t0, Clock::now());
            const double tps = double(B) * STEPS / dt;
            if (B == 1) base_tps = tps;
            std::printf("  %5lld  %6.1f   %5.1fx\n", (long long)B, tps, tps / base_tps);
        }

        std::printf(parity_ok ? "\nrun_batch: ok\n" : "\nrun_batch: FAIL\n");
        return parity_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_batch: exception: %s\n", e.what());
        return 1;
    }
}
