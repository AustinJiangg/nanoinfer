// G4a parity + throughput: batched decode on the GPU (forward_batch with the CUDA
// backend). The projection GEMMs fuse over the N rows into one big matmul — exactly
// what a GPU wants — while attention stays a per-sequence loop over each sequence's own
// CudaKVCache. Row s of a forward_batch result must equal a standalone forward of that
// sequence's token, bit-for-bit, because the linear kernel computes each output row
// independently (so batching cannot change any row) and attention is per-sequence.
//
// Throughput: decode tok/s vs batch size — on the GPU the batched GEMM should scale far
// better than batch-1, since one launch over N rows beats N launches over 1 row.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   ./build/run_cuda_batch weights/qwen2.5-0.5b
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cuda/cuda.hpp"
#include "model.hpp"
#include "parity_util.hpp"

using namespace ni;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_batch <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        if (!cuda_available()) {
            std::printf("run_cuda_batch: no CUDA device visible — skipping\n");
            return 0;
        }
        Model model(dir, QuantMode::None, Device::CUDA);
        const int64_t vocab = model.config().vocab_size;
        std::printf("engine: CUDA backend\n");

        std::vector<int64_t> base = read_ids(dir + "/ref_ids.txt");
        if (base.empty()) {
            std::printf("run_cuda_batch: ref_ids.txt is empty\n");
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

        std::vector<std::unique_ptr<KVCacheBase>> single, batched;
        std::vector<KVCacheBase*> batched_ptrs;
        std::vector<int64_t> cur(static_cast<size_t>(N));
        for (int64_t s = 0; s < N; ++s) {
            const int64_t cap = static_cast<int64_t>(prompts[s].size()) + STEPS + 1;
            single.push_back(model.make_kv_cache(cap));
            batched.push_back(model.make_kv_cache(cap));
        }
        for (int64_t s = 0; s < N; ++s) {
            model.forward(prompts[s], single[s].get());
            Tensor lp = model.forward(prompts[s], batched[s].get());  // identical prefill
            cur[s] = argmax_row(lp, lp.size(0) - 1, vocab);
            batched_ptrs.push_back(batched[s].get());
        }

        double maxd = 0.0;
        int tok_mism = 0;
        for (int step = 0; step < STEPS; ++step) {
            Tensor lb = model.forward_batch(cur, batched_ptrs);  // [N, vocab]
            std::vector<int64_t> next(static_cast<size_t>(N));
            for (int64_t s = 0; s < N; ++s) {
                Tensor ls = model.forward({cur[static_cast<size_t>(s)]}, single[s].get());  // [1, vocab]
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

        // --- Throughput: decode tok/s vs batch size (the batched-GEMM win on GPU) ---
        std::printf("\nbatched decode throughput (greedy, %d steps each):\n", STEPS);
        std::printf("  batch   tok/s   speedup\n");
        double base_tps = 0.0;
        for (int64_t B : {1, 2, 4, 8, 16, 32}) {
            std::vector<std::unique_ptr<KVCacheBase>> caches;
            std::vector<KVCacheBase*> ptrs;
            std::vector<int64_t> tok(static_cast<size_t>(B));
            const int64_t cap = static_cast<int64_t>(base.size()) + STEPS + 1;
            for (int64_t b = 0; b < B; ++b) caches.push_back(model.make_kv_cache(cap));
            for (int64_t b = 0; b < B; ++b) {
                Tensor lp = model.forward(base, caches[static_cast<size_t>(b)].get());  // prefill (untimed)
                tok[static_cast<size_t>(b)] = argmax_row(lp, lp.size(0) - 1, vocab);
                ptrs.push_back(caches[static_cast<size_t>(b)].get());
            }
            model.forward_batch(tok, ptrs);  // warm before timing
            Clock::time_point t0 = Clock::now();
            for (int step = 0; step < STEPS; ++step) {
                Tensor lb = model.forward_batch(tok, ptrs);
                for (int64_t b = 0; b < B; ++b) tok[static_cast<size_t>(b)] = argmax_row(lb, b, vocab);
            }
            const double dt = secs(t0, Clock::now());
            const double tps = double(B) * STEPS / dt;
            if (B == 1) base_tps = tps;
            std::printf("  %5lld  %6.1f   %5.1fx\n", (long long)B, tps, tps / base_tps);
        }

        std::printf(parity_ok ? "\nrun_cuda_batch: ok\n" : "\nrun_cuda_batch: FAIL\n");
        return parity_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_batch: exception: %s\n", e.what());
        return 1;
    }
}
