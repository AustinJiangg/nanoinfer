// Paged KV cache parity + pool utilization (C++ stage F8b).
//
// Parity: the paged cache (K/V in fixed-size blocks via a block table) must produce
// EXACTLY the same logits as the contiguous cache — its update() gathers the filled
// prefix into the same contiguous [n_kv, len, head_dim] the attention kernel already
// consumes, so paging changes only WHERE the K/V live, not the math. Checked for the
// single-sequence path (forward) and the batched path (forward_batch), with a small
// block_size so prompts span several blocks (exercising allocation across boundaries).
//
// Utilization: a small pool shared across sequences — allocate, grow, free, and watch
// blocks return to the pool for reuse (the win paging buys over per-sequence [max_seq]
// preallocation). A timing/illustration tool gated on an exported model.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   ./build/run_paged weights/qwen2.5-0.5b
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.hpp"
#include "model.hpp"
#include "paged.hpp"
#include "parity_util.hpp"

// Max abs diff between two equally-shaped logit tensors.
static double max_abs_diff(const ni::Tensor& a, const ni::Tensor& b) {
    double m = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i)
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_paged <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir);
        const ni::Config& cfg = model.config();
        const int64_t vocab = cfg.vocab_size;
        std::vector<int64_t> base = read_ids(dir + "/ref_ids.txt");
        if (base.empty()) {
            std::printf("run_paged: ref_ids.txt is empty\n");
            return 2;
        }

        const int64_t BLOCK = 4;   // small on purpose: prompts span several blocks
        const int STEPS = 16;
        bool ok = true;

        // --- 1. Single-sequence: paged forward == contiguous forward ---
        {
            ni::KVCache cont = model.make_cache(static_cast<int64_t>(base.size()) + STEPS + 1);
            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, BLOCK, /*num_blocks=*/64);
            ni::PagedKVCache paged(&pool);

            double maxd = 0.0;
            int tok_mism = 0;
            ni::Tensor lc = model.forward(base, &cont);
            ni::Tensor lp = model.forward(base, &paged);   // prefill both
            maxd = std::max(maxd, max_abs_diff(lc, lp));
            int64_t tc = argmax_row(lc, lc.size(0) - 1, vocab);
            int64_t tp = argmax_row(lp, lp.size(0) - 1, vocab);
            if (tc != tp) ++tok_mism;
            int64_t next = tc;
            for (int s = 0; s < STEPS; ++s) {
                ni::Tensor dc = model.forward({next}, &cont);
                ni::Tensor dp = model.forward({next}, &paged);
                maxd = std::max(maxd, max_abs_diff(dc, dp));
                if (argmax_row(dc, 0, vocab) != argmax_row(dp, 0, vocab)) ++tok_mism;
                next = argmax_row(dc, 0, vocab);
            }
            const bool pass = (maxd == 0.0) && (tok_mism == 0);
            ok = ok && pass;
            std::printf("single : block_size=%lld  prefill+%d decode  max|diff|=%g  "
                        "token mismatches=%d  blocks=%lld -> %s\n",
                        (long long)BLOCK, STEPS, maxd, tok_mism, (long long)paged.num_blocks(),
                        pass ? "BIT-IDENTICAL" : "MISMATCH");
        }

        // --- 2. Batched: forward_batch over paged caches == over contiguous caches ---
        {
            std::vector<std::vector<int64_t>> prompts = {
                base,
                std::vector<int64_t>(base.begin(), base.begin() + std::max<size_t>(1, base.size() - 1)),
                std::vector<int64_t>(base.begin(), base.begin() + std::max<size_t>(1, base.size() / 2)),
            };
            const int64_t N = static_cast<int64_t>(prompts.size());
            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, BLOCK, 256);

            std::vector<ni::KVCache> cont;
            std::vector<std::unique_ptr<ni::PagedKVCache>> paged;
            std::vector<ni::KVCacheBase*> cont_ptrs, paged_ptrs;
            std::vector<int64_t> cur(static_cast<size_t>(N));
            for (int64_t s = 0; s < N; ++s)
                cont.push_back(model.make_cache(static_cast<int64_t>(prompts[s].size()) + STEPS + 1));
            for (int64_t s = 0; s < N; ++s) paged.push_back(std::make_unique<ni::PagedKVCache>(&pool));
            for (int64_t s = 0; s < N; ++s) {
                model.forward(prompts[s], &cont[s]);
                ni::Tensor lp = model.forward(prompts[s], paged[s].get());
                cur[s] = argmax_row(lp, lp.size(0) - 1, vocab);
                cont_ptrs.push_back(&cont[s]);
                paged_ptrs.push_back(paged[s].get());
            }

            double maxd = 0.0;
            int tok_mism = 0;
            for (int step = 0; step < STEPS; ++step) {
                ni::Tensor bc = model.forward_batch(cur, cont_ptrs);
                ni::Tensor bp = model.forward_batch(cur, paged_ptrs);
                maxd = std::max(maxd, max_abs_diff(bc, bp));
                std::vector<int64_t> next(static_cast<size_t>(N));
                for (int64_t s = 0; s < N; ++s) {
                    if (argmax_row(bc, s, vocab) != argmax_row(bp, s, vocab)) ++tok_mism;
                    next[static_cast<size_t>(s)] = argmax_row(bc, s, vocab);
                }
                cur = next;
            }
            const bool pass = (maxd == 0.0) && (tok_mism == 0);
            ok = ok && pass;
            std::printf("batched: %lld paged seqs x %d steps  max|diff|=%g  token mismatches=%d "
                        " pool used=%lld/%lld -> %s\n",
                        (long long)N, STEPS, maxd, tok_mism, (long long)pool.used_blocks(),
                        (long long)pool.num_blocks(), pass ? "BIT-IDENTICAL" : "MISMATCH");
        }

        // --- 3. Utilization: a small shared pool, blocks freed and reused ---
        {
            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, BLOCK, /*num_blocks=*/8);
            std::printf("\npool utilization (block_size=%lld, %lld blocks):\n",
                        (long long)BLOCK, (long long)pool.num_blocks());
            std::printf("  start            free=%lld\n", (long long)pool.free_blocks());
            {
                ni::PagedKVCache a(&pool);
                model.forward(base, &a);  // fills ceil(len/block) blocks
                std::printf("  seq A prefilled  free=%lld (A uses %lld)\n",
                            (long long)pool.free_blocks(), (long long)a.num_blocks());
                {
                    ni::PagedKVCache b(&pool);
                    model.forward(base, &b);
                    std::printf("  seq B prefilled  free=%lld (B uses %lld)\n",
                                (long long)pool.free_blocks(), (long long)b.num_blocks());
                }  // B finishes -> its blocks return to the pool
                std::printf("  seq B freed      free=%lld (reusable by the next request)\n",
                            (long long)pool.free_blocks());
            }
            std::printf("  seq A freed      free=%lld\n", (long long)pool.free_blocks());
            ok = ok && (pool.free_blocks() == pool.num_blocks());
        }

        std::printf(ok ? "\nrun_paged: ok\n" : "\nrun_paged: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_paged: exception: %s\n", e.what());
        return 1;
    }
}
