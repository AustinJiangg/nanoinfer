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
#include <chrono>
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

        // --- 4. Throughput: paged attend (block-indexed) vs contiguous (gather +
        //     repeat_kv). Same projections; the difference is the K/V read path. The
        //     paged kernel skips the per-step gather and the n_rep-fold KV copy, so
        //     the gap widens with context length. ---
        {
            using Clock = std::chrono::steady_clock;
            const int64_t P = 64, D = 64;  // prefill P tokens, time D decode steps
            std::vector<int64_t> prompt(static_cast<size_t>(P));
            for (int64_t i = 0; i < P; ++i) prompt[static_cast<size_t>(i)] = base[i % base.size()];

            auto bench = [&](ni::KVCacheBase* cache) {
                model.forward(prompt, cache);  // prefill (untimed)
                int64_t tok = base[0];
                Clock::time_point t0 = Clock::now();
                for (int64_t s = 0; s < D; ++s)
                    tok = argmax_row(model.forward({tok}, cache), 0, vocab);
                return std::chrono::duration<double>(Clock::now() - t0).count();
            };

            { ni::KVCache warm = model.make_cache(P + 2); model.forward(prompt, &warm); }
            ni::KVCache cont = model.make_cache(P + D + 1);
            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, 16, 64);
            ni::PagedKVCache paged(&pool);
            const double cont_s = bench(&cont);
            const double paged_s = bench(&paged);
            std::printf("\ndecode throughput (prefill %lld, %lld steps, context -> %lld):\n",
                        (long long)P, (long long)D, (long long)(P + D));
            std::printf("  contiguous (gather + repeat_kv): %.1f tok/s\n", D / cont_s);
            std::printf("  paged (block-indexed attend)   : %.1f tok/s  (%.2fx)\n",
                        D / paged_s, cont_s / paged_s);
        }

        // --- 5. Prefix sharing (RadixAttention): a sequence seeded with another's
        //     prefix blocks, then prefilling only its suffix, must match a full fresh
        //     prefill bit-for-bit (the shared KV equals recomputing it), and the
        //     shared blocks are refcounted (held by both sequences). ---
        {
            const int64_t BS = 4;
            std::vector<int64_t> prompt(16);
            for (int64_t i = 0; i < 16; ++i) prompt[static_cast<size_t>(i)] = base[i % base.size()];
            const int64_t SHARED = 2, shared_len = SHARED * BS;  // share 2 blocks = 8 tok

            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, BS, 64);
            // A prefills only the shared prefix; its blocks then hold that prefix's KV.
            ni::PagedKVCache a(&pool);
            model.forward(std::vector<int64_t>(prompt.begin(), prompt.begin() + shared_len), &a);
            // B shares A's blocks and prefills only the suffix.
            ni::PagedKVCache b(&pool);
            b.share_prefix(a.block_table(), shared_len);
            ni::Tensor lb = model.forward(
                std::vector<int64_t>(prompt.begin() + shared_len, prompt.end()), &b);
            // C is the reference: a full fresh prefill of the whole prompt.
            ni::PagedKVCache c(&pool);
            ni::Tensor lc = model.forward(prompt, &c);

            double d = 0.0;  // compare the last-position logits (both at position 15)
            for (int64_t j = 0; j < vocab; ++j)
                d = std::max(d, std::fabs(double(lb.at(lb.size(0) - 1, j)) -
                                          double(lc.at(lc.size(0) - 1, j))));
            const int64_t rc = pool.refcount(a.block_table()[0]);
            const bool pass = (d == 0.0) && (rc == 2) &&
                              argmax_row(lb, lb.size(0) - 1, vocab) ==
                                  argmax_row(lc, lc.size(0) - 1, vocab);
            ok = ok && pass;
            std::printf("\nprefix sharing: shared %lld blocks (%lld tok) + %lld suffix tok; "
                        "last-row max|diff|=%g shared-block refcount=%lld -> %s\n",
                        (long long)SHARED, (long long)shared_len,
                        (long long)(16 - shared_len), d, (long long)rc,
                        pass ? "BIT-IDENTICAL" : "MISMATCH");
        }

        // --- 6. Rollback (S1): truncate(L) + replay is bit-identical to a cache that only
        //     ever decoded the accepted tokens. Speculative decode's verify forward writes a
        //     tentative tail; truncate discards the rejected part. Contiguous just moves the
        //     length pointer; paged FREES the tail blocks — which the next forward re-allocates
        //     and reuses, so a small block_size + a tail longer than the accepted tail exercises
        //     that reuse. Checked for BOTH cache types; both must equal the never-had-a-tail run
        //     to the last bit (the retained K/V is untouched, the stale tail is overwritten). ---
        {
            const int64_t BS = 4;
            const size_t plen = std::min<size_t>(base.size(), 6);
            std::vector<int64_t> prompt(base.begin(), base.begin() + plen);
            // A long "rejected" tentative tail and a shorter "accepted" continuation (arbitrary
            // token positions — a draft guesses wrong, so the rejected values never matter).
            const std::vector<int64_t> tail = {40, 100, 12095, 785, 11, 42, 7};
            const std::vector<int64_t> accepted = {40, 100, 999};

            // roll: prefill, write the tentative tail, truncate back, replay the accepted tail.
            auto roll = [&](ni::KVCacheBase* c) {
                model.forward(prompt, c);
                const int64_t L = c->length();
                model.forward(tail, c);       // tentative K/V for the rejected tail
                c->truncate(L);               // roll back to the confirmed prefix
                return model.forward(accepted, c);
            };
            // only: prefill, then decode ONLY the accepted tail — the reference state.
            auto only = [&](ni::KVCacheBase* c) {
                model.forward(prompt, c);
                return model.forward(accepted, c);
            };

            ni::KVCache rc = model.make_cache((int64_t)(plen + tail.size() + accepted.size() + 1));
            ni::KVCache oc = model.make_cache((int64_t)(plen + accepted.size() + 1));
            const double dc = max_abs_diff(roll(&rc), only(&oc));

            ni::BlockPool pool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, BS, /*num_blocks=*/64);
            double dp;
            {
                ni::PagedKVCache rp(&pool), op(&pool);
                dp = max_abs_diff(roll(&rp), only(&op));
            }  // both paged caches finish -> every block must return to the pool
            const bool blocks_recovered = pool.free_blocks() == pool.num_blocks();

            const bool pass = (dc == 0.0) && (dp == 0.0) && blocks_recovered;
            ok = ok && pass;
            std::printf("\nrollback (S1): truncate+replay == decode-only  contiguous max|diff|=%g  "
                        "paged max|diff|=%g  blocks recovered=%s -> %s\n",
                        dc, dp, blocks_recovered ? "yes" : "no",
                        pass ? "BIT-IDENTICAL" : "MISMATCH");
        }

        std::printf(ok ? "\nrun_paged: ok\n" : "\nrun_paged: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_paged: exception: %s\n", e.what());
        return 1;
    }
}
