// P2 — long-context decode sweep on 1.5B. The G-track parked Flash-Decoding (split-KV, G5g) and the
// paged cache as levers that GROW with context: split adds decode parallelism (nq=H warps at sq=1 leave
// the GPU idle; split fans them to nq*num_splits) and paging drops the contiguous cache's O(ctx) cat_seq
// copy. On 0.5B (head_dim 64, KV small) their end-to-end decode win was modest and, past a point, the
// contiguous cache OOMed. This bench RE-RUNS the ctx sweep on 1.5B (head_dim 128, KV ~2.3x/token) to
// find WHERE paged+split now dominates — the P2 question.
//
// It loads the model ONCE and, for each context C, measures end-to-end decode tok/s under the four
// configs {contiguous, paged} x {split off, on}: prefill C tokens (untimed), then time a window of
// single-token decode steps (the KV sits at ~C throughout). The device pool is trimmed between configs
// so the contiguous cache's growing cat_seq buffers can't accumulate across the sweep, and an OOM on one
// config (expected for the contiguous cache at long ctx) is caught and reported, not fatal — the paged
// configs keep going. Correctness gate: every config's greedy tokens must match the paged split-off
// reference (paged is bit-identical to contiguous per run_cuda_paged; split reorders the reduction but
// greedy argmax is robust — the CLAUDE.md GPU bar of tolerance + tokens).
//
//   python ../tools/export_weights.py weights/qwen2.5-1.5b
//   ./build-cuda/run_cuda_ctx_sweep weights/qwen2.5-1.5b [window] [ctx1 ctx2 ...]
//   NI_FP16W=1 ./build-cuda/run_cuda_ctx_sweep weights/qwen2.5-1.5b   # fp16 weights (fit longer ctx)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "model.hpp"
#include "parity_util.hpp"

using namespace ni;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

struct Result {
    double decode_tps = 0.0;      // window / decode_seconds; -1 if OOM
    std::vector<int64_t> toks;    // the greedy tokens decoded (for the correctness gate)
    bool oom = false;
    bool skip = false;            // config not run (paged_only) → printed "skip", excluded from the gate
};

// Prefill `prompt` (untimed), then greedily decode `window` tokens through the chosen cache/kernel path,
// timing only the decode loop. use_split flips Flash-Decoding on; the split_count gate (cuda_runtime.cu)
// then engages at long context and no-ops at short. Catches OOM (the contiguous cat_seq at long ctx) so
// the sweep continues; the caller trims the pool afterward to reclaim whatever was retained.
static Result measure(const Model& model, const std::vector<int64_t>& prompt, int64_t window,
                      int64_t vocab, bool use_paged, bool use_split) {
    cuda_policy().use_split_attn = use_split;
    Result r;
    try {
        const int64_t ctx = static_cast<int64_t>(prompt.size());
        const int64_t max_seq = ctx + window + 8;
        std::unique_ptr<KVCacheBase> cont;
        std::unique_ptr<CudaBlockPool> pool;
        std::unique_ptr<CudaPagedKVCache> paged;
        KVCacheBase* cache = nullptr;
        if (use_paged) {
            const Config& c = model.config();
            const int64_t block_size = 16;
            const int64_t num_blocks = (max_seq + block_size - 1) / block_size + 4;
            pool = std::make_unique<CudaBlockPool>(c.num_layers, c.num_kv_heads, c.head_dim, block_size,
                                                   num_blocks);
            paged = std::make_unique<CudaPagedKVCache>(pool.get());
            cache = paged.get();
        } else {
            cont = model.make_kv_cache(max_seq);
            cache = cont.get();
        }
        Tensor l = model.forward(prompt, cache);  // prefill (untimed)
        int64_t next = argmax_row(l, l.size(0) - 1, vocab);
        r.toks.reserve(static_cast<size_t>(window));
        Clock::time_point t0 = Clock::now();
        for (int64_t d = 0; d < window; ++d) {
            r.toks.push_back(next);
            Tensor ld = model.forward({next}, cache);
            next = argmax_row(ld, 0, vocab);
        }
        r.decode_tps = static_cast<double>(window) / secs(t0, Clock::now());
    } catch (const std::exception& e) {
        r.oom = true;
        r.decode_tps = -1.0;
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_ctx_sweep <weights_dir> [window] [ctx1 ctx2 ...]\n");
        return 2;
    }
    const std::string dir = argv[1];
    int64_t window = 32;
    std::vector<int64_t> ctxs;
    try {
        if (argc > 2) window = std::stoll(argv[2]);
        for (int i = 3; i < argc; ++i) ctxs.push_back(std::stoll(argv[i]));
    } catch (const std::exception& e) {
        std::printf("bad argument: %s\n", e.what());
        return 2;
    }
    if (window < 1) {
        std::printf("need window >= 1\n");
        return 2;
    }
    if (ctxs.empty()) ctxs = {512, 1024, 2048, 4096, 8192, 16384};

    try {
        if (!cuda_available()) {
            std::printf("run_cuda_ctx_sweep: no CUDA device visible — skipping\n");
            return 0;
        }
        // NI_FP16W honored like run_cuda_decode_bench (must be set before the Model is built — the
        // conversion happens at the once-per-load upload). fp16 is golden-lossless on 1.5B (P1) and
        // frees ~3 GB, which is what lets the top of the sweep fit alongside a long-context KV.
        BackendConfig cfg;
        if (const char* e = std::getenv("NI_FP16W")) cfg.fp16_weights = (e[0] == '1');
        // NI_PAGED_ONLY=1 skips the two CONTIGUOUS configs. The CUDA contiguous cache grows by an
        // un-reusing cat_seq (a distinct-size buffer every step), so decoding it at long context is an
        // O(ctx)-per-step crawl that OOMs after a handful of steps regardless of weight dtype — that
        // non-scaling IS the P2 finding, not a number worth waiting minutes to (not) measure. So run the
        // long-context rows paged-only; measure contiguous only where it fits cheaply (short/mid ctx).
        bool paged_only = false;
        if (const char* e = std::getenv("NI_PAGED_ONLY")) paged_only = (e[0] == '1');
        Model model(dir, QuantMode::None, Device::CUDA, cfg);
        const int64_t vocab = model.config().vocab_size;

        // Synthesize each prompt by cycling the reference ids (any valid ids decode the same shape).
        std::vector<int64_t> seed;
        try {
            seed = read_ids(dir + "/ref_ids.txt");
        } catch (...) {
        }
        if (seed.empty()) seed = {1, 2, 3, 4, 5};
        auto make_prompt = [&](int64_t n) {
            std::vector<int64_t> p(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) p[static_cast<size_t>(i)] = seed[static_cast<size_t>(i % (int64_t)seed.size())];
            return p;
        };

        // Warm the CUDA context / JIT / device pool once so the first timed cell isn't cold-biased
        // (decode kernels are shape-stable at sq=1, so one warmup covers every context's decode path).
        { Result w = measure(model, make_prompt(64), 4, vocab, true, true); (void)w; device_pool_trim(); }

        std::printf("run_cuda_ctx_sweep: %s  weights=%s  window=%lld  (CUDA backend)\n", dir.c_str(),
                    cfg.fp16_weights ? "fp16 (G5d)" : "fp32", (long long)window);
        std::printf("decode tok/s at context C (prefill C, time %lld decode steps):\n\n", (long long)window);
        std::printf("%8s %12s %12s %12s %12s %12s %8s\n", "ctx", "contig-off", "contig-on", "paged-off",
                    "paged-on", "pg+spl/base", "tokens");
        std::printf("%8s %12s %12s %12s %12s %12s %8s\n", "----", "----------", "---------", "---------",
                    "--------", "-----------", "------");

        bool all_ok = true;
        for (int64_t ctx : ctxs) {
            std::vector<int64_t> prompt = make_prompt(ctx);
            // Order matters: run paged split-off FIRST so it is the correctness reference even when the
            // contiguous configs OOM. Trim the pool after each config so the contiguous cat_seq buffers
            // (a distinct size every step) don't pile up in the free list and OOM a later config.
            Result pg_off = measure(model, prompt, window, vocab, /*paged=*/true, /*split=*/false);
            device_pool_trim();
            Result pg_on = measure(model, prompt, window, vocab, true, true);
            device_pool_trim();
            Result ct_off, ct_on;  // default {0, {}, oom=false} → printed as "skip" when paged_only
            if (paged_only) {
                ct_off.skip = ct_on.skip = true;
            } else {
                ct_off = measure(model, prompt, window, vocab, false, false);
                device_pool_trim();
                ct_on = measure(model, prompt, window, vocab, false, true);
                device_pool_trim();
            }

            // Correctness gate: every config that actually ran (not skipped, not OOM) must greedy-match
            // the paged split-off reference (paged==contiguous bit-exact; split reorders but argmax holds).
            const std::vector<int64_t>& ref = pg_off.toks;
            auto matches = [&](const Result& x) { return x.skip || x.oom || x.toks == ref; };
            const bool tok_ok = !pg_off.oom && matches(pg_on) && matches(ct_off) && matches(ct_on);
            all_ok = all_ok && tok_ok;

            // paged+split speedup vs the baseline (contiguous split-off — the pre-P2 default path).
            const double base = (ct_off.oom || ct_off.skip) ? -1.0 : ct_off.decode_tps;
            const double speedup = (base > 0 && pg_on.decode_tps > 0) ? pg_on.decode_tps / base : -1.0;
            auto cell = [](const Result& x, char* buf) {
                if (x.skip) std::snprintf(buf, 16, "%s", "skip");
                else if (x.oom) std::snprintf(buf, 16, "%s", "OOM");
                else std::snprintf(buf, 16, "%.1f", x.decode_tps);
                return buf;
            };
            char b1[16], b2[16], b3[16], b4[16], sp[16];
            if (speedup > 0) std::snprintf(sp, 16, "%.2fx", speedup);
            else std::snprintf(sp, 16, "%s", "n/a");
            std::printf("%8lld %12s %12s %12s %12s %12s %8s\n", (long long)ctx, cell(ct_off, b1),
                        cell(ct_on, b2), cell(pg_off, b3), cell(pg_on, b4), sp, tok_ok ? "MATCH" : "DIFF");
        }
        cuda_policy().use_split_attn = false;
        std::printf("\nrun_cuda_ctx_sweep: %s\n", all_ok ? "ok" : "FAIL (token mismatch)");
        return all_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_ctx_sweep: exception: %s\n", e.what());
        return 1;
    }
}
