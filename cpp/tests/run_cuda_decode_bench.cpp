// G5b end-to-end: confirm the warp-GEMV's decode win at the MODEL level (not just the
// linear microbench). Loads the GPU model once and decodes the same way twice — forcing
// the naive GEMM, then the GEMV (g_cuda_force_naive_gemm) — so the only thing that changes
// between the two runs is the decode kernel. Reports prefill/decode tok/s for both and the
// decode speedup. Prefill is naive in BOTH runs (m>16 won't hit the GEMV until G5c), a
// built-in control: it should barely move while decode jumps.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   ./build/run_cuda_decode_bench weights/qwen2.5-0.5b [prefill_len] [decode_len]
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

// Prefill `prompt`, then greedily decode `decode_len` tokens, timing each phase. The decode
// kernel is selected by `force_naive` — the only thing that differs between the two calls.
struct Timing {
    double prefill_s, decode_s;
};
static Timing run_one(const Model& model, const std::vector<int64_t>& prompt, int64_t decode_len,
                      int64_t vocab, bool force_naive, bool use_paged) {
    g_cuda_force_naive_gemm = force_naive;
    const int64_t max_seq = static_cast<int64_t>(prompt.size()) + decode_len + 8;
    // Contiguous (G3, default) or paged (G4b) KV cache. The contiguous cache's cat_seq copies the whole
    // history each step (O(ctx)); the paged cache writes only the new token into a block — so NI_PAGED=1
    // is where the Flash-Decoding (split-KV) decode win shows up undiluted.
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
    Clock::time_point t0 = Clock::now();
    Tensor l = model.forward(prompt, cache);  // prefill (naive in both runs: m>16)
    Clock::time_point t1 = Clock::now();
    int64_t next = argmax_row(l, l.size(0) - 1, vocab);
    Clock::time_point t2 = Clock::now();
    for (int64_t d = 0; d < decode_len; ++d) {  // decode one token at a time (m=1)
        Tensor ld = model.forward({next}, cache);
        next = argmax_row(ld, 0, vocab);
    }
    Clock::time_point t3 = Clock::now();
    return {secs(t0, t1), secs(t2, t3)};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_decode_bench <weights_dir> [prefill_len] [decode_len]\n");
        return 2;
    }
    const std::string dir = argv[1];
    int64_t prefill_len = 32, decode_len = 64;
    try {
        if (argc > 2) prefill_len = std::stoll(argv[2]);
        if (argc > 3) decode_len = std::stoll(argv[3]);
    } catch (const std::exception& e) {
        std::printf("bad length argument: %s\n", e.what());
        return 2;
    }
    if (prefill_len < 1 || decode_len < 1) {
        std::printf("need prefill_len >= 1 and decode_len >= 1\n");
        return 2;
    }

    try {
        if (!cuda_available()) {
            std::printf("run_cuda_decode_bench: no CUDA device visible — skipping\n");
            return 0;
        }
        // NI_FP16W=1 uploads the layer weights as fp16 (G5d) — must be set before the Model
        // is built, since the conversion happens at the once-per-load upload.
        if (const char* e = std::getenv("NI_FP16W")) g_cuda_fp16_weights = (e[0] == '1');
        // NI_NAIVE_ATTN=1 forces the naive one-thread-per-query attention (G5e A/B): hold the GEMM
        // path fixed and toggle only attention to isolate its prefill/decode contribution.
        if (const char* e = std::getenv("NI_NAIVE_ATTN")) g_cuda_force_naive_attn = (e[0] == '1');
        // NI_TILE_ATTN=1 opts into the shared-memory K/V tiled kernel at prefill (G5f A/B): isolate
        // the tiling's prefill contribution (bit-identical output either way; default is non-tiled).
        if (const char* e = std::getenv("NI_TILE_ATTN")) g_cuda_use_tiled_attn = (e[0] == '1');
        // NI_SPLIT_ATTN=1 opts into Flash-Decoding / split-KV attention (G5g): the long-context decode
        // lever. A/B the decode tok/s by running this bench at a long context with and without it set.
        if (const char* e = std::getenv("NI_SPLIT_ATTN")) g_cuda_use_split_attn = (e[0] == '1');
        // NI_PAGED=1 decodes through the paged KV cache (G4b) instead of the contiguous one — no
        // O(ctx) cat_seq copy per step, so it pairs with NI_SPLIT_ATTN to show the undiluted decode win.
        bool use_paged = false;
        if (const char* e = std::getenv("NI_PAGED")) use_paged = (e[0] == '1');
        Model model(dir, QuantMode::None, Device::CUDA);
        const int64_t vocab = model.config().vocab_size;
        std::printf("layer weights: %s; KV cache: %s\n", g_cuda_fp16_weights ? "fp16 (G5d)" : "fp32",
                    use_paged ? "paged (G4b)" : "contiguous (G3)");

        // Synthesize a prompt by cycling the reference ids (any valid ids time the same).
        std::vector<int64_t> seed;
        try {
            seed = read_ids(dir + "/ref_ids.txt");
        } catch (...) {
        }
        if (seed.empty()) seed = {1, 2, 3, 4, 5};
        std::vector<int64_t> prompt(static_cast<size_t>(prefill_len));
        for (int64_t i = 0; i < prefill_len; ++i)
            prompt[static_cast<size_t>(i)] = seed[static_cast<size_t>(i) % seed.size()];

        std::printf("run_cuda_decode_bench: %s  prefill=%lld decode=%lld  (CUDA backend)\n",
                    dir.c_str(), (long long)prefill_len, (long long)decode_len);

        run_one(model, prompt, 4, vocab, false, use_paged);  // warm CUDA context / reach steady state

        const Timing naive = run_one(model, prompt, decode_len, vocab, true, use_paged);
        const Timing gemv = run_one(model, prompt, decode_len, vocab, false, use_paged);
        g_cuda_force_naive_gemm = false;

        auto tps = [](int64_t n, double s) { return static_cast<double>(n) / s; };
        std::printf("\n%-8s %14s %14s\n", "kernel", "prefill tok/s", "decode tok/s");
        std::printf("%-8s %14.1f %14.1f\n", "naive", tps(prefill_len, naive.prefill_s),
                    tps(decode_len, naive.decode_s));
        std::printf("%-8s %14.1f %14.1f\n", "GEMV", tps(prefill_len, gemv.prefill_s),
                    tps(decode_len, gemv.decode_s));
        std::printf("\ndecode: %.1f -> %.1f tok/s (%.2fx)   prefill control: %.1f -> %.1f tok/s\n",
                    tps(decode_len, naive.decode_s), tps(decode_len, gemv.decode_s),
                    naive.decode_s / gemv.decode_s, tps(prefill_len, naive.prefill_s),
                    tps(prefill_len, gemv.prefill_s));
        std::printf("run_cuda_decode_bench: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("run_cuda_decode_bench: exception: %s\n", e.what());
        return 1;
    }
}
