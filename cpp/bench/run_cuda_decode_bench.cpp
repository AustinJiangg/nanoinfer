// G5b end-to-end: confirm the warp-GEMV's decode win at the MODEL level (not just the
// linear microbench). Loads the GPU model once and decodes the same way twice — forcing
// the naive GEMM, then the GEMV (cuda_policy().force_naive_gemm) — so the only thing that changes
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

// NI_QEMBED mode (set in main): A/B the weight-only int8 lm_head kernel instead of the layer GEMM.
static bool g_ab_int8_lmhead = false;
// NI_W8A8 mode (set in main): the layer projections are int8×int8 DP4A (CudaW8A8Weight → cuda_linear_w8a8),
// which force_naive_gemm can't reach (that only toggles CudaBackend::linear's fp32/fp16 path). So the
// W8A8 decode A/B is the tiled DP4A GEMM (P1's 0.64× decode regime) vs the new warp-per-output GEMV.
static bool g_ab_w8a8 = false;

// Prefill `prompt`, then greedily decode `decode_len` tokens, timing each phase. `force_slow` selects
// the SLOW variant of the kernel under test — the only thing that differs between the two calls.
struct Timing {
    double prefill_s, decode_s;
};
static Timing run_one(const Model& model, const std::vector<int64_t>& prompt, int64_t decode_len,
                      int64_t vocab, bool force_slow, bool use_paged,
                      std::vector<int64_t>* out_toks = nullptr) {
    // Default (G5b): the layer-projection GEMM (naive vs warp-GEMV). NI_QEMBED: the int8 lm_head
    // (prefill-tiled vs decode GEMV) — layers stay on the GEMV so only the lm_head kernel changes.
    if (g_ab_int8_lmhead)
        cuda_policy().force_tiled_q8 = force_slow;
    else if (g_ab_w8a8)
        cuda_policy().force_tiled_w8a8 = force_slow;  // W8A8 layer projections: tiled (slow) vs GEMV
    else
        cuda_policy().force_naive_gemm = force_slow;
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
    Tensor l = model.forward(prompt, cache);  // prefill: the control (kernel under test unchanged at m>16)
    Clock::time_point t1 = Clock::now();
    int64_t next = argmax_row(l, l.size(0) - 1, vocab);
    Clock::time_point t2 = Clock::now();
    for (int64_t d = 0; d < decode_len; ++d) {  // decode one token at a time (m=1)
        if (out_toks) out_toks->push_back(next);
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
        BackendConfig cfg;
        if (const char* e = std::getenv("NI_FP16W")) cfg.fp16_weights = (e[0] == '1');
        // NI_WMMA=1 routes the prefill (m>16) projection GEMMs through the tensor-core wmma-h kernel
        // (P0: it TIED on 0.5B but wins on 1.5B's wide MLP GEMMs — gate/up 1.61×, down 1.25× isolated).
        // Pairs with NI_FP16W (wmma only wins reading fp16 weight storage, not fp32+per-tile convert).
        if (const char* e = std::getenv("NI_WMMA")) cuda_policy().use_wmma = (e[0] == '1');
        // NI_NAIVE_ATTN=1 forces the naive one-thread-per-query attention (G5e A/B): hold the GEMM
        // path fixed and toggle only attention to isolate its prefill/decode contribution.
        if (const char* e = std::getenv("NI_NAIVE_ATTN")) cuda_policy().force_naive_attn = (e[0] == '1');
        // NI_TILE_ATTN=1 opts into the shared-memory K/V tiled kernel at prefill (G5f A/B): isolate
        // the tiling's prefill contribution (bit-identical output either way; default is non-tiled).
        if (const char* e = std::getenv("NI_TILE_ATTN")) cuda_policy().use_tiled_attn = (e[0] == '1');
        // NI_NO_TILE_ATTN=1 forces tiling OFF — the A/B baseline at head_dim>=128, where tiling is
        // the DEFAULT since P0 (on both KV paths; pairs with NI_PAGED=1 for the paged tiled kernel).
        if (const char* e = std::getenv("NI_NO_TILE_ATTN")) cuda_policy().no_tiled_attn = (e[0] == '1');
        // NI_SPLIT_ATTN=1 opts into Flash-Decoding / split-KV attention (G5g): the long-context decode
        // lever. A/B the decode tok/s by running this bench at a long context with and without it set.
        if (const char* e = std::getenv("NI_SPLIT_ATTN")) cuda_policy().use_split_attn = (e[0] == '1');
        // NI_PAGED=1 decodes through the paged KV cache (G4b) instead of the contiguous one — no
        // O(ctx) cat_seq copy per step, so it pairs with NI_SPLIT_ATTN to show the undiluted decode win.
        bool use_paged = false;
        if (const char* e = std::getenv("NI_PAGED")) use_paged = (e[0] == '1');
        // NI_QEMBED=1 quantizes the tied embedding / lm_head to weight-only int8 (G5d) and shifts the
        // A/B to that lm_head kernel: prefill-tiled vs the decode GEMV (the layer GEMM stays on the
        // GEMV in both runs). Must be set before the Model is built (the quantize happens at load).
        if (const char* e = std::getenv("NI_QEMBED")) {
            g_ab_int8_lmhead = (e[0] == '1');
            cfg.quantize_embed = g_ab_int8_lmhead;
        }
        // NI_W8A8_LMHEAD=1 (backlog follow-up, pairs with NI_QEMBED): route the int8 lm_head's PREFILL
        // through int8×int8 DP4A (the compute win) instead of weight-only int8. Reflected in the prefill
        // tok/s (run with/without to A/B the e2e prefill delta); decode is unaffected (stays weight-only).
        if (const char* e = std::getenv("NI_W8A8_LMHEAD")) cuda_policy().use_w8a8_lmhead = (e[0] == '1');
        // NI_W8A8=1 runs the layer projections as int8×int8 DP4A (P1, the COMPUTE lever, 4:1 MACs):
        // it won isolated on 1.5B's wide GEMMs (gate/up 1.94×, lm_head 2.16× — run_cuda_bench 1.5b),
        // and unlike fp16/wmma (byte levers) it also cuts the projection FLOPs — does it translate
        // e2e where wmma washed out? Combine with NI_QEMBED for the full-int8 model. Must precede the
        // Model build (the quantize happens at the once-per-load upload, like fp16).
        QuantMode qmode = QuantMode::None;
        if (const char* e = std::getenv("NI_W8A8"); e && e[0] == '1') {
            qmode = QuantMode::W8A8;
            g_ab_w8a8 = true;  // A/B the W8A8 layer projections: tiled DP4A (slow) vs decode GEMV
        }
        Model model(dir, qmode, Device::CUDA, cfg);
        const int64_t vocab = model.config().vocab_size;
        std::printf("layer weights: %s%s; KV cache: %s; A/B: %s\n",
                    qmode == QuantMode::W8A8 ? "int8 W8A8 (DP4A)" : (cfg.fp16_weights ? "fp16 (G5d)" : "fp32"),
                    cfg.quantize_embed ? " + int8 embed/lm_head" : "",
                    use_paged ? "paged (G4b)" : "contiguous (G3)",
                    g_ab_int8_lmhead ? "int8 lm_head tiled vs GEMV"
                                     : (g_ab_w8a8 ? "W8A8 layers tiled vs GEMV" : "layer GEMM naive vs GEMV"));

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

        // NI_GRAPH=1 (G6): A/B the CUDA-graph decode against eager paged decode. The graph folds a
        // decode step's ~360 per-op launches into one cudaGraphLaunch. Greedy tokens MUST match (the
        // graph is bit-identical to the eager paged forward); the headline is decode tok/s. Both use
        // the paged cache (the graph needs its fixed block addresses + device-side length/token).
        if (const char* e = std::getenv("NI_GRAPH"); e && e[0] == '1') {
            const Config& c = model.config();
            const int64_t block_size = 16, max_seq = prefill_len + decode_len + 8;
            const int64_t num_blocks = (max_seq + block_size - 1) / block_size + 4;
            auto greedy = [&](bool use_graph, std::vector<int64_t>& toks) -> double {
                CudaBlockPool pool(c.num_layers, c.num_kv_heads, c.head_dim, block_size, num_blocks);
                CudaPagedKVCache cache(&pool);
                Tensor l = model.forward(prompt, &cache);  // prefill runs eager on both paths
                int64_t next = argmax_row(l, l.size(0) - 1, vocab);
                std::unique_ptr<CudaGraphDecoder> dec;
                if (use_graph) dec = std::make_unique<CudaGraphDecoder>(model, cache, vocab);
                Clock::time_point t0 = Clock::now();
                for (int64_t d = 0; d < decode_len; ++d) {
                    toks.push_back(next);
                    Tensor ld = use_graph ? dec->decode(next) : model.forward({next}, &cache);
                    next = argmax_row(ld, 0, vocab);
                }
                return secs(t0, Clock::now());
            };
            std::vector<int64_t> te, tg;
            greedy(false, te);
            greedy(true, tg);  // warm both (CUDA context + graph capture path)
            te.clear();
            tg.clear();
            const double se = greedy(false, te), sg = greedy(true, tg);
            const bool match = (te == tg);
            std::printf("\nCUDA graph decode (G6) vs eager paged decode (decode=%lld):\n",
                        (long long)decode_len);
            std::printf("  eager: %6.1f tok/s   graph: %6.1f tok/s   (%.2fx)\n", decode_len / se,
                        decode_len / sg, se / sg);
            std::printf("  golden: graph greedy tokens %s eager (%zu tokens)\n", match ? "==" : "!=",
                        te.size());
            std::printf("run_cuda_decode_bench: %s\n", match ? "ok" : "FAIL");
            return match ? 0 : 1;
        }

        run_one(model, prompt, 4, vocab, false, use_paged);  // warm CUDA context / reach steady state

        std::vector<int64_t> slow_toks, fast_toks;
        const Timing slow = run_one(model, prompt, decode_len, vocab, true, use_paged, &slow_toks);
        const Timing fast = run_one(model, prompt, decode_len, vocab, false, use_paged, &fast_toks);
        cuda_policy().force_naive_gemm = false;
        cuda_policy().force_tiled_q8 = false;
        cuda_policy().force_tiled_w8a8 = false;

        // Correctness gate for the decode A/B (previously only NI_GRAPH token-checked): the two kernels
        // must emit the same greedy stream. For W8A8 (int32-exact GEMV) and the fp32 warp-GEMV this is
        // BIT-IDENTICAL (max|diff|=0 → the full forward can't diverge); the weight-only q8 GEMV reorders
        // its fp32 sum to ~1e-6, so a knife-edge argmax could differ — informational there, hard for W8A8.
        const bool toks_match = (slow_toks == fast_toks);
        std::printf("decode tokens slow==fast: %s (%zu tokens)\n", toks_match ? "MATCH" : "DIFFER",
                    fast_toks.size());
        if (g_ab_w8a8 && !toks_match) {  // W8A8 GEMV is provably bit-identical — a mismatch is a real bug
            std::printf("run_cuda_decode_bench: FAIL (W8A8 GEMV must be bit-identical to tiled)\n");
            return 1;
        }

        // the slow kernel under test: int8 lm_head / W8A8 layers A/B the tiled GEMM, the fp32 path the naive
        const char* slow_label = (g_ab_int8_lmhead || g_ab_w8a8) ? "tiled" : "naive";
        auto tps = [](int64_t n, double s) { return static_cast<double>(n) / s; };
        std::printf("\n%-8s %14s %14s\n", "kernel", "prefill tok/s", "decode tok/s");
        std::printf("%-8s %14.1f %14.1f\n", slow_label, tps(prefill_len, slow.prefill_s),
                    tps(decode_len, slow.decode_s));
        std::printf("%-8s %14.1f %14.1f\n", "GEMV", tps(prefill_len, fast.prefill_s),
                    tps(decode_len, fast.decode_s));
        std::printf("\ndecode: %.1f -> %.1f tok/s (%.2fx)   prefill control: %.1f -> %.1f tok/s\n",
                    tps(decode_len, slow.decode_s), tps(decode_len, fast.decode_s),
                    slow.decode_s / fast.decode_s, tps(prefill_len, slow.prefill_s),
                    tps(prefill_len, fast.prefill_s));
        std::printf("run_cuda_decode_bench: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("run_cuda_decode_bench: exception: %s\n", e.what());
        return 1;
    }
}
