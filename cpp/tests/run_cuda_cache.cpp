// G3 parity: the device-resident KV cache (CudaKVCache) on the GPU. Two checks:
//   1. cached == uncached, per position (the C3 / run_cache bar) — prefill a prefix,
//      decode the rest one token at a time, plus a multi-token chunk at nonzero length;
//   2. real incremental greedy decode (prefill once, then one new token per step, reusing
//      the cache) reproduces nanoinfer's golden continuation — unlike run_cuda_parity's
//      full-recompute greedy, this is the actual decode path.
//
// Within float tolerance, not bit-identical (the GPU kernels accumulate in float). The
// CPU/nanoinfer reference is the oracle.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_cuda_cache weights/qwen2.5-0.5b
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"  // g_cuda_use_split_attn (Flash-Decoding check, G5g)
#include "model.hpp"
#include "parity_util.hpp"

using namespace ni;

// Max abs diff between row `ra` of `a` and row `rb` of `b` (both [*, vocab]).
static double row_maxdiff(const Tensor& a, int64_t ra, const Tensor& b, int64_t rb, int64_t vocab) {
    double m = 0.0;
    for (int64_t j = 0; j < vocab; ++j) {
        const double d = std::fabs(double(a.at(ra, j)) - double(b.at(rb, j)));
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_cache <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        if (!cuda_available()) {
            std::printf("run_cuda_cache: no CUDA device visible — skipping\n");
            return 0;
        }
        Model model(dir, QuantMode::None, Device::CUDA);
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        std::vector<int64_t> ref_gen = read_ids(dir + "/ref_gen_ids.txt");
        if (ids.size() < 2 || ref_gen.empty()) {
            std::printf("run_cuda_cache: need >= 2 prompt tokens and a ref continuation\n");
            return 2;
        }
        const int64_t seq = static_cast<int64_t>(ids.size());

        // Uncached reference (GPU): row p = logits at position p given ids[0..p].
        Tensor ref = model.forward(ids);
        const int64_t vocab = ref.size(1);

        // 1a. Cached: prefill the first `split` tokens, then decode the rest one at a time.
        const int64_t split = seq / 2 > 0 ? seq / 2 : 1;
        std::unique_ptr<KVCacheBase> cache = model.make_kv_cache(seq + ref_gen.size() + 8);
        std::vector<int64_t> pre(ids.begin(), ids.begin() + split);
        Tensor lp = model.forward(pre, cache.get());
        double maxd = row_maxdiff(lp, split - 1, ref, split - 1, vocab);
        for (int64_t p = split; p < seq; ++p) {
            Tensor ld = model.forward({ids[static_cast<size_t>(p)]}, cache.get());
            maxd = std::fmax(maxd, row_maxdiff(ld, 0, ref, p, vocab));
        }

        // 1b. Chunked: prefill `split`, then the rest as ONE chunk at nonzero length (t>1,
        // start_pos>0) — the path single-token decode never exercises.
        {
            std::unique_ptr<KVCacheBase> c2 = model.make_kv_cache(seq + 8);
            std::vector<int64_t> pre2(ids.begin(), ids.begin() + split);
            model.forward(pre2, c2.get());
            std::vector<int64_t> chunk(ids.begin() + split, ids.end());
            Tensor lc = model.forward(chunk, c2.get());
            for (int64_t r = 0; r < seq - split; ++r)
                maxd = std::fmax(maxd, row_maxdiff(lc, r, ref, split + r, vocab));
        }
        std::printf("cached vs uncached: max|diff|=%g over %lld positions (split=%lld, +chunk)\n",
                    maxd, (long long)seq, (long long)split);

        // 2. Real incremental greedy: prefill the prompt, then one token per step reusing
        // the cache (the decode path), vs nanoinfer's golden continuation.
        std::unique_ptr<KVCacheBase> gen_cache = model.make_kv_cache(seq + ref_gen.size() + 8);
        Tensor pl = model.forward(ids, gen_cache.get());
        int64_t tok = argmax_row(pl, pl.size(0) - 1, vocab);
        std::vector<int64_t> got{tok};
        for (size_t t = 1; t < ref_gen.size(); ++t) {
            Tensor ld = model.forward({tok}, gen_cache.get());  // one cached decode step
            tok = argmax_row(ld, 0, vocab);
            got.push_back(tok);
        }
        print_ids("ref   :", ref_gen);
        print_ids("cuda  :", got);
        int genmism = 0;
        for (size_t i = 0; i < ref_gen.size() && i < got.size(); ++i)
            if (ref_gen[i] != got[i]) ++genmism;
        const bool greedy_ok = got.size() == ref_gen.size() && genmism == 0;
        std::printf("incremental greedy: %s (%d/%zu mismatches)\n", greedy_ok ? "MATCH" : "MISMATCH",
                    genmism, ref_gen.size());

        // 3. Flash-Decoding (G5g): on the real decode path, enabling split-KV must not change the
        // greedy tokens. The golden continuation is only a handful of tokens (context < the split
        // threshold), so we build a long synthetic context (the prompt repeated past 512 tokens, so
        // split_count engages from the first decode step) and require the split-ON token stream to
        // equal the split-OFF (warp-kernel) one — the decode-path analogue of "tiled == non-tiled".
        // Split matches the oracle to ~1e-8, so argmax must be unmoved; a mismatch is a real bug.
        std::vector<int64_t> longctx;
        while (static_cast<int64_t>(longctx.size()) < 512)
            longctx.insert(longctx.end(), ids.begin(), ids.end());
        auto decode_split = [&](bool use_split) {
            g_cuda_use_split_attn = use_split;
            const int steps = 16;
            std::unique_ptr<KVCacheBase> c =
                model.make_kv_cache(static_cast<int64_t>(longctx.size()) + steps + 8);
            Tensor pl = model.forward(longctx, c.get());  // long prefill (split no-ops: sq large)
            int64_t t = argmax_row(pl, pl.size(0) - 1, vocab);
            std::vector<int64_t> out{t};
            for (int s = 1; s < steps; ++s) {  // decode: sq=1, sk>=512 -> split engages when ON
                Tensor ld = model.forward({t}, c.get());
                t = argmax_row(ld, 0, vocab);
                out.push_back(t);
            }
            g_cuda_use_split_attn = false;
            return out;
        };
        std::vector<int64_t> tok_off = decode_split(false);  // warp kernel (default)
        std::vector<int64_t> tok_on = decode_split(true);    // split-KV
        int splitmism = 0;
        for (size_t i = 0; i < tok_off.size() && i < tok_on.size(); ++i)
            if (tok_off[i] != tok_on[i]) ++splitmism;
        const bool split_ok = tok_off.size() == tok_on.size() && splitmism == 0;
        std::printf("flash-decoding: split-on vs split-off greedy %s (%d/%zu mism, ctx=%zu)\n",
                    split_ok ? "MATCH" : "MISMATCH", splitmism, tok_off.size(), longctx.size());

        const bool ok = maxd < 1e-3 && greedy_ok && split_ok;
        std::printf(ok ? "run_cuda_cache: ok\n" : "run_cuda_cache: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_cache: exception: %s\n", e.what());
        return 1;
    }
}
