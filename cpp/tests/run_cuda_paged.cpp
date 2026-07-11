// G4b parity: paged KV cache (CudaBlockPool + CudaPagedKVCache + a paged-attention
// kernel) == contiguous KV cache (CudaKVCache), on the GPU. The two caches read the
// same K/V in the same order — paged just indexes fixed-size blocks via a block table
// instead of a contiguous buffer — so the logits must match bit-for-bit, and both must
// reproduce nanoinfer's golden continuation. This is the GPU sibling of run_paged.cpp.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_cuda_paged weights/qwen2.5-0.5b
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "model.hpp"
#include "parity_util.hpp"

using namespace ni;

// NaN-aware (the G5f lesson): std::fmax(x, NaN) silently returns x, so a NaN-poisoned logit row
// would otherwise vanish from the reduction and the gate would pass on garbage. A NaN diff is
// returned as NaN, which fails every `< tol` / `== 0.0` check downstream.
static double full_maxdiff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i) {
        const double d = std::fabs(double(a[i]) - double(b[i]));
        if (std::isnan(d)) return d;
        m = std::fmax(m, d);
    }
    return m;
}

// Accumulate diffs without fmax's NaN-swallowing: once any operand is NaN, stay NaN.
static double worse(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
    return std::fmax(a, b);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_paged <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        if (!cuda_available()) {
            std::printf("run_cuda_paged: no CUDA device visible — skipping\n");
            return 0;
        }
        Model model(dir, QuantMode::None, Device::CUDA);
        const Config& c = model.config();
        const int64_t vocab = c.vocab_size;
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        std::vector<int64_t> ref_gen = read_ids(dir + "/ref_gen_ids.txt");
        if (ids.size() < 2 || ref_gen.empty()) {
            std::printf("run_cuda_paged: need >= 2 prompt tokens and a ref continuation\n");
            return 2;
        }

        // Two caches for the same model: contiguous (G3) and paged (G4b). A small
        // block_size=4 makes the prompt+continuation span several blocks, so the block
        // table indexing and boundary crossings are actually exercised (prefill alone
        // spills past the first block); 256 blocks is far more than this needs.
        const int64_t block_size = 4, num_blocks = 256;
        std::unique_ptr<KVCacheBase> cont = model.make_kv_cache(ids.size() + ref_gen.size() + 8);
        CudaBlockPool pool(c.num_layers, c.num_kv_heads, c.head_dim, block_size, num_blocks);
        CudaPagedKVCache paged(&pool);

        // Prefill both with the prompt, comparing the full [seq, vocab] logits.
        Tensor lc = model.forward(ids, cont.get());
        Tensor lp = model.forward(ids, &paged);
        double maxd = full_maxdiff(lc, lp);

        // Decode in lockstep, feeding both the same (greedy) token each step.
        int64_t tok = argmax_row(lc, lc.size(0) - 1, vocab);
        std::vector<int64_t> got{tok};
        for (size_t t = 1; t < ref_gen.size(); ++t) {
            Tensor dc = model.forward({tok}, cont.get());
            Tensor dp = model.forward({tok}, &paged);
            maxd = worse(maxd, full_maxdiff(dc, dp));
            tok = argmax_row(dc, 0, vocab);
            got.push_back(tok);
        }

        int genmism = 0;
        for (size_t i = 0; i < ref_gen.size() && i < got.size(); ++i)
            if (ref_gen[i] != got[i]) ++genmism;
        const bool greedy_ok = got.size() == ref_gen.size() && genmism == 0;

        print_ids("ref   :", ref_gen);
        print_ids("cuda  :", got);
        std::printf("paged vs contiguous: max|diff|=%g over %zu decode steps\n", maxd,
                    ref_gen.size());
        std::printf("blocks: %lld used / %lld total (block_size=%lld)\n",
                    (long long)pool.used_blocks(), (long long)pool.num_blocks(),
                    (long long)pool.block_size());
        std::printf("greedy vs golden: %s (%d/%zu mismatches)\n", greedy_ok ? "MATCH" : "MISMATCH",
                    genmism, ref_gen.size());

        // Flash-Decoding (G5g) on the paged path: with split-KV ON, paged must STILL equal contiguous
        // bit-for-bit — the split kernels read the same chunks in the same order, only the K/V address
        // differs (contiguous buffer vs blocks), exactly like the warp kernels above. The golden prompt
        // is too short to engage split_count, so prefill a long synthetic context (>512, split engages
        // from the first decode step) into a fresh pair of caches and compare in lockstep.
        cuda_policy().use_split_attn = true;
        std::vector<int64_t> longctx;
        while (static_cast<int64_t>(longctx.size()) < 512)
            longctx.insert(longctx.end(), ids.begin(), ids.end());
        const int split_steps = 8;
        std::unique_ptr<KVCacheBase> cont2 =
            model.make_kv_cache(static_cast<int64_t>(longctx.size()) + split_steps + 8);
        CudaBlockPool pool2(c.num_layers, c.num_kv_heads, c.head_dim, block_size, num_blocks);
        CudaPagedKVCache paged2(&pool2);
        Tensor lc2 = model.forward(longctx, cont2.get());
        Tensor lp2 = model.forward(longctx, &paged2);
        double splitd = full_maxdiff(lc2, lp2);
        int64_t st = argmax_row(lc2, lc2.size(0) - 1, vocab);
        for (int s = 1; s < split_steps; ++s) {  // decode: sq=1, sk>=512 -> split engages on both paths
            Tensor dc = model.forward({st}, cont2.get());
            Tensor dp = model.forward({st}, &paged2);
            splitd = worse(splitd, full_maxdiff(dc, dp));
            st = argmax_row(dc, 0, vocab);
        }
        cuda_policy().use_split_attn = false;
        const bool split_ok = splitd < 1e-4;
        std::printf("flash-decoding: paged-split == contiguous-split max|diff|=%g over %d steps (ctx=%zu)\n",
                    splitd, split_steps, longctx.size());

        // Paged shared-mem K/V tiling (the backlog mirror of G5f-tiled, default at head_dim>=128
        // since P0): force tiling on (this model may be D=64, where it isn't the default) and re-run
        // prefill on fresh caches, then a verify-shaped multi-query forward (the S0 primitive: t>1
        // onto a populated cache at a non-block-aligned offset — the tile gather starts mid-block).
        // The tiled paged kernel stages the same keys in the same lane order as the non-tiled one,
        // so BOTH must be bit-identical: paged-tiled == contiguous-tiled in lockstep, and the
        // paged-tiled prefill == the non-tiled paged prefill logits (lp) from the main run above.
        cuda_policy().use_tiled_attn = true;
        std::unique_ptr<KVCacheBase> cont3 = model.make_kv_cache(ids.size() + 16);
        CudaBlockPool pool3(c.num_layers, c.num_kv_heads, c.head_dim, block_size, num_blocks);
        CudaPagedKVCache paged3(&pool3);
        Tensor lc3 = model.forward(ids, cont3.get());
        Tensor lp3 = model.forward(ids, &paged3);
        double tiledd = worse(full_maxdiff(lc3, lp3), full_maxdiff(lp3, lp));
        std::vector<int64_t> vtoks(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 5));
        Tensor vc3 = model.forward(vtoks, cont3.get());
        Tensor vp3 = model.forward(vtoks, &paged3);
        tiledd = worse(tiledd, full_maxdiff(vc3, vp3));
        cuda_policy().use_tiled_attn = false;
        const bool tiled_ok = tiledd == 0.0;
        std::printf("paged tiled: prefill+verify == contiguous-tiled / == non-tiled paged  "
                    "max|diff|=%g -> %s\n",
                    tiledd, tiled_ok ? "BIT-IDENTICAL" : "MISMATCH");

        // Rollback (S1): truncate(L) + replay is bit-identical to a cache that only ever decoded
        // the accepted tokens — the same kernels in the same order, so max|diff|==0 even on the
        // GPU. The contiguous CudaKVCache slices its cat_seq-grown history; the paged cache FREES
        // the rejected tail blocks, which the replay re-allocates and reuses (small block_size + a
        // tail longer than the accepted tail forces that reuse). Both must equal the never-had-a-
        // tail run to the last bit. This is S1's "done when".
        {
            const size_t plen = std::min<size_t>(ids.size(), 6);
            std::vector<int64_t> prompt(ids.begin(), ids.begin() + plen);
            const std::vector<int64_t> tail = {40, 100, 12095, 785, 11, 42, 7};  // rejected tentative tail
            const std::vector<int64_t> accepted = {40, 100, 999};                // confirmed continuation
            auto roll = [&](KVCacheBase* cache) {
                model.forward(prompt, cache);
                const int64_t L = cache->length();
                model.forward(tail, cache);     // tentative K/V for the rejected tail
                cache->truncate(L);             // roll back to the confirmed prefix
                return model.forward(accepted, cache);
            };
            auto only = [&](KVCacheBase* cache) {
                model.forward(prompt, cache);
                return model.forward(accepted, cache);
            };
            std::unique_ptr<KVCacheBase> rc =
                model.make_kv_cache(ids.size() + tail.size() + accepted.size() + 1);
            std::unique_ptr<KVCacheBase> oc = model.make_kv_cache(ids.size() + accepted.size() + 1);
            const double rdc = full_maxdiff(roll(rc.get()), only(oc.get()));

            CudaBlockPool rpool(c.num_layers, c.num_kv_heads, c.head_dim, block_size, num_blocks);
            double rdp;
            {
                CudaPagedKVCache rp(&rpool), op(&rpool);
                rdp = full_maxdiff(roll(&rp), only(&op));
            }  // both caches finish -> every block returns to the pool
            const bool recovered = rpool.free_blocks() == rpool.num_blocks();
            const bool rollback_ok = rdc == 0.0 && rdp == 0.0 && recovered;
            std::printf("rollback (S1): truncate+replay == decode-only  contiguous max|diff|=%g  "
                        "paged max|diff|=%g  blocks recovered=%s -> %s\n",
                        rdc, rdp, recovered ? "yes" : "no",
                        rollback_ok ? "BIT-IDENTICAL" : "MISMATCH");

            // Paging is pure indexing — same K/V, same order — so expect bit-identical (split path too).
            const bool ok = maxd < 1e-4 && greedy_ok && split_ok && tiled_ok && rollback_ok;
            std::printf(ok ? "run_cuda_paged: ok\n" : "run_cuda_paged: FAIL\n");
            return ok ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::printf("run_cuda_paged: exception: %s\n", e.what());
        return 1;
    }
}
