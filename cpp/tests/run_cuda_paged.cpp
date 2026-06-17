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

static double full_maxdiff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i) m = std::fmax(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
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
            maxd = std::fmax(maxd, full_maxdiff(dc, dp));
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

        // Paging is pure indexing — same K/V, same order — so expect bit-identical.
        const bool ok = maxd < 1e-4 && greedy_ok;
        std::printf(ok ? "run_cuda_paged: ok\n" : "run_cuda_paged: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_paged: exception: %s\n", e.what());
        return 1;
    }
}
