// G2 parity: the WHOLE forward on the GPU (CUDA backend) reproduces the CPU/nanoinfer
// reference. Two checks, both the GPU analogue of the existing CPU e2e tests:
//   1. logits vs ref_logits.bin  (run_parity, but every op ran on the 4070S)
//   2. greedy continuation vs ref_gen_ids.txt  (run_generate)
//
// NOT bit-identical: the kernels accumulate in float and in a different order than the
// CPU's double-accumulated SIMD reductions, and that drift compounds over 24 layers —
// so the real correctness signal is the golden tokens (argmax is robust to the logit
// wiggle), with max|diff| reported as a loose guard. The CPU backend is the oracle.
//
// Greedy here is a FULL RECOMPUTE each step (no KV cache): the cache moves to the GPU in
// G3, so the uncached forward is all we can drive on the device today.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_cuda_parity weights/qwen2.5-0.5b
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "model.hpp"
#include "parity_util.hpp"
#include "serialize.hpp"

using namespace ni;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cuda_parity <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        if (!cuda_available()) {
            std::printf("run_cuda_parity: no CUDA device visible — skipping\n");
            return 0;
        }
        Model model(dir, QuantMode::None, Device::CUDA);
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        std::vector<int64_t> ref_gen = read_ids(dir + "/ref_gen_ids.txt");
        if (ids.empty() || ref_gen.empty()) {
            std::printf("run_cuda_parity: empty ref_ids.txt / ref_gen_ids.txt\n");
            return 2;
        }
        std::printf("model: %lld layers, vocab %lld, hidden %lld; seq %zu (CUDA backend)\n",
                    (long long)model.config().num_layers, (long long)model.config().vocab_size,
                    (long long)model.config().hidden_size, ids.size());

        // --- 1. logit parity vs nanoinfer's dump (run_parity bar, on the GPU) ---
        Tensor logits = model.forward(ids);  // returns CPU logits (forward D2Hs at the edge)
        Tensor ref = load_bin(dir + "/ref_logits.bin");
        if (logits.numel() != ref.numel()) {
            std::printf("FAIL: logits shape mismatch %lld vs %lld\n",
                        (long long)logits.numel(), (long long)ref.numel());
            return 1;
        }
        const int64_t seq = logits.size(0), vocab = logits.size(1);
        double maxd = 0.0, sumd = 0.0;
        bool has_nan = false;
        for (int64_t i = 0; i < logits.numel(); ++i) {
            if (std::isnan(logits[i])) has_nan = true;
            const double d = std::fabs(double(logits[i]) - double(ref[i]));
            maxd = d > maxd ? d : maxd;
            sumd += d;
        }
        int argmism = 0;
        for (int64_t s = 0; s < seq; ++s)
            if (argmax_row(logits, s, vocab) != argmax_row(ref, s, vocab)) ++argmism;
        const int64_t cpp_tok = argmax_row(logits, seq - 1, vocab);
        const int64_t ref_tok = argmax_row(ref, seq - 1, vocab);
        std::printf("logits: max|diff|=%g  mean|diff|=%g  argmax mism=%d/%lld  next-token %s\n",
                    maxd, sumd / double(logits.numel()), argmism, (long long)seq,
                    cpp_tok == ref_tok ? "MATCH" : "MISMATCH");

        // --- 2. uncached greedy vs nanoinfer's continuation (run_generate bar) ---
        std::vector<int64_t> ctx = ids, got;
        for (size_t t = 0; t < ref_gen.size(); ++t) {
            Tensor lg = model.forward(ctx);  // full recompute (no cache yet)
            const int64_t tok = argmax_row(lg, lg.size(0) - 1, lg.size(1));
            got.push_back(tok);
            ctx.push_back(tok);
        }
        print_ids("ref   :", ref_gen);
        print_ids("cuda  :", got);
        int genmism = 0;
        for (size_t i = 0; i < ref_gen.size() && i < got.size(); ++i)
            if (ref_gen[i] != got[i]) ++genmism;
        const bool greedy_ok = got.size() == ref_gen.size() && genmism == 0;
        std::printf("greedy match: %s (%d/%zu mismatches)\n", greedy_ok ? "MATCH" : "MISMATCH",
                    genmism, ref_gen.size());

        // --- 3. fp16 weights (G5d), informational: the same uncached greedy with the layer
        // projections uploaded as fp16. Does the whole model at fp16 weights still match the
        // golden tokens? Not gated — fp16 may flip a close argmax; that's the measured cost. ---
        g_cuda_fp16_weights = true;
        Model model16(dir, QuantMode::None, Device::CUDA);
        g_cuda_fp16_weights = false;
        double maxd16 = 0.0;
        {
            Tensor lg = model16.forward(ids);
            for (int64_t i = 0; i < lg.numel() && i < ref.numel(); ++i)
                maxd16 = std::fmax(maxd16, std::fabs(double(lg[i]) - double(ref[i])));
        }
        std::vector<int64_t> ctx16 = ids, got16;
        for (size_t t = 0; t < ref_gen.size(); ++t) {
            Tensor lg = model16.forward(ctx16);
            const int64_t tok = argmax_row(lg, lg.size(0) - 1, lg.size(1));
            got16.push_back(tok);
            ctx16.push_back(tok);
        }
        int gen16 = 0;
        for (size_t i = 0; i < ref_gen.size() && i < got16.size(); ++i)
            if (ref_gen[i] != got16[i]) ++gen16;
        print_ids("fp16w :", got16);
        std::printf("fp16 weights: greedy %s (%d/%zu mism), logits max|diff|=%g vs fp32 ref\n",
                    gen16 == 0 ? "MATCH" : "DIFFERS", gen16, ref_gen.size(), maxd16);

        if (has_nan) std::printf("WARNING: GPU logits contain NaN\n");
        // Correctness = the tokens (argmax + greedy); maxd is a loose drift guard, looser
        // than the CPU's 0.1 because float error compounds across the 24 layers.
        const bool ok = !has_nan && argmism == 0 && cpp_tok == ref_tok && greedy_ok && maxd < 0.5;
        std::printf(ok ? "run_cuda_parity: ok\n" : "run_cuda_parity: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_parity: exception: %s\n", e.what());
        return 1;
    }
}
