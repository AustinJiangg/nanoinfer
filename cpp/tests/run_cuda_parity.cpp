// G2 parity: the WHOLE forward on the GPU (CUDA backend) reproduces the CPU/nanoinfer
// reference. Two checks, both the GPU analogue of the existing CPU e2e tests:
//   1. logits vs ref_logits.bin  (run_parity, but every op ran on the 4070S)
//   2. greedy continuation vs ref_gen_ids.txt  (run_generate)
//
// NOT bit-identical: the kernels accumulate in float and in a different order than the
// CPU's double-accumulated SIMD reductions, and that drift compounds over the layers —
// so the real correctness signal is the golden tokens (argmax is robust to the logit
// wiggle), with max|diff| reported as a loose guard. The CPU backend is the oracle.
//
// Greedy here is a FULL RECOMPUTE each step (no KV cache): the cache moves to the GPU in
// G3, so the uncached forward is all we can drive on the device today.
//
// Each weight format (fp32 / fp16 / bf16 / W8A8 / int8-embed / full-int8) runs in its OWN scope and
// device_pool_trim() reclaims its device buffers before the next Model is built. The pool is a
// caching allocator that never cudaFrees on its own, so without the trim the formats' weights
// would sum in VRAM: fine at 0.5B (~6 GB total) but an OOM at 1.5B (fp32 alone ~6 GB, five
// formats ~18 GB > the 4070S's 12 GB).
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_cuda_parity weights/qwen2.5-0.5b
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
        // NI_DBUF=1 routes the fp32 prefill projections through the double-buffered GEMM (G5 micro-gain)
        // — only fires when the prompt is >16 tokens (m>16); bit-identical, so golden tokens must hold.
        if (const char* e = std::getenv("NI_DBUF")) cuda_policy().use_dbuf = (e[0] == '1');

        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        std::vector<int64_t> ref_gen = read_ids(dir + "/ref_gen_ids.txt");
        if (ids.empty() || ref_gen.empty()) {
            std::printf("run_cuda_parity: empty ref_ids.txt / ref_gen_ids.txt\n");
            return 2;
        }
        Tensor ref = load_bin(dir + "/ref_logits.bin");  // CPU reference; reused by every format below

        // Carried out of the fp32 block: the gated result (below) and the storage-ratio denominator
        // (the only cross-format dependencies) — everything else lives and dies inside its own scope.
        double maxd = 0.0, maxd16 = 0.0, maxde = 0.0;
        int argmism = 0;
        int64_t cpp_tok = 0, ref_tok = 0;
        bool greedy_ok = false, has_nan = false;
        double wb32_bytes = 0.0;  // fp32 device weight bytes — the ratio denominator for fp16 / full-int8

        // === 1 + 2. fp32 GPU forward: the GATED parity + golden-token checks ===
        {
            Model model(dir, QuantMode::None, Device::CUDA);
            std::printf("model: %lld layers, vocab %lld, hidden %lld; seq %zu (CUDA backend)\n",
                        (long long)model.config().num_layers, (long long)model.config().vocab_size,
                        (long long)model.config().hidden_size, ids.size());

            // 1. logit parity vs nanoinfer's dump (run_parity bar, on the GPU)
            Tensor logits = model.forward(ids);  // returns CPU logits (forward D2Hs at the edge)
            if (logits.numel() != ref.numel()) {
                std::printf("FAIL: logits shape mismatch %lld vs %lld\n",
                            (long long)logits.numel(), (long long)ref.numel());
                return 1;
            }
            const int64_t seq = logits.size(0), vocab = logits.size(1);
            double sumd = 0.0;
            for (int64_t i = 0; i < logits.numel(); ++i) {
                if (std::isnan(logits[i])) has_nan = true;
                const double d = std::fabs(double(logits[i]) - double(ref[i]));
                maxd = d > maxd ? d : maxd;
                sumd += d;
            }
            for (int64_t s = 0; s < seq; ++s)
                if (argmax_row(logits, s, vocab) != argmax_row(ref, s, vocab)) ++argmism;
            cpp_tok = argmax_row(logits, seq - 1, vocab);
            ref_tok = argmax_row(ref, seq - 1, vocab);
            std::printf("logits: max|diff|=%g  mean|diff|=%g  argmax mism=%d/%lld  next-token %s\n",
                        maxd, sumd / double(logits.numel()), argmism, (long long)seq,
                        cpp_tok == ref_tok ? "MATCH" : "MISMATCH");

            // 2. uncached greedy vs nanoinfer's continuation (run_generate bar)
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
            greedy_ok = got.size() == ref_gen.size() && genmism == 0;
            std::printf("greedy match: %s (%d/%zu mismatches)\n", greedy_ok ? "MATCH" : "MISMATCH",
                        genmism, ref_gen.size());
            wb32_bytes = double(model.weight_bytes().first);  // capture before `model` frees
        }
        device_pool_trim();

        // --- 3. fp16 weights (G5d), informational: the same uncached greedy with the big weights
        // uploaded as fp16 — the layer projections AND the token embedding / tied lm_head (the
        // single largest weight). Does the whole model at fp16 still match the golden tokens? Not
        // gated — fp16 may flip a close argmax (esp. now the lm_head logits are fp16); the measured
        // cost is the logit max|diff| and whether the tokens hold. Also reports the storage win. ---
        {
            BackendConfig cfg16;
            cfg16.fp16_weights = true;
            Model model16(dir, QuantMode::None, Device::CUDA, cfg16);
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
            // The headline win: half the bytes for the layer projections + the tied embedding.
            const auto wb16 = model16.weight_bytes();  // fp16 model: actual reflects the half buffers
            std::printf("fp16 weights: device weight storage %.0f MB -> %.0f MB (%.2fx smaller)\n",
                        wb32_bytes / 1e6, double(wb16.first) / 1e6,
                        wb32_bytes / double(wb16.first));
        }
        device_pool_trim();

        // --- 3b. bf16 weights (B1), informational: the same uncached greedy with the big weights
        // uploaded as bf16 — the same 2× storage win as fp16, but byte-exact to the bf16 the
        // checkpoint ships (the fp32 export is a lossless bf16 upcast, and RNE f32->bf16 inverts
        // it exactly), so the weights carry ZERO quantization error vs the checkpoint. The logit
        // diff vs the fp32 ref is then the fp32-vs-bf16-WEIGHT gap the model itself induced —
        // expected ~8× fp16's (8 vs 11 mantissa bits). Like fp16: not gated, tokens reported. ---
        {
            BackendConfig cfgb;
            cfgb.bf16_weights = true;
            Model modelb(dir, QuantMode::None, Device::CUDA, cfgb);
            double maxdb = 0.0;
            {
                Tensor lg = modelb.forward(ids);
                for (int64_t i = 0; i < lg.numel() && i < ref.numel(); ++i)
                    maxdb = std::fmax(maxdb, std::fabs(double(lg[i]) - double(ref[i])));
            }
            std::vector<int64_t> ctxb = ids, gotb;
            for (size_t t = 0; t < ref_gen.size(); ++t) {
                Tensor lg = modelb.forward(ctxb);
                const int64_t tok = argmax_row(lg, lg.size(0) - 1, lg.size(1));
                gotb.push_back(tok);
                ctxb.push_back(tok);
            }
            int genb = 0;
            for (size_t i = 0; i < ref_gen.size() && i < gotb.size(); ++i)
                if (ref_gen[i] != gotb[i]) ++genb;
            print_ids("bf16w :", gotb);
            std::printf("bf16 weights: greedy %s (%d/%zu mism), logits max|diff|=%g vs fp32 ref\n",
                        genb == 0 ? "MATCH" : "DIFFERS", genb, ref_gen.size(), maxdb);
            const auto wbb = modelb.weight_bytes();
            std::printf("bf16 weights: device weight storage %.0f MB -> %.0f MB (%.2fx smaller)\n",
                        wb32_bytes / 1e6, double(wbb.first) / 1e6,
                        wb32_bytes / double(wbb.first));
        }
        device_pool_trim();

        // --- 4. W8A8 (G5d), informational: the layer projections run as int8×int8 DP4A on the GPU
        // (a device CudaW8A8Weight in qweights_); embedding/lm_head/norms stay fp32. Activation quant
        // is lossy, so — like run_quant w8a8 on CPU — it may flip a close argmax. Not gated; the
        // point is the int8 COMPUTE running end-to-end on the GPU, and how close the tokens stay. ---
        {
            Model modelq(dir, QuantMode::W8A8, Device::CUDA);
            std::vector<int64_t> ctxq = ids, gotq;
            for (size_t t = 0; t < ref_gen.size(); ++t) {
                Tensor lg = modelq.forward(ctxq);
                gotq.push_back(argmax_row(lg, lg.size(0) - 1, lg.size(1)));
                ctxq.push_back(gotq.back());
            }
            int genq = 0;
            for (size_t i = 0; i < ref_gen.size() && i < gotq.size(); ++i)
                if (ref_gen[i] != gotq[i]) ++genq;
            const auto wbq = modelq.weight_bytes();
            const bool nextq = !ref_gen.empty() && !gotq.empty() && ref_gen[0] == gotq[0];
            print_ids("w8a8  :", gotq);
            std::printf("W8A8 (GPU int8 DP4A): greedy vs fp32 %d/%zu differ, next-token %s; weights %.0f MB\n",
                        genq, ref_gen.size(), nextq ? "preserved" : "CHANGED", double(wbq.first) / 1e6);
        }
        device_pool_trim();

        // --- 5. int8 embed/lm_head on the GPU (G5d): the tied token-embedding / output-projection run
        // weight-only int8 on the device (cuda_embedding_q8 gather + cuda_linear_q8), the GPU mirror of
        // the CPU oracle. Like W8A8 it feeds argmax, so this is the token guard (informational, not
        // gated). First the embed/lm_head int8 ALONE (layers fp32), then the full int8 GPU model
        // (W8A8 layers + int8 embed) for the memory headline. ---
        {
            BackendConfig cfge;
            cfge.quantize_embed = true;
            auto greedy = [&](Model& mdl) {
                std::vector<int64_t> ctx = ids, out;
                for (size_t t = 0; t < ref_gen.size(); ++t) {
                    Tensor lg = mdl.forward(ctx);
                    out.push_back(argmax_row(lg, lg.size(0) - 1, lg.size(1)));
                    ctx.push_back(out.back());
                }
                return out;
            };
            {  // int8 embed/lm_head ALONE (layers fp32)
                Model modele(dir, QuantMode::None, Device::CUDA, cfge);
                {
                    Tensor lg = modele.forward(ids);
                    for (int64_t i = 0; i < lg.numel() && i < ref.numel(); ++i)
                        maxde = std::fmax(maxde, std::fabs(double(lg[i]) - double(ref[i])));
                }
                std::vector<int64_t> gote = greedy(modele);
                int gene = 0;
                for (size_t i = 0; i < ref_gen.size() && i < gote.size(); ++i)
                    if (ref_gen[i] != gote[i]) ++gene;
                const bool nexte = !ref_gen.empty() && !gote.empty() && ref_gen[0] == gote[0];
                print_ids("emb8  :", gote);
                std::printf("int8 embed/lm_head ALONE: greedy vs fp32 %d/%zu differ, next-token %s, "
                            "logits max|diff|=%g; weights %.0f MB\n",
                            gene, ref_gen.size(), nexte ? "preserved" : "CHANGED", maxde,
                            double(modele.weight_bytes().first) / 1e6);

                // W8A8 lm_head (backlog follow-up): the same int8 embed/lm_head, but its PREFILL linear
                // runs int8×int8 DP4A (the compute win) instead of weight-only int8. The lm_head feeds
                // argmax and the activation quant is lossy, so this is the TOKEN GUARD — informational,
                // not gated (like the W8A8 layers). Decode stays the weight-only q8 GEMV (unchanged).
                cuda_policy().use_w8a8_lmhead = true;
                std::vector<int64_t> gotw = greedy(modele);
                cuda_policy().use_w8a8_lmhead = false;
                int genw = 0;
                for (size_t i = 0; i < ref_gen.size() && i < gotw.size(); ++i)
                    if (ref_gen[i] != gotw[i]) ++genw;
                const bool nextw = !ref_gen.empty() && !gotw.empty() && ref_gen[0] == gotw[0];
                print_ids("w8lm  :", gotw);
                std::printf("W8A8 lm_head (int8 prefill compute): greedy vs fp32 %d/%zu differ, "
                            "next-token %s\n",
                            genw, ref_gen.size(), nextw ? "preserved" : "CHANGED");
            }
            device_pool_trim();
            {  // full int8 GPU model (W8A8 layers + int8 embed) — the memory headline
                Model modelfull(dir, QuantMode::W8A8, Device::CUDA, cfge);
                std::vector<int64_t> gotf = greedy(modelfull);
                int genf = 0;
                for (size_t i = 0; i < ref_gen.size() && i < gotf.size(); ++i)
                    if (ref_gen[i] != gotf[i]) ++genf;
                const bool nextf = !ref_gen.empty() && !gotf.empty() && ref_gen[0] == gotf[0];
                print_ids("full8 :", gotf);
                std::printf("full int8 GPU model (W8A8 layers + int8 embed): greedy vs fp32 %d/%zu differ, "
                            "next-token %s; weights %.0f MB (%.2fx vs fp32)\n",
                            genf, ref_gen.size(), nextf ? "preserved" : "CHANGED",
                            double(modelfull.weight_bytes().first) / 1e6,
                            wb32_bytes / double(modelfull.weight_bytes().first));
                // R5: the same Weight::format breakdown on DEVICE weights — the full int8 GPU model holds
                // W8A8 projections + a Q8 embed/lm_head, named through the one shared format_name() the CPU
                // run_quant diagnostic uses. The Format tag is backend-agnostic: the R5 symmetry made legible.
                std::printf("formats (full int8, GPU):");
                for (const auto& fb : modelfull.weight_format_breakdown())
                    std::printf(" %s=%.1fMB", format_name(fb.first), fb.second / 1e6);
                std::printf("\n");
            }
        }
        device_pool_trim();

        if (has_nan) std::printf("WARNING: GPU logits contain NaN\n");
        // Correctness = the tokens (argmax + greedy); maxd is a loose drift guard, looser
        // than the CPU's 0.1 because float error compounds across the layers.
        const bool ok = !has_nan && argmism == 0 && cpp_tok == ref_tok && greedy_ok && maxd < 0.5;
        std::printf(ok ? "run_cuda_parity: ok\n" : "run_cuda_parity: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cuda_parity: exception: %s\n", e.what());
        return 1;
    }
}
