// Run the engine with Q8 weight quantization and measure the cost: weight memory
// vs fp32, the logit error vs the fp32 reference, and whether greedy decoding
// still tracks the fp32 engine.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b
//   ./build/run_quant weights/qwen2.5-0.5b [q8|q4|q4g]   (default q8)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "generate.hpp"
#include "model.hpp"
#include "parity_util.hpp"
#include "serialize.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_quant <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string mode_str = argc > 2 ? argv[2] : "q8";
    ni::QuantMode mode = ni::QuantMode::Q8;
    if (mode_str == "q4") mode = ni::QuantMode::Q4;
    else if (mode_str == "q4g") mode = ni::QuantMode::Q4G;
    else if (mode_str == "w8a8") mode = ni::QuantMode::W8A8;
    else if (mode_str != "q8") {
        std::printf("unknown mode '%s' (use q8, q4, q4g, or w8a8)\n", mode_str.c_str());
        return 2;
    }
    try {
        ni::Model model(dir, mode);
        std::printf("quant mode: %s\n", mode_str.c_str());
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");

        auto bytes = model.weight_bytes();
        std::printf("weights: %.1f MB (%s) vs %.1f MB (fp32) -> %.2fx smaller\n",
                    bytes.first / 1e6, mode_str.c_str(), bytes.second / 1e6,
                    double(bytes.second) / double(bytes.first));

        // Logit error vs the fp32 reference.
        ni::Tensor logits = model.forward(ids);
        ni::Tensor ref = ni::load_bin(dir + "/ref_logits.bin");
        const int64_t seq = logits.size(0), vocab = logits.size(1);
        double maxd = 0.0;
        int mism = 0;
        for (int64_t s = 0; s < seq; ++s) {
            for (int64_t j = 0; j < vocab; ++j) {
                const double d = std::fabs(double(logits.at(s, j)) - double(ref.at(s, j)));
                if (d > maxd) maxd = d;
            }
            if (argmax_row(logits, s, vocab) != argmax_row(ref, s, vocab)) ++mism;
        }
        std::printf("logits vs fp32: max|diff|=%g; per-position argmax mismatches %d/%lld\n",
                    maxd, mism, (long long)seq);

        // Greedy decoding under Q8 vs the fp32 reference continuation.
        std::vector<int64_t> ref_gen = read_ids(dir + "/ref_gen_ids.txt");
        ni::GenerateConfig gc;
        gc.max_tokens = static_cast<int>(ref_gen.size());
        gc.eos_id = -1;
        std::vector<int64_t> got = ni::generate(model, ids, gc);
        int gen_mism = 0;
        for (size_t i = 0; i < ref_gen.size() && i < got.size(); ++i)
            if (ref_gen[i] != got[i]) ++gen_mism;
        print_ids("fp32  greedy:", ref_gen);
        print_ids((mode_str + " greedy:").c_str(), got);
        std::printf("greedy tokens matching fp32: %zu/%zu\n", ref_gen.size() - gen_mism,
                    ref_gen.size());

        // Informational: does the quant mode preserve the greedy next token? Q8
        // does; naive per-channel Q4 does not (its scale can't absorb outliers —
        // group-wise quant is the fix). This is a measurement tool, so a quality
        // loss is reported, not a failure.
        const bool next_ok = !ref_gen.empty() && !got.empty() && ref_gen[0] == got[0];
        std::printf("next-token under %s: %s\n", mode_str.c_str(),
                    next_ok ? "preserved" : "CHANGED (lossy)");
        std::printf("run_quant: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("run_quant: exception: %s\n", e.what());
        return 1;
    }
}
