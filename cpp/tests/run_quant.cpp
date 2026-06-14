// Run the engine with Q8 weight quantization and measure the cost: weight memory
// vs fp32, the logit error vs the fp32 reference, and whether greedy decoding
// still tracks the fp32 engine.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b
//   ./build/run_quant weights/qwen2.5-0.5b
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

static int64_t argmax_row(const ni::Tensor& t, int64_t row, int64_t vocab) {
    int64_t best = 0;
    float bv = t.at(row, 0);
    for (int64_t j = 1; j < vocab; ++j)
        if (t.at(row, j) > bv) { bv = t.at(row, j); best = j; }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_quant <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir, /*quantize=*/true);
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");

        auto bytes = model.weight_bytes();
        std::printf("weights: %.1f MB (Q8) vs %.1f MB (fp32) -> %.2fx smaller\n",
                    bytes.first / 1e6, bytes.second / 1e6,
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
        print_ids("fp32 greedy:", ref_gen);
        print_ids("Q8   greedy:", got);
        std::printf("greedy tokens matching fp32: %zu/%zu\n", ref_gen.size() - gen_mism,
                    ref_gen.size());

        // The next-token argmax matching is the practical "still works" signal.
        const bool next_ok = !ref_gen.empty() && !got.empty() && ref_gen[0] == got[0];
        std::printf("run_quant: %s (next-token %s under Q8)\n", next_ok ? "ok" : "FAIL",
                    next_ok ? "preserved" : "changed");
        return next_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_quant: exception: %s\n", e.what());
        return 1;
    }
}
