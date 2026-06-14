// End-to-end parity: load the exported Qwen2.5, run the C++ forward over the
// reference token ids, and compare the logits against nanoinfer's dump. This is
// the C1 milestone — the C++ engine reproducing the Python engine's logits.
//
// Run after exporting weights + dumping a reference:
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_parity weights/qwen2.5-0.5b
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.hpp"
#include "parity_util.hpp"
#include "serialize.hpp"

static int64_t argmax_row(const ni::Tensor& t, int64_t row, int64_t vocab) {
    int64_t best = 0;
    float bv = t.at(row, 0);
    for (int64_t j = 1; j < vocab; ++j) {
        const float x = t.at(row, j);
        if (x > bv) { bv = x; best = j; }
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_parity <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir);
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        if (ids.empty()) {
            std::printf("run_parity: ref_ids.txt is empty\n");
            return 2;
        }
        std::printf("model: %lld layers, vocab %lld, hidden %lld; seq %zu\n",
                    (long long)model.config().num_layers, (long long)model.config().vocab_size,
                    (long long)model.config().hidden_size, ids.size());

        ni::Tensor logits = model.forward(ids);
        ni::Tensor ref = ni::load_bin(dir + "/ref_logits.bin");
        if (logits.numel() != ref.numel()) {
            std::printf("FAIL: logits shape mismatch %lld vs %lld\n",
                        (long long)logits.numel(), (long long)ref.numel());
            return 1;
        }

        double maxd = 0.0, sumd = 0.0;
        bool has_nan = false;
        for (int64_t i = 0; i < logits.numel(); ++i) {
            if (std::isnan(logits[i])) has_nan = true;
            const double d = std::fabs(double(logits[i]) - double(ref[i]));
            maxd = d > maxd ? d : maxd;
            sumd += d;
        }
        const int64_t seq = logits.size(0), vocab = logits.size(1);

        int mism = 0;
        for (int64_t s = 0; s < seq; ++s)
            if (argmax_row(logits, s, vocab) != argmax_row(ref, s, vocab)) ++mism;

        const int64_t cpp_tok = argmax_row(logits, seq - 1, vocab);
        const int64_t ref_tok = argmax_row(ref, seq - 1, vocab);

        std::printf("max|diff|=%g  mean|diff|=%g\n", maxd, sumd / double(logits.numel()));
        std::printf("per-position argmax mismatches: %d / %lld\n", mism, (long long)seq);
        std::printf("next-token argmax: cpp=%lld ref=%lld  %s\n", (long long)cpp_tok,
                    (long long)ref_tok, cpp_tok == ref_tok ? "MATCH" : "MISMATCH");

        if (has_nan) std::printf("WARNING: C++ logits contain NaN\n");
        // Correctness is the argmax agreement (the tokens both engines decode);
        // the abs-diff bound is a loose guard on accumulated float drift.
        const bool ok = !has_nan && (mism == 0) && (cpp_tok == ref_tok) && (maxd < 0.1);
        std::printf(ok ? "run_parity: ok\n" : "run_parity: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_parity: exception: %s\n", e.what());
        return 1;
    }
}
