// End-to-end generation: greedily continue the reference prompt with the C++
// engine and compare, token-for-token, against nanoinfer's greedy continuation
// (ref_gen_ids.txt). Greedy is deterministic and RNG-independent, so it must
// match exactly. Also prints a sampled continuation as a demo.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
//   ./build/run_generate weights/qwen2.5-0.5b
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "generate.hpp"
#include "model.hpp"
#include "parity_util.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_generate <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir);
        std::vector<int64_t> prompt = read_ids(dir + "/ref_ids.txt");
        std::vector<int64_t> ref = read_ids(dir + "/ref_gen_ids.txt");
        if (prompt.empty() || ref.empty()) {
            std::printf("run_generate: empty ref_ids.txt / ref_gen_ids.txt\n");
            return 2;
        }

        // Greedy parity: same length as the reference, no EOS stop.
        ni::GenerateConfig gc;
        gc.max_tokens = static_cast<int>(ref.size());
        gc.eos_id = -1;  // default greedy params (temperature 0)
        std::vector<int64_t> got = ni::generate(model, prompt, gc);

        print_ids("ref   :", ref);
        print_ids("cpp   :", got);

        int mism = 0;
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i)
            if (ref[i] != got[i]) ++mism;
        const bool ok = (got.size() == ref.size()) && (mism == 0);
        std::printf("greedy match: %s (%d/%zu mismatches)\n", ok ? "MATCH" : "MISMATCH",
                    mism, ref.size());

        // Demo: a sampled continuation (not parity-checked — RNG differs from torch).
        ni::GenerateConfig sc;
        sc.max_tokens = static_cast<int>(ref.size());
        sc.seed = 1234;
        sc.params.temperature = 0.8f;
        sc.params.top_p = 0.95f;
        sc.params.top_k = 50;
        sc.eos_id = model.config().eos_token_id;
        print_ids("sample:", ni::generate(model, prompt, sc));

        std::printf(ok ? "run_generate: ok\n" : "run_generate: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_generate: exception: %s\n", e.what());
        return 1;
    }
}
