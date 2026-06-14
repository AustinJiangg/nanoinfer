// Cached == uncached, on the real model (C3 correctness milestone). Runs the
// full-recompute forward over the reference prompt, then replays it through the
// KV cache (prefill a prefix, decode the rest one token at a time) and checks the
// per-position logits match. Counterpart of nanoinfer test_cache.py.
//
//   python ../tools/export_weights.py weights/qwen2.5-0.5b
//   python ../tools/dump_reference.py  weights/qwen2.5-0.5b
//   ./build/run_cache weights/qwen2.5-0.5b
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.hpp"
#include "model.hpp"
#include "parity_util.hpp"

// Max abs diff between row `r` of two [seq, vocab] tensors.
static double row_maxdiff(const ni::Tensor& a, int64_t ra, const ni::Tensor& b, int64_t rb,
                          int64_t vocab) {
    double m = 0.0;
    for (int64_t j = 0; j < vocab; ++j) {
        const double d = std::fabs(double(a.at(ra, j)) - double(b.at(rb, j)));
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: run_cache <weights_dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    try {
        ni::Model model(dir);
        std::vector<int64_t> ids = read_ids(dir + "/ref_ids.txt");
        if (ids.size() < 2) {
            std::printf("run_cache: need >= 2 prompt tokens\n");
            return 2;
        }
        const int64_t seq = static_cast<int64_t>(ids.size());

        // Uncached reference: row p = logits at position p given ids[0..p].
        ni::Tensor ref = model.forward(ids);
        const int64_t vocab = ref.size(1);

        // Cached: prefill the first `split` tokens, then decode the rest.
        const int64_t split = seq / 2 > 0 ? seq / 2 : 1;
        ni::KVCache cache = model.make_cache(seq);

        std::vector<int64_t> pre(ids.begin(), ids.begin() + split);
        ni::Tensor lp = model.forward(pre, &cache);
        double maxd = row_maxdiff(lp, split - 1, ref, split - 1, vocab);

        for (int64_t p = split; p < seq; ++p) {
            ni::Tensor ld = model.forward({ids[static_cast<size_t>(p)]}, &cache);
            const double d = row_maxdiff(ld, 0, ref, p, vocab);
            if (d > maxd) maxd = d;
        }

        std::printf("cached vs uncached: max|diff|=%g over %lld positions (split=%lld)\n",
                    maxd, (long long)seq, (long long)split);
        const bool ok = maxd < 1e-3;
        std::printf(ok ? "run_cache: ok\n" : "run_cache: FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("run_cache: exception: %s\n", e.what());
        return 1;
    }
}
