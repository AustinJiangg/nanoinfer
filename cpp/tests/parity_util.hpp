// Shared helpers for the weights-gated parity runners (run_parity, run_generate).
#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

// Argmax over row `row` of a [seq, vocab] logits tensor (first max on ties,
// matching torch.argmax).
inline int64_t argmax_row(const ni::Tensor& t, int64_t row, int64_t vocab) {
    int64_t best = 0;
    float bv = t.at(row, 0);
    for (int64_t j = 1; j < vocab; ++j)
        if (t.at(row, j) > bv) { bv = t.at(row, j); best = j; }
    return best;
}

// Read whitespace-separated token ids from a text file (ref_ids.txt etc.).
inline std::vector<int64_t> read_ids(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int64_t> ids;
    int64_t x;
    while (f >> x) ids.push_back(x);
    return ids;
}

inline void print_ids(const char* label, const std::vector<int64_t>& ids) {
    std::printf("%s", label);
    for (int64_t id : ids) std::printf(" %lld", static_cast<long long>(id));
    std::printf("\n");
}
