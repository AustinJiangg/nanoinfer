// Fast unit tests for the KV cache mechanics (no model). The cached==uncached
// numerical check on the real model lives in run_cache.cpp.
#include <cstdio>

#include "cache.hpp"
#include "tensor.hpp"
#include "test_util.hpp"

using ni::KVCache;
using ni::Tensor;

static Tensor filled(std::vector<int64_t> shape, float base) {
    Tensor t(std::move(shape));
    for (int64_t i = 0; i < t.numel(); ++i) t[i] = base + float(i);
    return t;
}

int main() {
    KVCache cache(/*layers=*/2, /*n_kv=*/2, /*head_dim=*/8, /*max_seq=*/16);
    CHECK(cache.length() == 0);
    CHECK(cache.max_seq() == 16);

    // Append 3 positions to layer 0; read-back equals what was written.
    Tensor k = filled({2, 3, 8}, 1.0f);
    Tensor v = filled({2, 3, 8}, 100.0f);
    auto kv = cache.update(0, k, v);
    CHECK(kv.first.size(0) == 2 && kv.first.size(1) == 3 && kv.first.size(2) == 8);
    for (int64_t h = 0; h < 2; ++h)
        for (int64_t i = 0; i < 3; ++i)
            for (int64_t d = 0; d < 8; ++d) {
                CHECK_CLOSE(kv.first.at(h, i, d), k.at(h, i, d), 1e-9);
                CHECK_CLOSE(kv.second.at(h, i, d), v.at(h, i, d), 1e-9);
            }

    // update() does not advance length — the model does, once per pass.
    CHECK(cache.length() == 0);
    cache.advance(3);
    CHECK(cache.length() == 3);

    // Appending one more token reads back all 4 positions, earlier ones intact.
    Tensor k2 = filled({2, 1, 8}, 7.0f);
    Tensor v2 = filled({2, 1, 8}, 70.0f);
    auto kv2 = cache.update(0, k2, v2);
    CHECK(kv2.first.size(1) == 4);
    CHECK_CLOSE(kv2.first.at(0, 0, 0), 1.0, 1e-9);   // original position 0 kept
    CHECK_CLOSE(kv2.first.at(0, 3, 0), 7.0, 1e-9);   // newly written position 3

    // Layers are independent: layer 1 still empty at this length view.
    Tensor k3 = filled({2, 1, 8}, -1.0f);
    auto kv3 = cache.update(1, k3, k3);
    CHECK(kv3.first.size(1) == 4);  // sees the shared length (3) + 1

    // Overflow throws.
    KVCache small(1, 2, 8, 4);
    Tensor big = filled({2, 5, 8}, 0.0f);
    bool threw = false;
    try { small.update(0, big, big); } catch (const std::exception&) { threw = true; }
    CHECK(threw);

    std::printf(g_failures ? "test_cache: %d failures\n" : "test_cache: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
