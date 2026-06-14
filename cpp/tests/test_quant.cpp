// Fast tests for Q8 weight quantization (no model). Verify the quantization-error
// bound, the int8 linear == dequantized-matmul, and zero-row handling.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "ops.hpp"
#include "quant.hpp"
#include "tensor.hpp"
#include "test_util.hpp"

using ni::Tensor;

static Tensor make(std::vector<int64_t> shape, std::vector<float> vals) {
    Tensor t(std::move(shape));
    for (int64_t i = 0; i < t.numel(); ++i) t[i] = vals[static_cast<size_t>(i)];
    return t;
}

int main() {
    // scale = absmax/127 per row; codes are in [-127, 127]; the row max maps to 127.
    {
        Tensor w = make({2, 3}, {1, 2, 4, -8, 0, 8});
        ni::QTensor q = ni::quantize_q8(w);
        CHECK(q.out == 2 && q.in == 3);
        CHECK_CLOSE(q.scale[0], 4.0 / 127.0, 1e-9);
        CHECK_CLOSE(q.scale[1], 8.0 / 127.0, 1e-9);
        CHECK(q.q[2] == 127);    // row 0: w=4 == absmax -> +127
        CHECK(q.q[3] == -127);   // row 1: w=-8 == -absmax -> -127
        CHECK(q.q[4] == 0);      // row 1: w=0 -> 0
        for (int8_t c : q.q) CHECK(c >= -127 && c <= 127);
    }

    // Quantization error is bounded by half a step (scale_o / 2) per weight.
    {
        std::mt19937_64 rng(0);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        Tensor w({8, 32});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        ni::QTensor q = ni::quantize_q8(w);
        Tensor dq = ni::dequantize_q8(q);
        for (int64_t o = 0; o < 8; ++o) {
            const double bound = q.scale[static_cast<size_t>(o)] / 2.0 + 1e-6;
            for (int64_t j = 0; j < 32; ++j)
                CHECK(std::fabs(double(dq.at(o, j)) - w.at(o, j)) <= bound);
        }
    }

    // linear_q8(x, q) == linear(x, dequant(q)) (same numbers, up to fp ordering).
    {
        std::mt19937_64 rng(1);
        std::normal_distribution<float> nd;
        Tensor x({3, 16}), w({5, 16}), bias({5});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = nd(rng);

        ni::QTensor q = ni::quantize_q8(w);
        Tensor ref = ni::linear(x, ni::dequantize_q8(q), &bias);
        Tensor got = ni::linear_q8(x, q, &bias);
        for (int64_t i = 0; i < ref.numel(); ++i) CHECK_CLOSE(got[i], ref[i], 1e-4);
    }

    // linear_q8 with NO bias (the path o/gate/up/down projections take).
    {
        std::mt19937_64 rng(2);
        std::normal_distribution<float> nd;
        Tensor x({2, 12}), w({7, 12});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        ni::QTensor q = ni::quantize_q8(w);
        Tensor ref = ni::linear(x, ni::dequantize_q8(q));  // no bias
        Tensor got = ni::linear_q8(x, q);                  // no bias
        for (int64_t i = 0; i < ref.numel(); ++i) CHECK_CLOSE(got[i], ref[i], 1e-4);
    }

    // Codes always stay in [-127, 127], even for adversarial magnitudes near
    // absmax or with a huge dynamic range (the clamp must hold).
    {
        Tensor w = make({2, 4}, {5, 5, 5, 5, 1e30f, -1e30f, 1e-30f, 0});
        ni::QTensor q = ni::quantize_q8(w);
        for (int8_t c : q.q) CHECK(c >= -127 && c <= 127);
    }

    // An all-zero row quantizes to scale 0 / codes 0 / dequant 0 (no divide-by-zero).
    {
        Tensor w = make({2, 2}, {0, 0, 3, -3});
        ni::QTensor q = ni::quantize_q8(w);
        CHECK_CLOSE(q.scale[0], 0.0, 1e-12);
        CHECK(q.q[0] == 0 && q.q[1] == 0);
        Tensor dq = ni::dequantize_q8(q);
        CHECK_CLOSE(dq.at(0, 0), 0.0, 1e-12);
        CHECK_CLOSE(dq.at(0, 1), 0.0, 1e-12);
    }

    // --- Q4 (int4) ---

    // Q4 error is bounded by half a step (scale_o/2 = absmax_o/14) per weight,
    // and dequant round-trips the packed nibbles.
    {
        std::mt19937_64 rng(3);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        Tensor w({6, 24});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        ni::Q4Tensor q = ni::quantize_q4(w);
        Tensor dq = ni::dequantize_q4(q);
        for (int64_t o = 0; o < 6; ++o) {
            const double bound = q.scale[static_cast<size_t>(o)] / 2.0 + 1e-6;
            for (int64_t j = 0; j < 24; ++j)
                CHECK(std::fabs(double(dq.at(o, j)) - w.at(o, j)) <= bound);
        }
    }

    // linear_q4(x, q) == linear(x, dequant(q)) with and without bias.
    {
        std::mt19937_64 rng(4);
        std::normal_distribution<float> nd;
        Tensor x({3, 20}), w({5, 20}), bias({5});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = nd(rng);
        ni::Q4Tensor q = ni::quantize_q4(w);
        Tensor dq = ni::dequantize_q4(q);
        Tensor refb = ni::linear(x, dq, &bias);
        Tensor gotb = ni::linear_q4(x, q, &bias);
        Tensor refn = ni::linear(x, dq);
        Tensor gotn = ni::linear_q4(x, q);
        for (int64_t i = 0; i < refb.numel(); ++i) {
            CHECK_CLOSE(gotb[i], refb[i], 1e-4);
            CHECK_CLOSE(gotn[i], refn[i], 1e-4);
        }
    }

    // Odd in-features: the last byte uses only its low nibble; round-trip holds.
    {
        Tensor w = make({2, 3}, {1, -2, 4, 7, 0, -7});
        ni::Q4Tensor q = ni::quantize_q4(w);
        CHECK(q.q.size() == static_cast<size_t>(2 * 2));  // 2 rows * ceil(3/2)=2 bytes
        Tensor dq = ni::dequantize_q4(q);
        for (int64_t o = 0; o < 2; ++o)
            for (int64_t j = 0; j < 3; ++j)
                CHECK(std::fabs(double(dq.at(o, j)) - w.at(o, j)) <=
                      q.scale[static_cast<size_t>(o)] / 2.0 + 1e-6);
    }

    std::printf(g_failures ? "test_quant: %d failures\n" : "test_quant: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
