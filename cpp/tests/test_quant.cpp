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

    // --- W8A8 (int8 activation × int8 weight) ---

    // The W8A8 product equals dequant(x)·dequant(w): factoring the two scales out of the integer
    // dot is the same value as multiplying the dequantized operands (up to float ordering). Running
    // quantize_q8 on x reproduces W8A8's own per-row activation quant, so this pins the dual-scale
    // dequant and the integer core together.
    {
        std::mt19937_64 rng(10);
        std::normal_distribution<float> nd;
        Tensor x({3, 16}), w({5, 16}), bias({5});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = nd(rng);
        ni::QTensor qw = ni::quantize_q8(w);
        Tensor x_deq = ni::dequantize_q8(ni::quantize_q8(x));  // per-row int8 activations, dequantized
        Tensor ref = ni::linear(x_deq, ni::dequantize_q8(qw), &bias);
        Tensor got = ni::linear_w8a8(x, qw, &bias);
        for (int64_t i = 0; i < ref.numel(); ++i) CHECK_CLOSE(got[i], ref[i], 1e-3);
    }

    // W8A8 quantizes the activations too, so it genuinely differs from weight-only Q8 (the
    // activation quant is real, not a no-op), while staying a small approximation of fp32. (It is
    // worse than Q8 in aggregate/expectation, but not necessarily on the max element — the two
    // independent roundings can partially cancel — so that is not asserted.)
    {
        std::mt19937_64 rng(11);
        std::normal_distribution<float> nd;
        Tensor x({4, 64}), w({6, 64});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        ni::QTensor qw = ni::quantize_q8(w);
        Tensor fp32 = ni::linear(x, w);
        Tensor q8 = ni::linear_q8(x, qw);      // weight-only int8
        Tensor w8a8 = ni::linear_w8a8(x, qw);  // + dynamic int8 activations
        double e_w8a8 = 0, peak = 0, vs_q8 = 0;
        for (int64_t i = 0; i < fp32.numel(); ++i) {
            const double f = std::fabs(double(fp32[i]));
            if (f > peak) peak = f;
            const double dw = std::fabs(double(w8a8[i]) - double(fp32[i]));
            if (dw > e_w8a8) e_w8a8 = dw;
            const double dvs = std::fabs(double(w8a8[i]) - double(q8[i]));
            if (dvs > vs_q8) vs_q8 = dvs;
        }
        CHECK(vs_q8 > 0.0);          // activation quant actually changes the result (not a no-op)
        CHECK(e_w8a8 < 0.2 * peak);  // but W8A8 stays a small approximation (quantization is ~0.4%/operand)
    }

    // Factory routes W8A8 to linear_w8a8; storage matches Q8 (same int8 weight — the win is compute).
    {
        std::mt19937_64 rng(12);
        std::normal_distribution<float> nd;
        Tensor x({2, 16}), w({4, 16});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        auto qw8 = ni::make_quantized(w, ni::QuantMode::W8A8);
        Tensor ref = ni::linear_w8a8(x, ni::quantize_q8(w), nullptr);
        Tensor got = qw8->linear(x, nullptr);
        for (int64_t i = 0; i < ref.numel(); ++i) CHECK_CLOSE(got[i], ref[i], 1e-9);
        CHECK(qw8->bytes() == ni::make_quantized(w, ni::QuantMode::Q8)->bytes());
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

    // linear_q4(x, q) == linear(x, dequant(q)) with and without bias, at m below and at/above
    // kQ4UnpackMinRows (8) — covering BOTH the fused-scalar decode path and the unpack+dot_qf32
    // prefill path. in=40 exercises the SIMD unpack's 32-code block plus its 8-wide tail.
    for (int64_t m : {3, 16}) {
        std::mt19937_64 rng(4);
        std::normal_distribution<float> nd;
        Tensor x({m, 40}), w({5, 40}), bias({5});
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

    // Packed byte layout is exact (pins the +8 bias and the low/high nibble
    // choice — a serialization contract once Q4 weights hit disk). w={7,-7} with
    // absmax 7 -> scale 1 -> codes 7,-7 -> nibbles 15,1 -> byte 0x0F | (0x01<<4).
    {
        Tensor w = make({1, 2}, {7, -7});
        ni::Q4Tensor q = ni::quantize_q4(w);
        CHECK_CLOSE(q.scale[0], 1.0, 1e-9);
        CHECK(q.q.size() == 1);
        CHECK(q.q[0] == 0x1F);  // low nibble 0x0F (code 7), high nibble 0x10 (code -7)
        Tensor dq = ni::dequantize_q4(q);
        CHECK_CLOSE(dq.at(0, 0), 7.0, 1e-9);
        CHECK_CLOSE(dq.at(0, 1), -7.0, 1e-9);
    }

    // Codes stay in [-7, 7] even with a huge outlier: |dequant| <= absmax per row.
    {
        Tensor w = make({1, 4}, {100, 1, -1, 0});
        ni::Q4Tensor q = ni::quantize_q4(w);
        Tensor dq = ni::dequantize_q4(q);
        for (int64_t j = 0; j < 4; ++j) CHECK(std::fabs(dq.at(0, j)) <= 100.0 + 1e-4);
    }

    // All-zero Q4 row: scale 0, dequant 0 (no divide-by-zero).
    {
        Tensor w = make({2, 2}, {0, 0, 3, -3});
        ni::Q4Tensor q = ni::quantize_q4(w);
        CHECK_CLOSE(q.scale[0], 0.0, 1e-12);
        Tensor dq = ni::dequantize_q4(q);
        CHECK_CLOSE(dq.at(0, 0), 0.0, 1e-12);
        CHECK_CLOSE(dq.at(0, 1), 0.0, 1e-12);
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

    // --- Q4G (group-wise int4) ---

    // Q4G error is bounded per block (scale_block/2) and dequant round-trips.
    {
        std::mt19937_64 rng(5);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        Tensor w({4, 40});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        const int64_t group = 8, blocks = (40 + group - 1) / group;
        ni::Q4GTensor q = ni::quantize_q4g(w, group);
        CHECK(q.group == group);
        CHECK(static_cast<int64_t>(q.scale.size()) == 4 * blocks);
        Tensor dq = ni::dequantize_q4g(q);
        for (int64_t o = 0; o < 4; ++o)
            for (int64_t j = 0; j < 40; ++j) {
                const double bound = q.scale[static_cast<size_t>(o * blocks + j / group)] / 2.0 + 1e-6;
                CHECK(std::fabs(double(dq.at(o, j)) - w.at(o, j)) <= bound);
            }
    }

    // linear_q4g(x, q) == linear(x, dequant(q)), with and without bias, at m below and at/above
    // kQ4UnpackMinRows — both the fused-scalar decode path and the unpack+per-group dot_qf32 path.
    for (int64_t m : {3, 16}) {
        std::mt19937_64 rng(6);
        std::normal_distribution<float> nd;
        Tensor x({m, 32}), w({5, 32}), bias({5});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = nd(rng);
        ni::Q4GTensor q = ni::quantize_q4g(w, 8);
        Tensor dq = ni::dequantize_q4g(q);
        Tensor refb = ni::linear(x, dq, &bias), gotb = ni::linear_q4g(x, q, &bias);
        Tensor refn = ni::linear(x, dq), gotn = ni::linear_q4g(x, q);
        for (int64_t i = 0; i < refb.numel(); ++i) {
            CHECK_CLOSE(gotb[i], refb[i], 1e-4);
            CHECK_CLOSE(gotn[i], refn[i], 1e-4);
        }
    }

    // The whole point: with an outlier in one block, group-wise quantizes the
    // OTHER block accurately — per-channel crushes it because one global scale is
    // set by the distant outlier. Measure error on the outlier-free block (j>=8).
    {
        Tensor w({1, 16});
        for (int64_t j = 0; j < 16; ++j) w[j] = (j == 0) ? 100.0f : 0.5f;  // outlier in block 0
        auto max_err_block1 = [&](const Tensor& dq) {
            double e = 0;
            for (int64_t j = 8; j < 16; ++j) e = std::max(e, std::fabs(double(dq.at(0, j)) - w[j]));
            return e;
        };
        const double q4_err = max_err_block1(ni::dequantize_q4(ni::quantize_q4(w)));
        const double q4g_err = max_err_block1(ni::dequantize_q4g(ni::quantize_q4g(w, 8)));
        CHECK(q4g_err < q4_err / 5.0);  // block 1 (no outlier) is far tighter under Q4G
    }

    // Partial last block (in not divisible by group) round-trips.
    {
        Tensor w({2, 10});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = float(i) - 5.0f;
        ni::Q4GTensor q = ni::quantize_q4g(w, 4);  // blocks of 4,4,2
        CHECK(static_cast<int64_t>(q.scale.size()) == 2 * 3);
        Tensor dq = ni::dequantize_q4g(q);
        for (int64_t o = 0; o < 2; ++o)
            for (int64_t j = 0; j < 10; ++j)
                CHECK(std::fabs(double(dq.at(o, j)) - w.at(o, j)) <=
                      q.scale[static_cast<size_t>(o * 3 + j / 4)] / 2.0 + 1e-6);
    }

    // group == in collapses to per-channel: q4g(w, in) == q4(w) elementwise.
    {
        std::mt19937_64 rng(8);
        std::normal_distribution<float> nd;
        Tensor w({4, 16});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        Tensor a = ni::dequantize_q4g(ni::quantize_q4g(w, 16));  // one block per row
        Tensor b = ni::dequantize_q4(ni::quantize_q4(w));
        for (int64_t i = 0; i < a.numel(); ++i) CHECK_CLOSE(a[i], b[i], 1e-6);
    }

    // group == 1: each weight is its own block, so it maps to +-7 at its own
    // absmax and dequant round-trips it exactly.
    {
        Tensor w = make({1, 5}, {3, -1, 0, 7, -2});
        ni::Q4GTensor q = ni::quantize_q4g(w, 1);
        CHECK(static_cast<int64_t>(q.scale.size()) == 5);
        Tensor dq = ni::dequantize_q4g(q);
        for (int64_t j = 0; j < 5; ++j) CHECK_CLOSE(dq.at(0, j), w[j], 1e-5);
    }

    // check_q4gtensor rejects a malformed tensor before indexing out of bounds.
    {
        ni::Q4GTensor bad;
        bad.out = 2;
        bad.in = 8;
        bad.group = 4;
        bad.q.assign(2 * 4, 0);    // ceil(8/2)=4 bytes/row -> ok
        bad.scale.assign(1, 0.0f);  // wrong: should be out*ceil(in/group) = 2*2 = 4
        bool threw = false;
        try { ni::dequantize_q4g(bad); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    // The polymorphic factory routes linear() to the matching free function.
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<float> nd;
        Tensor x({2, 16}), w({4, 16});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = nd(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = nd(rng);
        auto qg = ni::make_quantized(w, ni::QuantMode::Q4G);
        Tensor ref = ni::linear_q4g(x, ni::quantize_q4g(w, 32));  // in<32 -> one block
        Tensor got = qg->linear(x, nullptr);
        for (int64_t i = 0; i < ref.numel(); ++i) CHECK_CLOSE(got[i], ref[i], 1e-9);
        CHECK(qg->bytes() < qg->fp32_bytes());
        CHECK(ni::make_quantized(w, ni::QuantMode::None) == nullptr);
    }

    // R5: every factory-built Weight reports its storage Format — the one tag CPU and CUDA share, so
    // a caller reads the representation without RTTI / knowing the subclass. (None builds no Weight.)
    {
        Tensor w({4, 16});
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = float(i) - 8.0f;
        CHECK(ni::make_quantized(w, ni::QuantMode::Q8)->format() == ni::Format::Q8);
        CHECK(ni::make_quantized(w, ni::QuantMode::Q4)->format() == ni::Format::Q4);
        CHECK(ni::make_quantized(w, ni::QuantMode::Q4G)->format() == ni::Format::Q4G);
        CHECK(ni::make_quantized(w, ni::QuantMode::W8A8)->format() == ni::Format::W8A8);
        CHECK(ni::make_q8_embed(w)->format() == ni::Format::Q8);  // weight-only int8 embed/lm_head
    }

    // --- embedding_q8 (weight-only int8 token embedding / tied lm_head) ---

    // Gathering rows of the int8 table (dequantized on the fly) equals gathering rows of
    // the dequantized fp32 table — embedding_q8 IS the row-gather of dequantize_q8, using
    // the same per-row scale the tied lm_head's linear_q8 applies. One int8 weight, two ops.
    {
        std::mt19937_64 rng(20);
        std::normal_distribution<float> nd;
        Tensor table({6, 8});  // [vocab=6, hidden=8]
        for (int64_t i = 0; i < table.numel(); ++i) table[i] = nd(rng);
        ni::QTensor q = ni::quantize_q8(table);
        Tensor dq = ni::dequantize_q8(q);                 // fp32 reconstruction
        std::vector<int64_t> ids = {0, 5, 3, 3, 1};       // a repeat + the last row
        Tensor got = ni::embedding_q8(q, ids);
        CHECK(got.size(0) == 5 && got.size(1) == 8);
        for (size_t r = 0; r < ids.size(); ++r)
            for (int64_t c = 0; c < 8; ++c)
                CHECK_CLOSE(got.at(static_cast<int64_t>(r), c), dq.at(ids[r], c), 1e-6);
    }

    // An out-of-range id is rejected (the gather indexes the table directly).
    {
        Tensor table({3, 4});
        for (int64_t i = 0; i < table.numel(); ++i) table[i] = float(i);
        ni::QTensor q = ni::quantize_q8(table);
        bool threw = false;
        try { ni::embedding_q8(q, {3}); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
    }

    std::printf(g_failures ? "test_quant: %d failures\n" : "test_quant: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
