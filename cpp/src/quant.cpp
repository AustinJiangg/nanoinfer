#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ni {

namespace {
constexpr float kQ8Max = 127.0f;

void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

// Round half-to-even (matches numpy's np.round / IEEE default) and clamp.
int8_t quant_code(float w, float scale) {
    if (scale <= 0.0f) return 0;  // all-zero row: code is 0, dequant is 0
    float r = std::nearbyint(w / scale);
    // Clamp BEFORE the cast: float division can nudge the max element past 127.
    if (r > kQ8Max) r = kQ8Max;
    if (r < -kQ8Max) r = -kQ8Max;
    // A NaN weight slips through the clamps (all comparisons are false); casting
    // it to int8 is UB, so map it to 0 instead. (+/-inf is caught by the clamp.)
    if (std::isnan(r)) return 0;
    return static_cast<int8_t>(r);
}

// A QTensor from quantize_q8 is always consistent, but guard against a
// hand-built or (future) deserialized one before indexing into its buffers.
void check_qtensor(const QTensor& w) {
    require(static_cast<int64_t>(w.q.size()) == w.out * w.in, "QTensor: q size != out*in");
    require(static_cast<int64_t>(w.scale.size()) == w.out, "QTensor: scale size != out");
}
}  // namespace

QTensor quantize_q8(const Tensor& w) {
    require(w.ndim() == 2, "quantize_q8 expects a 2-D [out, in] weight");
    const int64_t out = w.size(0), in = w.size(1);

    QTensor qt;
    qt.out = out;
    qt.in = in;
    qt.q.resize(static_cast<size_t>(out * in));
    qt.scale.resize(static_cast<size_t>(out));

    for (int64_t o = 0; o < out; ++o) {
        float absmax = 0.0f;
        for (int64_t j = 0; j < in; ++j) absmax = std::max(absmax, std::fabs(w.at(o, j)));
        const float scale = absmax / kQ8Max;  // 0 for an all-zero row
        qt.scale[static_cast<size_t>(o)] = scale;
        for (int64_t j = 0; j < in; ++j)
            qt.q[static_cast<size_t>(o * in + j)] = quant_code(w.at(o, j), scale);
    }
    return qt;
}

Tensor dequantize_q8(const QTensor& w) {
    check_qtensor(w);
    Tensor out({w.out, w.in});
    for (int64_t o = 0; o < w.out; ++o) {
        const float scale = w.scale[static_cast<size_t>(o)];
        for (int64_t j = 0; j < w.in; ++j)
            out.at(o, j) = float(w.q[static_cast<size_t>(o * w.in + j)]) * scale;
    }
    return out;
}

Tensor linear_q8(const Tensor& x, const QTensor& w, const Tensor* bias) {
    check_qtensor(w);
    require(x.ndim() == 2, "linear_q8 expects 2-D x");
    require(x.size(1) == w.in, "linear_q8: x cols must match weight in-features");
    if (bias) require(bias->numel() == w.out, "linear_q8: bias must match out-features");

    const int64_t m = x.size(0), in = w.in, out = w.out;
    Tensor y({m, out});
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        for (int64_t o = 0; o < out; ++o) {
            const int8_t* qr = w.q.data() + o * in;
            double acc = 0.0;  // int8 weights, fp32 activations -> fp accumulation
            for (int64_t j = 0; j < in; ++j) acc += double(xr[j]) * qr[j];
            // The per-row scale is applied once, after the inner product.
            acc = acc * w.scale[static_cast<size_t>(o)];
            if (bias) acc += double((*bias)[o]);
            y.at(i, o) = float(acc);
        }
    }
    return y;
}

}  // namespace ni
