#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "parallel.hpp"
#include "simd.hpp"

namespace ni {

namespace {
constexpr float kQ8Max = 127.0f;
constexpr float kQ4Max = 7.0f;

void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}

// Round half-to-even (matches numpy's np.round / IEEE default), clamp to
// [-qmax, qmax], and map a NaN weight to 0 (returns the integer code as float).
// Clamp BEFORE any int cast: float division can nudge the max element past qmax,
// and a NaN slips through the clamps (all comparisons false) — casting that is UB.
float round_clamp(float w, float scale, float qmax) {
    if (scale <= 0.0f) return 0.0f;  // all-zero row: code is 0, dequant is 0
    float r = std::nearbyint(w / scale);
    if (r > qmax) r = qmax;
    if (r < -qmax) r = -qmax;
    if (std::isnan(r)) return 0.0f;
    return r;
}

int64_t q4_row_bytes(int64_t in) { return (in + 1) / 2; }  // 2 nibbles per byte

// Unpack the int4 code (in [-7, 7]) at column j of a packed row.
int q4_code(const uint8_t* row, int64_t j) {
    const uint8_t byte = row[j / 2];
    const uint8_t nib = (j & 1) ? (byte >> 4) : (byte & 0x0F);
    return int(nib) - 8;
}

// Consistency guards for a hand-built / (future) deserialized quantized tensor.
void check_qtensor(const QTensor& w) {
    require(static_cast<int64_t>(w.q.size()) == w.out * w.in, "QTensor: q size != out*in");
    require(static_cast<int64_t>(w.scale.size()) == w.out, "QTensor: scale size != out");
}
void check_q4tensor(const Q4Tensor& w) {
    require(static_cast<int64_t>(w.q.size()) == w.out * q4_row_bytes(w.in),
            "Q4Tensor: q size != out*ceil(in/2)");
    require(static_cast<int64_t>(w.scale.size()) == w.out, "Q4Tensor: scale size != out");
}

int64_t n_blocks(int64_t in, int64_t group) { return (in + group - 1) / group; }  // ceil

void check_q4gtensor(const Q4GTensor& w) {
    require(w.group > 0, "Q4GTensor: group must be positive");
    require(static_cast<int64_t>(w.q.size()) == w.out * q4_row_bytes(w.in),
            "Q4GTensor: q size != out*ceil(in/2)");
    require(static_cast<int64_t>(w.scale.size()) == w.out * n_blocks(w.in, w.group),
            "Q4GTensor: scale size != out*ceil(in/group)");
}

// Pack one int4 code (in [-7, 7]) into the nibble for column j of a packed row.
void pack_nibble(uint8_t* row, int64_t j, int code) {
    const uint8_t nib = static_cast<uint8_t>(code + 8);  // [1, 15]
    if (j & 1) row[j / 2] |= static_cast<uint8_t>(nib << 4);
    else row[j / 2] |= nib;
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
            qt.q[static_cast<size_t>(o * in + j)] =
                static_cast<int8_t>(round_clamp(w.at(o, j), scale, kQ8Max));
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
    const int8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        float* yr = y.data() + i * out;
        // Threaded over output channels (bit-identical to serial); dot_qf32
        // vectorizes the int8×fp32 inner product, accumulating in double.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
        for (int64_t o = 0; o < out; ++o) {
            // Per-row scale applied once, after the inner product.
            double acc = simd::dot_qf32(xr, qp + o * in, in) * double(sp[o]);
            if (bp) acc += double(bp[o]);
            yr[o] = float(acc);
        }
    }
    return y;
}

Q4Tensor quantize_q4(const Tensor& w) {
    require(w.ndim() == 2, "quantize_q4 expects a 2-D [out, in] weight");
    const int64_t out = w.size(0), in = w.size(1), row_bytes = q4_row_bytes(in);

    Q4Tensor qt;
    qt.out = out;
    qt.in = in;
    qt.q.assign(static_cast<size_t>(out * row_bytes), 0);  // zero-init; we OR nibbles in
    qt.scale.resize(static_cast<size_t>(out));

    for (int64_t o = 0; o < out; ++o) {
        float absmax = 0.0f;
        for (int64_t j = 0; j < in; ++j) absmax = std::max(absmax, std::fabs(w.at(o, j)));
        const float scale = absmax / kQ4Max;
        qt.scale[static_cast<size_t>(o)] = scale;

        uint8_t* row = qt.q.data() + o * row_bytes;
        for (int64_t j = 0; j < in; ++j)
            pack_nibble(row, j, int(round_clamp(w.at(o, j), scale, kQ4Max)));  // [-7, 7]
    }
    return qt;
}

Tensor dequantize_q4(const Q4Tensor& w) {
    check_q4tensor(w);
    const int64_t row_bytes = q4_row_bytes(w.in);
    Tensor out({w.out, w.in});
    for (int64_t o = 0; o < w.out; ++o) {
        const uint8_t* row = w.q.data() + o * row_bytes;
        const float scale = w.scale[static_cast<size_t>(o)];
        for (int64_t j = 0; j < w.in; ++j) out.at(o, j) = float(q4_code(row, j)) * scale;
    }
    return out;
}

Tensor linear_q4(const Tensor& x, const Q4Tensor& w, const Tensor* bias) {
    check_q4tensor(w);
    require(x.ndim() == 2, "linear_q4 expects 2-D x");
    require(x.size(1) == w.in, "linear_q4: x cols must match weight in-features");
    if (bias) require(bias->numel() == w.out, "linear_q4: bias must match out-features");

    const int64_t m = x.size(0), in = w.in, out = w.out, row_bytes = q4_row_bytes(in);
    Tensor y({m, out});
    const uint8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        float* yr = y.data() + i * out;
        // Threaded over output channels (bit-identical to serial). The packed
        // nibble unpack stays scalar — threading is the win that matters here.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t* row = qp + o * row_bytes;
            double acc = 0.0;
            for (int64_t j = 0; j < in; ++j) acc += double(xr[j]) * q4_code(row, j);
            acc = acc * double(sp[o]);
            if (bp) acc += double(bp[o]);
            yr[o] = float(acc);
        }
    }
    return y;
}

Q4GTensor quantize_q4g(const Tensor& w, int64_t group) {
    require(w.ndim() == 2, "quantize_q4g expects a 2-D [out, in] weight");
    require(group > 0, "quantize_q4g: group must be positive");
    const int64_t out = w.size(0), in = w.size(1);
    const int64_t row_bytes = q4_row_bytes(in), blocks = n_blocks(in, group);

    Q4GTensor qt;
    qt.out = out;
    qt.in = in;
    qt.group = group;
    qt.q.assign(static_cast<size_t>(out * row_bytes), 0);
    qt.scale.resize(static_cast<size_t>(out * blocks));

    for (int64_t o = 0; o < out; ++o) {
        uint8_t* row = qt.q.data() + o * row_bytes;
        for (int64_t b = 0; b < blocks; ++b) {
            const int64_t j0 = b * group, j1 = std::min(j0 + group, in);
            float absmax = 0.0f;
            for (int64_t j = j0; j < j1; ++j) absmax = std::max(absmax, std::fabs(w.at(o, j)));
            const float scale = absmax / kQ4Max;  // per-block scale
            qt.scale[static_cast<size_t>(o * blocks + b)] = scale;
            for (int64_t j = j0; j < j1; ++j)
                pack_nibble(row, j, int(round_clamp(w.at(o, j), scale, kQ4Max)));
        }
    }
    return qt;
}

Tensor dequantize_q4g(const Q4GTensor& w) {
    check_q4gtensor(w);
    const int64_t row_bytes = q4_row_bytes(w.in), blocks = n_blocks(w.in, w.group);
    Tensor out({w.out, w.in});
    for (int64_t o = 0; o < w.out; ++o) {
        const uint8_t* row = w.q.data() + o * row_bytes;
        for (int64_t j = 0; j < w.in; ++j) {
            const float scale = w.scale[static_cast<size_t>(o * blocks + j / w.group)];
            out.at(o, j) = float(q4_code(row, j)) * scale;
        }
    }
    return out;
}

Tensor linear_q4g(const Tensor& x, const Q4GTensor& w, const Tensor* bias) {
    check_q4gtensor(w);
    require(x.ndim() == 2, "linear_q4g expects 2-D x");
    require(x.size(1) == w.in, "linear_q4g: x cols must match weight in-features");
    if (bias) require(bias->numel() == w.out, "linear_q4g: bias must match out-features");

    const int64_t m = x.size(0), in = w.in, out = w.out;
    const int64_t row_bytes = q4_row_bytes(in), blocks = n_blocks(in, w.group);
    Tensor y({m, out});
    const uint8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        float* yr = y.data() + i * out;
        // Threaded over output channels (bit-identical to serial).
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t* row = qp + o * row_bytes;
            double acc = 0.0;
            // The scale varies per block, so apply it to each block's partial sum.
            for (int64_t b = 0; b < blocks; ++b) {
                const int64_t j0 = b * w.group, j1 = std::min(j0 + w.group, in);
                double bsum = 0.0;
                for (int64_t j = j0; j < j1; ++j) bsum += double(xr[j]) * q4_code(row, j);
                acc += bsum * double(sp[o * blocks + b]);
            }
            if (bp) acc += double(bp[o]);
            yr[o] = float(acc);
        }
    }
    return y;
}

namespace {
// Thin polymorphic wrappers — each holds one quantized tensor and routes linear()
// to the matching free function. Adding a mode is one subclass + one factory case.
class Q8Weight : public QuantizedWeight {
    QTensor t_;
public:
    explicit Q8Weight(QTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q8(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
class Q4Weight : public QuantizedWeight {
    Q4Tensor t_;
public:
    explicit Q4Weight(Q4Tensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q4(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
class Q4GWeight : public QuantizedWeight {
    Q4GTensor t_;
public:
    explicit Q4GWeight(Q4GTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q4g(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
}  // namespace

std::unique_ptr<QuantizedWeight> make_quantized(const Tensor& w, QuantMode mode) {
    switch (mode) {
        case QuantMode::Q8: return std::make_unique<Q8Weight>(quantize_q8(w));
        case QuantMode::Q4: return std::make_unique<Q4Weight>(quantize_q4(w));
        case QuantMode::Q4G: return std::make_unique<Q4GWeight>(quantize_q4g(w));  // group 32
        case QuantMode::None: return nullptr;
    }
    return nullptr;
}

}  // namespace ni
