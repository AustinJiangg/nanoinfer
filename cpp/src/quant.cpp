#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "parallel.hpp"
#include "simd.hpp"

namespace ni {

namespace {
constexpr float kQ8Max = 127.0f;
constexpr float kQ4Max = 7.0f;

// Q4/Q4G unpack the packed weight row to an int8 scratch buffer once and reuse it across the m
// activation rows (then SIMD dot_qf32). That extra unpack pass only pays off once it amortizes:
// below this many rows (decode / tiny batches) the materialization costs more than the SIMD dot
// saves, so we keep the fused scalar q4_code dot there. Prefill (m = prompt length) clears it.
constexpr int64_t kQ4UnpackMinRows = 8;

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

Tensor embedding_q8(const QTensor& table, const std::vector<int64_t>& ids) {
    check_qtensor(table);
    const int64_t vocab = table.out, hidden = table.in;  // table is [vocab, hidden]
    Tensor out({static_cast<int64_t>(ids.size()), hidden});
    for (size_t r = 0; r < ids.size(); ++r) {
        const int64_t id = ids[r];
        require(id >= 0 && id < vocab, "embedding_q8: id out of range");
        const int8_t* src = table.q.data() + id * hidden;          // int8 codes for token `id`
        const float scale = table.scale[static_cast<size_t>(id)];  // that row's dequant scale
        float* dst = out.data() + static_cast<int64_t>(r) * hidden;
        for (int64_t c = 0; c < hidden; ++c) dst[c] = float(src[c]) * scale;
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
    const float* xp = x.data();
    const int8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    float* yp = y.data();
    // Output-channel parallel, row loop inner so each weight row streams once per
    // call, not once per row (see linear() in ops.cpp). dot_qf32 vectorizes the
    // int8×fp32 inner product in double; the per-row scale is applied once.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
    for (int64_t o = 0; o < out; ++o) {
        const int8_t* qr = qp + o * in;
        const double scale = double(sp[o]);
        const double b = bp ? double(bp[o]) : 0.0;
        for (int64_t i = 0; i < m; ++i)
            yp[i * out + o] = float(simd::dot_qf32(xp + i * in, qr, in) * scale + b);
    }
    return y;
}

Tensor linear_w8a8(const Tensor& x, const QTensor& w, const Tensor* bias) {
    check_qtensor(w);
    require(x.ndim() == 2, "linear_w8a8 expects 2-D x");
    require(x.size(1) == w.in, "linear_w8a8: x cols must match weight in-features");
    if (bias) require(bias->numel() == w.out, "linear_w8a8: bias must match out-features");

    const int64_t m = x.size(0), in = w.in, out = w.out;
    Tensor y({m, out});
    const float* xp = x.data();
    const int8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    float* yp = y.data();

    // Dynamic per-row activation quantization (symmetric int8), recomputed every call since the
    // activations change. Same scheme as the weight (quantize_q8's per-output-row), applied to each
    // input row: a_scale[i] = max_j|x[i,j]|/127, xq = round_clamp(x / a_scale).
    std::vector<int8_t> xq(static_cast<size_t>(m) * static_cast<size_t>(in));
    std::vector<float> a_scale(static_cast<size_t>(m));
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = xp + i * in;
        float absmax = 0.0f;
        for (int64_t j = 0; j < in; ++j) absmax = std::max(absmax, std::fabs(xr[j]));
        const float scale = absmax / kQ8Max;  // 0 for an all-zero row
        a_scale[static_cast<size_t>(i)] = scale;
        int8_t* xqr = xq.data() + i * in;
        for (int64_t j = 0; j < in; ++j)
            xqr[j] = static_cast<int8_t>(round_clamp(xr[j], scale, kQ8Max));
    }

    // int8×int8 → int32 GEMM. Output-channel parallel, row loop inner so each weight row streams
    // once per call (like linear()). dot_qq is the exact integer core; the two scales fold in once.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
    for (int64_t o = 0; o < out; ++o) {
        const int8_t* qr = qp + o * in;
        const double ws = double(sp[o]);
        const double b = bp ? double(bp[o]) : 0.0;
        for (int64_t i = 0; i < m; ++i)
            yp[i * out + o] = float(double(simd::dot_qq(xq.data() + i * in, qr, in)) *
                                        double(a_scale[static_cast<size_t>(i)]) * ws +
                                    b);
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
    const float* xp = x.data();
    const uint8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    float* yp = y.data();
    // Output-channel parallel, row loop inner so the packed weight row streams once per call (see
    // linear()). Prefill (m large): unpack each row's nibbles to int8 ONCE (simd::unpack_q4) and reuse
    // across the m activation rows via the SIMD dot_qf32 — the unpack is vectorized and amortized over
    // m. Decode (m < kQ4UnpackMinRows): the unpack wouldn't amortize, so keep the fused scalar q4_code
    // dot (materializing the whole weight for one dot is net slower). codes is per-thread scratch.
    const bool unpack = m >= kQ4UnpackMinRows;
#if defined(_OPENMP)
#pragma omp parallel if (out >= kParallelMinRows)
#endif
    {
        std::vector<int8_t> codes(unpack ? static_cast<size_t>(in) : 0);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t* row = qp + o * row_bytes;
            const double scale = double(sp[o]);
            const double b = bp ? double(bp[o]) : 0.0;
            if (unpack) {
                simd::unpack_q4(row, codes.data(), in);
                for (int64_t i = 0; i < m; ++i)
                    yp[i * out + o] = float(simd::dot_qf32(xp + i * in, codes.data(), in) * scale + b);
            } else {
                for (int64_t i = 0; i < m; ++i) {
                    const float* xr = xp + i * in;
                    double acc = 0.0;
                    for (int64_t j = 0; j < in; ++j) acc += double(xr[j]) * q4_code(row, j);
                    yp[i * out + o] = float(acc * scale + b);
                }
            }
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
    const float* xp = x.data();
    const uint8_t* qp = w.q.data();
    const float* sp = w.scale.data();
    const float* bp = bias ? bias->data() : nullptr;
    float* yp = y.data();
    // Output-channel parallel, row loop inner so the packed weight row streams once per call (see
    // linear()). Prefill (m large): unpack each row's nibbles to int8 ONCE (simd::unpack_q4), then
    // dot_qf32 each group's slice — the per-block scale applies to each group's partial sum, the dot
    // is SIMD, the unpack amortizes over m. Decode (m < kQ4UnpackMinRows): fused scalar (the unpack
    // wouldn't amortize). codes is per-thread scratch. (Same m split as linear_q4.)
    const bool unpack = m >= kQ4UnpackMinRows;
#if defined(_OPENMP)
#pragma omp parallel if (out >= kParallelMinRows)
#endif
    {
        std::vector<int8_t> codes(unpack ? static_cast<size_t>(in) : 0);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int64_t o = 0; o < out; ++o) {
            const uint8_t* row = qp + o * row_bytes;
            const float* srow = sp + o * blocks;
            const double bias_o = bp ? double(bp[o]) : 0.0;
            if (unpack) simd::unpack_q4(row, codes.data(), in);
            for (int64_t i = 0; i < m; ++i) {
                const float* xr = xp + i * in;
                double acc = 0.0;
                // The scale varies per block, so apply it to each block's partial sum.
                for (int64_t bk = 0; bk < blocks; ++bk) {
                    const int64_t j0 = bk * w.group, j1 = std::min(j0 + w.group, in);
                    if (unpack)
                        acc += simd::dot_qf32(xr + j0, codes.data() + j0, j1 - j0) * double(srow[bk]);
                    else {
                        double bsum = 0.0;
                        for (int64_t j = j0; j < j1; ++j) bsum += double(xr[j]) * q4_code(row, j);
                        acc += bsum * double(srow[bk]);
                    }
                }
                yp[i * out + o] = float(acc + bias_o);
            }
        }
    }
    return y;
}

namespace {
// Thin polymorphic wrappers — each holds one quantized tensor and routes linear()
// to the matching free function. Adding a mode is one subclass + one factory case.
class Q8Weight : public Weight {
    QTensor t_;
public:
    explicit Q8Weight(QTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q8(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
class Q4Weight : public Weight {
    Q4Tensor t_;
public:
    explicit Q4Weight(Q4Tensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q4(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
class Q4GWeight : public Weight {
    Q4GTensor t_;
public:
    explicit Q4GWeight(Q4GTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q4g(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
// W8A8 stores the same int8 weight as Q8 (a QTensor) — the difference is the LINEAR (int8×int8,
// activations quantized at call time), so storage bytes match Q8; the win is compute, not memory.
class W8A8Weight : public Weight {
    QTensor t_;
public:
    explicit W8A8Weight(QTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_w8a8(x, t_, bias); }
    int64_t bytes() const override {
        return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4;
    }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
// R3b: the tied embedding / lm_head as weight-only int8 — a QTensor that BOTH gathers (embedding_q8,
// for embed_tokens) and projects (linear_q8, for the tied lm_head). The one quant weight that
// overrides gather(); the projection weights above keep the throwing default.
class EmbedQ8Weight : public Weight {
    QTensor t_;
public:
    explicit EmbedQ8Weight(QTensor t) : t_(std::move(t)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override { return linear_q8(x, t_, bias); }
    Tensor gather(const std::vector<int64_t>& ids) const override { return embedding_q8(t_, ids); }
    int64_t bytes() const override { return int64_t(t_.q.size()) + int64_t(t_.scale.size()) * 4; }
    int64_t fp32_bytes() const override { return t_.out * t_.in * 4; }
};
}  // namespace

std::unique_ptr<Weight> make_quantized(const Tensor& w, QuantMode mode) {
    switch (mode) {
        case QuantMode::Q8: return std::make_unique<Q8Weight>(quantize_q8(w));
        case QuantMode::Q4: return std::make_unique<Q4Weight>(quantize_q4(w));
        case QuantMode::Q4G: return std::make_unique<Q4GWeight>(quantize_q4g(w));  // group 32
        case QuantMode::W8A8: return std::make_unique<W8A8Weight>(quantize_q8(w));  // int8 weight, int8 act
        case QuantMode::None: return nullptr;
    }
    return nullptr;
}

// R3b: gather() is meaningful only for the embedding weight; a projection weight gathered by mistake
// fails loudly rather than silently. DenseWeight and EmbedQ8Weight (+ the CUDA mirror) override it.
Tensor Weight::gather(const std::vector<int64_t>&) const {
    throw std::runtime_error("Weight::gather: this weight is not an embedding table");
}

std::unique_ptr<Weight> make_q8_embed(const Tensor& w) {
    return std::make_unique<EmbedQ8Weight>(quantize_q8(w));
}

}  // namespace ni
