#include "ops.hpp"

#include <cmath>
#include <stdexcept>

namespace ni {

namespace {
// The ops fail loudly on a shape mismatch rather than reading out of bounds —
// the C++ analogue of nanoinfer's "be loud about anything unexpected".
void require(bool cond, const char* msg) {
    if (!cond) throw std::invalid_argument(msg);
}
}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    require(a.ndim() == 2 && b.ndim() == 2, "matmul expects 2-D tensors");
    require(a.size(1) == b.size(0), "matmul inner dimensions must match");

    const int64_t m = a.size(0), k = a.size(1), n = b.size(1);
    Tensor out({m, n});
    // Naive i-k-j order: accumulate into row i of out while walking row i of a
    // and row p of b. (i-k-j touches b and out contiguously — friendlier to the
    // cache than i-j-k — but it's still the plain O(m*k*n) triple loop.)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t p = 0; p < k; ++p) {
            const float aip = a.at(i, p);
            for (int64_t j = 0; j < n; ++j) {
                out.at(i, j) += aip * b.at(p, j);
            }
        }
    }
    return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    require(x.ndim() == 2 && weight.ndim() == 2, "linear expects 2-D x and weight");
    require(x.size(1) == weight.size(1), "linear: x cols must match weight in-features");

    const int64_t m = x.size(0), in = x.size(1), out = weight.size(0);
    if (bias) require(bias->numel() == out, "linear: bias must match out-features");

    Tensor y({m, out});
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        for (int64_t o = 0; o < out; ++o) {
            const float* wr = weight.data() + o * in;  // row o = the o-th output's weights
            double acc = bias ? double((*bias)[o]) : 0.0;
            for (int64_t p = 0; p < in; ++p) acc += double(xr[p]) * wr[p];
            y.at(i, o) = float(acc);
        }
    }
    return y;
}

Tensor embedding(const Tensor& table, const std::vector<int64_t>& ids) {
    require(table.ndim() == 2, "embedding table must be 2-D [vocab, hidden]");
    const int64_t vocab = table.size(0), hidden = table.size(1);

    Tensor out({static_cast<int64_t>(ids.size()), hidden});
    for (size_t r = 0; r < ids.size(); ++r) {
        const int64_t id = ids[r];
        require(id >= 0 && id < vocab, "embedding id out of range");
        const float* src = table.data() + id * hidden;
        float* dst = out.data() + static_cast<int64_t>(r) * hidden;
        for (int64_t c = 0; c < hidden; ++c) dst[c] = src[c];
    }
    return out;
}

Tensor silu(const Tensor& x) {
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i) {
        const float v = x[i];
        out[i] = v / (1.0f + std::exp(-v));
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    require(a.shape() == b.shape(), "mul expects equally-shaped tensors");
    Tensor out(a.shape());
    for (int64_t i = 0; i < a.numel(); ++i) out[i] = a[i] * b[i];
    return out;
}

Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    require(x.ndim() >= 1, "rmsnorm expects at least 1-D input");
    const int64_t d = x.size(x.ndim() - 1);
    require(weight.numel() == d, "rmsnorm weight must match the last dimension");

    const int64_t rows = x.numel() / d;
    Tensor out(x.shape());
    const float* xp = x.data();
    const float* wp = weight.data();
    float* op = out.data();

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = xp + r * d;
        float* orow = op + r * d;
        double sumsq = 0.0;  // accumulate in double for stability
        for (int64_t c = 0; c < d; ++c) sumsq += double(xr[c]) * double(xr[c]);
        const float scale = 1.0f / std::sqrt(float(sumsq / d) + eps);
        for (int64_t c = 0; c < d; ++c) orow[c] = xr[c] * scale * wp[c];
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    require(x.ndim() >= 1, "softmax expects at least 1-D input");
    const int64_t d = x.size(x.ndim() - 1);
    const int64_t rows = x.numel() / d;
    Tensor out(x.shape());
    const float* xp = x.data();
    float* op = out.data();

    for (int64_t r = 0; r < rows; ++r) {
        const float* xr = xp + r * d;
        float* orow = op + r * d;
        float maxv = xr[0];
        for (int64_t c = 1; c < d; ++c) maxv = xr[c] > maxv ? xr[c] : maxv;
        double sum = 0.0;
        for (int64_t c = 0; c < d; ++c) {
            float e = std::exp(xr[c] - maxv);
            orow[c] = e;
            sum += e;
        }
        const float inv = float(1.0 / sum);
        for (int64_t c = 0; c < d; ++c) orow[c] *= inv;
    }
    return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
    require(a.shape() == b.shape(), "add expects equally-shaped tensors");
    Tensor out(a.shape());
    for (int64_t i = 0; i < a.numel(); ++i) out[i] = a[i] + b[i];
    return out;
}

}  // namespace ni
