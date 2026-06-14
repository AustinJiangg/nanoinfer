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
