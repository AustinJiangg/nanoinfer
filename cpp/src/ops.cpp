#include "ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "simd.hpp"

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
    const float* wp = weight.data();
    const float* bp = bias ? bias->data() : nullptr;
    for (int64_t i = 0; i < m; ++i) {
        const float* xr = x.data() + i * in;
        float* yr = y.data() + i * out;
        // Threaded over output channels: each output o is computed in full by one
        // thread (no shared accumulator), so the result is identical to the serial
        // reduction. dot_f32 vectorizes the inner product, accumulating in double.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (out >= kParallelMinRows)
#endif
        for (int64_t o = 0; o < out; ++o) {
            double acc = simd::dot_f32(xr, wp + o * in, in);  // row o = output o's weights
            if (bp) acc += double(bp[o]);
            yr[o] = float(acc);
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

RopeCache build_rope_cache(int64_t seq_len, int64_t head_dim, float theta) {
    require(head_dim % 2 == 0, "rope head_dim must be even");
    const int64_t half = head_dim / 2;
    Tensor cos({seq_len, head_dim});
    Tensor sin({seq_len, head_dim});
    for (int64_t p = 0; p < seq_len; ++p) {
        for (int64_t i = 0; i < half; ++i) {
            // inv_freq_i = theta^(-(2i)/head_dim); angle = position * inv_freq.
            const double inv_freq =
                1.0 / std::pow(double(theta), double(2 * i) / double(head_dim));
            const double angle = double(p) * inv_freq;
            const float c = float(std::cos(angle));
            const float s = float(std::sin(angle));
            // Duplicate each frequency across both halves (the half-split layout).
            cos.at(p, i) = c;
            cos.at(p, i + half) = c;
            sin.at(p, i) = s;
            sin.at(p, i + half) = s;
        }
    }
    return {std::move(cos), std::move(sin)};
}

Tensor apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin, int64_t pos_offset) {
    require(x.ndim() == 3, "apply_rope expects [heads, seq, head_dim]");
    const int64_t heads = x.size(0), seq = x.size(1), dim = x.size(2);
    const int64_t half = dim / 2;
    require(cos.ndim() == 2 && cos.size(1) == dim && cos.size(0) >= pos_offset + seq,
            "apply_rope: cos shape");
    require(sin.ndim() == 2 && sin.size(1) == dim && sin.size(0) >= pos_offset + seq,
            "apply_rope: sin shape");

    Tensor out({heads, seq, dim});
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t p = 0; p < seq; ++p) {
            const int64_t pos = pos_offset + p;  // absolute position of token p
            for (int64_t d = 0; d < dim; ++d) {
                // rotate_half(x) = [-x2, x1]: the first half pulls -(second half),
                // the second half pulls +(first half).
                const float rot = (d < half) ? -x.at(h, p, d + half) : x.at(h, p, d - half);
                out.at(h, p, d) = x.at(h, p, d) * cos.at(pos, d) + rot * sin.at(pos, d);
            }
        }
    }
    return out;
}

Tensor split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) {
    require(x.ndim() == 2, "split_heads expects [seq, n_heads*head_dim]");
    require(x.size(1) == n_heads * head_dim, "split_heads: width must be n_heads*head_dim");
    const int64_t seq = x.size(0);
    Tensor out({n_heads, seq, head_dim});
    for (int64_t s = 0; s < seq; ++s)
        for (int64_t h = 0; h < n_heads; ++h)
            for (int64_t d = 0; d < head_dim; ++d)
                out.at(h, s, d) = x.at(s, h * head_dim + d);
    return out;
}

Tensor merge_heads(const Tensor& x) {
    require(x.ndim() == 3, "merge_heads expects [heads, seq, head_dim]");
    const int64_t heads = x.size(0), seq = x.size(1), dim = x.size(2);
    Tensor out({seq, heads * dim});
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t s = 0; s < seq; ++s)
            for (int64_t d = 0; d < dim; ++d)
                out.at(s, h * dim + d) = x.at(h, s, d);
    return out;
}

Tensor repeat_kv(const Tensor& x, int64_t n_rep) {
    require(x.ndim() == 3, "repeat_kv expects [n_kv, seq, head_dim]");
    require(n_rep >= 1, "repeat_kv: n_rep must be >= 1");
    const int64_t kv = x.size(0), seq = x.size(1), dim = x.size(2);
    Tensor out({kv * n_rep, seq, dim});
    for (int64_t j = 0; j < kv; ++j)
        for (int64_t r = 0; r < n_rep; ++r)
            for (int64_t s = 0; s < seq; ++s)
                for (int64_t d = 0; d < dim; ++d)
                    out.at(j * n_rep + r, s, d) = x.at(j, s, d);
    return out;
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                 int64_t query_offset) {
    require(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3, "attention expects 3-D q/k/v");
    const int64_t heads = q.size(0), sq = q.size(1), dim = q.size(2);
    const int64_t sk = k.size(1);
    require(k.size(0) == heads && v.size(0) == heads, "attention: head count must match");
    require(k.size(2) == dim && v.size(2) == dim, "attention: head_dim must match");
    require(v.size(1) == sk, "attention: k and v seq length must match");
    // Causal invariant: the sq new queries are the last positions, so their
    // absolute range [query_offset, query_offset+sq) must end exactly at the key
    // count. A mismatch means a stale/wrong offset — fail loud, don't silently
    // mask the wrong key set.
    require(!causal || query_offset + sq == sk,
            "attention: causal query_offset + query_len must equal key length");

    const float scale = 1.0f / std::sqrt(float(dim));
    Tensor out({heads, sq, dim});
    const float* qp = q.data();
    const float* kp = k.data();
    const float* vp = v.data();
    float* op = out.data();

    // Threaded over heads: head h reads/writes only its own slices, so the per-head
    // result is identical to serial. `scores` is declared per-iteration (was shared)
    // so each thread has its own — a correctness requirement under threading.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (heads >= 2)
#endif
    for (int64_t h = 0; h < heads; ++h) {
        std::vector<float> scores(static_cast<size_t>(sk));
        const float* qh = qp + h * sq * dim;
        const float* kh = kp + h * sk * dim;
        const float* vh = vp + h * sk * dim;
        for (int64_t i = 0; i < sq; ++i) {
            const float* qi = qh + i * dim;
            // Query i is at absolute position query_offset+i and attends keys
            // 0..(query_offset+i). The invariant above guarantees limit <= sk.
            const int64_t limit = causal ? (query_offset + i + 1) : sk;
            // scores_j = scale * (q_i . k_j); track the max for stable softmax.
            float maxv = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < limit; ++j) {
                const float s = float(simd::dot_f32(qi, kh + j * dim, dim)) * scale;
                scores[static_cast<size_t>(j)] = s;
                if (s > maxv) maxv = s;
            }
            // softmax over the visible keys
            double denom = 0.0;
            for (int64_t j = 0; j < limit; ++j) {
                const float e = std::exp(scores[static_cast<size_t>(j)] - maxv);
                scores[static_cast<size_t>(j)] = e;
                denom += e;
            }
            const float inv = float(1.0 / denom);
            // out_i = sum_j softmax_ij * v_j  (strided over keys, so kept scalar)
            float* oi = op + (h * sq + i) * dim;
            for (int64_t d = 0; d < dim; ++d) {
                double acc = 0.0;
                for (int64_t j = 0; j < limit; ++j)
                    acc += double(scores[static_cast<size_t>(j)]) * inv * vh[j * dim + d];
                oi[d] = float(acc);
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
