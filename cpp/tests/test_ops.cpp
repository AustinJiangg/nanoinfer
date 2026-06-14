// Numerical sanity for the ops on small, hand-verified inputs — no external
// fixtures. The numpy parity check lives in ops_parity.cpp.
#include "ops.hpp"
#include "test_util.hpp"
#include "tensor.hpp"

using ni::Tensor;

static Tensor make(std::vector<int64_t> shape, std::vector<float> vals) {
    Tensor t(std::move(shape));
    for (int64_t i = 0; i < t.numel(); ++i) t[i] = vals[static_cast<size_t>(i)];
    return t;
}

int main() {
    // matmul: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
    {
        Tensor a = make({2, 2}, {1, 2, 3, 4});
        Tensor b = make({2, 2}, {5, 6, 7, 8});
        Tensor c = ni::matmul(a, b);
        CHECK(c.ndim() == 2 && c.size(0) == 2 && c.size(1) == 2);
        CHECK_CLOSE(c.at(0, 0), 19.0, 1e-6);
        CHECK_CLOSE(c.at(0, 1), 22.0, 1e-6);
        CHECK_CLOSE(c.at(1, 0), 43.0, 1e-6);
        CHECK_CLOSE(c.at(1, 1), 50.0, 1e-6);
    }

    // Non-square matmul shape: [2,3] x [3,1] -> [2,1].
    {
        Tensor a = make({2, 3}, {1, 2, 3, 4, 5, 6});
        Tensor b = make({3, 1}, {1, 1, 1});
        Tensor c = ni::matmul(a, b);
        CHECK(c.size(0) == 2 && c.size(1) == 1);
        CHECK_CLOSE(c.at(0, 0), 6.0, 1e-6);   // 1+2+3
        CHECK_CLOSE(c.at(1, 0), 15.0, 1e-6);  // 4+5+6
    }

    // add: elementwise.
    {
        Tensor a = make({3}, {1, 2, 3});
        Tensor b = make({3}, {4, 5, 6});
        Tensor c = ni::add(a, b);
        CHECK_CLOSE(c[0], 5.0, 1e-6);
        CHECK_CLOSE(c[1], 7.0, 1e-6);
        CHECK_CLOSE(c[2], 9.0, 1e-6);
    }

    // softmax: uniform input -> uniform output; and a known 2-element case.
    {
        Tensor u = make({3}, {1, 1, 1});
        Tensor su = ni::softmax(u);
        CHECK_CLOSE(su[0], 1.0 / 3, 1e-6);
        CHECK_CLOSE(su[1], 1.0 / 3, 1e-6);
        CHECK_CLOSE(su[2], 1.0 / 3, 1e-6);

        Tensor p = make({2}, {0, 1});
        Tensor sp = ni::softmax(p);
        CHECK_CLOSE(sp[0], 0.2689414, 1e-5);
        CHECK_CLOSE(sp[1], 0.7310586, 1e-5);
        CHECK_CLOSE(sp[0] + sp[1], 1.0, 1e-6);  // normalized
    }

    // softmax over the last dim of a 2-D tensor (two independent rows).
    {
        Tensor x = make({2, 2}, {0, 1, 2, 2});
        Tensor s = ni::softmax(x);
        CHECK_CLOSE(s.at(0, 0), 0.2689414, 1e-5);
        CHECK_CLOSE(s.at(0, 1), 0.7310586, 1e-5);
        CHECK_CLOSE(s.at(1, 0), 0.5, 1e-6);  // equal logits -> equal probs
        CHECK_CLOSE(s.at(1, 1), 0.5, 1e-6);
    }

    // rmsnorm: x=[3,4], unit weight, eps=0 -> scale 1/sqrt(12.5); output RMS == 1.
    {
        Tensor x = make({2}, {3, 4});
        Tensor w = make({2}, {1, 1});
        Tensor y = ni::rmsnorm(x, w, 0.0f);
        CHECK_CLOSE(y[0], 3.0 / std::sqrt(12.5), 1e-5);
        CHECK_CLOSE(y[1], 4.0 / std::sqrt(12.5), 1e-5);
        double rms = std::sqrt((double(y[0]) * y[0] + double(y[1]) * y[1]) / 2.0);
        CHECK_CLOSE(rms, 1.0, 1e-5);
    }

    // linear: weight is [out, in] (nn.Linear), so y = x @ w^T + bias.
    {
        Tensor x = make({1, 2}, {1, 2});
        Tensor w = make({2, 2}, {1, 0, 0, 1});  // identity
        Tensor b = make({2}, {10, 20});
        Tensor y = ni::linear(x, w, &b);
        CHECK_CLOSE(y.at(0, 0), 11.0, 1e-6);  // 1 + 10
        CHECK_CLOSE(y.at(0, 1), 22.0, 1e-6);  // 2 + 20
        Tensor y2 = ni::linear(x, w);          // no bias
        CHECK_CLOSE(y2.at(0, 0), 1.0, 1e-6);
        CHECK_CLOSE(y2.at(0, 1), 2.0, 1e-6);

        // A non-identity row to confirm the weight is read row-per-output.
        Tensor w2 = make({1, 3}, {1, 2, 3});
        Tensor x2 = make({1, 3}, {1, 1, 1});
        CHECK_CLOSE(ni::linear(x2, w2).at(0, 0), 6.0, 1e-6);  // 1+2+3
    }

    // silu: silu(0)=0, silu(+large)~x, silu(-large)~0.
    {
        Tensor x = make({3}, {0, 20, -20});
        Tensor y = ni::silu(x);
        CHECK_CLOSE(y[0], 0.0, 1e-6);
        CHECK_CLOSE(y[1], 20.0, 1e-3);
        CHECK_CLOSE(y[2], 0.0, 1e-6);
    }

    // mul: elementwise.
    {
        Tensor a = make({3}, {1, 2, 3});
        Tensor b = make({3}, {4, 5, 6});
        Tensor c = ni::mul(a, b);
        CHECK_CLOSE(c[0], 4.0, 1e-6);
        CHECK_CLOSE(c[1], 10.0, 1e-6);
        CHECK_CLOSE(c[2], 18.0, 1e-6);
    }

    // embedding: gather rows by id.
    {
        Tensor table = make({3, 2}, {10, 11, 20, 21, 30, 31});
        Tensor e = ni::embedding(table, {2, 0});
        CHECK_CLOSE(e.at(0, 0), 30.0, 1e-6);
        CHECK_CLOSE(e.at(0, 1), 31.0, 1e-6);
        CHECK_CLOSE(e.at(1, 0), 10.0, 1e-6);
        CHECK_CLOSE(e.at(1, 1), 11.0, 1e-6);
    }

    // rope cache: position 0 is the identity rotation (cos=1, sin=0).
    {
        ni::RopeCache rc = ni::build_rope_cache(/*seq=*/2, /*head_dim=*/4, /*theta=*/10000.0f);
        CHECK(rc.cos.size(0) == 2 && rc.cos.size(1) == 4);
        for (int64_t d = 0; d < 4; ++d) {
            CHECK_CLOSE(rc.cos.at(0, d), 1.0, 1e-6);
            CHECK_CLOSE(rc.sin.at(0, d), 0.0, 1e-6);
        }
        // The half-split duplicates each frequency: column i and i+half match.
        CHECK_CLOSE(rc.cos.at(1, 0), rc.cos.at(1, 2), 1e-6);
        CHECK_CLOSE(rc.sin.at(1, 1), rc.sin.at(1, 3), 1e-6);
    }

    // apply_rope: position 0 leaves the vector unchanged; rotation preserves norm.
    {
        ni::RopeCache rc = ni::build_rope_cache(2, 4, 10000.0f);
        Tensor q = make({1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});  // [heads=1, seq=2, dim=4]
        Tensor r = ni::apply_rope(q, rc.cos, rc.sin);
        // pos 0 unchanged
        for (int64_t d = 0; d < 4; ++d) CHECK_CLOSE(r.at(0, 0, d), q.at(0, 0, d), 1e-6);
        // pos 1: norm preserved by the rotation
        auto rownorm = [](const Tensor& t, int64_t p) {
            double s = 0;
            for (int64_t d = 0; d < 4; ++d) s += double(t.at(0, p, d)) * t.at(0, p, d);
            return std::sqrt(s);
        };
        CHECK_CLOSE(rownorm(r, 1), rownorm(q, 1), 1e-5);
    }

    std::printf(g_failures ? "test_ops: %d failures\n" : "test_ops: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
