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

    std::printf(g_failures ? "test_ops: %d failures\n" : "test_ops: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
