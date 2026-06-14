// Shape / stride / indexing sanity for the Tensor class — no external inputs.
#include "test_util.hpp"
#include "tensor.hpp"

using ni::Tensor;

int main() {
    Tensor t({2, 3});
    CHECK(t.ndim() == 2);
    CHECK(t.numel() == 6);
    CHECK(t.size(0) == 2);
    CHECK(t.size(1) == 3);

    // Row-major strides for [2, 3] are [3, 1].
    CHECK(t.strides()[0] == 3);
    CHECK(t.strides()[1] == 1);

    // Fresh tensor is zero-filled.
    for (int64_t i = 0; i < t.numel(); ++i) CHECK(t[i] == 0.0f);

    // 2-D at() maps to the right flat offset (i*3 + j).
    t.at(1, 2) = 7.0f;
    CHECK(t[1 * 3 + 2] == 7.0f);
    t.at(0, 1) = 3.0f;
    CHECK(t.at(0, 1) == 3.0f);
    CHECK(t[1] == 3.0f);

    // A 1-D tensor has stride [1].
    Tensor v({4});
    CHECK(v.ndim() == 1);
    CHECK(v.strides()[0] == 1);

    std::printf(g_failures ? "test_tensor: %d failures\n" : "test_tensor: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
