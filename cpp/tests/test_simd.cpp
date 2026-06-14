// C5 unit test: the vectorized dot helpers reproduce a scalar double reference
// (across every SIMD-tail length 0..40), and the threaded linear() is
// deterministic and thread-count-independent — the latter is the whole reason we
// parallelize over output channels rather than over the reduction.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "ops.hpp"
#include "parallel.hpp"
#include "simd.hpp"
#include "tensor.hpp"
#include "test_util.hpp"

using ni::Tensor;

// Independent sequential-double references — whatever path simd.hpp compiled to
// (AVX2 or scalar) must match these to re-association error.
static double ref_dot(const float* a, const float* b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += double(a[i]) * double(b[i]);
    return s;
}
static double ref_dot_q(const float* x, const int8_t* q, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += double(x[i]) * double(q[i]);
    return s;
}

int main() {
    std::printf("simd target: %s; threads: %d (%s)\n", ni::simd::target(),
                ni::max_threads(), ni::threading_backend());

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> uf(-2.0f, 2.0f);
    std::uniform_int_distribution<int> ui(-127, 127);

    // Lengths 0..40 cover an empty input, sub-vector lengths, and every possible
    // 8-wide remainder (the scalar tail the SIMD loop peels off).
    for (int64_t n = 0; n <= 40; ++n) {
        std::vector<float> a(static_cast<size_t>(n)), b(static_cast<size_t>(n)),
            x(static_cast<size_t>(n));
        std::vector<int8_t> q(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            a[static_cast<size_t>(i)] = uf(rng);
            b[static_cast<size_t>(i)] = uf(rng);
            x[static_cast<size_t>(i)] = uf(rng);
            q[static_cast<size_t>(i)] = static_cast<int8_t>(ui(rng));
        }
        const double rf = ref_dot(a.data(), b.data(), n);
        CHECK_CLOSE(ni::simd::dot_f32(a.data(), b.data(), n), rf, 1e-6 + 1e-9 * std::fabs(rf));
        const double rq = ref_dot_q(x.data(), q.data(), n);
        CHECK_CLOSE(ni::simd::dot_qf32(x.data(), q.data(), n), rq, 1e-6 + 1e-9 * std::fabs(rq));
    }
    CHECK_CLOSE(ni::simd::dot_f32(nullptr, nullptr, 0), 0.0, 0.0);  // empty -> 0

    // Threaded linear is bit-identical run-to-run and across thread counts: each
    // output channel is one thread's complete reduction, so nothing depends on how
    // the channels are split. (in is not a multiple of 8 -> exercises the tail;
    // out > kParallelMinRows -> the parallel branch is actually taken.)
    {
        const int64_t m = 3, in = 137, out = 200;
        Tensor x({m, in}), w({out, in}), bias({out});
        for (int64_t i = 0; i < x.numel(); ++i) x[i] = uf(rng);
        for (int64_t i = 0; i < w.numel(); ++i) w[i] = uf(rng);
        for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = uf(rng);
        const size_t nbytes = static_cast<size_t>(out * m) * sizeof(float);

        Tensor y1 = ni::linear(x, w, &bias);
        Tensor y2 = ni::linear(x, w, &bias);
        CHECK(std::memcmp(y1.data(), y2.data(), nbytes) == 0);  // deterministic

#if defined(_OPENMP)
        const int orig = omp_get_max_threads();
        omp_set_num_threads(1);
        Tensor ys = ni::linear(x, w, &bias);  // serial
        omp_set_num_threads(orig > 1 ? orig : 2);
        Tensor yp = ni::linear(x, w, &bias);  // parallel
        CHECK(std::memcmp(ys.data(), yp.data(), nbytes) == 0);  // thread-count-independent
        omp_set_num_threads(orig);
#endif
    }

    std::printf(g_failures ? "test_simd: %d failures\n" : "test_simd: ok\n", g_failures);
    return g_failures ? 1 : 0;
}
