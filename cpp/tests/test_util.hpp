// Dependency-free test helpers (no GoogleTest/Catch2 — from scratch, like the rest).
// Each test is its own executable: run checks, return g_failures ? 1 : 0 so ctest
// sees a nonzero exit on failure.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "tensor.hpp"

inline int g_failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                \
        }                                                                \
    } while (0)

#define CHECK_CLOSE(a, b, tol)                                                          \
    do {                                                                                \
        double _d = std::fabs(double(a) - double(b));                                   \
        if (_d > (tol)) {                                                               \
            std::printf("FAIL %s:%d  |%g - %g| = %g > %g\n", __FILE__, __LINE__,        \
                        double(a), double(b), _d, double(tol));                         \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

// Elementwise allclose between two tensors; prints the first few mismatches.
// Returns the number of failures (0 == match).
inline int compare_tensors(const ni::Tensor& got, const ni::Tensor& exp, double tol,
                           const char* name) {
    if (got.numel() != exp.numel()) {
        std::printf("FAIL %s: numel %lld vs %lld\n", name, (long long)got.numel(),
                    (long long)exp.numel());
        return 1;
    }
    int fails = 0;
    for (int64_t i = 0; i < got.numel(); ++i) {
        double d = std::fabs(double(got[i]) - double(exp[i]));
        if (d > tol) {
            if (fails < 5)
                std::printf("FAIL %s[%lld]: |%g - %g| = %g > %g\n", name, (long long)i,
                            double(got[i]), double(exp[i]), d, tol);
            ++fails;
        }
    }
    if (!fails) std::printf("ok   %s (%lld elems)\n", name, (long long)got.numel());
    return fails;
}
