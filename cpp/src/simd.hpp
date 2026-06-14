// Vectorized inner products for stage C5 — the compute primitives the hot
// kernels (linear, the quant linears, attention) reduce through.
//
// The single most important design choice here is that EVERY path accumulates
// in **double**, exactly like the scalar code it replaces. Vectorizing a sum
// only re-associates it (several partial sums summed pairwise instead of one
// sequential chain); in double that perturbs the result by ~1e-16 relative, so
// the final cast back to float is bit-identical to the scalar version in
// practice. That is deliberate: nanoinfer is the oracle and the C++ engine is
// pinned to it at ~4e-5 / zero-argmax-flips. C5 is supposed to make the engine
// FASTER, not to move the numbers — so we buy speed from SIMD width + threads,
// never from dropping to float accumulation (that is the further, lossy step
// llama.cpp takes; left as a documented extension, not the default).
//
// AVX2+FMA is used when the compiler targets it (-march=native / -mavx2 -mfma,
// which define __AVX2__ and __FMA__); otherwise a plain scalar loop runs, so the
// engine stays correct on any target — a NEON path slots in at the marked #elif.
#pragma once

#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace ni {
namespace simd {

// A short label for benchmarks/reports — which kernel the build actually uses.
inline const char* target() {
#if defined(__AVX2__) && defined(__FMA__)
    return "avx2+fma";
#else
    return "scalar";
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
// Horizontal sum of a 4-wide double vector.
inline double hsum_pd(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);             // lanes 0,1
    __m128d hi = _mm256_extractf128_pd(v, 1);           // lanes 2,3
    __m128d s = _mm_add_pd(lo, hi);                     // [0+2, 1+3]
    s = _mm_add_pd(s, _mm_unpackhi_pd(s, s));           // +[1+3]
    return _mm_cvtsd_f64(s);
}
#endif

// dot(a, b) = sum_i a[i]*b[i], accumulated in double. Bit-for-bit a re-association
// of the scalar `double(a[i])*b[i]` sum (float->double widening is exact, the
// product is the same rounded double), so callers keep their parity floor.
inline double dot_f32(const float* a, const float* b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    // Two independent accumulators hide FMA latency (≥2 in flight per cycle).
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        // Widen the low and high 4 floats to double; every float is exactly
        // representable, so this loses nothing before the multiply.
        const __m256d alo = _mm256_cvtps_pd(_mm256_castps256_ps128(av));
        const __m256d blo = _mm256_cvtps_pd(_mm256_castps256_ps128(bv));
        const __m256d ahi = _mm256_cvtps_pd(_mm256_extractf128_ps(av, 1));
        const __m256d bhi = _mm256_cvtps_pd(_mm256_extractf128_ps(bv, 1));
        acc0 = _mm256_fmadd_pd(alo, blo, acc0);
        acc1 = _mm256_fmadd_pd(ahi, bhi, acc1);
    }
    double total = hsum_pd(_mm256_add_pd(acc0, acc1));
    for (; i < n; ++i) total += double(a[i]) * double(b[i]);  // scalar tail
    return total;
#else
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) total += double(a[i]) * double(b[i]);
    return total;
#endif
}

// dot(x, q) for float activations x and int8 weight codes q (the Q8 inner
// product), accumulated in double. int8 -> int32 -> float -> double is exact for
// |q| <= 127, so again this only re-associates the scalar sum.
inline double dot_qf32(const float* x, const int8_t* q, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        // Load 8 int8, sign-extend to 8 int32, convert to 8 float.
        const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + i));
        const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256d xlo = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
        const __m256d qlo = _mm256_cvtps_pd(_mm256_castps256_ps128(qf));
        const __m256d xhi = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
        const __m256d qhi = _mm256_cvtps_pd(_mm256_extractf128_ps(qf, 1));
        acc0 = _mm256_fmadd_pd(xlo, qlo, acc0);
        acc1 = _mm256_fmadd_pd(xhi, qhi, acc1);
    }
    double total = hsum_pd(_mm256_add_pd(acc0, acc1));
    for (; i < n; ++i) total += double(x[i]) * double(q[i]);
    return total;
#else
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) total += double(x[i]) * double(q[i]);
    return total;
#endif
}

}  // namespace simd
}  // namespace ni
