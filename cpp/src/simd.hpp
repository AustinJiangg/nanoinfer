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
// which define __AVX2__ and __FMA__); aarch64 (Apple M-series — the cross-platform
// leg) takes the NEON path below; any other target falls back to a plain scalar
// loop, so the engine stays correct everywhere. All three keep the double-accum
// discipline (dot_qq accumulates in int32, which is associative and therefore
// exact), so every path agrees with the scalar oracle to re-association error.
#pragma once

#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ni {
namespace simd {

// A short label for benchmarks/reports — which kernel the build actually uses.
inline const char* target() {
#if defined(__AVX2__) && defined(__FMA__)
    return "avx2+fma";
#elif defined(__aarch64__)
    return "neon";
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
#elif defined(__aarch64__)
    // NEON mirror of the AVX2 path: widen each float to double (exact) and FMA, so
    // the sum is re-associated only within double — the final cast back to float
    // lands on the same value as the scalar loop, just like AVX2. float64x2 NEON is
    // aarch64-only, which is exactly the target (Apple M-series).
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t av = vld1q_f32(a + i);
        const float32x4_t bv = vld1q_f32(b + i);
        acc0 = vfmaq_f64(acc0, vcvt_f64_f32(vget_low_f32(av)),  vcvt_f64_f32(vget_low_f32(bv)));
        acc1 = vfmaq_f64(acc1, vcvt_f64_f32(vget_high_f32(av)), vcvt_f64_f32(vget_high_f32(bv)));
    }
    double total = vaddvq_f64(vaddq_f64(acc0, acc1));
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
#elif defined(__aarch64__)
    // 8 per iteration like AVX2: widen the int8 codes int8->int16->int32->float, then
    // share dot_f32's double-FMA reduction. Same double-accum, same parity floor.
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const int16x8_t q16 = vmovl_s8(vld1_s8(q + i));                       // 8 int8 -> 8 int16
        const float32x4_t qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));  // q[0..3] -> float
        const float32x4_t qhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16))); // q[4..7] -> float
        const float32x4_t xlo = vld1q_f32(x + i);
        const float32x4_t xhi = vld1q_f32(x + i + 4);
        acc0 = vfmaq_f64(acc0, vcvt_f64_f32(vget_low_f32(xlo)),  vcvt_f64_f32(vget_low_f32(qlo)));
        acc1 = vfmaq_f64(acc1, vcvt_f64_f32(vget_high_f32(xlo)), vcvt_f64_f32(vget_high_f32(qlo)));
        acc0 = vfmaq_f64(acc0, vcvt_f64_f32(vget_low_f32(xhi)),  vcvt_f64_f32(vget_low_f32(qhi)));
        acc1 = vfmaq_f64(acc1, vcvt_f64_f32(vget_high_f32(xhi)), vcvt_f64_f32(vget_high_f32(qhi)));
    }
    double total = vaddvq_f64(vaddq_f64(acc0, acc1));
    for (; i < n; ++i) total += double(x[i]) * double(q[i]);
    return total;
#else
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i) total += double(x[i]) * double(q[i]);
    return total;
#endif
}

// dot(a, b) for two int8 code arrays, accumulated EXACTLY in int32 (the W8A8 inner product, G5d).
// Both operands are int8, so this runs at int8 throughput — _mm256_madd_epi16 here, vmull+vpadalq on
// NEON, __dp4a on the GPU — the compute win weight-only Q8 (dot_qf32, an fp inner product) can't get.
// Integer add is associative, so the SIMD re-association is bit-identical to the scalar loop AND to a
// GPU int32 accumulate (DP4A): the integer core is the same number everywhere; only the later float
// dequant drifts. Exact while k·127² < INT32_MAX (~133k contraction); our largest matmul is k=4864.
inline int32_t dot_qq(const int8_t* a, const int8_t* b, int64_t n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256i acc = _mm256_setzero_si256();  // 8 int32 lanes
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        // Sign-extend 16 int8 -> 16 int16, multiply-add adjacent pairs -> 8 int32, accumulate.
        const __m256i av = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
        const __m256i bv = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(av, bv));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));  // fold 4 -> 2 lanes
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));  // fold 2 -> 1 lane
    int32_t total = _mm_cvtsi128_si32(s);
    for (; i < n; ++i) total += int32_t(a[i]) * int32_t(b[i]);  // scalar tail
    return total;
#elif defined(__aarch64__)
    // 16 int8 per iteration. vmull_s8 gives the widening int8*int8 -> int16 products (max
    // 127*127 = 16129, fits int16); vpadalq_s16 pairwise-adds them into the int32 accumulator.
    // Integer add is associative, so this is bit-exact == the scalar loop == the AVX2 madd ==
    // a GPU DP4A int32 accumulate — the W8A8 integer core is the same number on every backend.
    int32x4_t acc = vdupq_n_s32(0);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const int8x16_t av = vld1q_s8(a + i);
        const int8x16_t bv = vld1q_s8(b + i);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(av),  vget_low_s8(bv)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(av), vget_high_s8(bv)));
    }
    int32_t total = vaddvq_s32(acc);
    for (; i < n; ++i) total += int32_t(a[i]) * int32_t(b[i]);  // scalar tail
    return total;
#else
    int32_t total = 0;
    for (int64_t i = 0; i < n; ++i) total += int32_t(a[i]) * int32_t(b[i]);
    return total;
#endif
}

// Unpack n packed int4 weight codes into int8 codes in [-7, 7]. The packing is two-per-byte:
// column 2i in the low nibble of byte i, column 2i+1 in the high nibble; the stored nibble is
// code+8, so a code is nibble-8 (quant.cpp's q4_code). The Q4/Q4G linears unpack a weight row
// ONCE with this, then run the existing int8×fp32 dot_qf32 over the buffer and reuse it across
// all m activation rows — so the nibble work is SIMD and amortized over m, instead of a scalar
// q4_code() per element *per row*. Integer-exact on every path (it must reproduce q4_code()
// bit-for-bit — test_simd checks it vs the scalar ref); the speed comes from the SIMD dot.
inline void unpack_q4(const uint8_t* packed, int8_t* out, int64_t n) {
    int64_t j = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const __m128i lomask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);
    for (; j + 32 <= n; j += 32) {  // 16 packed bytes -> 32 codes
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed + j / 2));
        const __m128i lo = _mm_and_si128(b, lomask);                    // even cols: low nibbles
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), lomask);  // odd cols: high nibbles
        const __m128i loc = _mm_sub_epi8(lo, eight);  // nibble-8 -> code in [-7, 7]
        const __m128i hic = _mm_sub_epi8(hi, eight);
        // Re-interleave to column order: out = [lo0,hi0,lo1,hi1,...] = cols 0,1,2,3,...
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + j), _mm_unpacklo_epi8(loc, hic));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + j + 16), _mm_unpackhi_epi8(loc, hic));
    }
#elif defined(__aarch64__)
    const uint8x16_t lomask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (; j + 32 <= n; j += 32) {
        const uint8x16_t b = vld1q_u8(packed + j / 2);
        const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, lomask)), eight);  // even cols
        const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);     // odd cols
        vst1q_s8(out + j, vzip1q_s8(lo, hi));       // cols 0..15 interleaved
        vst1q_s8(out + j + 16, vzip2q_s8(lo, hi));  // cols 16..31
    }
#endif
    for (; j + 2 <= n; j += 2) {  // scalar tail (and the whole loop on non-SIMD targets)
        const uint8_t byte = packed[j / 2];
        out[j] = static_cast<int8_t>((byte & 0x0F) - 8);
        out[j + 1] = static_cast<int8_t>((byte >> 4) - 8);
    }
    if (j < n) out[j] = static_cast<int8_t>((packed[j / 2] & 0x0F) - 8);  // odd n: final low nibble
}

}  // namespace simd
}  // namespace ni
