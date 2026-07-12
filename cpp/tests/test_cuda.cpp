// G1/G5 parity: the CUDA backend's linear() reproduces the CPU backend's linear() within
// tolerance, on synthetic data (no model weights). Covers every kernel the dispatch picks:
// warp-GEMV (m<=16, decode), register-tiled GEMM (m>16, prefill), tensor-core wmma (m>16,
// opt-in), and the half-WEIGHT paths (gemv-h / wmma-h — fp16, G5d; gemv-b / tiled-b — bf16,
// B1). The golden e2e tests (run_cuda_parity/cache) only use short sequences, so the prefill
// + half-storage kernels meet the oracle here.
//
// NOT bit-identical: the GPU accumulates in float and in a different order than the CPU's
// double-accumulated SIMD dot (~1e-4 drift). The wmma/fp16/bf16 paths are looser still — fp16
// rounds operands to ~3 decimal digits (bf16 to ~2), so their printed max|diff| IS the result:
// the half-precision accuracy cost vs the fp32 oracle. The CPU backend is the oracle (itself
// HF-parity-locked).
//
// Self-contained, so it joins ctest; skips at runtime if no GPU is visible. Only built when
// -DNI_CUDA=ON.
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <random>

#include "backend.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "quant.hpp"
#include "tensor.hpp"

using namespace ni;

// One shape: GPU linear() (with bias) vs the CPU oracle. wdt picks the weight upload dtype
// (F32 / F16 / BF16 — the last two exercise the half-storage kernels). ok if max|diff| < tol.
static bool check_linear(int64_t m, int64_t n, int64_t k, double tol, DType wdt,
                         std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);

    CpuBackend cpu;
    Tensor y_cpu = cpu.linear(x, w, &bias);  // the oracle (fp32 weight)

    CudaBackend gpu;
    Tensor xd = to_device(x), bd = to_device(bias);
    Tensor wd = wdt == DType::F16    ? to_device_f16(w)
                : wdt == DType::BF16 ? to_device_bf16(w)
                                     : to_device(w);
    Tensor y_gpu = to_host(gpu.linear(xd, wd, &bd));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]));
    const bool ok = maxdiff < tol;
    const bool aligned = (n % 128 == 0 && k % 16 == 0);
    const char* kern =
        wdt == DType::F16
            ? (m <= 16 ? "gemv-h" : (n >= 8192 ? "wmma-h" : "tiled-h"))
        : wdt == DType::BF16
            // B1 dispatch: GEMV / float4 tiled / the scalar-tiled ragged fallback (B2 adds wmma-b).
            ? (m <= 16 ? "gemv-b" : (aligned ? "tiled-b" : "stile-b"))
            : (cuda_policy().use_wmma ? "wmma" : (m <= 16 ? "gemv" : "tiled"));
    std::printf("test_cuda: [%4lld x %4lld] @ [%6lld x %4lld]^T (%-6s) max|diff|=%.3e (tol %.0e) %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, kern, maxdiff, tol,
                ok ? "ok" : "FAIL");
    return ok;
}

// Warp-per-query attention (G5e) vs the CPU oracle. Random q[H,sq,D], k/v[H,sk,D]; ok if the GPU
// output matches within tol (only the per-key sums are reordered vs the CPU's sequential add).
static bool check_attention(int64_t H, int64_t sq, int64_t sk, int64_t D, bool causal,
                            int64_t query_offset, double tol, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor q({H, sq, D}), k({H, sk, D}), v({H, sk, D});
    for (int64_t i = 0; i < q.numel(); ++i) q[i] = dist(rng);
    for (int64_t i = 0; i < k.numel(); ++i) k[i] = dist(rng);
    for (int64_t i = 0; i < v.numel(); ++i) v[i] = dist(rng);

    CpuBackend cpu;
    Tensor o_cpu = cpu.attention(q, k, v, causal, query_offset);  // the oracle

    CudaBackend gpu;
    Tensor qd = to_device(q), kd = to_device(k), vd = to_device(v);
    Tensor o_gpu = to_host(gpu.attention(qd, kd, vd, causal, query_offset));

    // NaN-aware: std::max(x, NaN) silently returns x, so a NaN output (e.g. an empty-lane combine
    // poisoning the softmax) would slip past a plain max reduction. Catch it explicitly and fail.
    double maxdiff = 0.0;
    bool nan = false;
    for (int64_t i = 0; i < o_cpu.numel(); ++i) {
        const double d = std::fabs(static_cast<double>(o_cpu[i]) - o_gpu[i]);
        if (std::isnan(d)) nan = true;
        else maxdiff = std::max(maxdiff, d);
    }
    if (nan) maxdiff = std::numeric_limits<double>::quiet_NaN();
    const bool ok = !nan && maxdiff < tol;
    std::printf("test_cuda: attn H=%lld sq=%4lld sk=%4lld D=%lld %-6s qoff=%-3lld max|diff|=%.3e (tol %.0e) %s\n",
                (long long)H, (long long)sq, (long long)sk, (long long)D, causal ? "causal" : "full",
                (long long)query_offset, maxdiff, tol, ok ? "ok" : "FAIL");
    return ok;
}

// Paged shared-mem tiled attention (the backlog mirror of G5f-tiled): the tiled paged kernel must be
// BIT-IDENTICAL to the non-tiled paged kernel — same keys in the same lane order, only staged through
// smem via a block-table gather. Drives a real CudaPagedKVCache (no weights): seed `ctx` positions,
// advance, then attend a multi-query block of `t` new tokens at query_offset=ctx (the prefill /
// spec-verify shape) twice — non-tiled (no_tiled_attn) vs tiled (use_tiled_attn, or the D>=128
// default when `force` is false). attend() re-writes the same slots while the length isn't advanced,
// so both runs see identical K/V. NaN-aware (the G5f empty-lane lesson: fmax swallows NaN).
static bool check_paged_tiled(int64_t H, int64_t n_kv, int64_t t, int64_t ctx, int64_t D, int64_t bs,
                              bool causal, bool force, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n_rep = H / n_kv;
    CudaBlockPool pool(/*num_layers=*/1, n_kv, D, bs, (ctx + t) / bs + 2);
    CudaPagedKVCache paged(&pool);
    if (ctx > 0) {  // seed the cache so the new queries attend across old blocks too
        Tensor qc({H, ctx, D}), kc({n_kv, ctx, D}), vc({n_kv, ctx, D});
        for (int64_t i = 0; i < qc.numel(); ++i) qc[i] = dist(rng);
        for (int64_t i = 0; i < kc.numel(); ++i) kc[i] = dist(rng);
        for (int64_t i = 0; i < vc.numel(); ++i) vc[i] = dist(rng);
        Tensor qd = to_device(qc), kd = to_device(kc), vd = to_device(vc);
        paged.attend(0, qd, kd, vd, n_rep, true, 0);
        paged.advance(ctx);
    }
    Tensor q({H, t, D}), k({n_kv, t, D}), v({n_kv, t, D});
    for (int64_t i = 0; i < q.numel(); ++i) q[i] = dist(rng);
    for (int64_t i = 0; i < k.numel(); ++i) k[i] = dist(rng);
    for (int64_t i = 0; i < v.numel(); ++i) v[i] = dist(rng);
    Tensor qd = to_device(q), kd = to_device(k), vd = to_device(v);

    cuda_policy().no_tiled_attn = true;  // baseline: the proven non-tiled warp kernel
    Tensor o_warp = to_host(paged.attend(0, qd, kd, vd, n_rep, causal, ctx));
    cuda_policy().no_tiled_attn = false;
    cuda_policy().use_tiled_attn = force;  // force=false leans on the D>=kTileMinHeadDim default
    Tensor o_tiled = to_host(paged.attend(0, qd, kd, vd, n_rep, causal, ctx));
    cuda_policy().use_tiled_attn = false;

    double maxdiff = 0.0;
    bool nan = false;
    for (int64_t i = 0; i < o_warp.numel(); ++i) {
        const double d = std::fabs(static_cast<double>(o_warp[i]) - o_tiled[i]);
        if (std::isnan(d)) nan = true;
        else maxdiff = std::max(maxdiff, d);
    }
    const bool ok = !nan && maxdiff == 0.0;
    std::printf("test_cuda: paged-tiled H=%lld nkv=%lld t=%3lld ctx=%3lld D=%3lld bs=%2lld %-6s %s "
                "max|diff|=%.3e %s\n",
                (long long)H, (long long)n_kv, (long long)t, (long long)ctx, (long long)D,
                (long long)bs, causal ? "causal" : "full", force ? "forced " : "default",
                nan ? std::numeric_limits<double>::quiet_NaN() : maxdiff, ok ? "ok" : "FAIL");
    return ok;
}

// Embedding gather vs the CPU oracle (G5d/B1). Random [vocab, hidden] table, random ids; an fp32
// table matches exactly (a pure copy — no arithmetic), an fp16/bf16 table within that dtype's
// rounding of the looked-up rows. Covers the half-table paths now that embed_tokens goes half.
static bool check_embedding(int64_t vocab, int64_t hidden, int64_t n, DType tdt, double tol,
                            std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor table({vocab, hidden});
    for (int64_t i = 0; i < table.numel(); ++i) table[i] = dist(rng);
    std::uniform_int_distribution<int64_t> idd(0, vocab - 1);
    std::vector<int64_t> ids(static_cast<size_t>(n));
    for (auto& v : ids) v = idd(rng);

    CpuBackend cpu;
    Tensor o_cpu = cpu.embedding(table, ids);  // oracle (fp32 table)

    CudaBackend gpu;
    Tensor td = tdt == DType::F16    ? to_device_f16(table)
                : tdt == DType::BF16 ? to_device_bf16(table)
                                     : to_device(table);
    Tensor o_gpu = to_host(gpu.embedding(td, ids));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < o_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(o_cpu[i]) - o_gpu[i]));
    const bool ok = maxdiff < tol;
    const char* lbl = tdt == DType::F16 ? "f16" : tdt == DType::BF16 ? "bf16" : "f32";
    std::printf("test_cuda: embed [%6lld x %4lld] gather n=%lld (%-4s) max|diff|=%.3e (tol %.0e) %s\n",
                (long long)vocab, (long long)hidden, (long long)n, lbl, maxdiff, tol,
                ok ? "ok" : "FAIL");
    return ok;
}

// W8A8 DP4A GEMM (G5d) vs the CPU linear_w8a8 oracle. quantize_q8(w) gives the int8 weight + per-row
// scales (uploaded to device); the GPU quantizes the activations itself. The integer core (dot_qq /
// __dp4a) is identical, so the only gap is the float-vs-double dequant — a tight tolerance, not the
// fp16 cost. k must be a multiple of 16 (the DP4A tile's K step).
static bool check_w8a8(int64_t m, int64_t n, int64_t k, bool with_bias, double tol,
                       std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);

    QTensor qw = quantize_q8(w);
    Tensor y_cpu = linear_w8a8(x, qw, with_bias ? &bias : nullptr);  // oracle

    Tensor ws({n});  // per-row weight scales as a tensor, to upload
    for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
    Tensor xd = to_device(x), wqd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
    Tensor bd = to_device(bias);
    Tensor y_gpu = to_host(cuda_linear_w8a8(xd, wqd, wsd, with_bias ? &bd : nullptr));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]));
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: w8a8 [%4lld x %4lld] @ [%5lld x %4lld]^T %-7s max|diff|=%.3e (tol %.0e) %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, with_bias ? "+bias" : "no-bias",
                maxdiff, tol, ok ? "ok" : "FAIL");
    return ok;
}

// The W8A8 decode GEMV (m<=kGemvMaxM) must be BIT-IDENTICAL to the tiled DP4A kernel, not merely within
// tolerance: a single cuda_linear_w8a8 call quantizes x to one int8 xq/a_scale, and both kernels sum
// the same int8×int8 products into an int32 (EXACT, associative — order-independent, no overflow) then
// do the identical float dequant, so only the reduction ORDER differs and the result cannot. A/B the
// two kernels on the same inputs via force_tiled_w8a8 and assert max|diff|==0 (the strong claim the
// weight-only q8 GEMV can't make — its fp32 accumulate reorders to ~1e-6).
static bool check_w8a8_gemv_bitident(int64_t m, int64_t n, int64_t k, bool with_bias,
                                     std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);
    QTensor qw = quantize_q8(w);
    Tensor ws({n});
    for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
    Tensor xd = to_device(x), wqd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
    Tensor bd = to_device(bias);
    const Tensor* bp = with_bias ? &bd : nullptr;

    cuda_policy().force_tiled_w8a8 = false;
    Tensor y_gemv = to_host(cuda_linear_w8a8(xd, wqd, wsd, bp));
    cuda_policy().force_tiled_w8a8 = true;
    Tensor y_tiled = to_host(cuda_linear_w8a8(xd, wqd, wsd, bp));
    cuda_policy().force_tiled_w8a8 = false;

    double d = 0.0;
    for (int64_t i = 0; i < y_gemv.numel(); ++i)
        d = std::max(d, std::fabs(static_cast<double>(y_gemv[i]) - y_tiled[i]));
    const bool ok = (d == 0.0);
    std::printf("test_cuda: w8a8 gemv==tiled [%4lld x %4lld] @ [%5lld x %4lld]^T %-7s max|diff|=%.3e %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, with_bias ? "+bias" : "no-bias",
                d, ok ? "ok (bit-identical)" : "FAIL");
    return ok;
}

// Weight-only int8 embedding gather (G5d) vs the CPU embedding_q8 oracle. quantize_q8(table) gives
// int8 codes + per-row scales (uploaded); the gather is one multiply per element (no reduction), so
// GPU and CPU compute the identical float — exact up to the single float multiply.
static bool check_embedding_q8(int64_t vocab, int64_t hidden, int64_t n, double tol,
                               std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor table({vocab, hidden});
    for (int64_t i = 0; i < table.numel(); ++i) table[i] = dist(rng);
    std::uniform_int_distribution<int64_t> idd(0, vocab - 1);
    std::vector<int64_t> ids(static_cast<size_t>(n));
    for (auto& v : ids) v = idd(rng);

    QTensor q = quantize_q8(table);
    Tensor o_cpu = embedding_q8(q, ids);  // oracle

    Tensor ws({vocab});
    for (int64_t o = 0; o < vocab; ++o) ws[o] = q.scale[static_cast<size_t>(o)];
    Tensor codesd = to_device_i8(q.q.data(), {vocab, hidden}), wsd = to_device(ws);
    Tensor o_gpu = to_host(cuda_embedding_q8(codesd, wsd, ids));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < o_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(o_cpu[i]) - o_gpu[i]));
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: embed_q8 [%6lld x %4lld] gather n=%lld max|diff|=%.3e (tol %.0e) %s\n",
                (long long)vocab, (long long)hidden, (long long)n, maxdiff, tol, ok ? "ok" : "FAIL");
    return ok;
}

// Weight-only int8 linear (G5d) vs the CPU linear_q8 oracle. quantize_q8(w) gives int8 codes + per-row
// scales (uploaded); x stays fp32. The codes are identical, so the only gap is the fp32 (GPU) vs
// double (CPU dot_qf32) reduction order — the same ~1e-3 drift as the fp32 tiled GEMM, not the
// fp16/W8A8 cost. The ragged-k case exercises the kernel's kk+c<k bound.
static bool check_linear_q8(int64_t m, int64_t n, int64_t k, bool with_bias, double tol,
                            std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);

    QTensor qw = quantize_q8(w);
    Tensor y_cpu = linear_q8(x, qw, with_bias ? &bias : nullptr);  // oracle

    Tensor ws({n});
    for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
    Tensor xd = to_device(x), codesd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
    Tensor bd = to_device(bias);
    Tensor y_gpu = to_host(cuda_linear_q8(xd, codesd, wsd, with_bias ? &bd : nullptr));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]));
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: linear_q8 [%4lld x %4lld] @ [%5lld x %4lld]^T %-7s max|diff|=%.3e (tol %.0e) %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, with_bias ? "+bias" : "no-bias",
                maxdiff, tol, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    if (!cuda_available()) {
        std::printf("test_cuda: no CUDA device visible — skipping\n");
        return 0;  // graceful skip on a CPU-only machine
    }
    std::mt19937 rng(1234);  // fixed seed: deterministic, per CLAUDE.md
    bool ok = true;
    ok &= check_linear(4, 896, 896, 5e-3, DType::F32, rng);     // warp-GEMV (decode)
    // Prefill tiled GEMM (G5c+ float4-vectorized). The dispatch picks the tile by output width n:
    // narrow n -> 64×64, huge n (>=8192) -> 128×128. Cover both, square + ragged m, to gate each.
    ok &= check_linear(128, 896, 896, 5e-3, DType::F32, rng);   // tiled-vec 64×64, square (prefill q/o)
    ok &= check_linear(128, 4864, 896, 5e-3, DType::F32, rng);  // tiled-vec 64×64, wide n (gate/up)
    ok &= check_linear(100, 896, 4864, 5e-3, DType::F32, rng);  // tiled-vec 64×64, ragged m + wide k (down)
    ok &= check_linear(128, 8192, 896, 5e-3, DType::F32, rng);  // tiled-vec 128×128, large n (lm_head path)
    ok &= check_linear(100, 8192, 896, 5e-3, DType::F32, rng);  // tiled-vec 128×128, large n + ragged m

    // Double-buffered projection GEMM (G5 micro-gain): bit-identical to the tiled-vec kernel (only the
    // load timing changes), so it must meet the same oracle on the narrow-n projection shapes it routes
    // — square, wide n, and the ragged-m + wide-k (down) case that exercises the prefetch's m-bound.
    cuda_policy().use_dbuf = true;
    ok &= check_linear(128, 896, 896, 5e-3, DType::F32, rng);   // dbuf 64×64, square (q/o)
    ok &= check_linear(128, 4864, 896, 5e-3, DType::F32, rng);  // dbuf 64×64, wide n (gate/up)
    ok &= check_linear(100, 896, 4864, 5e-3, DType::F32, rng);  // dbuf 64×64, ragged m + wide k (down)
    cuda_policy().use_dbuf = false;

    // Tensor-core (fp32 weight, fp16 staged) — the max|diff| is the fp16 cost. n<8192 hits the 64²
    // kernel; n>=8192 the 128² warp-tiled kernel (the lm_head path).
    cuda_policy().use_wmma = true;
    ok &= check_linear(128, 896, 896, 3e-1, DType::F32, rng);   // wmma 64², square
    ok &= check_linear(100, 896, 4864, 1e0, DType::F32, rng);   // wmma 64², ragged + wide k
    ok &= check_linear(128, 8192, 896, 5e-1, DType::F32, rng);  // wmma 128² warp-tiled, large n
    ok &= check_linear(100, 8192, 896, 5e-1, DType::F32, rng);  // wmma 128², large n + ragged m
    cuda_policy().use_wmma = false;

    // fp16 WEIGHTS (G5d): gemv-h (decode); prefill projections (n<8192) -> CUDA-core float4 tiled
    // (tiled-h: fp16-weight-only error, fp32 compute, so tighter than wmma's fp16-accumulate);
    // lm_head (n>=8192) -> 128² warp-tiled tensor cores (wmma-h).
    ok &= check_linear(4, 896, 896, 1e-1, DType::F16, rng);      // gemv-h (decode, fp16 weight)
    ok &= check_linear(128, 896, 896, 1e-1, DType::F16, rng);    // tiled-h (prefill projection, fp16 weight)
    ok &= check_linear(100, 896, 4864, 3e-1, DType::F16, rng);   // tiled-h, ragged m + wide k
    ok &= check_linear(128, 8192, 896, 5e-1, DType::F16, rng);   // wmma-h 128² warp-tiled, large n (lm_head)

    // bf16 WEIGHTS (B1): the same GEMV/tiled dispatch reading __nv_bfloat16 through ldf/load4 —
    // fp32 compute, so the max|diff| is purely the bf16 rounding of the weight (8 mantissa bits,
    // 2^3 coarser than fp16 → the tolerances sit ~8× above the fp16-weight rows). The ragged-n
    // case exercises the scalar-tiled fallback (the one path fp16 sends to wmma instead).
    ok &= check_linear(4, 896, 896, 8e-1, DType::BF16, rng);     // gemv-b (decode, bf16 weight)
    ok &= check_linear(128, 896, 896, 8e-1, DType::BF16, rng);   // tiled-b (prefill projection)
    ok &= check_linear(100, 896, 4864, 2e0, DType::BF16, rng);   // tiled-b, ragged m + wide k
    ok &= check_linear(128, 8192, 896, 8e-1, DType::BF16, rng);  // tiled-b 128-aligned huge n (lm_head, B1)
    ok &= check_linear(20, 100, 90, 8e-1, DType::BF16, rng);     // stile-b: ragged n+k fallback

    // Embedding gather (G5d/B1): fp32 table is an exact copy; fp16/bf16 tables (embed_tokens path)
    // cost only that dtype's rounding of the looked-up rows.
    ok &= check_embedding(2048, 896, 8, DType::F32, 1e-6, rng);   // fp32 table — exact
    ok &= check_embedding(2048, 896, 8, DType::F16, 1e-3, rng);   // fp16 table — fp16 rounding
    ok &= check_embedding(2048, 896, 8, DType::BF16, 8e-3, rng);  // bf16 table — bf16 rounding (8× fp16)

    // Warp-per-query + online-softmax attention (G5f): prefill (limit>32 → lane key-striding),
    // decode (sq=1, large sk), and full (non-causal). Tight tol — only the key sums reorder vs the
    // CPU oracle. The causal cases exercise the empty-lane combine: query i sees limit=i+1 keys, so
    // for i<31 the lanes i+1..31 have no keys (m=−inf) and must merge to 0, not NaN.
    ok &= check_attention(2, 40, 40, 64, true, 0, 1e-3, rng);    // prefill, causal, limit up to 40
    ok &= check_attention(2, 1, 100, 64, true, 99, 1e-3, rng);   // decode, large sk
    ok &= check_attention(3, 16, 16, 64, false, 0, 1e-3, rng);   // full (non-causal)
    ok &= check_attention(1, 64, 64, 64, true, 0, 1e-3, rng);    // longer prefill, multiple lane passes
    ok &= check_attention(4, 5, 5, 64, true, 0, 1e-3, rng);      // tiny causal: most lanes empty (limit 1..5)

    // Opt-in shared-memory K/V tiled kernel (G5f-tiled): same cases must meet the oracle (it's
    // bit-identical to the non-tiled path — same key order). Includes a ragged shape (sq,sk not
    // multiples of 32/warps-per-block) to exercise partial tiles and inactive warps.
    cuda_policy().use_tiled_attn = true;
    ok &= check_attention(2, 40, 40, 64, true, 0, 1e-3, rng);    // tiled prefill, multi-tile
    ok &= check_attention(3, 16, 16, 64, false, 0, 1e-3, rng);   // tiled full (non-causal)
    ok &= check_attention(4, 5, 5, 64, true, 0, 1e-3, rng);      // tiled ragged: sq < warps/block
    ok &= check_attention(2, 130, 130, 64, true, 0, 1e-3, rng);  // tiled ragged tiles + query blocks
    cuda_policy().use_tiled_attn = false;

    // Paged tiled kernel (backlog): bit-identical to the non-tiled paged kernel on a real
    // CudaPagedKVCache. D=128 rows exercise the P0 default-on dispatch (force=false); D=64 rows
    // force it (the 0.5B A/B shape). Small bs makes one 32-key smem slab gather across many blocks;
    // ctx not block-aligned starts the gather mid-block; small t leaves most lanes/warps empty.
    ok &= check_paged_tiled(12, 2, 40, 0, 128, 16, true, false, rng);   // 1.5B prefill, default-on
    ok &= check_paged_tiled(12, 2, 5, 13, 128, 4, true, false, rng);    // verify shape onto a populated cache
    ok &= check_paged_tiled(12, 2, 130, 30, 128, 16, true, false, rng); // ragged query blocks + long ctx
    ok &= check_paged_tiled(14, 2, 40, 23, 64, 8, true, true, rng);     // 0.5B shape, forced, odd offset
    ok &= check_paged_tiled(14, 2, 16, 8, 64, 4, false, true, rng);     // non-causal, forced

    // Flash-Decoding / split-KV (G5g): the KV is split across warps and recombined, so the reduction
    // order differs from the non-split kernel — match the CPU oracle within tol (same 1e-3, not
    // bit-identical). Long-context decode (sq=1) is where split_count engages many splits; short
    // context and prefill fall back to the warp kernel (split_count<2), exercised here too.
    cuda_policy().use_split_attn = true;
    ok &= check_attention(14, 1, 1024, 64, true, 1023, 1e-3, rng);  // decode, splits engage (~8)
    ok &= check_attention(14, 1, 4096, 64, true, 4095, 1e-3, rng);  // decode, more splits (~32)
    ok &= check_attention(2, 1, 100, 64, true, 99, 1e-3, rng);      // short decode: num_splits=1 fallback
    ok &= check_attention(2, 40, 40, 64, true, 0, 1e-3, rng);       // prefill: num_splits=1 fallback (no-op)
    cuda_policy().use_split_attn = false;

    // W8A8 DP4A int8 GEMM (G5d): the projection shapes, prefill + decode, with and without bias.
    // The integer core is identical to the CPU oracle, so the tolerance is just the float dequant.
    // m<=kGemvMaxM(16) routes to the decode GEMV (one warp per output, ¼ the fp32 weight bytes — the
    // P1 decode cap), m>16 to the tiled GEMM; cover both, incl. m=1 (true decode) + wide k.
    ok &= check_w8a8(128, 896, 896, true, 2e-3, rng);    // q/o prefill, +bias (tiled)
    ok &= check_w8a8(128, 4864, 896, false, 2e-3, rng);  // gate/up prefill, no bias (tiled)
    ok &= check_w8a8(100, 896, 4864, false, 2e-3, rng);  // down: ragged m + wide k (tiled)
    ok &= check_w8a8(1, 896, 896, true, 2e-3, rng);      // q/o decode, m=1 (GEMV), +bias
    ok &= check_w8a8(1, 4864, 896, false, 2e-3, rng);    // gate/up decode, m=1 (GEMV)
    ok &= check_w8a8(16, 896, 4864, false, 2e-3, rng);   // batched decode, m=16 wide k (GEMV boundary)
    // The GEMV must be BIT-IDENTICAL to the tiled kernel (int32 exact) — the strong parity claim.
    ok &= check_w8a8_gemv_bitident(1, 896, 896, true, rng);     // decode projection, +bias
    ok &= check_w8a8_gemv_bitident(8, 4864, 896, false, rng);   // batched decode, wide n
    ok &= check_w8a8_gemv_bitident(16, 896, 4864, false, rng);  // m=16 boundary, wide k

    // Weight-only int8 embed/lm_head (G5d): the gather is exact (one multiply); the linear matches the
    // CPU linear_q8 oracle within the fp32-reduction tolerance. m<=16 routes to the decode GEMV (one
    // warp per output, ¼ the fp32 bytes), m>16 to the tiled GEMM — cover both at the lm_head's huge n,
    // a projection shape with bias, ragged m + wide k, and a ragged k (k%16!=0). The GEMV cases include
    // m=1 (true decode) and a +bias case (the dequant-scale+bias store the tiled path also exercises).
    ok &= check_embedding_q8(2048, 896, 8, 1e-4, rng);        // int8 table gather — exact
    ok &= check_linear_q8(1, 8192, 896, false, 5e-3, rng);    // lm_head decode, true m=1 (GEMV, huge n)
    ok &= check_linear_q8(4, 8192, 896, false, 5e-3, rng);    // lm_head batched decode (GEMV, huge n)
    ok &= check_linear_q8(2, 896, 896, true, 5e-3, rng);      // GEMV + bias (decode projection shape)
    ok &= check_linear_q8(128, 8192, 896, false, 5e-3, rng);  // lm_head prefill (tiled, large m, huge n)
    ok &= check_linear_q8(128, 896, 896, true, 5e-3, rng);    // projection shape, +bias (tiled)
    ok &= check_linear_q8(100, 896, 4864, false, 5e-3, rng);  // ragged m + wide k (tiled)
    ok &= check_linear_q8(8, 100, 90, false, 5e-3, rng);      // GEMV, ragged k (k%16!=0 — the k-bound)

    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
