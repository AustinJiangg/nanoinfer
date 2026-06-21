// G1/G5 parity: the CUDA backend's linear() reproduces the CPU backend's linear() within
// tolerance, on synthetic data (no model weights). Covers every kernel the dispatch picks:
// warp-GEMV (m<=16, decode), register-tiled GEMM (m>16, prefill), tensor-core wmma (m>16,
// opt-in), and the fp16-WEIGHT paths (gemv-h / wmma-h, weight uploaded as half — G5d). The
// golden e2e tests (run_cuda_parity/cache) only use short sequences, so the prefill + fp16
// kernels meet the oracle here.
//
// NOT bit-identical: the GPU accumulates in float and in a different order than the CPU's
// double-accumulated SIMD dot (~1e-4 drift). The wmma/fp16 paths are looser still — fp16
// rounds operands to ~3 decimal digits, so their printed max|diff| IS the result: the fp16
// accuracy cost vs the fp32 oracle. The CPU backend is the oracle (itself HF-parity-locked).
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

// One shape: GPU linear() (with bias) vs the CPU oracle. f16w uploads the weight as half
// (exercising the fp16 kernels). ok if max|diff| < tol.
static bool check_linear(int64_t m, int64_t n, int64_t k, double tol, bool f16w,
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
    Tensor wd = f16w ? to_device_f16(w) : to_device(w);  // fp16 or fp32 weight buffer
    Tensor y_gpu = to_host(gpu.linear(xd, wd, &bd));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]));
    const bool ok = maxdiff < tol;
    const char* kern = f16w ? (m <= 16 ? "gemv-h" : (n >= 8192 ? "wmma-h" : "tiled-h"))
                            : (g_cuda_use_wmma ? "wmma" : (m <= 16 ? "gemv" : "tiled"));
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

// Embedding gather vs the CPU oracle (G5d). Random [vocab, hidden] table, random ids; an fp32
// table matches exactly (a pure copy — no arithmetic), an fp16 table within fp16 rounding (the
// looked-up rows are half-rounded). Covers the fp16-table path now that embed_tokens goes fp16.
static bool check_embedding(int64_t vocab, int64_t hidden, int64_t n, bool f16, double tol,
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
    Tensor td = f16 ? to_device_f16(table) : to_device(table);
    Tensor o_gpu = to_host(gpu.embedding(td, ids));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < o_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(o_cpu[i]) - o_gpu[i]));
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: embed [%6lld x %4lld] gather n=%lld (%-3s) max|diff|=%.3e (tol %.0e) %s\n",
                (long long)vocab, (long long)hidden, (long long)n, f16 ? "f16" : "f32", maxdiff, tol,
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
    ok &= check_linear(4, 896, 896, 5e-3, false, rng);     // warp-GEMV (decode)
    // Prefill tiled GEMM (G5c+ float4-vectorized). The dispatch picks the tile by output width n:
    // narrow n -> 64×64, huge n (>=8192) -> 128×128. Cover both, square + ragged m, to gate each.
    ok &= check_linear(128, 896, 896, 5e-3, false, rng);   // tiled-vec 64×64, square (prefill q/o)
    ok &= check_linear(128, 4864, 896, 5e-3, false, rng);  // tiled-vec 64×64, wide n (gate/up)
    ok &= check_linear(100, 896, 4864, 5e-3, false, rng);  // tiled-vec 64×64, ragged m + wide k (down)
    ok &= check_linear(128, 8192, 896, 5e-3, false, rng);  // tiled-vec 128×128, large n (lm_head path)
    ok &= check_linear(100, 8192, 896, 5e-3, false, rng);  // tiled-vec 128×128, large n + ragged m

    // Tensor-core (fp32 weight, fp16 staged) — the max|diff| is the fp16 cost. n<8192 hits the 64²
    // kernel; n>=8192 the 128² warp-tiled kernel (the lm_head path).
    g_cuda_use_wmma = true;
    ok &= check_linear(128, 896, 896, 3e-1, false, rng);   // wmma 64², square
    ok &= check_linear(100, 896, 4864, 1e0, false, rng);   // wmma 64², ragged + wide k
    ok &= check_linear(128, 8192, 896, 5e-1, false, rng);  // wmma 128² warp-tiled, large n
    ok &= check_linear(100, 8192, 896, 5e-1, false, rng);  // wmma 128², large n + ragged m
    g_cuda_use_wmma = false;

    // fp16 WEIGHTS (G5d): gemv-h (decode); prefill projections (n<8192) -> CUDA-core float4 tiled
    // (tiled-h: fp16-weight-only error, fp32 compute, so tighter than wmma's fp16-accumulate);
    // lm_head (n>=8192) -> 128² warp-tiled tensor cores (wmma-h).
    ok &= check_linear(4, 896, 896, 1e-1, true, rng);      // gemv-h (decode, fp16 weight)
    ok &= check_linear(128, 896, 896, 1e-1, true, rng);    // tiled-h (prefill projection, fp16 weight)
    ok &= check_linear(100, 896, 4864, 3e-1, true, rng);   // tiled-h, ragged m + wide k
    ok &= check_linear(128, 8192, 896, 5e-1, true, rng);   // wmma-h 128² warp-tiled, large n (lm_head)

    // Embedding gather (G5d): fp32 table is an exact copy; fp16 table (embed_tokens path) costs
    // only the fp16 rounding of the looked-up rows.
    ok &= check_embedding(2048, 896, 8, false, 1e-6, rng);  // fp32 table — exact
    ok &= check_embedding(2048, 896, 8, true, 1e-3, rng);   // fp16 table — fp16 rounding

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
    g_cuda_use_tiled_attn = true;
    ok &= check_attention(2, 40, 40, 64, true, 0, 1e-3, rng);    // tiled prefill, multi-tile
    ok &= check_attention(3, 16, 16, 64, false, 0, 1e-3, rng);   // tiled full (non-causal)
    ok &= check_attention(4, 5, 5, 64, true, 0, 1e-3, rng);      // tiled ragged: sq < warps/block
    ok &= check_attention(2, 130, 130, 64, true, 0, 1e-3, rng);  // tiled ragged tiles + query blocks
    g_cuda_use_tiled_attn = false;

    // W8A8 DP4A int8 GEMM (G5d): the projection shapes, prefill + small m, with and without bias.
    // The integer core is identical to the CPU oracle, so the tolerance is just the float dequant.
    ok &= check_w8a8(128, 896, 896, true, 2e-3, rng);    // q/o prefill, +bias
    ok &= check_w8a8(128, 4864, 896, false, 2e-3, rng);  // gate/up prefill, no bias
    ok &= check_w8a8(100, 896, 4864, false, 2e-3, rng);  // down: ragged m + wide k
    ok &= check_w8a8(4, 896, 896, true, 2e-3, rng);      // small m (decode-ish), +bias

    // Weight-only int8 embed/lm_head (G5d): the gather is exact (one multiply); the linear matches the
    // CPU linear_q8 oracle within the fp32-reduction tolerance. Cover the lm_head's huge n at decode +
    // prefill m, a projection shape with bias, ragged m + wide k, and a ragged k (k%16!=0).
    ok &= check_embedding_q8(2048, 896, 8, 1e-4, rng);        // int8 table gather — exact
    ok &= check_linear_q8(4, 8192, 896, false, 5e-3, rng);    // lm_head decode (small m, huge n)
    ok &= check_linear_q8(128, 8192, 896, false, 5e-3, rng);  // lm_head prefill (large m, huge n)
    ok &= check_linear_q8(128, 896, 896, true, 5e-3, rng);    // projection shape, +bias
    ok &= check_linear_q8(100, 896, 4864, false, 5e-3, rng);  // ragged m + wide k
    ok &= check_linear_q8(8, 100, 90, false, 5e-3, rng);      // ragged k (k%16!=0 — the k-bound)

    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
