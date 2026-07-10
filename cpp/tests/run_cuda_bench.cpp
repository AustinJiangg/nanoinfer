// G5a measurement scaffold: micro-benchmark the CUDA backend's linear() — a naive
// one-thread-per-output GEMM — against cuBLAS sgemm on the real Qwen2.5-0.5B projection
// shapes, to quantify "how far from cuBLAS" before sharpening the kernel (G5b/G5c).
//
// Synthetic data (no weights), seeded; skips if no GPU. Per shape it reports our op's
// GFLOP/s and effective GB/s (useful bytes / time) with their % of cuBLAS and of the
// 4070S roofline, the per-op alloc overhead our functional Tensor API pays (a
// cudaMalloc/free every call), and a correctness check vs cuBLAS (~1e-3, the float-
// accumulation tolerance — cuBLAS reorders the reduction just like our kernel does).
//
// cuBLAS is the YARDSTICK only — the engine's forward never calls it; the golden rule is
// that we run our own kernels. This bench is the one place the comparison lives.
//
// Shapes are selectable (P0, the 1.5B perf-retune): `run_cuda_bench [0.5b|1.5b]` sweeps that
// model's projection shapes — the GEMM analog of run_cuda_attn_bench's `<H> <D>` parameterization,
// so the wmma/dbuf/int8 A/Bs can be re-measured on the bigger model's ~3× matrices (head_dim 128,
// hidden 1536, intermediate 8960) that the roadmap predicted would wake the parked GEMM levers.
//
//   cmake -S . -B build -DNI_CUDA=ON && cmake --build build -j --target run_cuda_bench
//   ./build/run_cuda_bench            # 0.5B shapes (default)
//   ./build/run_cuda_bench 1.5b       # 1.5B shapes
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "quant.hpp"
#include "tensor.hpp"

using namespace ni;

namespace {

// RTX 4070 SUPER datasheet peaks — the roofline the kernel is measured against.
constexpr double kPeakFp32 = 35.5e12;  // FLOP/s (FMA counted as 2 ops)
constexpr double kPeakBW = 504.0e9;    // bytes/s (192-bit GDDR6X @ 21 Gbps)

void cublas_check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuBLAS error in ") + what);
}
void cuda_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

// ms/iter for a callable, timed on the GPU stream with CUDA events (after warmup).
template <class F>
double time_ms(F&& fn, int iters, int warmup) {
    for (int i = 0; i < warmup; ++i) fn();
    cudaEvent_t a, b;
    cuda_ck(cudaEventCreate(&a), "eventCreate");
    cuda_ck(cudaEventCreate(&b), "eventCreate");
    cuda_ck(cudaEventRecord(a), "eventRecord");
    for (int i = 0; i < iters; ++i) fn();
    cuda_ck(cudaEventRecord(b), "eventRecord");
    cuda_ck(cudaEventSynchronize(b), "eventSync");
    float ms = 0.0f;
    cuda_ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return static_cast<double>(ms) / iters;
}

struct Shape {
    const char* label;
    int64_t n, k;
};

}  // namespace

int main(int argc, char** argv) {
    if (!cuda_available()) {
        std::printf("run_cuda_bench: no CUDA device visible — skipping\n");
        return 0;
    }
    const std::string model = (argc > 1) ? argv[1] : "0.5b";  // "0.5b" (default) | "1.5b"

    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    // NI_WMMA=1 routes the prefill (m>16) rows through the tensor-core kernel (G5d) instead of
    // the fp32 tiled GEMM, so the GFLOP/s and max|diff| columns then report wmma vs cuBLAS.
    if (const char* e = std::getenv("NI_WMMA")) cuda_policy().use_wmma = (e[0] == '1');
    bool fp16w = false;
    if (const char* e = std::getenv("NI_FP16W")) fp16w = (e[0] == '1');
    if (const char* e = std::getenv("NI_DBUF")) cuda_policy().use_dbuf = (e[0] == '1');
    std::printf("prefill (m>16) kernel: %s\n",
                fp16w ? "wmma-h (fp16 weights)"
                      : (cuda_policy().use_wmma ? "wmma (fp32 weights, fp16 staged)"
                                                : "tiled (fp32)"));

    // Projection shapes {label, n, k} = y[m,n] = x[m,k] @ w[n,k]ᵀ, per model.
    //   0.5B: hidden=896,  intermediate=4864, n_heads=14, head_dim=64  (q/o n=896,  kv n=128)
    //   1.5B: hidden=1536, intermediate=8960, n_heads=12, head_dim=128 (q/o n=1536, kv n=256)
    // both share vocab=151936. The 1.5B shapes are the P0 target — bigger n/k, so the tensor-core
    // (wmma) and double-buffer levers that TIED on 0.5B get their predicted headroom re-measured.
    const std::vector<Shape> shapes_05 = {
        {"q/o_proj", 896, 896},   {"k/v_proj", 128, 896},     {"gate/up", 4864, 896},
        {"down", 896, 4864},      {"lm_head", 151936, 896},
    };
    const std::vector<Shape> shapes_15 = {
        {"q/o_proj", 1536, 1536}, {"k/v_proj", 256, 1536},    {"gate/up", 8960, 1536},
        {"down", 1536, 8960},     {"lm_head", 151936, 1536},
    };
    const std::vector<Shape>& shapes = (model == "1.5b") ? shapes_15 : shapes_05;
    const std::vector<int64_t> batch = {1, 16, 128};  // decode / batched-decode / prefill

    std::mt19937 rng(1234);  // fixed seed: deterministic, per CLAUDE.md
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CudaBackend gpu;
    const float alpha = 1.0f, beta = 0.0f;

    std::printf("run_cuda_bench: linear() vs cuBLAS sgemm, Qwen2.5-%s shapes\n", model.c_str());
    std::printf("roofline (RTX 4070 SUPER): %.1f TFLOP/s fp32, %.0f GB/s\n", kPeakFp32 / 1e12,
                kPeakBW / 1e9);
    std::printf("(m=1 decode, m=16 batched decode, m=128 prefill; ovhd = per-op alloc/free)\n\n");
    std::printf("%-9s %4s %7s %5s | %9s %8s %7s %5s %6s | %9s | %9s\n", "shape", "m", "n", "k",
                "GFLOP/s", "%cuBLAS", "GB/s", "%BW", "ovhd", "cuBLAS", "max|diff|");

    for (int64_t m : batch) {
        for (const Shape& sh : shapes) {
            const int64_t n = sh.n, k = sh.k;
            Tensor x({m, k}), w({n, k});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
            for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
            Tensor xd = to_device(x), wf32 = to_device(w), yd = to_device(Tensor({m, n}));
            // ours uses fp16 weights when opted in; cuBLAS always reads the fp32 copy, so the
            // max|diff| column then reports the fp16-weight cost vs the fp32 baseline.
            Tensor wd = fp16w ? to_device_f16(w) : wf32;
            float* xp = static_cast<float*>(xd.device_ptr());
            float* wp = static_cast<float*>(wf32.device_ptr());
            float* yp = static_cast<float*>(yd.device_ptr());

            // y[m,n] row-major = x[m,k] @ w[n,k]ᵀ. cuBLAS is column-major, where our
            // row-major y is the column-major matrix Cᵀ[n,m]; that equals opA(W)·opB(X)
            // with opA=T (w is [n,k], its transpose [k,n] read with lda=k), opB=N (x is
            // [m,k] = column-major [k,m] with ldb=k), ldc=n. Same arithmetic as our kernel.
            auto run_cublas = [&] {
                cublas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m, (int)k,
                                         &alpha, wp, (int)k, xp, (int)k, &beta, yp, (int)n),
                             "sgemm");
            };
            auto run_ours = [&] { Tensor y = gpu.linear(xd, wd, nullptr); };
            auto run_alloc = [&] { Tensor t = gpu.alloc({m, n}); };

            // Correctness: our kernel vs cuBLAS (both to host). cuBLAS overwrote yd above.
            run_cublas();
            cuda_ck(cudaDeviceSynchronize(), "sync");
            Tensor y_cublas = to_host(yd);
            Tensor y_ours = to_host(gpu.linear(xd, wd, nullptr));
            double maxdiff = 0.0;
            for (int64_t i = 0; i < y_ours.numel(); ++i)
                maxdiff = std::max(maxdiff, std::fabs((double)y_ours[i] - y_cublas[i]));

            const int iters = 30, warmup = 5;
            const double op_ms = time_ms(run_ours, iters, warmup);
            const double alloc_ms = time_ms(run_alloc, iters, warmup);
            const double cublas_ms = time_ms(run_cublas, iters, warmup);

            const double flop = 2.0 * m * n * k;
            const double bytes = 4.0 * (double(m) * k + double(n) * k + double(m) * n);
            const double ours_gflops = flop / (op_ms * 1e-3) / 1e9;
            const double cublas_gflops = flop / (cublas_ms * 1e-3) / 1e9;
            const double ours_gbps = bytes / (op_ms * 1e-3) / 1e9;

            std::printf("%-9s %4lld %7lld %5lld | %9.0f %7.1f%% %7.0f %4.0f%% %5.0f%% | %9.0f | %9.1e\n",
                        sh.label, (long long)m, (long long)n, (long long)k, ours_gflops,
                        100.0 * ours_gflops / cublas_gflops, ours_gbps,
                        100.0 * ours_gbps / (kPeakBW / 1e9), 100.0 * alloc_ms / op_ms,
                        cublas_gflops, maxdiff);
        }
        std::printf("\n");
    }

    // --- double-buffered projection GEMM (G5 micro-gain) vs the default tiled, prefill m=128. The
    // low-occupancy projections (down/q-o ~28 blocks, k/v 4) should gain most from hiding the K-tile
    // global load behind compute; gate/up (152 blocks) already hides it via cross-block occupancy.
    // Bit-identical (only the load TIMING moves, not the math), so the dbuf-tiled diff column must
    // read 0 — the speedup is the whole story. ---
    cuda_policy().use_wmma = false;
    fp16w = false;
    std::printf("\ndouble-buffered projection GEMM vs default tiled, prefill m=128 (A/B, bit-identical):\n");
    std::printf("%-9s %7s %5s | %10s | %10s | %8s | %10s\n", "shape", "n", "k", "tiled GF/s",
                "dbuf GF/s", "speedup", "dbuf-tiled");
    {
        const int64_t m = 128;
        for (const Shape& sh : shapes) {
            const int64_t n = sh.n, k = sh.k;
            if (n >= 8192 || n % 128 != 0 || k % 16 != 0) continue;  // only the narrow-n dbuf path
            Tensor x({m, k}), w({n, k});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
            for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
            Tensor xd = to_device(x), wd = to_device(w);
            auto run = [&] { Tensor y = gpu.linear(xd, wd, nullptr); };

            cuda_policy().use_dbuf = false;
            Tensor y_tiled = to_host(gpu.linear(xd, wd, nullptr));
            cuda_policy().use_dbuf = true;
            Tensor y_dbuf = to_host(gpu.linear(xd, wd, nullptr));
            double diff = 0.0;
            for (int64_t i = 0; i < y_tiled.numel(); ++i)
                diff = std::max(diff, std::fabs((double)y_tiled[i] - y_dbuf[i]));

            // These projection ops are microseconds at m=128, so GPU boost-clock drift between the two
            // timing blocks swamps the signal at low iter counts — take the MIN over several interleaved
            // rounds (min ≈ peak-clock, least-perturbed) to compare the kernels at the same clock.
            double tiled_ms = 1e30, dbuf_ms = 1e30;
            for (int r = 0; r < 8; ++r) {
                cuda_policy().use_dbuf = false;
                tiled_ms = std::min(tiled_ms, time_ms(run, 100, 20));
                cuda_policy().use_dbuf = true;
                dbuf_ms = std::min(dbuf_ms, time_ms(run, 100, 20));
            }
            cuda_policy().use_dbuf = false;

            const double flop = 2.0 * m * n * k;
            std::printf("%-9s %7lld %5lld | %10.0f | %10.0f | %7.2fx | %10.1e\n", sh.label,
                        (long long)n, (long long)k, flop / (tiled_ms * 1e-3) / 1e9,
                        flop / (dbuf_ms * 1e-3) / 1e9, tiled_ms / dbuf_ms, diff);
        }
    }

    // --- int8 W8A8 (DP4A) vs fp32 tiled, prefill m=128 — the COMPUTE win (fp16 only TIED here,
    // since the compute-bound projections aren't byte-bound; int8 cuts the MACs 4:1 via __dp4a).
    // The int8 time INCLUDES the per-call activation quant (realistic — the model quantizes the
    // activations every forward). "vs fp32" is the W8A8 quantization error, not a kernel check
    // (parity is test_cuda vs the CPU oracle). ---
    fp16w = false;
    cuda_policy().use_wmma = false;
    std::printf("\nint8 W8A8 (DP4A) vs fp32 tiled, prefill m=128 (int8 time incl. activation quant):\n");
    std::printf("%-9s %10s %10s %8s | %10s\n", "shape", "int8 GF/s", "fp32 GF/s", "speedup",
                "quant err");
    {
        const int64_t m = 128;
        for (const Shape& sh : shapes) {
            const int64_t n = sh.n, k = sh.k;
            if (k % 16 != 0) continue;  // DP4A tile needs k%16==0 (every Qwen linear qualifies)
            Tensor x({m, k}), w({n, k});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
            for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
            QTensor qw = quantize_q8(w);
            Tensor ws({n});
            for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
            Tensor xd = to_device(x), wf32 = to_device(w);
            Tensor wqd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
            auto run_fp32 = [&] { Tensor y = gpu.linear(xd, wf32, nullptr); };
            auto run_i8 = [&] { Tensor y = cuda_linear_w8a8(xd, wqd, wsd, nullptr); };
            Tensor y_f = to_host(gpu.linear(xd, wf32, nullptr));
            Tensor y_i = to_host(cuda_linear_w8a8(xd, wqd, wsd, nullptr));
            double qerr = 0.0;
            for (int64_t i = 0; i < y_f.numel(); ++i)
                qerr = std::max(qerr, std::fabs((double)y_f[i] - y_i[i]));
            const double fp32_ms = time_ms(run_fp32, 30, 5);
            const double i8_ms = time_ms(run_i8, 30, 5);
            const double flop = 2.0 * m * n * k;
            const double i8_gf = flop / (i8_ms * 1e-3) / 1e9;
            const double fp32_gf = flop / (fp32_ms * 1e-3) / 1e9;
            std::printf("%-9s %10.0f %10.0f %7.2fx | %10.1e\n", sh.label, i8_gf, fp32_gf,
                        fp32_ms / i8_ms, qerr);
        }
    }

    // --- weight-only int8 lm_head: decode GEMV vs the prefill-tuned tiled kernel (G5d follow-up).
    // The quantized lm_head (cuda_linear_q8) ran the tiled GEMM at every m; at decode (small m) its
    // 64-row tile leaves ~63/64 of the warps idle yet still streams the whole int8 weight, so the huge
    // lm_head runs under the bandwidth wall. The warp-per-output GEMV fixes that. A/B via
    // cuda_policy().force_tiled_q8 with everything else fixed; the two share the accumulate-then-scale math, so
    // GEMV==tiled within the fp32-reduction tolerance (the speed differs, not the result). Decode is
    // memory-bound, so effective GB/s (counting the int8 codes — the dominant traffic) is the metric. ---
    std::printf("\nweight-only int8 lm_head: decode GEMV vs tiled (cuda_linear_q8, A/B, %% of BW):\n");
    std::printf("%-9s %4s | %10s %5s | %10s %5s | %8s | %9s\n", "shape", "m", "GEMV GB/s", "%BW",
                "tiled GB/s", "%BW", "speedup", "GEMV-tiled");
    {
        const Shape lm = shapes.back();  // lm_head (151936 × hidden), per the selected model
        for (int64_t m : {int64_t(1), int64_t(16)}) {  // decode, batched decode
            const int64_t n = lm.n, k = lm.k;
            Tensor x({m, k}), w({n, k});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
            for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
            QTensor qw = quantize_q8(w);
            Tensor ws({n});
            for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
            Tensor xd = to_device(x);
            Tensor codesd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
            auto run_q8 = [&] { Tensor y = cuda_linear_q8(xd, codesd, wsd, nullptr); };

            cuda_policy().force_tiled_q8 = false;
            Tensor y_gemv = to_host(cuda_linear_q8(xd, codesd, wsd, nullptr));
            cuda_policy().force_tiled_q8 = true;
            Tensor y_tiled = to_host(cuda_linear_q8(xd, codesd, wsd, nullptr));
            double diff = 0.0;
            for (int64_t i = 0; i < y_gemv.numel(); ++i)
                diff = std::max(diff, std::fabs((double)y_gemv[i] - y_tiled[i]));

            cuda_policy().force_tiled_q8 = false;
            const double gemv_ms = time_ms(run_q8, 50, 10);
            cuda_policy().force_tiled_q8 = true;
            const double tiled_ms = time_ms(run_q8, 50, 10);
            cuda_policy().force_tiled_q8 = false;

            // int8 codes (n·k) + fp32 x (m·k) + fp32 scales (n) + fp32 y (m·n): total DRAM traffic.
            const double bytes = double(n) * k + 4.0 * (double(m) * k + n + double(m) * n);
            const double gemv_gbps = bytes / (gemv_ms * 1e-3) / 1e9;
            const double tiled_gbps = bytes / (tiled_ms * 1e-3) / 1e9;
            std::printf("%-9s %4lld | %10.0f %4.0f%% | %10.0f %4.0f%% | %7.2fx | %9.1e\n", lm.label,
                        (long long)m, gemv_gbps, 100.0 * gemv_gbps / (kPeakBW / 1e9), tiled_gbps,
                        100.0 * tiled_gbps / (kPeakBW / 1e9), tiled_ms / gemv_ms, diff);
        }
    }

    // --- W8A8 layer projections: decode GEMV vs the prefill-tuned tiled DP4A kernel (backlog, the
    // P1-named decode cap). cuda_linear_w8a8 ran the tiled DP4A GEMM at every m; at decode its 64-row
    // tile leaves ~63/64 of the warps idle yet still streams the whole int8 weight (P1 measured W8A8
    // decode 0.64× vs fp32). The warp-per-output DP4A GEMV fixes it — and because int32 accumulation is
    // exact, GEMV is BIT-IDENTICAL to the tiled kernel (diff column must print 0.0, unlike the q8 GEMV's
    // ~1e-6 fp32 reorder). A/B via cuda_policy().force_tiled_w8a8; the int8 time INCLUDES the per-call
    // activation quant. Decode is memory-bound, so effective GB/s (int8 codes = the dominant traffic). ---
    std::printf("\nint8 W8A8 layer projections: decode GEMV vs tiled (cuda_linear_w8a8, A/B, %% of BW):\n");
    std::printf("%-9s %4s | %10s %5s | %10s %5s | %8s | %9s\n", "shape", "m", "GEMV GB/s", "%BW",
                "tiled GB/s", "%BW", "speedup", "GEMV-tiled");
    for (size_t si = 0; si + 1 < shapes.size(); ++si) {  // projection shapes (skip lm_head — the q8 domain)
        const Shape& sh = shapes[si];
        const int64_t n = sh.n, k = sh.k;
        if (k % 16 != 0) continue;  // DP4A tile needs k%16==0 (every Qwen linear qualifies)
        for (int64_t m : {int64_t(1), int64_t(16)}) {  // decode, batched decode
            Tensor x({m, k}), w({n, k});
            for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
            for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
            QTensor qw = quantize_q8(w);
            Tensor ws({n});
            for (int64_t o = 0; o < n; ++o) ws[o] = qw.scale[static_cast<size_t>(o)];
            Tensor xd = to_device(x);
            Tensor wqd = to_device_i8(qw.q.data(), {n, k}), wsd = to_device(ws);
            auto run_w8a8 = [&] { Tensor y = cuda_linear_w8a8(xd, wqd, wsd, nullptr); };

            cuda_policy().force_tiled_w8a8 = false;
            Tensor y_gemv = to_host(cuda_linear_w8a8(xd, wqd, wsd, nullptr));
            cuda_policy().force_tiled_w8a8 = true;
            Tensor y_tiled = to_host(cuda_linear_w8a8(xd, wqd, wsd, nullptr));
            double diff = 0.0;
            for (int64_t i = 0; i < y_gemv.numel(); ++i)
                diff = std::max(diff, std::fabs((double)y_gemv[i] - y_tiled[i]));

            cuda_policy().force_tiled_w8a8 = false;
            const double gemv_ms = time_ms(run_w8a8, 50, 10);
            cuda_policy().force_tiled_w8a8 = true;
            const double tiled_ms = time_ms(run_w8a8, 50, 10);
            cuda_policy().force_tiled_w8a8 = false;

            // int8 codes (n·k) + int8 xq (m·k) + fp32 x read by the quant (m·k) + fp32 scales/y.
            const double bytes = double(n) * k + double(m) * k + 4.0 * (double(m) * k + n + double(m) * n);
            const double gemv_gbps = bytes / (gemv_ms * 1e-3) / 1e9;
            const double tiled_gbps = bytes / (tiled_ms * 1e-3) / 1e9;
            std::printf("%-9s %4lld | %10.0f %4.0f%% | %10.0f %4.0f%% | %7.2fx | %9.1e\n", sh.label,
                        (long long)m, gemv_gbps, 100.0 * gemv_gbps / (kPeakBW / 1e9), tiled_gbps,
                        100.0 * tiled_gbps / (kPeakBW / 1e9), tiled_ms / gemv_ms, diff);
        }
    }

    cublasDestroy(handle);
    std::printf("run_cuda_bench: ok\n");
    return 0;
}
