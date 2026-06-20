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
//   cmake -S . -B build -DNI_CUDA=ON && cmake --build build -j --target run_cuda_bench
//   ./build/run_cuda_bench
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

int main() {
    if (!cuda_available()) {
        std::printf("run_cuda_bench: no CUDA device visible — skipping\n");
        return 0;
    }

    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");

    // NI_WMMA=1 routes the prefill (m>16) rows through the tensor-core kernel (G5d) instead of
    // the fp32 tiled GEMM, so the GFLOP/s and max|diff| columns then report wmma vs cuBLAS.
    if (const char* e = std::getenv("NI_WMMA")) g_cuda_use_wmma = (e[0] == '1');
    if (const char* e = std::getenv("NI_FP16W")) g_cuda_fp16_weights = (e[0] == '1');
    std::printf("prefill (m>16) kernel: %s\n",
                g_cuda_fp16_weights ? "wmma-h (fp16 weights)"
                                    : (g_cuda_use_wmma ? "wmma (fp32 weights, fp16 staged)"
                                                       : "tiled (fp32)"));

    // Qwen2.5-0.5B linears (hidden=896, intermediate=4864, kv=128, vocab=151936).
    const std::vector<Shape> shapes = {
        {"q/o_proj", 896, 896},   {"k/v_proj", 128, 896},     {"gate/up", 4864, 896},
        {"down", 896, 4864},      {"lm_head", 151936, 896},
    };
    const std::vector<int64_t> batch = {1, 16, 128};  // decode / batched-decode / prefill

    std::mt19937 rng(1234);  // fixed seed: deterministic, per CLAUDE.md
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CudaBackend gpu;
    const float alpha = 1.0f, beta = 0.0f;

    std::printf("run_cuda_bench: naive linear() vs cuBLAS sgemm, Qwen2.5-0.5B shapes\n");
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
            Tensor wd = g_cuda_fp16_weights ? to_device_f16(w) : wf32;
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

    // --- int8 W8A8 (DP4A) vs fp32 tiled, prefill m=128 — the COMPUTE win (fp16 only TIED here,
    // since the compute-bound projections aren't byte-bound; int8 cuts the MACs 4:1 via __dp4a).
    // The int8 time INCLUDES the per-call activation quant (realistic — the model quantizes the
    // activations every forward). "vs fp32" is the W8A8 quantization error, not a kernel check
    // (parity is test_cuda vs the CPU oracle). ---
    g_cuda_fp16_weights = false;
    g_cuda_use_wmma = false;
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

    cublasDestroy(handle);
    std::printf("run_cuda_bench: ok\n");
    return 0;
}
