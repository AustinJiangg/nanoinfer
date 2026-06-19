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
#include <cstdio>
#include <random>

#include "backend.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
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
    const char* kern = f16w ? (m <= 16 ? "gemv-h" : "wmma-h")
                            : (g_cuda_use_wmma ? "wmma" : (m <= 16 ? "gemv" : "tiled"));
    std::printf("test_cuda: [%4lld x %4lld] @ [%6lld x %4lld]^T (%-6s) max|diff|=%.3e (tol %.0e) %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, kern, maxdiff, tol,
                ok ? "ok" : "FAIL");
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
    ok &= check_linear(128, 896, 896, 5e-3, false, rng);   // tiled, square (prefill q/o)
    ok &= check_linear(128, 4864, 896, 5e-3, false, rng);  // tiled, wide n (gate/up)
    ok &= check_linear(100, 896, 4864, 5e-3, false, rng);  // tiled, ragged m + wide k (down)

    // Tensor-core (fp32 weight, fp16 staged) — the max|diff| is the fp16 cost.
    g_cuda_use_wmma = true;
    ok &= check_linear(128, 896, 896, 3e-1, false, rng);   // wmma, square
    ok &= check_linear(100, 896, 4864, 1e0, false, rng);   // wmma, ragged + wide k
    g_cuda_use_wmma = false;

    // fp16 WEIGHTS (G5d): weight uploaded as half -> gemv-h (decode) / wmma-h (prefill).
    ok &= check_linear(4, 896, 896, 1e-1, true, rng);      // gemv-h (decode, fp16 weight)
    ok &= check_linear(128, 896, 896, 3e-1, true, rng);    // wmma-h (prefill, fp16 weight)
    ok &= check_linear(100, 896, 4864, 1e0, true, rng);    // wmma-h, ragged + wide k

    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
