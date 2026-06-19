// G1/G5 parity: the CUDA backend's linear() reproduces the CPU backend's linear() within
// tolerance, on synthetic data (no model weights). Covers every kernel the dispatch picks:
// the warp-GEMV (m<=16, decode), the register-tiled GEMM (m>16, prefill), and the tensor-core
// wmma GEMM (m>16, opt-in). The golden e2e tests (run_cuda_parity/cache) only use short
// sequences (m<=16), so this is where the prefill kernels meet the oracle.
//
// NOT bit-identical: the GPU accumulates in float and in a different order than the CPU's
// double-accumulated SIMD dot (~1e-4 drift, expected — CLAUDE.md's "GPU parity is not
// bit-identical"). The wmma path is looser still: fp16 rounds the inputs to ~3 decimal
// digits, so its printed max|diff| IS the measured result — the fp16 accuracy cost vs the
// fp32 oracle. The CPU backend is the oracle (itself HF-parity-locked).
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

// One shape: GPU linear() (with bias) vs the CPU oracle. ok if max|diff| < tol.
static bool check_linear(int64_t m, int64_t n, int64_t k, double tol, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);

    CpuBackend cpu;
    Tensor y_cpu = cpu.linear(x, w, &bias);  // the oracle

    CudaBackend gpu;
    Tensor xd = to_device(x), wd = to_device(w), bd = to_device(bias);
    Tensor y_gpu = to_host(gpu.linear(xd, wd, &bd));

    double maxdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i)
        maxdiff = std::max(maxdiff, std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]));
    const bool ok = maxdiff < tol;
    const char* kern = g_cuda_use_wmma ? "wmma" : (m <= 16 ? "gemv" : "tiled");
    std::printf("test_cuda: [%4lld x %4lld] @ [%6lld x %4lld]^T (%-5s)  max|diff|=%.3e (tol %.0e) %s\n",
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
    ok &= check_linear(4, 896, 896, 5e-3, rng);     // warp-GEMV (decode)
    ok &= check_linear(128, 896, 896, 5e-3, rng);   // tiled, square (prefill q/o shape)
    ok &= check_linear(128, 4864, 896, 5e-3, rng);  // tiled, wide n (gate/up shape)
    ok &= check_linear(100, 896, 4864, 5e-3, rng);  // tiled, ragged m + wide k (down shape)

    // Tensor-core path (G5d): fp16 inputs -> a much looser tolerance. The max|diff| printed
    // here is the result: the fp16 accuracy cost (vs the fp32 oracle), not a tight bound.
    g_cuda_use_wmma = true;
    ok &= check_linear(128, 896, 896, 3e-1, rng);   // wmma, square
    ok &= check_linear(128, 4864, 896, 3e-1, rng);  // wmma, wide n
    ok &= check_linear(100, 896, 4864, 1e0, rng);   // wmma, ragged m + wide k (more accumulation)
    g_cuda_use_wmma = false;

    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
