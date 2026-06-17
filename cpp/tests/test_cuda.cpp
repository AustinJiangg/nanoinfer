// G1 parity: the CUDA backend's linear() — a hand-written naive GEMM — reproduces the
// CPU backend's linear() within float tolerance, on synthetic data (no model weights).
//
// NOT bit-identical: the GPU accumulates each dot product in float and in a different
// order than the CPU's double-accumulated SIMD dot, so ~1e-4 of drift is expected and
// correct — this is the concrete demonstration of CLAUDE.md's "GPU parity is not
// bit-identical" rule. The CPU backend is the oracle (it is itself HF-parity-locked).
//
// Self-contained, so it joins ctest; skips at runtime if no GPU is visible. Only built
// when -DNI_CUDA=ON.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "backend.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "tensor.hpp"

using namespace ni;

int main() {
    if (!cuda_available()) {
        std::printf("test_cuda: no CUDA device visible — skipping\n");
        return 0;  // graceful skip on a CPU-only machine
    }

    // Shapes like a Qwen2.5-0.5B projection: a few rows, hidden=896 in and out.
    const int64_t m = 4, k = 896, n = 896;
    std::mt19937 rng(1234);  // fixed seed: deterministic, per CLAUDE.md
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Tensor x({m, k}), w({n, k}), bias({n});
    for (int64_t i = 0; i < x.numel(); ++i) x[i] = dist(rng);
    for (int64_t i = 0; i < w.numel(); ++i) w[i] = dist(rng);
    for (int64_t i = 0; i < bias.numel(); ++i) bias[i] = dist(rng);

    // CPU reference (the oracle).
    CpuBackend cpu;
    Tensor y_cpu = cpu.linear(x, w, &bias);

    // GPU: move inputs to the device, run the kernel, bring the result back.
    CudaBackend gpu;
    Tensor xd = to_device(x), wd = to_device(w), bd = to_device(bias);
    Tensor y_gpu = to_host(gpu.linear(xd, wd, &bd));

    double maxdiff = 0.0, sumdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        const double d = std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]);
        maxdiff = std::max(maxdiff, d);
        sumdiff += d;
    }
    const double meandiff = sumdiff / static_cast<double>(y_cpu.numel());

    const double tol = 5e-3;  // float-vs-double accumulation over k=896; well above noise
    std::printf("test_cuda: linear [%lld x %lld] @ [%lld x %lld]^T -> [%lld x %lld]\n",
                (long long)m, (long long)k, (long long)n, (long long)k, (long long)m, (long long)n);
    std::printf("test_cuda: max|diff|=%.3e  mean|diff|=%.3e  (tol %.0e)\n",
                maxdiff, meandiff, tol);
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
