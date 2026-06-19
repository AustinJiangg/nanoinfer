// G1/G5 parity: the CUDA backend's linear() reproduces the CPU backend's linear() within
// float tolerance, on synthetic data (no model weights). Covers both kernels the dispatch
// picks: the warp-GEMV (m<=16, decode) and the register-tiled GEMM (m>16, prefill) — the
// golden e2e tests (run_cuda_parity/cache) only ever use short sequences (m<=16), so this is
// where the tiled prefill kernel gets its correctness check against the oracle.
//
// NOT bit-identical: the GPU accumulates each dot product in float and in a different order
// than the CPU's double-accumulated SIMD dot, so ~1e-4 of drift is expected and correct —
// the concrete demonstration of CLAUDE.md's "GPU parity is not bit-identical" rule. The CPU
// backend is the oracle (it is itself HF-parity-locked).
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

// One shape: GPU linear() (with bias) vs the CPU oracle. Returns true if within tolerance.
static bool check_linear(int64_t m, int64_t n, int64_t k, std::mt19937& rng) {
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

    double maxdiff = 0.0, sumdiff = 0.0;
    for (int64_t i = 0; i < y_cpu.numel(); ++i) {
        const double d = std::fabs(static_cast<double>(y_cpu[i]) - y_gpu[i]);
        maxdiff = std::max(maxdiff, d);
        sumdiff += d;
    }
    const double tol = 5e-3;  // float-vs-double accumulation over k; well above noise
    const bool ok = maxdiff < tol;
    std::printf("test_cuda: [%4lld x %4lld] @ [%6lld x %4lld]^T (%-5s)  max|diff|=%.3e  %s\n",
                (long long)m, (long long)k, (long long)n, (long long)k, m <= 16 ? "gemv" : "tiled",
                maxdiff, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    if (!cuda_available()) {
        std::printf("test_cuda: no CUDA device visible — skipping\n");
        return 0;  // graceful skip on a CPU-only machine
    }
    std::mt19937 rng(1234);  // fixed seed: deterministic, per CLAUDE.md
    bool ok = true;
    ok &= check_linear(4, 896, 896, rng);     // warp-GEMV (decode)
    ok &= check_linear(128, 896, 896, rng);   // tiled, square (prefill q/o shape)
    ok &= check_linear(128, 4864, 896, rng);  // tiled, wide n (gate/up shape)
    ok &= check_linear(100, 896, 4864, rng);  // tiled, ragged m + wide k (down shape)
    std::printf("test_cuda: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
