// G5f isolated attention micro-bench — the sibling of run_cuda_bench (which times linear()). The
// full-step decode bench dilutes attention (it's a minority of a prefill step, and the GEMMs
// dominate), so to see what the shared-memory K/V tiling does to the KERNEL we time attention() on
// its own. Three kernels on the same inputs: naive (1 thread/query, G5e floor), non-tiled online
// (G5f-online), and shared-mem tiled (G5f-tiled). Also checks tiled == non-tiled bit-for-bit
// (max|diff|=0 — the tiling keeps the key order) and both ~1e-6 vs naive. Only built with -DNI_CUDA.
//
// Honest read of the result: for Qwen2.5-0.5B (H=14, D=64) the per-layer KV fits in the 4070S's
// large L2 even at long context, so the global→smem reuse the tiling buys is already served by L2 —
// expect the tiled kernel to roughly TIE the non-tiled one here, and to pull ahead only when K/V
// outgrow L2 (bigger model / batch, or a smaller-L2 GPU). The win that DID land is online softmax
// (one pass) over the naive two-pass-style floor.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.hpp"
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "tensor.hpp"

using namespace ni;

namespace {
void cuda_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}

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

double maxdiff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (int64_t i = 0; i < a.numel(); ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}
}  // namespace

int main() {
    if (!cuda_available()) {
        std::printf("run_cuda_attn_bench: no CUDA device visible — skipping\n");
        return 0;
    }
    // Qwen2.5-0.5B attention shapes after repeat_kv: H=14 query heads, head_dim=64, causal, the
    // query block sitting at the end of the context (query_offset = sk - sq). Prefill sweeps sq=sk;
    // the last row is a decode step (sq=1) over a long context — there tiling has no query reuse, so
    // it should fall back to the non-tiled kernel (the dispatch sends sq=1 there) and tie it exactly.
    const int64_t H = 14, D = 64;
    struct Cfg {
        int64_t sq, sk;
    };
    const std::vector<Cfg> cfgs = {{128, 128}, {512, 512}, {1024, 1024}, {2048, 2048}, {1, 4096}};

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CudaBackend gpu;

    std::printf("run_cuda_attn_bench: isolated attention(), Qwen2.5-0.5B shape (H=%lld, D=%lld), causal\n",
                (long long)H, (long long)D);
    std::printf("%5s %6s | %9s %9s %9s | %7s %7s | %10s %10s\n", "sq", "sk", "naive ms", "notile ms",
                "tiled ms", "t/notile", "t/naive", "tile==notl", "notl~naive");

    bool ok = true;
    for (const Cfg& c : cfgs) {
        const int64_t sq = c.sq, sk = c.sk, qoff = sk - sq;
        Tensor q({H, sq, D}), k({H, sk, D}), v({H, sk, D});
        for (int64_t i = 0; i < q.numel(); ++i) q[i] = dist(rng);
        for (int64_t i = 0; i < k.numel(); ++i) k[i] = dist(rng);
        for (int64_t i = 0; i < v.numel(); ++i) v[i] = dist(rng);
        Tensor qd = to_device(q), kd = to_device(k), vd = to_device(v);

        auto run = [&] { Tensor o = gpu.attention(qd, kd, vd, /*causal=*/true, qoff); };

        g_cuda_force_naive_attn = true;
        g_cuda_use_tiled_attn = false;
        Tensor o_naive = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_naive = time_ms(run, 30, 5);

        g_cuda_force_naive_attn = false;
        g_cuda_use_tiled_attn = false;  // non-tiled online (the default for all sq)
        Tensor o_notile = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_notile = time_ms(run, 30, 5);

        g_cuda_use_tiled_attn = true;  // opt-in shared-memory tiling (sq>1)
        Tensor o_tiled = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_tiled = time_ms(run, 30, 5);

        const double d_tn = maxdiff(o_tiled, o_notile);   // must be 0 (same key order)
        const double d_na = maxdiff(o_notile, o_naive);   // ~1e-6 (reduction reorder)
        ok &= (d_tn == 0.0) && (d_na < 1e-3);
        std::printf("%5lld %6lld | %9.3f %9.3f %9.3f | %6.2fx %6.2fx | %10.1e %10.1e\n", (long long)sq,
                    (long long)sk, ms_naive, ms_notile, ms_tiled, ms_notile / ms_tiled,
                    ms_naive / ms_tiled, d_tn, d_na);
    }
    g_cuda_force_naive_attn = false;
    g_cuda_use_tiled_attn = false;
    std::printf("run_cuda_attn_bench: %s\n", ok ? "ok" : "FAIL (parity)");
    return ok ? 0 : 1;
}
