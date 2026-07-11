// G5f isolated attention micro-bench — the sibling of run_cuda_bench (which times linear()). The
// full-step decode bench dilutes attention (it's a minority of a prefill step, and the GEMMs
// dominate), so to see what the attention kernels do on their own we time attention() in isolation.
// Four kernels on the same inputs: naive (1 thread/query, G5e floor), non-tiled online (G5f-online,
// the default), shared-mem tiled (G5f-tiled), and split-KV / Flash-Decoding (G5g). Checks tiled ==
// non-tiled bit-for-bit (max|diff|=0 — tiling keeps the key order), and split ~ non-tiled within tol
// (split reorders the reduction across chunks, so ~1e-6, not 0). Only built with -DNI_CUDA.
//
// Honest read: for Qwen2.5-0.5B (H=14, D=64) the per-layer KV fits in the 4070S's large L2 even at
// long context, so the global→smem reuse TILING buys is already served by L2 — expect tiled to TIE
// non-tiled here, winning only when K/V outgrow L2 (bigger model/batch, smaller-L2 GPU). SPLIT-KV is
// a different lever: it adds parallelism (H*sq warps fill ~3% of the GPU at decode), so it wins on
// the decode rows (sq=1, long sk) and no-ops on prefill (sq large already saturates → split_count=1).
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

int main(int argc, char** argv) {
    if (!cuda_available()) {
        std::printf("run_cuda_attn_bench: no CUDA device visible — skipping\n");
        return 0;
    }
    // Attention shapes after repeat_kv: H query heads, head_dim D, causal, the query block sitting
    // at the end of the context (query_offset = sk - sq). Prefill sweeps sq=sk; the last rows are
    // decode steps (sq=1) over a long context — there tiling has no query reuse, so it falls back to
    // the non-tiled kernel (the dispatch sends sq=1 there) and ties it exactly.
    //   default: Qwen2.5-0.5B (H=14, D=64).  Pass "H D" to sweep another model —
    //   e.g. `run_cuda_attn_bench 12 128` for Qwen2.5-1.5B (head_dim doubles: 2x the per-key bytes,
    //   so L2 pressure — the thing that decides whether shared-mem TILING wins — hits at half the sk).
    int64_t H = 14, D = 64;
    if (argc > 2) {
        H = std::stoll(argv[1]);
        D = std::stoll(argv[2]);
    }
    struct Cfg {
        int64_t sq, sk;
    };
    // Prefill rows (sq=sk): split no-ops (sq large → split_count=1), a built-in control. Decode rows
    // (sq=1, growing sk): the Flash-Decoding regime, where split-KV adds the parallelism that the
    // 14-warp non-split kernel lacks — the win should grow with context.
    const std::vector<Cfg> cfgs = {{128, 128}, {512, 512}, {1024, 1024}, {2048, 2048},
                                   {1, 1024},  {1, 4096},  {1, 16384}};

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    CudaBackend gpu;

    std::printf("run_cuda_attn_bench: isolated attention(), Qwen2.5-0.5B shape (H=%lld, D=%lld), causal\n",
                (long long)H, (long long)D);
    std::printf("%5s %6s | %9s %9s %9s %9s | %8s %8s | %10s %10s\n", "sq", "sk", "naive ms",
                "notile ms", "tiled ms", "split ms", "notl/spl", "t/notile", "tile==notl", "spl~notl");

    bool ok = true;
    for (const Cfg& c : cfgs) {
        const int64_t sq = c.sq, sk = c.sk, qoff = sk - sq;
        Tensor q({H, sq, D}), k({H, sk, D}), v({H, sk, D});
        for (int64_t i = 0; i < q.numel(); ++i) q[i] = dist(rng);
        for (int64_t i = 0; i < k.numel(); ++i) k[i] = dist(rng);
        for (int64_t i = 0; i < v.numel(); ++i) v[i] = dist(rng);
        Tensor qd = to_device(q), kd = to_device(k), vd = to_device(v);

        auto run = [&] { Tensor o = gpu.attention(qd, kd, vd, /*causal=*/true, qoff); };

        cuda_policy().force_naive_attn = true;
        cuda_policy().use_tiled_attn = false;
        Tensor o_naive = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_naive = time_ms(run, 30, 5);

        cuda_policy().force_naive_attn = false;
        cuda_policy().use_tiled_attn = false;
        cuda_policy().no_tiled_attn = true;  // force the non-tiled online kernel (tiling is the D>=128 default)
        Tensor o_notile = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_notile = time_ms(run, 30, 5);

        cuda_policy().use_tiled_attn = true;  // force shared-memory tiling on (sq>1), at any head_dim
        cuda_policy().no_tiled_attn = false;
        Tensor o_tiled = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_tiled = time_ms(run, 30, 5);

        cuda_policy().use_tiled_attn = false;
        cuda_policy().no_tiled_attn = true;   // keep tiling off so split (or its warp fallback) runs, not the D>=128 tile
        cuda_policy().use_split_attn = true;  // Flash-Decoding / split-KV (engages for small sq + long sk)
        Tensor o_split = to_host(gpu.attention(qd, kd, vd, true, qoff));
        const double ms_split = time_ms(run, 30, 5);
        cuda_policy().use_split_attn = false;
        cuda_policy().no_tiled_attn = false;

        const double d_tn = maxdiff(o_tiled, o_notile);   // must be 0 (same key order)
        const double d_na = maxdiff(o_notile, o_naive);   // ~1e-6 (reduction reorder)
        const double d_sn = maxdiff(o_split, o_notile);   // ~1e-6 (split reorders across chunks)
        ok &= (d_tn == 0.0) && (d_na < 1e-3) && (d_sn < 1e-3);
        std::printf("%5lld %6lld | %9.3f %9.3f %9.3f %9.3f | %7.2fx %7.2fx | %10.1e %10.1e\n",
                    (long long)sq, (long long)sk, ms_naive, ms_notile, ms_tiled, ms_split,
                    ms_notile / ms_split, ms_notile / ms_tiled, d_tn, d_sn);
    }
    cuda_policy().force_naive_attn = false;
    cuda_policy().use_tiled_attn = false;
    cuda_policy().no_tiled_attn = false;
    cuda_policy().use_split_attn = false;

    // PAGED attend, tiled vs non-tiled (the backlog mirror of G5f-tiled — the paged kernel the
    // 1.5B long-context default config actually runs at prefill). Drives a real CudaPagedKVCache
    // (block_size 16, the harness standard); each timed attend() includes the paged_write of the
    // sq new tokens on BOTH arms, so the ratio is fair (and the write is O(n_kv·sq·D), noise next
    // to the O(H·sq·sk·D) attention). Prefill rows only — at sq=1 tiling has no query reuse and
    // the dispatch never picks it. tiled == non-tiled must hold bit-for-bit (same lane/key order).
    std::printf("\npaged attend (block_size=16): tiled vs non-tiled, GQA n_kv=%lld\n",
                (long long)(H >= 2 ? 2 : 1));
    std::printf("%5s %6s | %9s %9s | %8s | %10s\n", "sq", "sk", "notile ms", "tiled ms", "t/notile",
                "tile==notl");
    const int64_t n_kv = H >= 2 ? 2 : 1, n_rep = H / n_kv;  // Qwen2.5 GQA shape (14/2 or 12/2)
    for (const Cfg& c : cfgs) {
        const int64_t sq = c.sq, sk = c.sk, ctx = sk - sq;
        if (sq <= 1) continue;  // decode: tiling never dispatches (no reuse) — nothing to A/B
        const int64_t bs = 16;
        CudaBlockPool pool(/*num_layers=*/1, n_kv, D, bs, sk / bs + 4);
        CudaPagedKVCache paged(&pool);
        Tensor q({H, sq, D}), k({n_kv, sq, D}), v({n_kv, sq, D});
        for (int64_t i = 0; i < q.numel(); ++i) q[i] = dist(rng);
        for (int64_t i = 0; i < k.numel(); ++i) k[i] = dist(rng);
        for (int64_t i = 0; i < v.numel(); ++i) v[i] = dist(rng);
        Tensor qd = to_device(q), kd = to_device(k), vd = to_device(v);
        if (ctx > 0) {  // seed the context so the queries attend across old blocks too
            Tensor qc({H, ctx, D}), kc({n_kv, ctx, D}), vc({n_kv, ctx, D});
            for (int64_t i = 0; i < kc.numel(); ++i) kc[i] = dist(rng);
            for (int64_t i = 0; i < vc.numel(); ++i) vc[i] = dist(rng);
            Tensor qcd = to_device(qc), kcd = to_device(kc), vcd = to_device(vc);
            paged.attend(0, qcd, kcd, vcd, n_rep, true, 0);
            paged.advance(ctx);
        }
        auto run = [&] { Tensor o = paged.attend(0, qd, kd, vd, n_rep, /*causal=*/true, ctx); };

        cuda_policy().no_tiled_attn = true;
        Tensor o_notile = to_host(paged.attend(0, qd, kd, vd, n_rep, true, ctx));
        const double ms_notile = time_ms(run, 30, 5);
        cuda_policy().no_tiled_attn = false;
        cuda_policy().use_tiled_attn = true;  // force on so the A/B also runs at D<128
        Tensor o_tiled = to_host(paged.attend(0, qd, kd, vd, n_rep, true, ctx));
        const double ms_tiled = time_ms(run, 30, 5);
        cuda_policy().use_tiled_attn = false;

        const double d_tn = maxdiff(o_tiled, o_notile);  // must be 0 (same key order)
        ok &= (d_tn == 0.0);
        std::printf("%5lld %6lld | %9.3f %9.3f | %7.2fx | %10.1e\n", (long long)sq, (long long)sk,
                    ms_notile, ms_tiled, ms_notile / ms_tiled, d_tn);
    }
    std::printf("run_cuda_attn_bench: %s\n", ok ? "ok" : "FAIL (parity)");
    return ok ? 0 : 1;
}
