// CUDA backend kernels + device memory plumbing (C++ stages G1–G2).
//
// The only file compiled by nvcc. It holds the device<->host transfer helpers
// (cuda.hpp), the CudaBackend methods (cuda_backend.hpp), and the kernels they launch.
// Everything the rest of the engine sees is plain C++ (Tensor in, Tensor out).
//
// Each kernel mirrors the arithmetic of its ops.cpp counterpart exactly (so the GPU
// reproduces the CPU oracle within float tolerance — never bit-identical, because the
// kernels accumulate in float and in a different order than the CPU's double-accumulated
// SIMD reductions; see CLAUDE.md "GPU parity is not bit-identical"). They are the naive,
// readable versions — one thread per output element, no shared-memory tiling, a
// cudaDeviceSynchronize after every launch for easy error attribution. Speed is G5.
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#include "cuda/cuda_internal.cuh"  // R4: shared device pool + grid/split helpers + dtype loaders

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.hpp"
#include "quant.hpp"
#include "tensor.hpp"

namespace ni {

namespace {

// Convert an fp32 buffer to fp16 (G5d weight upload): one thread per element.
__global__ void f32_to_f16_kernel(const float* __restrict__ src, half* __restrict__ dst,
                                  int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Scaled dot-product attention, causal. One thread per (head, query) — it walks all
// visible keys twice (max, then softmax+weighted-V), recomputing scores rather than
// storing them. Two-pass max-subtract mirrors ops.cpp; acc holds the output row.
constexpr int kMaxHeadDim = 128;
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, float* __restrict__ out,
                                 int64_t H, int64_t sq, int64_t sk, int64_t D,
                                 int causal, int64_t query_offset, float scale) {
    const int64_t hi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (hi >= H * sq) return;
    const int64_t h = hi / sq, i = hi % sq;
    const float* qi = q + (h * sq + i) * D;
    const float* kh = k + h * sk * D;
    const float* vh = v + h * sk * D;
    const int64_t limit = causal ? (query_offset + i + 1) : sk;

    float maxv = -INFINITY;
    for (int64_t j = 0; j < limit; ++j) {
        const float* kj = kh + j * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        s *= scale;
        if (s > maxv) maxv = s;
    }
    float acc[kMaxHeadDim];
    for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
    float denom = 0.0f;
    for (int64_t j = 0; j < limit; ++j) {
        const float* kj = kh + j * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        const float e = expf(s * scale - maxv);
        denom += e;
        const float* vj = vh + j * D;
        for (int64_t d = 0; d < D; ++d) acc[d] += e * vj[d];
    }
    const float inv = 1.0f / denom;
    float* oi = out + (h * sq + i) * D;
    for (int64_t d = 0; d < D; ++d) oi[d] = acc[d] * inv;
}

// Attention, WARP-per-query with ONLINE SOFTMAX (G5f): the naive kernel above runs one THREAD per
// (head,query) — only H*sq threads (14*128 = 1792 at prefill), ~3% of the GPU, each grinding a
// serial key loop. This gives each (head,query) a whole WARP: the 32 lanes stride the keys (lane l
// takes keys l, l+32, …), 32× the threads and 32× less serial work per thread. G5f then makes the
// softmax ONE PASS (FlashAttention's online-softmax): each lane keeps a running (m, l, acc) over its
// keys, rescaling acc/l by exp(m_old − m_new) when the running max moves — instead of a first pass
// for the global max + a second to re-score. Same D-vector op count as the two-pass kernel, but the
// score dot is computed ONCE so K is read once not twice; the win grows with context (decode walks
// the whole KV every step). The 32 lane-partials then merge by the associative FlashAttention
// combine (m=max; a=exp(m−m); l,acc scaled by a), reduced to lane 0. Online rescaling reorders the
// per-key sums a little more than the two-pass add (~1e-6 vs the CPU oracle), still within GPU
// tolerance, golden tokens hold. The empty-lane case is clean: m=−inf ⇒ a=exp(−inf)=0 contributes
// nothing. Full warps only (warpId guard is warp-uniform), so every __shfl_*_sync sees all 32 lanes.
// This one-pass body is the prerequisite for K/V tiling — you can only stream K in blocks because the
// running max is rescaled as later blocks arrive. The paged kernel runs this same body verbatim.
__global__ void attention_warp_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, float* __restrict__ out,
                                      int64_t H, int64_t sq, int64_t sk, int64_t D, int causal,
                                      int64_t query_offset, float scale) {
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int64_t warpId = static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warpId >= H * sq) return;  // warp-uniform: all 32 lanes return together
    const int64_t h = warpId / sq, i = warpId % sq;
    const float* qi = q + (h * sq + i) * D;
    const float* kh = k + h * sk * D;
    const float* vh = v + h * sk * D;
    const int64_t limit = causal ? (query_offset + i + 1) : sk;

    // ONE PASS: this lane keeps a running max m, denom l, and output acc over its strided keys,
    // rescaling acc/l whenever a key raises the max. corr = exp(m_old − m_new) ≤ 1 (well-conditioned);
    // on the first key m=−inf ⇒ corr=exp(−inf)=0 zeroes the empty start (no NaN).
    float m = -INFINITY, l = 0.0f;
    float lacc[kMaxHeadDim];
    for (int64_t d = 0; d < D; ++d) lacc[d] = 0.0f;
    for (int64_t j = lane; j < limit; j += 32) {
        const float* kj = kh + j * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * corr + p;
        const float* vj = vh + j * D;
        for (int64_t d = 0; d < D; ++d) lacc[d] = lacc[d] * corr + p * vj[d];
        m = m_new;
    }
    // Merge the 32 lane-partials (FlashAttention combine, associative) down to lane 0, which writes.
    // Guard the both-empty case: a query with limit<32 leaves lanes limit..31 with no keys (m=−inf),
    // and −inf−(−inf)=NaN would poison exp. When m_new=−inf the whole segment is empty and must
    // contribute nothing, so force the corrections to 0 (keeping the lane at m=−inf, l=0, acc=0).
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(0xffffffff, m, off);
        const float l_o = __shfl_down_sync(0xffffffff, l, off);
        const float m_new = fmaxf(m, m_o);
        const bool empty = (m_new == -INFINITY);
        const float a = empty ? 0.0f : expf(m - m_new), a_o = empty ? 0.0f : expf(m_o - m_new);
        l = l * a + l_o * a_o;
        for (int64_t d = 0; d < D; ++d)
            lacc[d] = lacc[d] * a + __shfl_down_sync(0xffffffff, lacc[d], off) * a_o;
        m = m_new;
    }
    if (lane == 0) {
        const float inv = 1.0f / l;
        float* oi = out + (h * sq + i) * D;
        for (int64_t d = 0; d < D; ++d) oi[d] = lacc[d] * inv;
    }
}

// FlashAttention with shared-memory K/V TILING (G5f-tiled) — the IO-aware lever. attention_warp_kernel
// gives each (head,query) a warp, but every warp re-reads its keys from global/L2; here a thread BLOCK
// handles one head and a tile of warpsPerBlock queries, and stages each Bc-key slab of K/V into shared
// memory ONCE so all the block's queries read it from SRAM — warpsPerBlock× fewer global K/V reads.
// Parity is free: with Bc=32 and lane l taking key (base+l), every lane processes exactly the keys it
// would in attention_warp_kernel (l, l+32, …) in the same order, so this kernel is BIT-IDENTICAL to the
// non-tiled online kernel (hence to the paged kernel) — run_cuda_paged stays max|diff|=0 with only the
// contiguous path tiled. Online softmax (running m,l,acc per lane), so K is still read once. Inactive
// warps (qi>=sq, ragged last block) must keep hitting the block-wide __syncthreads(), so they take part
// in the cooperative load and skip only the compute/write. smem = 2·Bc·D floats (dynamic). For this
// model the per-layer KV fits in L2, so the win is modest (L2 already caches the reuse) — see ROADMAP.
template <int Bc>
__global__ void attention_warp_tiled_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                            const float* __restrict__ v, float* __restrict__ out,
                                            int64_t H, int64_t sq, int64_t sk, int64_t D, int causal,
                                            int64_t query_offset, float scale) {
    extern __shared__ float smem[];
    float* Ks = smem;             // [Bc][D]
    float* Vs = smem + Bc * D;    // [Bc][D]
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int warpInBlock = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int64_t h = blockIdx.y;                                       // one head per block (grid.y=H)
    const int64_t q0 = static_cast<int64_t>(blockIdx.x) * warpsPerBlock;  // first query in this block
    const int64_t qi = q0 + warpInBlock;                               // this warp's query (maybe >=sq)
    const bool active = (qi < sq);
    const float* qrow = active ? (q + (h * sq + qi) * D) : nullptr;
    const float* kh = k + h * sk * D;
    const float* vh = v + h * sk * D;
    const int64_t limit = active ? (causal ? (query_offset + qi + 1) : sk) : 0;
    // The block only needs key tiles up to the largest limit among its queries (the last one's).
    const int64_t q_last = (q0 + warpsPerBlock - 1 < sq) ? (q0 + warpsPerBlock - 1) : (sq - 1);
    const int64_t block_limit = causal ? (query_offset + q_last + 1) : sk;

    float m = -INFINITY, l = 0.0f;
    float lacc[kMaxHeadDim];
    for (int64_t d = 0; d < D; ++d) lacc[d] = 0.0f;

    for (int64_t base = 0; base < block_limit; base += Bc) {
        // Stage this tile into smem. The tile [base, base+valid) × D is contiguous in kh/vh; the whole
        // block cooperates (coalesced). Guard the ragged last tile against reading past sk.
        const int64_t valid = (sk - base < Bc) ? (sk - base) : Bc;
        for (int idx = static_cast<int>(threadIdx.x); idx < valid * D;
             idx += static_cast<int>(blockDim.x)) {
            Ks[idx] = kh[base * D + idx];
            Vs[idx] = vh[base * D + idx];
        }
        __syncthreads();
        // This warp's lane l scores key (base+l) from smem and online-updates — same key, same order
        // as attention_warp_kernel's `for j=lane; j<limit; j+=32`.
        if (active) {
            const int64_t j = base + lane;
            if (j < limit) {
                const float* kj = Ks + lane * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qrow[d] * kj[d];
                s *= scale;
                const float m_new = fmaxf(m, s);
                const float corr = expf(m - m_new);
                const float p = expf(s - m_new);
                l = l * corr + p;
                const float* vj = Vs + lane * D;
                for (int64_t d = 0; d < D; ++d) lacc[d] = lacc[d] * corr + p * vj[d];
                m = m_new;
            }
        }
        __syncthreads();  // all warps, before the next tile overwrites smem
    }
    // Merge the 32 lane-partials (identical to attention_warp_kernel, same empty-segment guard).
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(0xffffffff, m, off);
        const float l_o = __shfl_down_sync(0xffffffff, l, off);
        const float m_new = fmaxf(m, m_o);
        const bool empty = (m_new == -INFINITY);
        const float a = empty ? 0.0f : expf(m - m_new), a_o = empty ? 0.0f : expf(m_o - m_new);
        l = l * a + l_o * a_o;
        for (int64_t d = 0; d < D; ++d)
            lacc[d] = lacc[d] * a + __shfl_down_sync(0xffffffff, lacc[d], off) * a_o;
        m = m_new;
    }
    if (active && lane == 0) {
        const float inv = 1.0f / l;
        float* oi = out + (h * sq + qi) * D;
        for (int64_t d = 0; d < D; ++d) oi[d] = lacc[d] * inv;
    }
}

// Flash-Decoding pass 1 (G5g — split-KV attention). The warp-per-query kernel above launches only
// H*sq warps; at decode (sq=1) that is H (14) warps for the whole GPU, each walking the entire KV
// serially — so at long context attention, not the GEMV, is the decode bottleneck (G5f-online halved
// the K reads but could add no parallelism). This gives each (head,query) `num_splits` warps, one per
// contiguous KV chunk [k0,k1). Each runs the SAME one-pass online softmax as attention_warp_kernel
// over just its chunk and writes a PARTIAL (m, l, acc) — UNNORMALIZED (acc=Σ p·v, l=Σ p, NOT divided)
// — to scratch; attention_combine_kernel merges them. The chunks tile [0,sk) disjointly and causal
// clamps each to `limit`, so the merged result is the exact softmax over [0,limit) — only the
// reduction ORDER differs (chunked, not lane-strided-over-all), so it is within GPU tolerance of the
// CPU oracle but NOT bit-identical to the non-split kernel. An empty split (k0>=k1, e.g. a causal
// chunk past `limit`) stays m=−inf,l=0 and the combine's exp(−inf−M)=0 drops it (no NaN).
__global__ void attention_split_kv_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                          const float* __restrict__ v, float* __restrict__ m_part,
                                          float* __restrict__ l_part, float* __restrict__ acc_part,
                                          int64_t H, int64_t sq, int64_t sk, int64_t D, int causal,
                                          int64_t query_offset, float scale, int num_splits,
                                          int64_t chunk) {
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int64_t gwarp = static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int64_t nq = H * sq;
    if (gwarp >= nq * num_splits) return;  // warp-uniform
    const int64_t hi = gwarp / num_splits;  // (head,query) pair; consecutive warps share hi (L2-friendly)
    const int64_t s = gwarp % num_splits;   // which KV split
    const int64_t h = hi / sq, i = hi % sq;
    const float* qi = q + (h * sq + i) * D;
    const float* kh = k + h * sk * D;
    const float* vh = v + h * sk * D;
    const int64_t limit = causal ? (query_offset + i + 1) : sk;
    const int64_t k0 = s * chunk;                                   // this split's key range [k0,k1)
    const int64_t k1raw = (k0 + chunk < sk) ? (k0 + chunk) : sk;
    const int64_t k1 = causal ? (k1raw < limit ? k1raw : limit) : k1raw;  // clamp to the causal limit

    // One-pass online softmax over THIS split's keys — the body is identical to attention_warp_kernel,
    // only the j-range is [k0,k1) instead of [0,limit).
    float m = -INFINITY, l = 0.0f;
    float lacc[kMaxHeadDim];
    for (int64_t d = 0; d < D; ++d) lacc[d] = 0.0f;
    for (int64_t j = k0 + lane; j < k1; j += 32) {
        const float* kj = kh + j * D;
        float sdot = 0.0f;
        for (int64_t d = 0; d < D; ++d) sdot += qi[d] * kj[d];
        sdot *= scale;
        const float m_new = fmaxf(m, sdot);
        const float corr = expf(m - m_new);
        const float p = expf(sdot - m_new);
        l = l * corr + p;
        const float* vj = vh + j * D;
        for (int64_t d = 0; d < D; ++d) lacc[d] = lacc[d] * corr + p * vj[d];
        m = m_new;
    }
    // Merge the 32 lane-partials to lane 0 (same associative combine + empty guard as the warp kernel).
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(0xffffffff, m, off);
        const float l_o = __shfl_down_sync(0xffffffff, l, off);
        const float m_new = fmaxf(m, m_o);
        const bool empty = (m_new == -INFINITY);
        const float a = empty ? 0.0f : expf(m - m_new), a_o = empty ? 0.0f : expf(m_o - m_new);
        l = l * a + l_o * a_o;
        for (int64_t d = 0; d < D; ++d)
            lacc[d] = lacc[d] * a + __shfl_down_sync(0xffffffff, lacc[d], off) * a_o;
        m = m_new;
    }
    if (lane == 0) {  // write the UNNORMALIZED partial; attention_combine_kernel divides
        m_part[gwarp] = m;
        l_part[gwarp] = l;
        float* ap = acc_part + gwarp * D;
        for (int64_t d = 0; d < D; ++d) ap[d] = lacc[d];
    }
}

// Flash-Decoding pass 2 (G5g): combine the num_splits partials per (head,query) into the final output
// with the associative FlashAttention rule — M=max_s m_s; out[d] = (Σ_s acc_s[d]·e_s)/(Σ_s l_s·e_s),
// e_s=exp(m_s−M). One thread per (head,query,dim). A real query always sees key 0, so M is finite and
// e_s∈[0,1] (an empty split has m_s=−inf ⇒ e_s=0, contributing nothing). num_splits is small, so the
// denom (d-independent) is recomputed per d rather than shared — this kernel is negligible vs pass 1.
__global__ void attention_combine_kernel(const float* __restrict__ m_part,
                                         const float* __restrict__ l_part,
                                         const float* __restrict__ acc_part, float* __restrict__ out,
                                         int64_t nq, int64_t D, int num_splits) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= nq * D) return;
    const int64_t hi = idx / D, d = idx % D;
    const float* mp = m_part + hi * num_splits;
    const float* lp = l_part + hi * num_splits;
    const float* ap = acc_part + hi * num_splits * D;
    float M = -INFINITY;
    for (int s = 0; s < num_splits; ++s) M = fmaxf(M, mp[s]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const float e = (mp[s] == -INFINITY) ? 0.0f : expf(mp[s] - M);
        num += ap[s * D + d] * e;
        den += lp[s] * e;
    }
    out[hi * D + d] = num / den;
}

// Write the t new tokens' K/V ([n_kv, t, hd]) into this sequence's blocks for one layer.
// One thread per (head, token, dim); the block + offset come from the block table.
__global__ void paged_write_kernel(const float* __restrict__ k, const float* __restrict__ v,
                                   float* __restrict__ Kbase, float* __restrict__ Vbase,
                                   const int64_t* __restrict__ block_table, int64_t t, int64_t n_kv,
                                   int64_t hd, int64_t block_size, int64_t start,
                                   const int64_t* __restrict__ d_pos) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_kv * t * hd) return;
    const int64_t d = idx % hd, i = (idx / hd) % t, h = idx / (hd * t);
    const int64_t pos = (d_pos ? *d_pos : start) + i;  // G6 graph mode: device-side write position
    const int64_t blk = block_table[pos / block_size], off = pos % block_size;
    const int64_t dst = ((blk * n_kv + h) * block_size + off) * hd + d;
    Kbase[dst] = k[idx];  // k/v are [n_kv, t, hd]: idx = (h*t+i)*hd + d
    Vbase[dst] = v[idx];
}

// Paged attention: one thread per (head, query). Reads each key/value straight from the
// blocks via the block table (logical pos -> block_table[pos/bs] at pos%bs), folding GQA
// into the read (query head h uses KV head h/n_rep). Same two-pass max-subtract softmax
// as attention_kernel, over the same keys in the same order, so the result is
// bit-identical to the contiguous path — paging changes WHERE K/V live, not the math.
__global__ void paged_attention_kernel(const float* __restrict__ q, const float* __restrict__ Kbase,
                                       const float* __restrict__ Vbase,
                                       const int64_t* __restrict__ block_table,
                                       float* __restrict__ out, int64_t n_heads, int64_t sq,
                                       int64_t hd, int64_t n_kv, int64_t n_rep, int64_t block_size,
                                       int causal, int64_t query_offset, int64_t end, float scale) {
    const int64_t hi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (hi >= n_heads * sq) return;
    const int64_t h = hi / sq, i = hi % sq;
    const int64_t kvh = h / n_rep;  // GQA: this query head's KV head
    const float* qi = q + (h * sq + i) * hd;
    const int64_t limit = causal ? (query_offset + i + 1) : end;

    float maxv = -INFINITY;
    for (int64_t j = 0; j < limit; ++j) {
        const int64_t blk = block_table[j / block_size], off = j % block_size;
        const float* kj = Kbase + ((blk * n_kv + kvh) * block_size + off) * hd;
        float s = 0.0f;
        for (int64_t d = 0; d < hd; ++d) s += qi[d] * kj[d];
        s *= scale;
        if (s > maxv) maxv = s;
    }
    float acc[kMaxHeadDim];
    for (int64_t d = 0; d < hd; ++d) acc[d] = 0.0f;
    float denom = 0.0f;
    for (int64_t j = 0; j < limit; ++j) {
        const int64_t blk = block_table[j / block_size], off = j % block_size;
        const float* kj = Kbase + ((blk * n_kv + kvh) * block_size + off) * hd;
        float s = 0.0f;
        for (int64_t d = 0; d < hd; ++d) s += qi[d] * kj[d];
        const float e = expf(s * scale - maxv);
        denom += e;
        const float* vj = Vbase + ((blk * n_kv + kvh) * block_size + off) * hd;
        for (int64_t d = 0; d < hd; ++d) acc[d] += e * vj[d];
    }
    const float inv = 1.0f / denom;
    float* oi = out + (h * sq + i) * hd;
    for (int64_t d = 0; d < hd; ++d) oi[d] = acc[d] * inv;
}

// Paged attention, WARP-per-query + online softmax (G5f): the warp parallelization of
// paged_attention_kernel, the same lane-stride-the-keys + ONE-PASS online softmax as
// attention_warp_kernel, but each key is read straight from its block (GQA folded in, kvh = h/n_rep).
// Identical key order and arithmetic to the contiguous warp kernel, so the paged path stays
// bit-identical to it (run_cuda_paged max|diff|=0).
__global__ void paged_attention_warp_kernel(
    const float* __restrict__ q, const float* __restrict__ Kbase, const float* __restrict__ Vbase,
    const int64_t* __restrict__ block_table, float* __restrict__ out, int64_t n_heads, int64_t sq,
    int64_t hd, int64_t n_kv, int64_t n_rep, int64_t block_size, int causal, int64_t query_offset,
    int64_t end, float scale, const int64_t* __restrict__ d_pos) {
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int64_t hi = static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (hi >= n_heads * sq) return;  // warp-uniform
    const int64_t h = hi / sq, i = hi % sq;
    const int64_t kvh = h / n_rep;  // GQA: this query head's KV head
    const float* qi = q + (h * sq + i) * hd;
    // G6 graph mode: read the KV length from device (d_pos) so one captured graph spans all decode
    // steps; eager passes nullptr → the host query_offset/end, bit-identical. (end−query_offset = t,
    // baked, so the device end stays *d_pos + t.)
    const int64_t qoff = d_pos ? *d_pos : query_offset;
    const int64_t kend = d_pos ? (*d_pos + (end - query_offset)) : end;
    const int64_t limit = causal ? (qoff + i + 1) : kend;

    // ONE PASS online softmax — identical arithmetic/order to attention_warp_kernel (only the K/V
    // addresses come from the block table), so the paged path stays bit-identical to the contiguous
    // one (run_cuda_paged max|diff|=0).
    float m = -INFINITY, l = 0.0f;
    float lacc[kMaxHeadDim];
    for (int64_t d = 0; d < hd; ++d) lacc[d] = 0.0f;
    for (int64_t j = lane; j < limit; j += 32) {
        const int64_t blk = block_table[j / block_size], boff = j % block_size;
        const float* kj = Kbase + ((blk * n_kv + kvh) * block_size + boff) * hd;
        float s = 0.0f;
        for (int64_t d = 0; d < hd; ++d) s += qi[d] * kj[d];
        s *= scale;
        const float m_new = fmaxf(m, s);
        const float corr = expf(m - m_new);
        const float p = expf(s - m_new);
        l = l * corr + p;
        const float* vj = Vbase + ((blk * n_kv + kvh) * block_size + boff) * hd;
        for (int64_t d = 0; d < hd; ++d) lacc[d] = lacc[d] * corr + p * vj[d];
        m = m_new;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(0xffffffff, m, off);
        const float l_o = __shfl_down_sync(0xffffffff, l, off);
        const float m_new = fmaxf(m, m_o);
        const bool empty = (m_new == -INFINITY);  // both segments empty — see attention_warp_kernel
        const float a = empty ? 0.0f : expf(m - m_new), a_o = empty ? 0.0f : expf(m_o - m_new);
        l = l * a + l_o * a_o;
        for (int64_t d = 0; d < hd; ++d)
            lacc[d] = lacc[d] * a + __shfl_down_sync(0xffffffff, lacc[d], off) * a_o;
        m = m_new;
    }
    if (lane == 0) {
        const float inv = 1.0f / l;
        float* oi = out + (h * sq + i) * hd;
        for (int64_t d = 0; d < hd; ++d) oi[d] = lacc[d] * inv;
    }
}

// Flash-Decoding pass 1 for the PAGED cache (G5g): the split-KV analogue of paged_attention_warp_kernel,
// exactly as attention_split_kv_kernel is to attention_warp_kernel. Each (head,query) gets num_splits
// warps; warp s runs the one-pass online softmax over its key chunk [k0,k1) — keys read straight from
// the blocks (GQA folded in, kvh=h/n_rep) — and writes an UNNORMALIZED partial (m,l,acc) to the SAME
// scratch layout the contiguous split kernel uses, so it shares attention_combine_kernel (pass 2). Same
// key order and arithmetic as the contiguous split kernel (only the K/V addresses differ), so the paged
// split path stays bit-identical to the contiguous split path (run_cuda_paged max|diff|=0). The chunks
// tile [0,end) and causal clamps each to `limit`.
__global__ void paged_attention_split_kv_kernel(
    const float* __restrict__ q, const float* __restrict__ Kbase, const float* __restrict__ Vbase,
    const int64_t* __restrict__ block_table, float* __restrict__ m_part, float* __restrict__ l_part,
    float* __restrict__ acc_part, int64_t n_heads, int64_t sq, int64_t hd, int64_t n_kv, int64_t n_rep,
    int64_t block_size, int causal, int64_t query_offset, int64_t end, float scale, int num_splits,
    int64_t chunk) {
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int64_t gwarp = static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int64_t nq = n_heads * sq;
    if (gwarp >= nq * num_splits) return;  // warp-uniform
    const int64_t hi = gwarp / num_splits, s = gwarp % num_splits;
    const int64_t h = hi / sq, i = hi % sq;
    const int64_t kvh = h / n_rep;  // GQA: this query head's KV head
    const float* qi = q + (h * sq + i) * hd;
    const int64_t limit = causal ? (query_offset + i + 1) : end;
    const int64_t k0 = s * chunk;  // this split's key range [k0,k1), tiling [0,end)
    const int64_t k1raw = (k0 + chunk < end) ? (k0 + chunk) : end;
    const int64_t k1 = causal ? (k1raw < limit ? k1raw : limit) : k1raw;

    float m = -INFINITY, l = 0.0f;
    float lacc[kMaxHeadDim];
    for (int64_t d = 0; d < hd; ++d) lacc[d] = 0.0f;
    for (int64_t j = k0 + lane; j < k1; j += 32) {
        const int64_t blk = block_table[j / block_size], boff = j % block_size;
        const float* kj = Kbase + ((blk * n_kv + kvh) * block_size + boff) * hd;
        float sdot = 0.0f;
        for (int64_t d = 0; d < hd; ++d) sdot += qi[d] * kj[d];
        sdot *= scale;
        const float m_new = fmaxf(m, sdot);
        const float corr = expf(m - m_new);
        const float p = expf(sdot - m_new);
        l = l * corr + p;
        const float* vj = Vbase + ((blk * n_kv + kvh) * block_size + boff) * hd;
        for (int64_t d = 0; d < hd; ++d) lacc[d] = lacc[d] * corr + p * vj[d];
        m = m_new;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float m_o = __shfl_down_sync(0xffffffff, m, off);
        const float l_o = __shfl_down_sync(0xffffffff, l, off);
        const float m_new = fmaxf(m, m_o);
        const bool empty = (m_new == -INFINITY);
        const float a = empty ? 0.0f : expf(m - m_new), a_o = empty ? 0.0f : expf(m_o - m_new);
        l = l * a + l_o * a_o;
        for (int64_t d = 0; d < hd; ++d)
            lacc[d] = lacc[d] * a + __shfl_down_sync(0xffffffff, lacc[d], off) * a_o;
        m = m_new;
    }
    if (lane == 0) {  // write the UNNORMALIZED partial; attention_combine_kernel divides
        m_part[gwarp] = m;
        l_part[gwarp] = l;
        float* ap = acc_part + gwarp * hd;
        for (int64_t d = 0; d < hd; ++d) ap[d] = lacc[d];
    }
}

}  // namespace

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Tensor to_device(const Tensor& host) {
    Tensor d = device_alloc(host.shape());
    cuda_check(cudaMemcpy(d.device_ptr(), host.data(),
                          static_cast<size_t>(host.numel()) * sizeof(float),
                          cudaMemcpyHostToDevice),
               "to_device H2D");
    return d;
}

Tensor to_host(const Tensor& dev) {
    Tensor h(dev.shape());  // CPU, zero-filled
    cuda_check(cudaMemcpy(h.data(), dev.device_ptr(),
                          static_cast<size_t>(dev.numel()) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "to_host D2H");
    return h;
}


// Upload an fp32 host tensor and convert it to fp16 on device (G5d): stage fp32 to a pooled
// temp, then one convert kernel into an owned half buffer. Done once per weight at load.
Tensor to_device_f16(const Tensor& host) {
    Tensor src = to_device(host);                       // fp32 on device (a pooled temp)
    Tensor d = device_alloc(host.shape(), DType::F16);  // fp16 destination
    const int64_t n = host.numel();
    if (n > 0)
        f32_to_f16_kernel<<<grid1d(n, kBlock), kBlock>>>(dptr(src),
                                                         static_cast<half*>(d.device_ptr()), n);
    launch_check("f32_to_f16_kernel");
    return d;  // `src` (fp32 temp) returns to the pool here
}

Tensor to_device_i8(const int8_t* host, const std::vector<int64_t>& shape) {
    Tensor d = device_alloc(shape, DType::I8);
    const int64_t n = d.numel();
    if (n > 0)
        cuda_check(cudaMemcpy(d.device_ptr(), host, static_cast<size_t>(n) * sizeof(int8_t),
                              cudaMemcpyHostToDevice),
                   "to_device_i8");
    return d;
}

Device CudaBackend::device() const { return Device::CUDA; }


// R2: the kernel-selection knobs, consolidated from seven scattered g_cuda_* globals into one
// CudaPolicy (see cuda_backend.hpp). Still a file-scope instance — the readers include the
// cuda_linear_q8 free function and CudaPagedKVCache::attend, which a per-instance backend member
// can't reach until R3 threads it through the weight seam. The A/B harness sets cuda_policy().<field>.
CudaPolicy g_cuda_policy;
CudaPolicy& cuda_policy() { return g_cuda_policy; }

// G6 (CUDA graphs): the per-step DECODE inputs, made device-resident so ONE captured graph spans all
// steps. The graph driver (CudaGraphDecoder) points these at its device int64 buffers before capture
// and updates the buffers' CONTENTS each step (the graph reads them by pointer). nullptr = eager mode
// (the kernels use their host-int args, bit-identical). g_cuda_graph_pos: the KV length / write
// position (read by rope, paged_write, paged_attention). g_cuda_graph_token: the decode token id
// (the embedding gathers from it instead of a host id-upload — the upload is a sync H2D, illegal in a
// capture). Set/cleared only by the single-threaded driver around a capture; not otherwise thread-safe.
const int64_t* g_cuda_graph_pos = nullptr;
const int64_t* g_cuda_graph_token = nullptr;
// G6: keep logits on device so a graph driver does its own D2H after replay (see cuda_backend.hpp).
bool g_cuda_keep_device_logits = false;

Tensor CudaBackend::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                              int64_t query_offset) {
    const int64_t H = q.size(0), sq = q.size(1), D = q.size(2), sk = k.size(1);
    if (D > kMaxHeadDim)
        throw std::runtime_error("CudaBackend::attention: head_dim > 128 not supported (G2)");
    Tensor out = device_alloc({H, sq, D});
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    constexpr int kBc = 32;  // tile width = warp width, so lane l ↔ key (base+l) preserves the order
    if (cuda_policy().force_naive_attn) {
        attention_kernel<<<grid1d(H * sq, kBlock), kBlock>>>(
            dptr(q), dptr(k), dptr(v), dptr(out), H, sq, sk, D, causal ? 1 : 0, query_offset, scale);
        launch_check("attention_kernel");
    } else if (sq > 1 && cuda_policy().use_tiled_attn) {
        // Opt-in: shared-memory K/V tiling (G5f-tiled). A block of 8 warps shares each K/V tile, so
        // grid is (query-tiles per head, H). Default off — see cuda_policy().use_tiled_attn. Decode (sq=1)
        // has no query reuse anyway, so it always takes the non-tiled path below.
        const int threads = 256, warpsPerBlock = threads / 32;  // 8 queries/block
        const dim3 grid(static_cast<unsigned>((sq + warpsPerBlock - 1) / warpsPerBlock),
                        static_cast<unsigned>(H));
        const size_t smemBytes = 2u * kBc * static_cast<size_t>(D) * sizeof(float);
        attention_warp_tiled_kernel<kBc><<<grid, threads, smemBytes>>>(
            dptr(q), dptr(k), dptr(v), dptr(out), H, sq, sk, D, causal ? 1 : 0, query_offset, scale);
        launch_check("attention_warp_tiled_kernel");
    } else if (cuda_policy().use_split_attn && split_count(H * sq, sk) >= 2) {
        // Flash-Decoding (G5g): split the KV so nq*num_splits warps fill the GPU. Pass 1 writes each
        // split's partial (m,l,acc) to pooled scratch; pass 2 combines them. split_count returns >=2
        // only when worth it (small sq + long context); otherwise the warp kernel below runs instead.
        const int64_t nq = H * sq;
        const int num_splits = split_count(nq, sk);
        const int64_t chunk = (sk + num_splits - 1) / num_splits;  // keys per split, tiling [0,sk)
        Tensor m_part = device_alloc({nq * num_splits});           // pooled scratch (tiny, reused)
        Tensor l_part = device_alloc({nq * num_splits});
        Tensor acc_part = device_alloc({nq * num_splits, D});
        const int threads = 256, warpsPerBlock = threads / 32;  // one warp per (head,query,split)
        const int blocks = static_cast<int>((nq * num_splits + warpsPerBlock - 1) / warpsPerBlock);
        attention_split_kv_kernel<<<blocks, threads>>>(
            dptr(q), dptr(k), dptr(v), dptr(m_part), dptr(l_part), dptr(acc_part), H, sq, sk, D,
            causal ? 1 : 0, query_offset, scale, num_splits, chunk);
        launch_check("attention_split_kv_kernel");
        attention_combine_kernel<<<grid1d(nq * D, kBlock), kBlock>>>(
            dptr(m_part), dptr(l_part), dptr(acc_part), dptr(out), nq, D, num_splits);
        launch_check("attention_combine_kernel");
    } else {
        const int threads = 256;  // 8 warps/block, one warp per (head,query)
        const int blocks = static_cast<int>((H * sq + threads / 32 - 1) / (threads / 32));
        attention_warp_kernel<<<blocks, threads>>>(dptr(q), dptr(k), dptr(v), dptr(out), H, sq, sk, D,
                                                   causal ? 1 : 0, query_offset, scale);
        launch_check("attention_warp_kernel");
    }
    return out;
}

// R1: the model's make_kv_cache / forward-tail D2H, now behind the Backend (no #ifdef in model.cpp).
std::unique_ptr<KVCacheBase> CudaBackend::make_kv_cache(int64_t num_layers, int64_t n_kv_heads,
                                                        int64_t head_dim, int64_t /*max_seq*/) {
    // The device cache grows by concatenation, so it ignores max_seq (unlike the CPU cache).
    return std::make_unique<CudaKVCache>(this, num_layers, n_kv_heads, head_dim);
}

Tensor CudaBackend::finalize_logits(Tensor logits) {
    // D2H at the edge, unless a graph driver (CudaGraphDecoder) asked to keep the logits on the
    // device for its own post-replay copy — a sync D2H can't run inside a stream capture.
    if (logits.device() == Device::CUDA && !g_cuda_keep_device_logits) return to_host(logits);
    return logits;
}

Tensor CudaBackend::to_resident(Tensor weight, bool fp16_eligible) {
    // Upload a host weight to the GPU once, at load. The big eligible weights (layer projections /
    // embedding) go up as fp16 when config_.fp16_weights is set — half the DRAM bytes — and the
    // linear/embedding dispatch reads the F16 dtype directly; everything else (norms, biases, the
    // RoPE tables) stays fp32. The fp16 decision is a per-instance config field now, not a global.
    return (fp16_eligible && config_.fp16_weights) ? to_device_f16(weight) : to_device(weight);
}

// --- Device-resident KV cache (G3) ---

CudaKVCache::CudaKVCache(Backend* backend, int64_t num_layers, int64_t n_kv_heads,
                        int64_t head_dim)
    : backend_(backend), n_kv_heads_(n_kv_heads), head_dim_(head_dim),
      k_(static_cast<size_t>(num_layers)), v_(static_cast<size_t>(num_layers)) {}

Tensor CudaKVCache::attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                           int64_t n_rep, bool causal, int64_t query_offset) {
    // Append this layer's new K/V to its growing history, then attend over the whole
    // history — the same append + repeat_kv + attention the CPU cache does, on the GPU.
    Tensor& kh = k_[static_cast<size_t>(layer)];
    Tensor& vh = v_[static_cast<size_t>(layer)];
    kh = cat_seq(kh, k);
    vh = cat_seq(vh, v);
    Tensor kk = backend_->repeat_kv(kh, n_rep);
    Tensor vv = backend_->repeat_kv(vh, n_rep);
    return backend_->attention(q, kk, vv, causal, query_offset);
}

void CudaKVCache::advance(int64_t t) { length_ += t; }
int64_t CudaKVCache::length() const { return length_; }

// --- Paged KV cache (G4b) ---

CudaBlockPool::CudaBlockPool(int64_t num_layers, int64_t n_kv_heads, int64_t head_dim,
                            int64_t block_size, int64_t num_blocks)
    : num_layers_(num_layers), n_kv_heads_(n_kv_heads), head_dim_(head_dim),
      block_size_(block_size), num_blocks_(num_blocks) {
    if (num_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || block_size <= 0 || num_blocks <= 0)
        throw std::invalid_argument("CudaBlockPool: all dimensions must be positive");
    block_stride_ = n_kv_heads_ * block_size_ * head_dim_;
    const size_t total = static_cast<size_t>(num_layers_) * num_blocks_ * block_stride_;
    void* pk = nullptr;
    void* pv = nullptr;
    cuda_check(cudaMalloc(&pk, total * sizeof(float)), "CudaBlockPool K");
    cuda_check(cudaMalloc(&pv, total * sizeof(float)), "CudaBlockPool V");
    cuda_check(cudaMemset(pk, 0, total * sizeof(float)), "CudaBlockPool K zero");
    cuda_check(cudaMemset(pv, 0, total * sizeof(float)), "CudaBlockPool V zero");
    dK_ = std::shared_ptr<void>(pk, [](void* q) { cudaFree(q); });
    dV_ = std::shared_ptr<void>(pv, [](void* q) { cudaFree(q); });
    free_list_.reserve(static_cast<size_t>(num_blocks_));
    for (int64_t i = num_blocks_ - 1; i >= 0; --i) free_list_.push_back(i);
    refcount_.assign(static_cast<size_t>(num_blocks_), 0);
}

int64_t CudaBlockPool::allocate() {
    if (free_list_.empty())
        throw std::runtime_error("CudaBlockPool: out of blocks — raise num_blocks or evict");
    const int64_t b = free_list_.back();
    free_list_.pop_back();
    refcount_[static_cast<size_t>(b)] = 1;
    return b;
}

void CudaBlockPool::incref(int64_t block) {
    if (block < 0 || block >= num_blocks_) throw std::out_of_range("CudaBlockPool::incref");
    if (refcount_[static_cast<size_t>(block)] <= 0)
        throw std::runtime_error("CudaBlockPool::incref: block is free");
    ++refcount_[static_cast<size_t>(block)];
}

void CudaBlockPool::free(int64_t block) {
    if (block < 0 || block >= num_blocks_) throw std::out_of_range("CudaBlockPool::free");
    int64_t& rc = refcount_[static_cast<size_t>(block)];
    if (rc <= 0) throw std::runtime_error("CudaBlockPool::free: double free");
    if (--rc == 0) free_list_.push_back(block);
}

CudaPagedKVCache::CudaPagedKVCache(CudaBlockPool* pool) : pool_(pool) {
    if (pool == nullptr) throw std::invalid_argument("CudaPagedKVCache: null pool");
}

CudaPagedKVCache::~CudaPagedKVCache() {
    if (pool_)
        for (int64_t b : block_table_) pool_->free(b);
}

void CudaPagedKVCache::ensure_capacity(int64_t positions) {
    const int64_t bs = pool_->block_size();
    while (static_cast<int64_t>(block_table_.size()) * bs < positions)
        block_table_.push_back(pool_->allocate());
}

// Refresh the device block table from the host one. The table only GROWS (entries never change
// once set), so upload just the appended tail. Returns early with NO CUDA call when nothing grew —
// which is what lets it sit inside a graph-captured attend(): the driver pre-grows the table for
// the step's length before BeginCapture / before each replay, so during capture/replay this is a
// pure no-op and records nothing. The async copy (eager path) is ordered before the paged kernels
// on the one per-thread stream, so they read the fresh table.
void CudaPagedKVCache::sync_device_block_table() {
    const int64_t n = static_cast<int64_t>(block_table_.size());
    if (n == d_bt_count_) return;  // nothing appended — no CUDA work (safe under capture)
    if (!d_block_table_) {
        void* p = nullptr;  // sized to the pool's max blocks: a sequence never exceeds it
        cuda_check(cudaMalloc(&p, static_cast<size_t>(pool_->num_blocks()) * sizeof(int64_t)),
                   "paged d_block_table malloc");
        d_block_table_ = std::shared_ptr<void>(p, [](void* q) { cudaFree(q); });
    }
    cuda_check(cudaMemcpyAsync(static_cast<int64_t*>(d_block_table_.get()) + d_bt_count_,
                               block_table_.data() + d_bt_count_,
                               static_cast<size_t>(n - d_bt_count_) * sizeof(int64_t),
                               cudaMemcpyHostToDevice),
               "paged d_block_table H2D");
    d_bt_count_ = n;
}

// Host-side per-step prep (block allocation + device-table refresh) the graph driver runs OUTSIDE
// a capture; attend() calls it too for the eager path. `end` = length after this step.
void CudaPagedKVCache::prepare(int64_t end) {
    ensure_capacity(end);
    sync_device_block_table();
}

void CudaPagedKVCache::share_prefix(const std::vector<int64_t>& blocks, int64_t length) {
    if (!block_table_.empty() || length_ != 0)
        throw std::runtime_error("CudaPagedKVCache::share_prefix: must seed a fresh cache");
    const int64_t bs = pool_->block_size();
    if (length != static_cast<int64_t>(blocks.size()) * bs)
        throw std::invalid_argument("share_prefix: length must be a block boundary");
    // Hold each shared block; the sequence writes its own blocks past `length` (a block
    // boundary), so the shared blocks stay read-only — no copy-on-write.
    block_table_ = blocks;
    for (int64_t b : block_table_) pool_->incref(b);
    length_ = length;
}

Tensor CudaPagedKVCache::attend(int64_t layer, const Tensor& q, const Tensor& k, const Tensor& v,
                                int64_t n_rep, bool causal, int64_t query_offset) {
    const int64_t t = k.size(1);
    const int64_t n_kv = pool_->n_kv_heads(), hd = pool_->head_dim(), bs = pool_->block_size();
    const int64_t start = length_, end = start + t;
    prepare(end);  // grow blocks + refresh the device block table (idempotent across layers)

    // The block table lives in a STABLE device buffer (G6: so a captured graph reads it by
    // pointer while prepare() updates its contents between replays) — no per-call malloc/H2D/free.
    const int64_t* d_bt = device_block_table();

    float* Kbase = pool_->k_base() + layer * pool_->layer_stride();
    float* Vbase = pool_->v_base() + layer * pool_->layer_stride();

    paged_write_kernel<<<grid1d(n_kv * t * hd, kBlock), kBlock>>>(
        dptr(k), dptr(v), Kbase, Vbase, d_bt, t, n_kv, hd, bs, start, g_cuda_graph_pos);
    launch_check("paged_write_kernel");

    const int64_t n_heads = q.size(0), sq = q.size(1), D = q.size(2);
    if (D > kMaxHeadDim)
        throw std::runtime_error("CudaPagedKVCache::attend: head_dim > 128 not supported (G4b)");
    Tensor out = device_alloc({n_heads, sq, D});
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    if (cuda_policy().force_naive_attn) {
        paged_attention_kernel<<<grid1d(n_heads * sq, kBlock), kBlock>>>(
            dptr(q), Kbase, Vbase, d_bt, dptr(out), n_heads, sq, D, n_kv, n_rep, bs, causal ? 1 : 0,
            query_offset, end, scale);
        launch_check("paged_attention_kernel");
    } else if (cuda_policy().use_split_attn && split_count(n_heads * sq, end) >= 2) {
        // Flash-Decoding (G5g) on the paged path — the split-KV analogue, sharing attention_combine_kernel
        // with the contiguous path. The paged cache has no O(ctx) cat_seq copy, so this is where the split
        // win shows up end-to-end (the contiguous cache's growing per-step copy diluted it).
        const int64_t nq = n_heads * sq;
        const int num_splits = split_count(nq, end);
        const int64_t chunk = (end + num_splits - 1) / num_splits;  // keys per split, tiling [0,end)
        Tensor m_part = device_alloc({nq * num_splits});
        Tensor l_part = device_alloc({nq * num_splits});
        Tensor acc_part = device_alloc({nq * num_splits, D});
        const int threads = 256, warpsPerBlock = threads / 32;  // one warp per (head,query,split)
        const int blocks = static_cast<int>((nq * num_splits + warpsPerBlock - 1) / warpsPerBlock);
        paged_attention_split_kv_kernel<<<blocks, threads>>>(
            dptr(q), Kbase, Vbase, d_bt, dptr(m_part), dptr(l_part), dptr(acc_part), n_heads, sq, D,
            n_kv, n_rep, bs, causal ? 1 : 0, query_offset, end, scale, num_splits, chunk);
        launch_check("paged_attention_split_kv_kernel");
        attention_combine_kernel<<<grid1d(nq * D, kBlock), kBlock>>>(
            dptr(m_part), dptr(l_part), dptr(acc_part), dptr(out), nq, D, num_splits);
        launch_check("attention_combine_kernel");
    } else {
        const int threads = 256;  // 8 warps/block, one warp per (head,query)
        const int blocks = static_cast<int>((n_heads * sq + threads / 32 - 1) / (threads / 32));
        paged_attention_warp_kernel<<<blocks, threads>>>(
            dptr(q), Kbase, Vbase, d_bt, dptr(out), n_heads, sq, D, n_kv, n_rep, bs, causal ? 1 : 0,
            query_offset, end, scale, g_cuda_graph_pos);
        launch_check("paged_attention_warp_kernel");
    }

    return out;
}

// --- CUDA graph decode driver (G6) ---

CudaGraphDecoder::CudaGraphDecoder(const Model& model, CudaPagedKVCache& cache, int64_t vocab)
    : model_(model), cache_(cache), vocab_(vocab) {
    cuda_check(cudaMalloc(&d_pos_, sizeof(int64_t)), "graph d_pos malloc");
    cuda_check(cudaMalloc(&d_token_, sizeof(int64_t)), "graph d_token malloc");
}

CudaGraphDecoder::~CudaGraphDecoder() {
    // Clear the graph-mode globals in case this decoder owned them (single-threaded; defensive).
    g_cuda_graph_pos = nullptr;
    g_cuda_graph_token = nullptr;
    if (exec_) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(exec_));
    if (graph_) cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
    if (d_pos_) cudaFree(d_pos_);
    if (d_token_) cudaFree(d_token_);
}

// Record the decode forward into a graph. Stream capture RECORDS the kernels (host code runs, kernels
// do NOT execute), so: (1) the pool must already be warm (no cudaMalloc may run under capture — the
// first decode step ran eager to warm it); (2) the forward's advance() bumps the cache length during
// the capture pass with no real work done, so we roll it back; (3) the kernels read the step's
// token/length from d_token_/d_pos_ (set as graph-mode globals), so the SAME graph replays at any
// length. keep_device_logits leaves the logits on device for our own D2H (a sync D2H can't be captured).
void CudaGraphDecoder::capture(int64_t token) {
    const int64_t saved = cache_.length();
    g_cuda_graph_pos = d_pos_;
    g_cuda_graph_token = d_token_;
    g_cuda_keep_device_logits = true;
    // Values are read at LAUNCH time, but set them now so the capture pass is a valid forward.
    cuda_check(cudaMemcpy(d_pos_, &saved, sizeof(int64_t), cudaMemcpyHostToDevice), "graph cap pos");
    cuda_check(cudaMemcpy(d_token_, &token, sizeof(int64_t), cudaMemcpyHostToDevice), "graph cap tok");

    cudaGraph_t g = nullptr;
    cuda_check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal),
               "graph begin capture");
    Tensor dlog = model_.forward({token}, &cache_);  // records the ~360 kernels; none execute yet
    cuda_check(cudaStreamEndCapture(cudaStreamPerThread, &g), "graph end capture");

    cache_.set_length(saved);            // undo the capture pass's advance (no KV was written)
    d_logits_ = dptr(dlog);              // the lm_head output address the graph writes every launch
    captured_logits_ = std::move(dlog);  // keep that pool buffer alive so nothing reuses it

    cudaGraphExec_t e = nullptr;
    cuda_check(cudaGraphInstantiate(&e, g, 0), "graph instantiate");
    graph_ = g;
    exec_ = e;
    captured_ = true;
    g_cuda_keep_device_logits = false;  // forward() isn't called on replays; only the return needed it
    // g_cuda_graph_pos / g_cuda_graph_token stay set — the captured kernels read them on every launch.
}

Tensor CudaGraphDecoder::decode(int64_t token) {
    const int64_t pos = cache_.length();  // this token's write position
    cache_.prepare(pos + 1);              // grow blocks + refresh the device block table (host-side)

    if (!warmed_) {
        // First decode step runs EAGER: it executes (writes KV, makes logits) and warms the device
        // pool so the later capture allocates nothing. forward() D2Hs the logits and advances length.
        warmed_ = true;
        return model_.forward({token}, &cache_);
    }
    if (!captured_) capture(token);  // one-time: build the graph (cache length left unchanged)

    cuda_check(cudaMemcpy(d_pos_, &pos, sizeof(int64_t), cudaMemcpyHostToDevice), "graph pos H2D");
    cuda_check(cudaMemcpy(d_token_, &token, sizeof(int64_t), cudaMemcpyHostToDevice), "graph tok H2D");
    cuda_check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec_), cudaStreamPerThread), "graph launch");
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread), "graph sync");

    Tensor h({1, vocab_});  // D2H the logits the replayed graph just wrote
    cuda_check(cudaMemcpy(h.data(), d_logits_, static_cast<size_t>(vocab_) * sizeof(float),
                          cudaMemcpyDeviceToHost), "graph logits D2H");
    cache_.advance(1);  // the driver owns the length now (the captured advance was rolled back)
    return h;
}

}  // namespace ni
