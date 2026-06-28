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

// Round-half-to-even + clamp to [-127,127], NaN->0 — mirrors quant.cpp round_clamp EXACTLY (rintf
// == std::nearbyint under the default rounding mode), so device activation quant produces the same
// int8 codes as the CPU oracle and the integer GEMM core is identical end-to-end.
__device__ inline int8_t quant_round(float v, float scale) {
    if (!(scale > 0.0f)) return 0;                  // all-zero row (scale 0) or NaN scale -> 0
    float r = rintf(v / scale);
    r = fmaxf(-127.0f, fminf(127.0f, r));           // clamp BEFORE the cast
    if (isnan(r)) return 0;                          // NaN input slips the clamps -> 0
    return static_cast<int8_t>(r);
}

// Dynamic per-row int8 activation quantization (W8A8, G5d): a_scale[i] = max_j|x[i,j]|/127, then
// xq = round(x/a_scale). One block per row; threads stride the row for a shared-memory max-reduction
// (max is exact/order-free, so a_scale matches the CPU's sequential max bit-for-bit), then quantize.
// blockDim must be a power of two for the reduction. xq is bit-identical to quant.cpp's per-row quant.
__global__ void activation_quant_kernel(const float* __restrict__ x, int8_t* __restrict__ xq,
                                        float* __restrict__ a_scale, int m, int k) {
    const int row = blockIdx.x;
    if (row >= m) return;
    const float* xr = x + static_cast<size_t>(row) * k;
    int8_t* qr = xq + static_cast<size_t>(row) * k;
    const int tid = static_cast<int>(threadIdx.x), nt = static_cast<int>(blockDim.x);

    __shared__ float red[256];
    float local = 0.0f;
    for (int j = tid; j < k; j += nt) local = fmaxf(local, fabsf(xr[j]));
    red[tid] = local;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float scale = red[0] / 127.0f;
    if (tid == 0) a_scale[row] = scale;
    for (int j = tid; j < k; j += nt) qr[j] = quant_round(xr[j], scale);
}

// W8A8 GEMM: int8×int8 → int32 via __dp4a, dual-scale dequant (G5d) — the int8-COMPUTE win that
// weight-only Q8/fp16 can't get (fp16 only ties on the compute-bound projections). xq [m,k] and
// wq [n,k] are int8 row-major (K contiguous). Both tiles stage K-contiguous (As [BM][BK], Bs N-major
// [BN][BK]) so each thread packs 4 int8 into an int and __dp4a does 4 MACs at once into an int32
// accumulator — exact, identical to the CPU dot_qq. Dequant folds in at the store:
// y = acc · a_scale[row] · w_scale[col] + bias. BK%4==0 and k%BK==0 (dispatch); m,n ragged (checked).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void linear_w8a8_kernel(const int8_t* __restrict__ xq, const int8_t* __restrict__ wq,
                                   const float* __restrict__ a_scale,
                                   const float* __restrict__ w_scale, const float* __restrict__ bias,
                                   float* __restrict__ y, int m, int n, int k) {
    constexpr int numThreads = (BM / TM) * (BN / TN);
    __shared__ int8_t As[BM * BK];  // xq tile [BM][BK], K-contiguous
    __shared__ int8_t Bs[BN * BK];  // wq tile [BN][BK], N-major + K-contiguous (for DP4A packing)

    const int tid = static_cast<int>(threadIdx.x);
    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);
    const int rowBase = static_cast<int>(blockIdx.y) * BM, colBase = static_cast<int>(blockIdx.x) * BN;

    int32_t acc[TM * TN] = {0};

    for (int kk = 0; kk < k; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += numThreads) {  // stage xq (K-contiguous: coalesced)
            const int r = idx / BK, c = idx % BK, gr = rowBase + r;
            As[idx] = (gr < m) ? xq[static_cast<size_t>(gr) * k + kk + c] : int8_t(0);
        }
        for (int idx = tid; idx < BN * BK; idx += numThreads) {  // stage wq (K-contiguous: coalesced)
            const int o = idx / BK, c = idx % BK, go = colBase + o;
            Bs[idx] = (go < n) ? wq[static_cast<size_t>(go) * k + kk + c] : int8_t(0);
        }
        __syncthreads();
        for (int g = 0; g < BK; g += 4) {  // DP4A over 4-int8 groups
            int aPack[TM], bPack[TN];
            for (int i = 0; i < TM; ++i)
                aPack[i] = *reinterpret_cast<const int*>(&As[(threadRow * TM + i) * BK + g]);
            for (int j = 0; j < TN; ++j)
                bPack[j] = *reinterpret_cast<const int*>(&Bs[(threadCol * TN + j) * BK + g]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] = __dp4a(aPack[i], bPack[j], acc[i * TN + j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= m) continue;
        const float as = a_scale[gr];
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + threadCol * TN + j;
            if (gc >= n) continue;
            y[static_cast<size_t>(gr) * n + gc] =
                float(acc[i * TN + j]) * as * w_scale[gc] + (bias ? bias[gc] : 0.0f);
        }
    }
}

// Weight-only int8 GEMM (G5d): x fp32 [m,k], codes int8 [n,k] (K-contiguous), w_scale fp32 [n].
// y[i,o] = (sum_j x[i,j]*float(codes[o,j])) * w_scale[o] + bias[o]. fp32 accumulate (x is NOT
// quantized — unlike linear_w8a8_kernel's int8 activations), so it matches the CPU linear_q8 oracle
// within accelerator tolerance. One correct tiled kernel for decode (m=1) and prefill; a decode GEMV
// / warp-tiling for the huge lm_head are later levers (G5's staging: a correct kernel first, then
// tune). The k-bounds check lets any k through (the synthetic parity tests use small, ragged k).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void linear_q8_kernel(const float* __restrict__ x, const int8_t* __restrict__ codes,
                                 const float* __restrict__ w_scale, const float* __restrict__ bias,
                                 float* __restrict__ y, int m, int n, int k) {
    constexpr int numThreads = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK];  // x tile [BM][BK]
    __shared__ float Bs[BN * BK];  // codes tile [BN][BK], dequant-to-float on stage (scale at store)
    const int tid = static_cast<int>(threadIdx.x);
    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);
    const int rowBase = static_cast<int>(blockIdx.y) * BM, colBase = static_cast<int>(blockIdx.x) * BN;
    float acc[TM * TN] = {0.0f};
    for (int kk = 0; kk < k; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += numThreads) {
            const int r = idx / BK, c = idx % BK, gr = rowBase + r;
            As[idx] = (gr < m && kk + c < k) ? x[static_cast<size_t>(gr) * k + kk + c] : 0.0f;
        }
        for (int idx = tid; idx < BN * BK; idx += numThreads) {
            const int o = idx / BK, c = idx % BK, go = colBase + o;
            Bs[idx] = (go < n && kk + c < k) ? float(codes[static_cast<size_t>(go) * k + kk + c]) : 0.0f;
        }
        __syncthreads();
        for (int g = 0; g < BK; ++g) {
            float aReg[TM], bReg[TN];
            for (int i = 0; i < TM; ++i) aReg[i] = As[(threadRow * TM + i) * BK + g];
            for (int j = 0; j < TN; ++j) bReg[j] = Bs[(threadCol * TN + j) * BK + g];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= m) continue;
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + threadCol * TN + j;
            if (gc >= n) continue;
            y[static_cast<size_t>(gr) * n + gc] = acc[i * TN + j] * w_scale[gc] + (bias ? bias[gc] : 0.0f);
        }
    }
}

// Weight-only int8 decode GEMV (G5d follow-up): the linear_q8 mirror of linear_gemv_kernel, for the
// quantized lm_head/embed at decode (m<=kGemvMaxM). The tiled linear_q8_kernel is prefill-tuned —
// at m=1 its 64-row tile leaves ~63/64 of the warps idle yet still streams the whole int8 weight,
// so the huge lm_head (n=151936) runs far under the bandwidth wall. Here one WARP owns output o: the
// 32 lanes coalesce-stride the int8 codes[o,:] (¼ the DRAM bytes of fp32, ½ of fp16 — the decode
// memory win int8 was leaving on the table), a __shfl reduction folds the 32 partials, and the
// per-channel dequant scale + bias apply once at the store — the SAME accumulate-then-scale order as
// linear_q8_kernel and the CPU linear_q8 oracle (dot_qf32 · scale + b). x is NOT quantized (weight-
// only, fp32 accumulate), so only the warp-reduce reorders the sum: ~1e-6 vs the tiled kernel, the
// same reorder the fp32 GEMV already takes — golden tokens unchanged. m>1 loops the rows so codes[o]
// streams once and is reused across them from L2 (the batched-decode weight reuse, as in G5b).
__global__ void linear_q8_gemv_kernel(const float* __restrict__ x, const int8_t* __restrict__ codes,
                                      const float* __restrict__ w_scale,
                                      const float* __restrict__ bias, float* __restrict__ y,
                                      int m, int n, int k) {
    const int o = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // one warp per output channel
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (o >= n) return;
    const size_t wbase = static_cast<size_t>(o) * k;
    const float scale = w_scale[o];
    const float b = bias ? bias[o] : 0.0f;
    for (int i = 0; i < m; ++i) {
        const float* xr = x + static_cast<size_t>(i) * k;
        float partial = 0.0f;
        for (int j = lane; j < k; j += 32) partial += xr[j] * float(codes[wbase + j]);  // coalesced
        for (int off = 16; off > 0; off >>= 1)  // warp-reduce the 32 partials
            partial += __shfl_down_sync(0xffffffff, partial, off);
        if (lane == 0) y[static_cast<size_t>(i) * n + o] = partial * scale + b;  // dequant once
    }
}

// out[r,c] = table[ids[r], c]. One thread per output element; ids already on device. Templated
// on the table type so an fp16 embedding (G5d: embed_tokens, the largest weight, uploaded as
// half) reads through ldf and converts to fp32 on store — activations stay fp32, only the
// looked-up table is half. On a float* table ldf is the identity, so fp32 is unchanged.
template <typename WT>
__global__ void embedding_kernel(const WT* __restrict__ table, const int64_t* __restrict__ ids,
                                 float* __restrict__ out, int64_t n, int64_t hidden) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * hidden) return;
    const int64_t r = idx / hidden, c = idx % hidden;
    out[idx] = ldf(table, static_cast<size_t>(ids[r] * hidden + c));
}

// Weight-only int8 embedding gather (G5d): out[r,c] = float(codes[ids[r],c]) * scale[ids[r]]. The
// per-row scale means this can't be a plain embedding_kernel<int8_t> instantiation (ldf carries no
// scale), so it's its own kernel — the GPU mirror of the CPU embedding_q8.
__global__ void embedding_q8_kernel(const int8_t* __restrict__ codes, const float* __restrict__ scale,
                                    const int64_t* __restrict__ ids, float* __restrict__ out,
                                    int64_t n, int64_t hidden) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * hidden) return;
    const int64_t r = idx / hidden, c = idx % hidden, id = ids[r];
    out[idx] = float(codes[id * hidden + c]) * scale[id];
}

// RMSNorm over the last dim: out = x / sqrt(mean(x²)+eps) * weight. One thread per row;
// sum-of-squares in double to match ops.cpp (precision matters across 24 layers).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ out, int64_t rows, int64_t d, float eps) {
    const int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + r * d;
    float* orow = out + r * d;
    double sumsq = 0.0;
    for (int64_t c = 0; c < d; ++c) sumsq += static_cast<double>(xr[c]) * xr[c];
    const float scale = 1.0f / sqrtf(static_cast<float>(sumsq / d) + eps);
    for (int64_t c = 0; c < d; ++c) orow[c] = xr[c] * scale * w[c];
}

__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    out[i] = v / (1.0f + expf(-v));
}

__global__ void mul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// Convert an fp32 buffer to fp16 (G5d weight upload): one thread per element.
__global__ void f32_to_f16_kernel(const float* __restrict__ src, half* __restrict__ dst,
                                  int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// [seq, H*D] -> [H, seq, D]: out[h,s,d] = x[s, h*D+d]. One thread per output element.
__global__ void split_heads_kernel(const float* __restrict__ x, float* __restrict__ out,
                                   int64_t seq, int64_t H, int64_t D) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= H * seq * D) return;
    const int64_t h = idx / (seq * D), rem = idx % (seq * D), s = rem / D, d = rem % D;
    out[idx] = x[s * (H * D) + h * D + d];
}

// [H, seq, D] -> [seq, H*D]: out[s, h*D+d] = x[h,s,d]. One thread per input element (scatter).
__global__ void merge_heads_kernel(const float* __restrict__ x, float* __restrict__ out,
                                   int64_t H, int64_t seq, int64_t D) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= H * seq * D) return;
    const int64_t h = idx / (seq * D), rem = idx % (seq * D), s = rem / D, d = rem % D;
    out[s * (H * D) + h * D + d] = x[idx];
}

// GQA: [kv, seq, D] -> [kv*n_rep, seq, D]: out head oh reads source head oh/n_rep.
__global__ void repeat_kv_kernel(const float* __restrict__ x, float* __restrict__ out,
                                 int64_t n_rep, int64_t seq, int64_t D, int64_t out_heads) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= out_heads * seq * D) return;
    const int64_t oh = idx / (seq * D), rem = idx % (seq * D), s = rem / D, d = rem % D;
    const int64_t j = oh / n_rep;  // source kv head
    out[idx] = x[(j * seq + s) * D + d];
}

// RoPE (neox / half-split): out = x*cos + rotate_half(x)*sin, where rotate_half =
// [-x2, x1]. cos/sin are [maxpos, D]; token p sits at absolute position pos_offset+p.
// One thread per output element. Mirrors ops.cpp apply_rope line-for-line.
__global__ void rope_kernel(const float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, float* __restrict__ out,
                            int64_t H, int64_t seq, int64_t D, int64_t pos_offset,
                            const int64_t* __restrict__ d_pos) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= H * seq * D) return;
    const int64_t h = idx / (seq * D), rem = idx % (seq * D), p = rem / D, d = rem % D;
    const int64_t half = D / 2;
    // G6 graph mode: read the position from device (d_pos) so one captured graph spans all decode
    // steps; eager passes nullptr → the host pos_offset, bit-identical to before.
    const int64_t pos = (d_pos ? *d_pos : pos_offset) + p;
    const int64_t base = h * seq * D + p * D;
    const float rot = (d < half) ? -x[base + (d + half)] : x[base + (d - half)];
    out[idx] = x[idx] * cosT[pos * D + d] + rot * sinT[pos * D + d];
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

Tensor cuda_linear_w8a8(const Tensor& x, const Tensor& wq, const Tensor& w_scale,
                        const Tensor* bias) {
    const int64_t m = x.size(0), k = x.size(1), n = wq.size(0);
    Tensor xq = device_alloc({m, k}, DType::I8);
    Tensor a_scale = device_alloc({m}, DType::F32);
    activation_quant_kernel<<<static_cast<unsigned>(m), 256>>>(
        dptr(x), static_cast<int8_t*>(xq.device_ptr()), dptr(a_scale), static_cast<int>(m),
        static_cast<int>(k));
    launch_check("activation_quant_kernel");

    Tensor y = device_alloc({m, n});
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    constexpr int threads = (BM / TM) * (BN / TN);  // 256
    const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                    (static_cast<unsigned>(m) + BM - 1) / BM);
    linear_w8a8_kernel<BM, BN, BK, TM, TN><<<grid, threads>>>(
        static_cast<const int8_t*>(xq.device_ptr()), static_cast<const int8_t*>(wq.device_ptr()),
        dptr(a_scale), dptr(w_scale), bias ? dptr(*bias) : nullptr, dptr(y), static_cast<int>(m),
        static_cast<int>(n), static_cast<int>(k));
    launch_check("linear_w8a8_kernel");
    return y;
}

Tensor cuda_embedding_q8(const Tensor& codes, const Tensor& scale, const std::vector<int64_t>& ids) {
    const int64_t n = static_cast<int64_t>(ids.size()), hidden = codes.size(1);
    Tensor out = device_alloc({n, hidden});
    int64_t* d_ids = nullptr;
    cuda_check(cudaMalloc(&d_ids, n * sizeof(int64_t)), "embedding_q8 ids malloc");
    cuda_check(cudaMemcpy(d_ids, ids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice),
               "embedding_q8 ids H2D");
    embedding_q8_kernel<<<grid1d(n * hidden, kBlock), kBlock>>>(
        static_cast<const int8_t*>(codes.device_ptr()), dptr(scale), d_ids, dptr(out), n, hidden);
    launch_check("embedding_q8_kernel");
    cudaFree(d_ids);
    return out;
}

Tensor cuda_linear_q8(const Tensor& x, const Tensor& codes, const Tensor& scale, const Tensor* bias) {
    const int64_t m = x.size(0), k = x.size(1), n = codes.size(0);
    Tensor y = device_alloc({m, n});
    const int8_t* cp = static_cast<const int8_t*>(codes.device_ptr());
    const float* bp = bias ? dptr(*bias) : nullptr;
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    // Decode (small m): memory-bound, so the warp-per-output GEMV (¼ the fp32 bytes for the huge
    // lm_head); prefill (large m): compute-bound, the tiled GEMM. Same m split as CudaBackend::linear.
    if (m <= kGemvMaxM && !cuda_policy().force_tiled_q8) {
        const int threads = 128;  // 4 warps/block, one output channel per warp
        const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
        linear_q8_gemv_kernel<<<blocks, threads>>>(dptr(x), cp, dptr(scale), bp, dptr(y), mi, ni, ki);
        launch_check("linear_q8_gemv_kernel");
        return y;
    }
    constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    constexpr int threads = (BM / TM) * (BN / TN);  // 256
    const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN, (static_cast<unsigned>(m) + BM - 1) / BM);
    linear_q8_kernel<BM, BN, BK, TM, TN><<<grid, threads>>>(dptr(x), cp, dptr(scale), bp, dptr(y), mi,
                                                            ni, ki);
    launch_check("linear_q8_kernel");
    return y;
}

// Device-resident W8A8 weight (G5d): int8 codes + per-row scales on the GPU; linear() runs the DP4A
// kernel via cuda_linear_w8a8. Lives in the model's qweights_ for the CUDA + W8A8 path, so
// Model::project drives int8 compute through the Weight interface (no forward change).
namespace {
class CudaW8A8Weight : public Weight {
    Tensor wq_;       // device int8 [out, in]
    Tensor w_scale_;  // device fp32 [out]
    int64_t out_, in_;

public:
    CudaW8A8Weight(Tensor wq, Tensor w_scale, int64_t out, int64_t in)
        : wq_(std::move(wq)), w_scale_(std::move(w_scale)), out_(out), in_(in) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override {
        return cuda_linear_w8a8(x, wq_, w_scale_, bias);
    }
    int64_t bytes() const override { return out_ * in_ + out_ * 4; }  // int8 codes + fp32 scales
    int64_t fp32_bytes() const override { return out_ * in_ * 4; }
};
}  // namespace

std::unique_ptr<Weight> make_cuda_w8a8(const Tensor& w) {
    QTensor q = quantize_q8(w);  // per-channel int8 (same as Q8), then upload codes + scales once
    Tensor wq = to_device_i8(q.q.data(), {q.out, q.in});
    Tensor ws({q.out});
    for (int64_t o = 0; o < q.out; ++o) ws[o] = q.scale[static_cast<size_t>(o)];
    return std::make_unique<CudaW8A8Weight>(std::move(wq), to_device(ws), q.out, q.in);
}

// R3b: the device int8-embed Weight (the CUDA mirror of EmbedQ8Weight). Holds the codes + per-row
// scale on the GPU; gather() runs cuda_embedding_q8 (dequant a looked-up row), linear() runs the
// weight-only int8 cuda_linear_q8 (the tied lm_head) — fp32 activations into argmax.
namespace {
class CudaEmbedQ8Weight : public Weight {
    Tensor codes_;  // device int8 [vocab, hidden]
    Tensor scale_;  // device fp32 [vocab]
public:
    CudaEmbedQ8Weight(Tensor codes, Tensor scale) : codes_(std::move(codes)), scale_(std::move(scale)) {}
    Tensor linear(const Tensor& x, const Tensor* bias) const override {
        return cuda_linear_q8(x, codes_, scale_, bias);
    }
    Tensor gather(const std::vector<int64_t>& ids) const override {
        return cuda_embedding_q8(codes_, scale_, ids);
    }
    int64_t bytes() const override { return codes_.numel() + scale_.numel() * 4; }
    int64_t fp32_bytes() const override { return codes_.numel() * 4; }
};
}  // namespace

std::unique_ptr<Weight> make_cuda_q8_embed(const Tensor& host) {
    QTensor q = quantize_q8(host);  // per-channel int8, then upload codes + per-row scale once
    Tensor codes = to_device_i8(q.q.data(), {q.out, q.in});
    Tensor sc({q.out});
    for (int64_t o = 0; o < q.out; ++o) sc[o] = q.scale[static_cast<size_t>(o)];
    return std::make_unique<CudaEmbedQ8Weight>(std::move(codes), to_device(sc));
}

Device CudaBackend::device() const { return Device::CUDA; }


// R2: the kernel-selection knobs, consolidated from seven scattered g_cuda_* globals into one
// CudaPolicy (see cuda_backend.hpp). Still a file-scope instance — the readers include the
// cuda_linear_q8 free function and CudaPagedKVCache::attend, which a per-instance backend member
// can't reach until R3 threads it through the weight seam. The A/B harness sets cuda_policy().<field>.
CudaPolicy g_cuda_policy;
CudaPolicy& cuda_policy() { return g_cuda_policy; }

// Opt-in switch (see cuda_backend.hpp) to upload the layer weights as fp16 (G5d). Load-config, not
// kernel selection — folds into the R3 weight seam alongside g_quantize_embed.
bool g_cuda_fp16_weights = false;

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

Tensor CudaBackend::embedding(const Tensor& table, const std::vector<int64_t>& ids) {
    const int64_t n = static_cast<int64_t>(ids.size()), hidden = table.size(1);
    Tensor out = device_alloc({n, hidden});  // fp32 activations, even from an fp16 table
    // G6 graph mode: gather straight from the device-resident decode token (the driver updates it
    // each step), skipping the per-call sync H2D id-upload that a capture can't record. Eager
    // (g_cuda_graph_token == nullptr): upload the host ids as before — bit-identical.
    const int64_t* d_ids = g_cuda_graph_token;
    int64_t* owned = nullptr;
    if (d_ids == nullptr) {
        cuda_check(cudaMalloc(&owned, n * sizeof(int64_t)), "embedding ids malloc");
        cuda_check(cudaMemcpy(owned, ids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice),
                   "embedding ids H2D");
        d_ids = owned;
    }
    const int blocks = grid1d(n * hidden, kBlock);
    if (table.dtype() == DType::F16)  // G5d: embed_tokens uploaded as half (the largest weight)
        embedding_kernel<half><<<blocks, kBlock>>>(static_cast<const half*>(table.device_ptr()),
                                                   d_ids, dptr(out), n, hidden);
    else
        embedding_kernel<float><<<blocks, kBlock>>>(dptr(table), d_ids, dptr(out), n, hidden);
    launch_check("embedding_kernel");
    if (owned) cudaFree(owned);
    return out;
}

Tensor CudaBackend::rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    const int64_t d = x.size(x.ndim() - 1), rows = x.numel() / d;
    Tensor out = device_alloc(x.shape());
    rmsnorm_kernel<<<grid1d(rows, kBlock), kBlock>>>(dptr(x), dptr(weight), dptr(out), rows, d, eps);
    launch_check("rmsnorm_kernel");
    return out;
}

Tensor CudaBackend::silu(const Tensor& x) {
    Tensor out = device_alloc(x.shape());
    silu_kernel<<<grid1d(x.numel(), kBlock), kBlock>>>(dptr(x), dptr(out), x.numel());
    launch_check("silu_kernel");
    return out;
}

Tensor CudaBackend::mul(const Tensor& a, const Tensor& b) {
    Tensor out = device_alloc(a.shape());
    mul_kernel<<<grid1d(a.numel(), kBlock), kBlock>>>(dptr(a), dptr(b), dptr(out), a.numel());
    launch_check("mul_kernel");
    return out;
}

Tensor CudaBackend::add(const Tensor& a, const Tensor& b) {
    Tensor out = device_alloc(a.shape());
    add_kernel<<<grid1d(a.numel(), kBlock), kBlock>>>(dptr(a), dptr(b), dptr(out), a.numel());
    launch_check("add_kernel");
    return out;
}

Tensor CudaBackend::split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) {
    const int64_t seq = x.size(0);
    Tensor out = device_alloc({n_heads, seq, head_dim});
    split_heads_kernel<<<grid1d(n_heads * seq * head_dim, kBlock), kBlock>>>(
        dptr(x), dptr(out), seq, n_heads, head_dim);
    launch_check("split_heads_kernel");
    return out;
}

Tensor CudaBackend::merge_heads(const Tensor& x) {
    const int64_t H = x.size(0), seq = x.size(1), D = x.size(2);
    Tensor out = device_alloc({seq, H * D});
    merge_heads_kernel<<<grid1d(H * seq * D, kBlock), kBlock>>>(dptr(x), dptr(out), H, seq, D);
    launch_check("merge_heads_kernel");
    return out;
}

Tensor CudaBackend::repeat_kv(const Tensor& x, int64_t n_rep) {
    const int64_t kv = x.size(0), seq = x.size(1), D = x.size(2), out_heads = kv * n_rep;
    Tensor out = device_alloc({out_heads, seq, D});
    repeat_kv_kernel<<<grid1d(out_heads * seq * D, kBlock), kBlock>>>(
        dptr(x), dptr(out), n_rep, seq, D, out_heads);
    launch_check("repeat_kv_kernel");
    return out;
}

Tensor CudaBackend::apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                               int64_t pos_offset) {
    const int64_t H = x.size(0), seq = x.size(1), D = x.size(2);
    Tensor out = device_alloc({H, seq, D});
    rope_kernel<<<grid1d(H * seq * D, kBlock), kBlock>>>(dptr(x), dptr(cos), dptr(sin), dptr(out),
                                                         H, seq, D, pos_offset, g_cuda_graph_pos);
    launch_check("rope_kernel");
    return out;
}

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

Tensor CudaBackend::alloc(const std::vector<int64_t>& shape) { return device_alloc(shape); }

Tensor CudaBackend::extract_row(const Tensor& x, int64_t s, int64_t heads, int64_t dim) {
    // Row s of [n, heads*dim] is contiguous and equals [heads,1,dim] flattened — one D2D copy.
    const int64_t width = heads * dim;
    Tensor out = device_alloc({heads, 1, dim});
    cuda_check(cudaMemcpy(out.device_ptr(), dptr(x) + s * width,
                          static_cast<size_t>(width) * sizeof(float), cudaMemcpyDeviceToDevice),
               "extract_row");
    return out;
}

void CudaBackend::place_row(Tensor& dst, int64_t s, const Tensor& row) {
    const int64_t width = dst.size(1);
    cuda_check(cudaMemcpy(dptr(dst) + s * width, row.device_ptr(),
                          static_cast<size_t>(width) * sizeof(float), cudaMemcpyDeviceToDevice),
               "place_row");
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
    // embedding) go up as fp16 when g_cuda_fp16_weights is set — half the DRAM bytes — and the
    // linear/embedding dispatch reads the F16 dtype directly; everything else (norms, biases, the
    // RoPE tables) stays fp32. The g_cuda_fp16_weights read lives here, not in the model (R3).
    return (fp16_eligible && g_cuda_fp16_weights) ? to_device_f16(weight) : to_device(weight);
}

std::unique_ptr<Weight> CudaBackend::make_quant_weight(const Tensor& host, QuantMode mode) {
    // W8A8 is the GPU quant path — a device-resident int8 weight whose linear() runs int8×int8 DP4A
    // (the compute win). The other modes fall back to the CPU quant (their linear can't take a device
    // tensor) — unchanged from the pre-seam behavior, just routed through the factory now.
    return mode == QuantMode::W8A8 ? make_cuda_w8a8(host) : make_quantized(host, mode);
}

std::unique_ptr<Weight> CudaBackend::make_embed_weight(const Tensor& host) {
    return make_cuda_q8_embed(host);  // device int8 embed/lm_head (codes+scale resident)
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
