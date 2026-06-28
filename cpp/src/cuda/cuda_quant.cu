// CUDA backend: int8 quantization kernels + weights (R4 — split out of cuda_backend.cu).
//
// The W8A8 (int8 activation × int8 weight, DP4A) and weight-only-int8 (Q8 lm_head / embed) compute:
// the kernels (activation quant, the DP4A GEMM, the int8 GEMV, the int8 embed gather), the host
// entry points (cuda_linear_w8a8 / cuda_linear_q8 / cuda_embedding_q8), the device weight wrappers
// (CudaW8A8Weight, CudaEmbedQ8Weight) + their factories, and the CudaBackend quant-weight factory
// methods. Kernels launch only from this TU's own host functions. Shared helpers from cuda_internal.cuh.
#include "cuda/cuda.hpp"           // cuda_linear_w8a8/q8 + cuda_embedding_q8 + to_device* (declared)
#include "cuda/cuda_backend.hpp"   // CudaBackend, cuda_policy(), make_cuda_w8a8 / make_cuda_q8_embed
#include "cuda/cuda_internal.cuh"  // device_alloc, dptr, launch_check, kGemvMaxM, kBlock

#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "quant.hpp"   // quantize_q8, make_quantized, Weight, QTensor, QuantMode
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
}  // namespace

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

std::unique_ptr<Weight> CudaBackend::make_quant_weight(const Tensor& host, QuantMode mode) {
    // W8A8 is the GPU quant path — a device-resident int8 weight whose linear() runs int8×int8 DP4A
    // (the compute win). The other modes fall back to the CPU quant (their linear can't take a device
    // tensor) — unchanged from the pre-seam behavior, just routed through the factory now.
    return mode == QuantMode::W8A8 ? make_cuda_w8a8(host) : make_quantized(host, mode);
}

std::unique_ptr<Weight> CudaBackend::make_embed_weight(const Tensor& host) {
    return make_cuda_q8_embed(host);  // device int8 embed/lm_head (codes+scale resident)
}
}  // namespace ni
