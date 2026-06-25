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

#include "quant.hpp"
#include "tensor.hpp"

namespace ni {

namespace {

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

// The device buffer is stored type-erased as void* on the Tensor; the backend is the
// only place that knows it is really a float*.
float* dptr(const Tensor& t) { return static_cast<float*>(t.device_ptr()); }

// Device memory pool (G5): kill the per-op cudaMalloc. A forward allocates ~360 short-lived
// output buffers; a raw cudaMalloc each SYNCHRONIZES the device, so that alloc traffic — not
// the matmul — dominates GPU decode (run_cuda_decode_bench showed the GEMV win Amdahl-capped
// by it). This is a caching allocator: acquire() reuses a freed buffer of the same byte size
// or cudaMallocs once; release() (a Tensor's deleter) returns it to the free list, never
// cudaFree. Buffers live to process exit — a leaked singleton, so a Tensor outliving any
// wrapper still finds the pool (the CUDA context reclaims the memory at teardown). Reuse is
// safe with NO per-op sync because every kernel runs on the one default stream in submission
// order: a buffer handed to a later op is overwritten only after the earlier op's kernel
// (ordered before it) has read it. Single-threaded, like the rest of the host path. (The
// contiguous KV cache's cat_seq grows a new size each token, so those buffers don't reuse —
// the paged cache avoids that; the win here is the fixed-size activation buffers.)
struct DevicePool {
    std::unordered_map<size_t, std::vector<void*>> free_;  // byte size -> reusable buffers
    std::unordered_map<void*, size_t> size_of_;            // size of every buffer we own

    void* acquire(size_t nbytes) {
        auto it = free_.find(nbytes);
        if (it != free_.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, nbytes), "DevicePool::acquire");
        size_of_[p] = nbytes;
        return p;
    }
    void release(void* p) {
        auto it = size_of_.find(p);
        if (it == size_of_.end()) {  // foreign ptr — shouldn't happen
            cudaFree(p);
            return;
        }
        free_[it->second].push_back(p);
    }
};
DevicePool& pool() {
    static DevicePool* p = new DevicePool();  // leaked on purpose (see above)
    return *p;
}

// Allocate a device tensor of `shape` (+ dtype) from the pool and hand the Tensor a deleter
// that returns the buffer to the pool (not cudaFree) — so steady-state forwards do no
// cudaMalloc. F16 buffers (fp16 weights, G5d) are half the bytes; the pool keys on bytes.
Tensor device_alloc(const std::vector<int64_t>& shape, DType dt = DType::F32) {
    Tensor d(shape, Device::CUDA);
    d.set_dtype(dt);
    const size_t elem =
        (dt == DType::F16) ? sizeof(half) : (dt == DType::I8) ? sizeof(int8_t) : sizeof(float);
    const size_t nbytes = static_cast<size_t>(d.numel()) * elem;
    if (nbytes == 0) return d;  // empty tensor (e.g. an unfilled KV history): no buffer
    void* p = pool().acquire(nbytes);
    d.set_device_ptr(std::shared_ptr<void>(p, [](void* q) { pool().release(q); }));
    return d;
}

// Concatenate two [n_kv, *, head_dim] tensors along the seq (middle) axis into a fresh
// device buffer — how the GPU KV cache appends new tokens to a layer's history. `a` is
// the existing history ([] on the first append); `b` is the new token(s) [n_kv, t, hd].
Tensor cat_seq(const Tensor& a, const Tensor& b) {
    const int64_t nkv = b.size(0), t = b.size(1), hd = b.size(2);
    if (a.numel() == 0) {  // first token(s): copy b into an owned device buffer
        Tensor out = device_alloc({nkv, t, hd});
        cuda_check(cudaMemcpy(out.device_ptr(), b.device_ptr(),
                              static_cast<size_t>(b.numel()) * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cat_seq copy");
        return out;
    }
    const int64_t L = a.size(1), newL = L + t;
    Tensor out = device_alloc({nkv, newL, hd});
    // Per head: the old L rows, then the new t rows. (Heads interleave in memory, so this
    // is one device-to-device copy per head per side, not a single contiguous copy.)
    for (int64_t h = 0; h < nkv; ++h) {
        cuda_check(cudaMemcpy(dptr(out) + h * newL * hd, dptr(a) + h * L * hd,
                              static_cast<size_t>(L * hd) * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cat_seq old");
        cuda_check(cudaMemcpy(dptr(out) + (h * newL + L) * hd, dptr(b) + h * t * hd,
                              static_cast<size_t>(t * hd) * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "cat_seq new");
    }
    return out;
}

// Round a flat element count up to a whole number of `block`-sized 1-D grid blocks.
int grid1d(int64_t total, int block) { return static_cast<int>((total + block - 1) / block); }
constexpr int kBlock = 256;

// The m at/below which linear() is memory-bound and runs the warp-per-output GEMV (G5b) instead of
// the compute-bound tiled GEMM (G5c): decode (m=1) and small batched decode. Shared by the fp32/fp16
// linear() dispatch and the weight-only int8 cuda_linear_q8 (so the int8 lm_head decodes via GEMV too).
constexpr int64_t kGemvMaxM = 16;

// SM count of the current device, queried once. Flash-Decoding (G5g) sizes its split count from this
// rather than hardcoding a number — the engine reads its dimensions, it doesn't bake in magic ones.
int sm_count() {
    static int n = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);
        return p.multiProcessorCount;
    }();
    return n;
}

// How many KV splits Flash-Decoding should use, given nq = H*sq (head,query) pairs and key length sk.
// The non-split kernel launches nq warps; at decode nq=H (14 here) leaves the GPU nearly idle, so we
// split the KV to launch nq*splits warps (~kTargetWavesPerSM per SM) — but never finer than
// kMinKeysPerSplit keys per split (below that the split + combine overhead outweighs the parallelism).
// Returns 1 when splitting isn't worth it (short context, or prefill where nq already saturates the
// GPU); the caller then runs the plain warp kernel. Scales with context: more keys -> more (capped)
// splits, so a longer decode — where attention dominates — gets more parallelism.
int split_count(int64_t nq, int64_t sk) {
    constexpr int kMinKeysPerSplit = 128;
    constexpr int kTargetWavesPerSM = 8;
    const int64_t by_keys = sk / kMinKeysPerSplit;  // cap: keep >= kMinKeysPerSplit keys per split
    if (by_keys <= 1 || nq <= 0) return 1;
    int64_t splits = (static_cast<int64_t>(sm_count()) * kTargetWavesPerSM + nq - 1) / nq;  // ceil
    if (splits > by_keys) splits = by_keys;
    if (splits < 1) splits = 1;
    return static_cast<int>(splits);
}

// Weight loaders that fold the dtype into the read, so one templated kernel serves both fp32
// and fp16 weights (G5d). ldf returns float (for the fp32-accumulate GEMV/tiled paths); ldh
// returns half (for the wmma staging). On a float* both are the identity / the same round the
// fp32 kernels already did, so the fp32 instantiations stay bit-identical to before.
__device__ inline float ldf(const float* w, size_t i) { return w[i]; }
__device__ inline float ldf(const half* w, size_t i) { return __half2float(w[i]); }
__device__ inline half ldh(const float* w, size_t i) { return __float2half(w[i]); }
__device__ inline half ldh(const half* w, size_t i) { return w[i]; }

// Vectorized load of 4 consecutive weights as a float4, dtype-aware (G5d): fp32 reads one 16-byte
// float4; fp16 reads 8 bytes (4 halfs) and converts. Lets one templated tiled GEMM stage either
// weight dtype through the same float4 register path — fp16 just halves the weight DRAM bytes
// (CUDA-core compute, no tensor-core / per-element-convert overhead). i is a multiple of 4 (the
// tiled kernel's loadCol stride), so the fp16 half2 reads are 4-byte aligned.
__device__ inline float4 load4(const float* w, size_t i) {
    return *reinterpret_cast<const float4*>(w + i);
}
__device__ inline float4 load4(const half* w, size_t i) {
    const half2* h = reinterpret_cast<const half2*>(w + i);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}

// y[i,o] = sum_j x[i,j]*w[o,j] + (bias?bias[o]:0). One thread per output (i,o).
// w has one row per output feature (nn.Linear: y = x @ wᵀ + bias), so no transpose.
__global__ void linear_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ bias, float* __restrict__ y,
                              int m, int n, int k) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || o >= n) return;
    const float* xr = x + static_cast<size_t>(i) * k;
    const float* wr = w + static_cast<size_t>(o) * k;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += xr[j] * wr[j];
    if (bias) acc += bias[o];
    y[static_cast<size_t>(i) * n + o] = acc;
}

// Decode GEMV / skinny GEMM (G5b): one WARP per output channel o, vs linear_kernel's one
// thread per (row,output). The 32 lanes stride the k dim, so consecutive lanes read
// consecutive w[o,j] — a coalesced 128-byte transaction, where the naive kernel's threads
// read w[o,j] for consecutive o (stride-k, one transaction PER thread). A warp-shuffle
// then reduces the partial dots in log2(32) steps instead of one serial k-loop. For m>1
// (batched decode) the warp loops the rows, so w[o] is streamed once and reused across
// them from L2 — the batched weight reuse. Same float math as linear_kernel (a different
// reduction ORDER, so ~1e-6 off it, well within GPU tolerance); row s is independent of m,
// so forward_batch row s stays bit-identical to a standalone m=1 forward (run_cuda_batch).
template <typename WT>
__global__ void linear_gemv_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                   const float* __restrict__ bias, float* __restrict__ y,
                                   int m, int n, int k) {
    const int o = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // one warp per output
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (o >= n) return;
    const size_t wbase = static_cast<size_t>(o) * k;
    for (int i = 0; i < m; ++i) {
        const float* xr = x + static_cast<size_t>(i) * k;
        float partial = 0.0f;
        for (int j = lane; j < k; j += 32) partial += xr[j] * ldf(w, wbase + j);  // coalesced
        for (int off = 16; off > 0; off >>= 1)  // warp-reduce the 32 partials
            partial += __shfl_down_sync(0xffffffff, partial, off);
        if (lane == 0) y[static_cast<size_t>(i) * n + o] = partial + (bias ? bias[o] : 0.0f);
    }
}

// Prefill tiled GEMM (G5c): m>16 rows. Replaces the naive one-thread-per-output kernel
// (which re-read x and w from DRAM for every output — ~0.25 FLOP/byte, ~3% of cuBLAS). A
// thread BLOCK cooperatively stages a BM×BK tile of x and a BK×BN tile of w into shared
// memory; each THREAD then computes a TM×TN micro-tile of outputs by outer products held in
// registers, so every shared value is reused TM (or TN) times and every global load is
// amortized across the block. That lifts arithmetic intensity over the roofline ridge —
// prefill becomes compute-bound, as it should be. Same float math as the naive/GEMV kernels
// (a tiled reduction order), within GPU tolerance; golden tokens unchanged.
//
// Layout: y[m,n] = x[m,k] @ w[n,k]ᵀ. x is row-major [m,k]; w is row-major [n,k], so a w ROW
// (one output's weights) is contiguous — both staging loops stride the k dim fastest, which
// keeps the global reads coalesced. OOB rows/cols (ragged m,n) load 0 and skip the store; a
// thread reads only its own As rows / Bs cols, so the zero padding never contaminates a
// valid output.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void linear_tiled_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                    const float* __restrict__ bias, float* __restrict__ y, int m,
                                    int n, int k) {
    constexpr int numThreads = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK];  // a tile of x: [BM rows][BK cols]
    __shared__ float Bs[BK * BN];  // a tile of w viewed [BK k-rows][BN output-cols]

    const int tid = static_cast<int>(threadIdx.x);
    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);  // this thread's tile
    const int rowBase = static_cast<int>(blockIdx.y) * BM;  // global y-row base of the block
    const int colBase = static_cast<int>(blockIdx.x) * BN;  // global y-col (output) base

    // Cooperative-load indices: k fastest in both (innerCol = tid % BK), so consecutive
    // threads read consecutive memory (x within a row; w within a row).
    const int innerColA = tid % BK, innerRowA = tid / BK;
    const int innerColB = tid % BK, innerRowB = tid / BK;
    constexpr int strideA = numThreads / BK;  // rows of x staged per pass
    constexpr int strideB = numThreads / BK;  // output cols of w staged per pass

    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int kIdx = 0; kIdx < k; kIdx += BK) {
        for (int off = 0; off < BM; off += strideA) {
            const int r = innerRowA + off, gr = rowBase + r, gc = kIdx + innerColA;
            As[r * BK + innerColA] = (gr < m && gc < k) ? x[gr * k + gc] : 0.0f;
        }
        for (int off = 0; off < BN; off += strideB) {
            const int o = innerRowB + off, go = colBase + o, gk = kIdx + innerColB;
            Bs[innerColB * BN + o] = (go < n && gk < k) ? w[go * k + gk] : 0.0f;
        }
        __syncthreads();
        for (int dot = 0; dot < BK; ++dot) {
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + (threadCol * TN + j)];
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= m) continue;
        for (int j = 0; j < TN; ++j) {
            const int gc = colBase + threadCol * TN + j;
            if (gc >= n) continue;
            y[static_cast<size_t>(gr) * n + gc] = acc[i * TN + j] + (bias ? bias[gc] : 0.0f);
        }
    }
}

// Prefill tiled GEMM, VECTORIZED (G5c+): linear_tiled_kernel sharpened with the three standard
// levers from the canonical CUDA-GEMM progression (Simon Boehm's "kernel 6"). Same tiled-reduction
// float math, within GPU tolerance; golden tokens unchanged. The wins over the scalar G5c kernel:
//   1. float4 (128-bit) global loads/stores — one LDG.128 moves 4 contiguous floats, quartering
//      the load/store instruction count and saturating the memory pipe.
//   2. As staged TRANSPOSED in shared memory ([BK][BM], not [BM][BK]), so the inner-loop register
//      reads As[dot][·] are contiguous too — a float4 LDS instead of TM strided loads, and the
//      strided-by-BK bank conflicts the scalar kernel hit on its regM reads are gone.
//   3. a larger 128×128 block tile (TM=TN=8 → 64 outputs/thread): each staged value is reused
//      more, lifting arithmetic intensity further past the roofline ridge.
//
// Preconditions (the dispatch checks them, else falls back to the scalar tiled kernel): n % BN == 0
// and k % BK == 0, so only the m (row) dimension is ragged. EVERY float4 runs along k (the two
// staging loads) or n (the store) — both always multiples of 4 for a Qwen linear — so no float4
// ever straddles a ragged edge; only whole rows are bounds-checked (gr < m), which guards the m
// tail (e.g. the ragged m=100 parity case). Both x[m,k] and w[n,k] are row-major with k contiguous,
// so BOTH stage by float4-reading along k and scattering transposed into a [BK][*] tile (As[kl][ml],
// Bs[kl][nl]); symmetric. Alignment holds because k,n are multiples of 4 and every tile base lands
// on a 4-float boundary, so each reinterpret_cast<float4*> address is 16-byte aligned.
template <typename WT, int BM, int BN, int BK, int TM, int TN>
__global__ void linear_tiled_vec_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                        const float* __restrict__ bias, float* __restrict__ y, int m,
                                        int n, int k) {
    // WT = float (fp32 weights) or half (fp16 weights, G5d). x (activations) is always fp32; only
    // the weight global read differs (load4), and the staged Bs / register math stay fp32.
    constexpr int numThreads = (BM / TM) * (BN / TN);
    __shared__ float As[BK * BM];  // x tile, TRANSPOSED: As[kl * BM + ml]
    __shared__ float Bs[BK * BN];  // w tile, TRANSPOSED: Bs[kl * BN + nl]

    const int tid = static_cast<int>(threadIdx.x);
    const int rowBase = static_cast<int>(blockIdx.y) * BM;  // global y-row base of the block
    const int colBase = static_cast<int>(blockIdx.x) * BN;  // global y-col (output) base

    // Cooperative float4-load indices: each tile row is BK wide = BK/4 float4, so a thread owns
    // float4 (loadRow, loadCol) and strides loadRow by rowStride to cover all BM/BN rows. (At
    // 128×128×8 / 256 threads this is one float4 each, single pass — strideA/B == BM/BN.)
    constexpr int f4PerRow = BK / 4;
    const int loadRow = tid / f4PerRow, loadCol = (tid % f4PerRow) * 4;
    constexpr int rowStrideA = numThreads / f4PerRow;
    constexpr int rowStrideB = numThreads / f4PerRow;

    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);  // this thread's micro-tile
    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];

    for (int kIdx = 0; kIdx < k; kIdx += BK) {
        for (int off = 0; off < BM; off += rowStrideA) {  // stage x, transposed into As
            const int ml = loadRow + off, gr = rowBase + ml, gc = kIdx + loadCol;
            float4 t = {0.0f, 0.0f, 0.0f, 0.0f};  // ragged m tail loads 0 (whole row OOB)
            if (gr < m) t = *reinterpret_cast<const float4*>(&x[static_cast<size_t>(gr) * k + gc]);
            As[(loadCol + 0) * BM + ml] = t.x;
            As[(loadCol + 1) * BM + ml] = t.y;
            As[(loadCol + 2) * BM + ml] = t.z;
            As[(loadCol + 3) * BM + ml] = t.w;
        }
        for (int off = 0; off < BN; off += rowStrideB) {  // stage w, transposed into Bs
            const int nl = loadRow + off, go = colBase + nl, gk = kIdx + loadCol;
            // n % BN == 0 (precondition) → go is always valid; only k is strided, never ragged.
            float4 t = load4(w, static_cast<size_t>(go) * k + gk);  // fp32 float4 or fp16 4-half load
            Bs[(loadCol + 0) * BN + nl] = t.x;
            Bs[(loadCol + 1) * BN + nl] = t.y;
            Bs[(loadCol + 2) * BN + nl] = t.z;
            Bs[(loadCol + 3) * BN + nl] = t.w;
        }
        __syncthreads();
        for (int dot = 0; dot < BK; ++dot) {  // outer-product accumulate, all reads float4
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<const float4*>(&As[dot * BM + threadRow * TM + i]);
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<const float4*>(&Bs[dot * BN + threadCol * TN + j]);
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= m) continue;  // ragged m tail: skip the whole row (n is never ragged here)
        for (int j = 0; j < TN; j += 4) {
            const int gc = colBase + threadCol * TN + j;
            float4 t;
            t.x = acc[i * TN + j + 0] + (bias ? bias[gc + 0] : 0.0f);
            t.y = acc[i * TN + j + 1] + (bias ? bias[gc + 1] : 0.0f);
            t.z = acc[i * TN + j + 2] + (bias ? bias[gc + 2] : 0.0f);
            t.w = acc[i * TN + j + 3] + (bias ? bias[gc + 3] : 0.0f);
            *reinterpret_cast<float4*>(&y[static_cast<size_t>(gr) * n + gc]) = t;
        }
    }
}

// Prefill tiled GEMM, DOUBLE-BUFFERED (G5 micro-gain): software-pipelines linear_tiled_vec_kernel so
// the next K-tile's global load overlaps the current tile's compute, hiding the ~400-cycle DRAM
// latency that a single-buffered tile eats at each __syncthreads. Aimed at the LOW-OCCUPANCY
// projections — down (n=896,k=4864) and q/o (n=896,k=896) launch only 28 blocks (m=128/64 × n=896/64)
// on a 56-SM GPU, so ~half the SMs sit idle and there is no second resident block to hide the stall
// behind (the lm_head's >1000 blocks and gate/up's 152 already hide it via cross-block occupancy —
// which is why this is scoped to the narrow-n path, not lm_head). The textbook lever here is cp.async
// (LDGSTS: async global→smem, bypassing registers); we deliberately use REGISTER-prefetch double-
// buffering instead, because cp.async copies a CONTIGUOUS gmem chunk to a CONTIGUOUS smem chunk and
// CANNOT do the transpose-scatter this kernel relies on — linear_tiled_vec_kernel stages As/Bs
// TRANSPOSED ([BK][BM]/[BK][BN]) precisely so the inner-loop reads are conflict-free float4. A cp.async
// version would need the natural [BM][BK] layout, whose inner reads stride by BK·TN ≡ 0 (mod 32 banks)
// → a 16-way bank conflict, reintroducing exactly what G5c+'s transpose removed; reconciling the two
// needs smem swizzling, out of scope for a micro-gain. Register-prefetch hits the SAME overlap while
// keeping the transposed layout, so it is BIT-IDENTICAL to linear_tiled_vec_kernel: same smem
// contents, same ascending-k accumulation, only the LOAD TIMING moves. Two smem stages ping-pong;
// the loop prefetches tile kt+1 into registers (LDGs issued early), computes tile kt, then stores the
// prefetched registers into the other stage. Preconditions as G5c+ (n%BN==0, k%BK==0; only m ragged).
template <typename WT, int BM, int BN, int BK, int TM, int TN>
__global__ void linear_tiled_db_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                       const float* __restrict__ bias, float* __restrict__ y, int m,
                                       int n, int k) {
    constexpr int numThreads = (BM / TM) * (BN / TN);
    __shared__ float As[2][BK * BM];  // x tile, TRANSPOSED, DOUBLE-BUFFERED: As[stage][kl * BM + ml]
    __shared__ float Bs[2][BK * BN];  // w tile, TRANSPOSED, DOUBLE-BUFFERED: Bs[stage][kl * BN + nl]

    const int tid = static_cast<int>(threadIdx.x);
    const int rowBase = static_cast<int>(blockIdx.y) * BM;
    const int colBase = static_cast<int>(blockIdx.x) * BN;

    // Same cooperative float4-load mapping as linear_tiled_vec_kernel (one float4 each at 64×64×16/256).
    constexpr int f4PerRow = BK / 4;
    const int loadRow = tid / f4PerRow, loadCol = (tid % f4PerRow) * 4;
    constexpr int rowStrideA = numThreads / f4PerRow;
    constexpr int rowStrideB = numThreads / f4PerRow;
    constexpr int nA = BM / rowStrideA;  // float4s of x this thread stages per K-tile
    constexpr int nB = BN / rowStrideB;  // float4s of w this thread stages per K-tile

    const int threadRow = tid / (BN / TN), threadCol = tid % (BN / TN);  // this thread's micro-tile
    float acc[TM * TN] = {0.0f};
    float regM[TM], regN[TN];
    float4 ra[nA], rb[nB];  // the double buffer's "next tile" prefetch registers

    const int numKtiles = k / BK;  // k % BK == 0 (precondition)

    // --- prologue: load K-tile 0 into registers, store it transposed into smem stage 0 ---
#pragma unroll
    for (int p = 0; p < nA; ++p) {
        const int ml = loadRow + p * rowStrideA, gr = rowBase + ml;
        ra[p] = (gr < m) ? *reinterpret_cast<const float4*>(&x[static_cast<size_t>(gr) * k + loadCol])
                         : float4{0.0f, 0.0f, 0.0f, 0.0f};  // ragged m tail loads 0 (whole row OOB)
    }
#pragma unroll
    for (int p = 0; p < nB; ++p) {
        const int nl = loadRow + p * rowStrideB, go = colBase + nl;
        rb[p] = load4(w, static_cast<size_t>(go) * k + loadCol);  // n%BN==0 → go always valid
    }
#pragma unroll
    for (int p = 0; p < nA; ++p) {
        const int ml = loadRow + p * rowStrideA;
        As[0][(loadCol + 0) * BM + ml] = ra[p].x;
        As[0][(loadCol + 1) * BM + ml] = ra[p].y;
        As[0][(loadCol + 2) * BM + ml] = ra[p].z;
        As[0][(loadCol + 3) * BM + ml] = ra[p].w;
    }
#pragma unroll
    for (int p = 0; p < nB; ++p) {
        const int nl = loadRow + p * rowStrideB;
        Bs[0][(loadCol + 0) * BN + nl] = rb[p].x;
        Bs[0][(loadCol + 1) * BN + nl] = rb[p].y;
        Bs[0][(loadCol + 2) * BN + nl] = rb[p].z;
        Bs[0][(loadCol + 3) * BN + nl] = rb[p].w;
    }
    __syncthreads();

    for (int kt = 0; kt < numKtiles; ++kt) {
        const int cur = kt & 1;
        // Prefetch K-tile kt+1 into registers NOW — the LDGs are issued and their long DRAM latency
        // overlaps the compute below (which reads only smem[cur], not ra/rb).
        if (kt + 1 < numKtiles) {
            const int kIdx = (kt + 1) * BK;
#pragma unroll
            for (int p = 0; p < nA; ++p) {
                const int ml = loadRow + p * rowStrideA, gr = rowBase + ml, gc = kIdx + loadCol;
                ra[p] = (gr < m)
                            ? *reinterpret_cast<const float4*>(&x[static_cast<size_t>(gr) * k + gc])
                            : float4{0.0f, 0.0f, 0.0f, 0.0f};
            }
#pragma unroll
            for (int p = 0; p < nB; ++p) {
                const int nl = loadRow + p * rowStrideB, go = colBase + nl, gk = kIdx + loadCol;
                rb[p] = load4(w, static_cast<size_t>(go) * k + gk);
            }
        }
        // Compute tile kt on smem[cur] — identical math/order to linear_tiled_vec_kernel.
#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
            for (int i = 0; i < TM; i += 4)
                *reinterpret_cast<float4*>(&regM[i]) =
                    *reinterpret_cast<const float4*>(&As[cur][dot * BM + threadRow * TM + i]);
#pragma unroll
            for (int j = 0; j < TN; j += 4)
                *reinterpret_cast<float4*>(&regN[j]) =
                    *reinterpret_cast<const float4*>(&Bs[cur][dot * BN + threadCol * TN + j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
        }
        // Store the prefetched registers into the OTHER stage (it isn't read until kt+1 — the closing
        // __syncthreads orders this write before that read; the WAR on smem[cur] is fenced too).
        if (kt + 1 < numKtiles) {
            const int nxt = cur ^ 1;
#pragma unroll
            for (int p = 0; p < nA; ++p) {
                const int ml = loadRow + p * rowStrideA;
                As[nxt][(loadCol + 0) * BM + ml] = ra[p].x;
                As[nxt][(loadCol + 1) * BM + ml] = ra[p].y;
                As[nxt][(loadCol + 2) * BM + ml] = ra[p].z;
                As[nxt][(loadCol + 3) * BM + ml] = ra[p].w;
            }
#pragma unroll
            for (int p = 0; p < nB; ++p) {
                const int nl = loadRow + p * rowStrideB;
                Bs[nxt][(loadCol + 0) * BN + nl] = rb[p].x;
                Bs[nxt][(loadCol + 1) * BN + nl] = rb[p].y;
                Bs[nxt][(loadCol + 2) * BN + nl] = rb[p].z;
                Bs[nxt][(loadCol + 3) * BN + nl] = rb[p].w;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {  // epilogue: identical to linear_tiled_vec_kernel
        const int gr = rowBase + threadRow * TM + i;
        if (gr >= m) continue;  // ragged m tail: skip the whole row (n is never ragged here)
        for (int j = 0; j < TN; j += 4) {
            const int gc = colBase + threadCol * TN + j;
            float4 t;
            t.x = acc[i * TN + j + 0] + (bias ? bias[gc + 0] : 0.0f);
            t.y = acc[i * TN + j + 1] + (bias ? bias[gc + 1] : 0.0f);
            t.z = acc[i * TN + j + 2] + (bias ? bias[gc + 2] : 0.0f);
            t.w = acc[i * TN + j + 3] + (bias ? bias[gc + 3] : 0.0f);
            *reinterpret_cast<float4*>(&y[static_cast<size_t>(gr) * n + gc]) = t;
        }
    }
}

// Prefill tiled GEMM, WARP-TILED (G5c+ second lever, Boehm "kernel 10"): inserts a WARP tile
// between the block tile and the per-thread micro-tile. The block's BM×BN output splits into
// (BM/WM)×(BN/WN) warp tiles, one per warp; each warp sweeps its WM×WN tile as a WMITER×WNITER
// grid of subtiles, each thread owning a TM×TN micro-tile in every subtile. Why it beats the flat
// thread tiling (G5c+): a warp's 32 threads now reuse the SAME small As/Bs slice across WMITER·WNITER
// subtiles held in registers, so each shared-memory word feeds far more FMAs — the compute-to-smem
// ratio that decides a compute-bound GEMM. It's aimed at the big lm_head matmul (n=151936): the most
// compute-bound projection with the most headroom (G5c+ left it at ~51% of cuBLAS) and enough work
// to fill a 128×128 tile with >1000 blocks. Same ascending-k accumulation per output as every tiled
// kernel here, so still bit-identical to the scalar one; golden tokens unchanged. Preconditions as
// G5c+ (n%BN==0, k%BK==0; only m ragged, bounds-checked at the store).
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS>
__global__ void linear_warptiled_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ bias, float* __restrict__ y, int m,
                                        int n, int k) {
    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);  // M-subtiles per warp
    constexpr int WSUBM = WM / WMITER;                           // a warp subtile is WSUBM×WSUBN
    constexpr int WSUBN = WN / WNITER;
    __shared__ float As[BK * BM];  // x tile, TRANSPOSED: As[kl * BM + ml]
    __shared__ float Bs[BK * BN];  // w tile, TRANSPOSED: Bs[kl * BN + nl]

    const int tid = static_cast<int>(threadIdx.x);
    const int rowBase = static_cast<int>(blockIdx.y) * BM;
    const int colBase = static_cast<int>(blockIdx.x) * BN;

    const int warpIdx = tid / 32;                         // which warp tile this thread is in
    const int warpRow = warpIdx / (BN / WN), warpCol = warpIdx % (BN / WN);
    const int laneIdx = tid % 32;                         // thread placement inside the warp subtile
    const int threadRowInWarp = laneIdx / (WSUBN / TN);   // (WSUBM/TM)×(WSUBN/TN) == 32
    const int threadColInWarp = laneIdx % (WSUBN / TN);

    // Cooperative float4-load indices (same transposed staging as G5c+, but multi-pass: NUM_THREADS
    // float4 cover BM×BK / BK×BN over (NUM_THREADS*4)/BK-row strides).
    const int innerRowA = tid / (BK / 4), innerColA = tid % (BK / 4);
    const int innerRowB = tid / (BK / 4), innerColB = tid % (BK / 4);
    constexpr int rowStrideA = (NUM_THREADS * 4) / BK;
    constexpr int rowStrideB = (NUM_THREADS * 4) / BK;

    float acc[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM];
    float regN[WNITER * TN];

    for (int kIdx = 0; kIdx < k; kIdx += BK) {
        for (int off = 0; off < BM; off += rowStrideA) {  // stage x, transposed into As
            const int ml = innerRowA + off, gr = rowBase + ml, gc = kIdx + innerColA * 4;
            float4 t = {0.0f, 0.0f, 0.0f, 0.0f};  // ragged m tail loads 0 (whole row OOB)
            if (gr < m) t = *reinterpret_cast<const float4*>(&x[static_cast<size_t>(gr) * k + gc]);
            As[(innerColA * 4 + 0) * BM + ml] = t.x;
            As[(innerColA * 4 + 1) * BM + ml] = t.y;
            As[(innerColA * 4 + 2) * BM + ml] = t.z;
            As[(innerColA * 4 + 3) * BM + ml] = t.w;
        }
        for (int off = 0; off < BN; off += rowStrideB) {  // stage w, transposed into Bs
            const int nl = innerRowB + off, go = colBase + nl, gk = kIdx + innerColB * 4;
            float4 t = *reinterpret_cast<const float4*>(&w[static_cast<size_t>(go) * k + gk]);
            Bs[(innerColB * 4 + 0) * BN + nl] = t.x;
            Bs[(innerColB * 4 + 1) * BN + nl] = t.y;
            Bs[(innerColB * 4 + 2) * BN + nl] = t.z;
            Bs[(innerColB * 4 + 3) * BN + nl] = t.w;
        }
        __syncthreads();
        for (int dot = 0; dot < BK; ++dot) {
            for (int wm = 0; wm < WMITER; ++wm)  // load this warp's M-fragments (float4 from smem)
                for (int i = 0; i < TM; i += 4)
                    *reinterpret_cast<float4*>(&regM[wm * TM + i]) = *reinterpret_cast<const float4*>(
                        &As[dot * BM + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i]);
            for (int wn = 0; wn < WNITER; ++wn)  // load this warp's N-fragments
                for (int j = 0; j < TN; j += 4)
                    *reinterpret_cast<float4*>(&regN[wn * TN + j]) = *reinterpret_cast<const float4*>(
                        &Bs[dot * BN + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j]);
            for (int wm = 0; wm < WMITER; ++wm)  // outer product over all subtiles
                for (int wn = 0; wn < WNITER; ++wn)
                    for (int i = 0; i < TM; ++i)
                        for (int j = 0; j < TN; ++j)
                            acc[(wm * TM + i) * (WNITER * TN) + wn * TN + j] +=
                                regM[wm * TM + i] * regN[wn * TN + j];
        }
        __syncthreads();
    }

    for (int wm = 0; wm < WMITER; ++wm)  // store each subtile's TM×TN micro-tile, float4 along n
        for (int wn = 0; wn < WNITER; ++wn)
            for (int i = 0; i < TM; ++i) {
                const int gr = rowBase + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
                if (gr >= m) continue;  // ragged m tail (n never ragged here)
                for (int j = 0; j < TN; j += 4) {
                    const int gc = colBase + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
                    const int a = (wm * TM + i) * (WNITER * TN) + wn * TN + j;
                    float4 t;
                    t.x = acc[a + 0] + (bias ? bias[gc + 0] : 0.0f);
                    t.y = acc[a + 1] + (bias ? bias[gc + 1] : 0.0f);
                    t.z = acc[a + 2] + (bias ? bias[gc + 2] : 0.0f);
                    t.w = acc[a + 3] + (bias ? bias[gc + 3] : 0.0f);
                    *reinterpret_cast<float4*>(&y[static_cast<size_t>(gr) * n + gc]) = t;
                }
            }
}

// Prefill on tensor cores (G5d): the same tiled GEMM, but each warp's 16×16×16 multiply runs
// on the Ada tensor cores via the wmma API. x/w stay fp32 in DRAM (the engine is fp32); each
// BM×BK / BK×BN tile is converted to half as it is staged into shared memory, and the matmul
// accumulates in fp32 (wmma half×half→float). Mixed precision: fp16 inputs, fp32 accumulate —
// far faster compute than the fp32 CUDA-core kernel, but lossy (fp16 rounds inputs to ~3
// decimal digits). Opt-in via g_cuda_use_wmma; the accuracy cost is measured (test_cuda /
// run_cuda_bench). Block = 2×2 warps over a 64×64 tile (each warp a 2×2 grid of 16×16 frags);
// n is a multiple of 64 for every Qwen linear, m is bounds-checked at the store.
template <typename WT>
__global__ void linear_wmma_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                   const float* __restrict__ bias, float* __restrict__ y, int m,
                                   int n, int k) {
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, BK = 16;
    constexpr int WM = 16, WN = 16, WK = 16;  // wmma fragment shape
    constexpr int numThreads = 128;           // 2×2 warps
    __shared__ half As[BM * BK];              // x tile [BM][BK], row-major
    __shared__ half Bs[BK * BN];              // w tile [BK][BN] = [k][o], row-major
    __shared__ float Cs[4][WM * WN];          // per-warp 16×16 store staging (bias + bounds)

    const int tid = static_cast<int>(threadIdx.x);
    const int warpId = tid / 32, lane = tid % 32;
    const int warpRow = warpId / 2, warpCol = warpId % 2;  // 2×2 warp grid
    const int rowBase = static_cast<int>(blockIdx.y) * BM;
    const int colBase = static_cast<int>(blockIdx.x) * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int kk = 0; kk < k; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += numThreads) {  // stage x (k fastest: coalesced)
            const int r = idx / BK, c = idx % BK, gr = rowBase + r, gc = kk + c;
            As[r * BK + c] = (gr < m && gc < k) ? __float2half(x[gr * k + gc]) : __float2half(0.0f);
        }
        for (int idx = tid; idx < BK * BN; idx += numThreads) {  // stage w (k fastest: coalesced)
            const int o = idx / BK, c = idx % BK, go = colBase + o, gc = kk + c;
            Bs[c * BN + o] =
                (go < n && gc < k) ? ldh(w, static_cast<size_t>(go) * k + gc) : __float2half(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> bFrag;
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(aFrag, As + (warpRow * 32 + i * 16) * BK, BK);
            for (int j = 0; j < 2; ++j) {
                wmma::load_matrix_sync(bFrag, Bs + (warpCol * 32 + j * 16), BN);
                wmma::mma_sync(acc[i][j], aFrag, bFrag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(Cs[warpId], acc[i][j], WN, wmma::mem_row_major);
            __syncwarp();
            const int r0 = rowBase + warpRow * 32 + i * 16, c0 = colBase + warpCol * 32 + j * 16;
            for (int e = lane; e < WM * WN; e += 32) {
                const int gr = r0 + e / WN, gc = c0 + e % WN;
                if (gr < m && gc < n)
                    y[static_cast<size_t>(gr) * n + gc] = Cs[warpId][e] + (bias ? bias[gc] : 0.0f);
            }
            __syncwarp();
        }
    }
}

// Prefill on tensor cores, 128×128 WARP-TILED (G5d). The 64×64 linear_wmma_kernel above starves the
// Ada tensor cores: a 64×64 / BK=16 tile is only ~32 FLOP/byte, far under the fp16 roofline ridge
// (~282 FLOP/byte at 142 TFLOP/s), so it loses even to the fp32 tiled GEMM. This stages a 128×128
// block tile and gives each of 8 warps a 64×32 region = a 4×2 grid of 16×16×16 wmma fragments, so
// each staged smem word feeds 8 tensor-core MMAs (the compute-to-smem ratio a TC GEMM lives on) and
// each warp reloads only 4 A- + 2 B-fragments per K-step. Aimed at the huge lm_head (n=151936):
// >1000 blocks, so cross-block occupancy hides the per-K-tile load-sync stall even without cp.async/
// double-buffering (which IS that lever, for the lower-occupancy projections — explored in
// linear_tiled_db_kernel, a tie on this model). x (activations) is always fp32,
// staged to half; w is fp32 or fp16 (WT) read via ldh. fp16 inputs, fp32 accumulate — same mixed
// precision as the 64² kernel, so the same ~fp16 accuracy cost. n%128==0 and k%16==0 (the dispatch
// guarantees it); only m is ragged, bounds-checked at the store.
template <typename WT>
__global__ void linear_wmma_tiled_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                         const float* __restrict__ bias, float* __restrict__ y,
                                         int m, int n, int k) {
    using namespace nvcuda;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WARPS_M = 2, WARPS_N = 4, NUM_WARPS = WARPS_M * WARPS_N;  // 8 warps = 256 threads
    constexpr int WMROWS = BM / WARPS_M, WNCOLS = BN / WARPS_N;             // 64×32 per warp
    constexpr int MFRAG = WMROWS / 16, NFRAG = WNCOLS / 16;                 // 4×2 frags per warp
    __shared__ half As[BM * BK];              // x tile [BM][BK], row-major (M-major)
    __shared__ half Bs[BK * BN];              // w tile [BK][BN], row-major (K-major)
    __shared__ float Cs[NUM_WARPS][16 * 16];  // per-warp 16×16 epilogue staging (bias + bounds)

    const int tid = static_cast<int>(threadIdx.x);
    const int warpId = tid / 32, lane = tid % 32;
    const int warpRow = warpId / WARPS_N, warpCol = warpId % WARPS_N;  // 2×4 warp grid
    const int rowBase = static_cast<int>(blockIdx.y) * BM;
    const int colBase = static_cast<int>(blockIdx.x) * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MFRAG][NFRAG];
    for (int i = 0; i < MFRAG; ++i)
        for (int j = 0; j < NFRAG; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int kk = 0; kk < k; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += NUM_WARPS * 32) {  // stage x (k fastest: coalesced)
            const int r = idx / BK, c = idx % BK, gr = rowBase + r;
            As[r * BK + c] = (gr < m) ? __float2half(x[static_cast<size_t>(gr) * k + kk + c])
                                      : __float2half(0.0f);
        }
        for (int idx = tid; idx < BK * BN; idx += NUM_WARPS * 32) {  // stage w (k fastest: coalesced)
            const int o = idx / BK, c = idx % BK, go = colBase + o;
            Bs[c * BN + o] =
                (go < n) ? ldh(w, static_cast<size_t>(go) * k + kk + c) : __float2half(0.0f);
        }
        __syncthreads();
        // Reload this warp's A/B fragments once, then 8 MMAs reuse them (the warp-tile leverage).
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag[MFRAG];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag[NFRAG];
        for (int i = 0; i < MFRAG; ++i)
            wmma::load_matrix_sync(aFrag[i], As + (warpRow * WMROWS + i * 16) * BK, BK);
        for (int j = 0; j < NFRAG; ++j)
            wmma::load_matrix_sync(bFrag[j], Bs + (warpCol * WNCOLS + j * 16), BN);
        for (int i = 0; i < MFRAG; ++i)
            for (int j = 0; j < NFRAG; ++j) wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        __syncthreads();
    }

    for (int i = 0; i < MFRAG; ++i)
        for (int j = 0; j < NFRAG; ++j) {
            wmma::store_matrix_sync(Cs[warpId], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            const int r0 = rowBase + warpRow * WMROWS + i * 16;
            const int c0 = colBase + warpCol * WNCOLS + j * 16;
            for (int e = lane; e < 16 * 16; e += 32) {
                const int gr = r0 + e / 16, gc = c0 + e % 16;
                if (gr < m && gc < n)
                    y[static_cast<size_t>(gr) * n + gc] = Cs[warpId][e] + (bias ? bias[gc] : 0.0f);
            }
            __syncwarp();
        }
}

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
                            int64_t H, int64_t seq, int64_t D, int64_t pos_offset) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= H * seq * D) return;
    const int64_t h = idx / (seq * D), rem = idx % (seq * D), p = rem / D, d = rem % D;
    const int64_t half = D / 2;
    const int64_t pos = pos_offset + p;
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
                                   int64_t hd, int64_t block_size, int64_t start) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_kv * t * hd) return;
    const int64_t d = idx % hd, i = (idx / hd) % t, h = idx / (hd * t);
    const int64_t pos = start + i;
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
    int64_t end, float scale) {
    const int warpsPerBlock = static_cast<int>(blockDim.x) / 32;
    const int64_t hi = static_cast<int64_t>(blockIdx.x) * warpsPerBlock + threadIdx.x / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (hi >= n_heads * sq) return;  // warp-uniform
    const int64_t h = hi / sq, i = hi % sq;
    const int64_t kvh = h / n_rep;  // GQA: this query head's KV head
    const float* qi = q + (h * sq + i) * hd;
    const int64_t limit = causal ? (query_offset + i + 1) : end;

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

// launch_check (the post-launch error check) is defined below with the CudaBackend ops;
// forward-declare it for the fp16 upload helper, which launches the convert kernel here.
static void launch_check(const char* what);

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
    if (m <= kGemvMaxM && !g_cuda_force_tiled_q8) {
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
// Model::project drives int8 compute through the QuantizedWeight interface (no forward change).
namespace {
class CudaW8A8Weight : public QuantizedWeight {
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

std::unique_ptr<QuantizedWeight> make_cuda_w8a8(const Tensor& w) {
    QTensor q = quantize_q8(w);  // per-channel int8 (same as Q8), then upload codes + scales once
    Tensor wq = to_device_i8(q.q.data(), {q.out, q.in});
    Tensor ws({q.out});
    for (int64_t o = 0; o < q.out; ++o) ws[o] = q.scale[static_cast<size_t>(o)];
    return std::make_unique<CudaW8A8Weight>(std::move(wq), to_device(ws), q.out, q.in);
}

Device CudaBackend::device() const { return Device::CUDA; }

// After a launch, check the launch config (cheap, immediate). The device sync that used to
// follow was dropped in G5: kernels run ordered on the one default stream, so correctness
// needs no per-op sync — results are read only through to_host's (synchronizing) D2H copy.
// The trade is attribution: an in-kernel fault now surfaces at the next sync, not its call.
static void launch_check(const char* what) {
    cuda_check(cudaGetLastError(), what);
}

// Bench-only switch (see cuda_backend.hpp) to force the naive GEMM path for A/B timing.
bool g_cuda_force_naive_gemm = false;
// Bench-only switch (see cuda_backend.hpp) to force the tiled int8 kernel for A/B timing the GEMV.
bool g_cuda_force_tiled_q8 = false;
// Opt-in switch (see cuda_backend.hpp) to run the prefill GEMM on the tensor cores (G5d).
bool g_cuda_use_wmma = false;
// Opt-in switch (see cuda_backend.hpp) to upload the layer weights as fp16 (G5d).
bool g_cuda_fp16_weights = false;
// Bench/diagnostic knob (G5e): force the naive one-thread-per-query attention for A/B timing.
bool g_cuda_force_naive_attn = false;
// Opt-in knob (G5f): use the shared-memory K/V tiled kernel at prefill (sq>1). Default OFF — on
// Qwen2.5-0.5B the per-layer KV fits in L2, so tiling only TIES the non-tiled online kernel (and
// slightly regresses small prefill from the smem staging); it's the correct FlashAttention structure
// that wins once K/V outgrow L2 (bigger model/batch, smaller-L2 GPU). Bit-identical either way.
bool g_cuda_use_tiled_attn = false;
// Opt-in knob (G5g): Flash-Decoding / split-KV attention at decode (see cuda_backend.hpp). Default
// OFF; even when on, split_count() gates it to the shapes where it helps (small sq + long context).
bool g_cuda_use_split_attn = false;
// Opt-in knob (G5 micro-gain): double-buffered fp32 projection GEMM (see cuda_backend.hpp). Default OFF.
bool g_cuda_use_dbuf = false;

Tensor CudaBackend::linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    const int64_t m = x.size(0), k = x.size(1), n = weight.size(0);
    Tensor y = device_alloc({m, n});
    const float* xp = dptr(x);
    const float* wp = dptr(weight);
    const float* bp = bias ? dptr(*bias) : nullptr;
    float* yp = dptr(y);
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    // Pick the kernel. fp16 weights (G5d) read half directly: GEMV-h for decode; for prefill, the
    // 128² tensor-core kernel for the huge lm_head (where TC win) and the CUDA-core float4 tiled GEMM
    // for the projections (where wmma's small-n fragment overhead loses) — both stream ½ the weight
    // bytes. Otherwise the bench knob forces the naive GEMM (the A/B baseline); else small m (decode)
    // is memory-bound → warp-GEMV (G5b), large m (prefill) is compute-bound → tiled GEMM (G5c).
    // Each output row is independent in all paths, so batching never changes a row. kGemvMaxM is
    // the shared decode/prefill split (file scope — cuda_linear_q8 routes the int8 lm_head the same).
    if (weight.dtype() == DType::F16) {
        const half* wh = static_cast<const half*>(weight.device_ptr());
        if (m <= kGemvMaxM) {
            const int threads = 128;
            const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
            linear_gemv_kernel<half><<<blocks, threads>>>(xp, wh, bp, yp, mi, ni, ki);
        } else if (n >= 8192 && n % 128 == 0 && k % 16 == 0) {
            // lm_head: huge n, compute-bound — 128² warp-tiled tensor cores (8 frags/warp). >1000
            // blocks, so cross-block occupancy hides the load-sync stall (double-buffering — the cp.async
            // lever — is in linear_tiled_db_kernel for the low-occupancy projections; a tie on this model).
            constexpr int BM = 128, BN = 128;
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_tiled_kernel<half><<<grid, 256>>>(xp, wh, bp, yp, mi, ni, ki);
        } else if (n % 128 == 0 && k % 16 == 0) {
            // Projections (narrow n): fp16 weights through the CUDA-core float4 tiled GEMM — read ½
            // the bytes (load4), convert to fp32 in-register, fp32 compute. Beats the 64² wmma-h
            // here (small n + wmma fragment overhead make tensor cores lose); the same tuned tiled
            // kernel as fp32, just halving the weight DRAM traffic. lm_head stays on wmma above.
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_tiled_vec_kernel<half, BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wh, bp, yp, mi,
                                                                                 ni, ki);
        } else {
            constexpr int BM = 64, BN = 64;  // ragged fallback: wmma bounds-checks every edge
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_kernel<half><<<grid, 128>>>(xp, wh, bp, yp, mi, ni, ki);
        }
    } else if (g_cuda_force_naive_gemm) {
        const dim3 block(16, 16);
        const dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
                        (static_cast<unsigned>(m) + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(xp, wp, bp, yp, mi, ni, ki);
    } else if (m <= kGemvMaxM) {
        const int threads = 128;  // 4 warps/block, one output channel per warp
        const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
        linear_gemv_kernel<float><<<blocks, threads>>>(xp, wp, bp, yp, mi, ni, ki);
    } else if (g_cuda_use_wmma) {
        if (n >= 8192 && n % 128 == 0 && k % 16 == 0) {  // large n: 128² warp-tiled tensor cores
            constexpr int BM = 128, BN = 128;
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_tiled_kernel<float><<<grid, 256>>>(xp, wp, bp, yp, mi, ni, ki);
        } else {
            constexpr int BM = 64, BN = 64;  // 2×2 warps = 128 threads compute a 64×64 tile
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_kernel<float><<<grid, 128>>>(xp, wp, bp, yp, mi, ni, ki);
        }
    } else if (n % 128 == 0 && k % 16 == 0) {
        // G5c+: float4-vectorized tiled GEMM (transposed-As smem, 128-bit loads/stores). Two tile
        // sizes, picked by output width n — every real Qwen linear meets the n%128, k%16 gate, and
        // only the m (row) dimension is ever ragged (bounds-checked inside the kernel):
        //   • huge n (lm_head, n=151936): a WARP-TILED 128×128 tile — the most compute-bound matmul,
        //     enough work for >1000 blocks, so warp-tiling's extra register reuse pays off (G5c+).
        //   • narrow n (the projections): a 64×64 / 4×4 tile — at m=128 a 128-wide tile would launch
        //     only n/128 blocks (n=896 → 7) and starve a 56-SM GPU; the 64-wide tile keeps ~4× more
        //     blocks resident. BK=16 so 256 threads still load one float4 of x and w apiece per pass.
        if (n >= 8192) {
            constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4;
            constexpr int NT = 128;  // 4 warps, each a 64×64 warp tile (2×2 grid over the block)
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_warptiled_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
                <<<grid, NT>>>(xp, wp, bp, yp, mi, ni, ki);
        } else {
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            // G5 micro-gain: the low-occupancy projections (down/q/o, ~28 blocks) can hide the K-tile
            // load latency with a double-buffered software pipeline; bit-identical, A/B'd via NI_DBUF.
            if (g_cuda_use_dbuf)
                linear_tiled_db_kernel<float, BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp,
                                                                                     mi, ni, ki);
            else
                linear_tiled_vec_kernel<float, BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp,
                                                                                      mi, ni, ki);
        }
    } else {
        // Fallback for shapes the vectorized kernel's divisibility preconditions don't cover
        // (off-checkpoint dims): the scalar G5c tiled kernel, which bounds-checks every edge.
        constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        constexpr int threads = (BM / TM) * (BN / TN);  // 256
        const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                        (static_cast<unsigned>(m) + BM - 1) / BM);
        linear_tiled_kernel<BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp, mi, ni, ki);
    }
    launch_check("linear_kernel");
    return y;
}

Tensor CudaBackend::embedding(const Tensor& table, const std::vector<int64_t>& ids) {
    const int64_t n = static_cast<int64_t>(ids.size()), hidden = table.size(1);
    Tensor out = device_alloc({n, hidden});  // fp32 activations, even from an fp16 table
    int64_t* d_ids = nullptr;
    cuda_check(cudaMalloc(&d_ids, n * sizeof(int64_t)), "embedding ids malloc");
    cuda_check(cudaMemcpy(d_ids, ids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice),
               "embedding ids H2D");
    const int blocks = grid1d(n * hidden, kBlock);
    if (table.dtype() == DType::F16)  // G5d: embed_tokens uploaded as half (the largest weight)
        embedding_kernel<half><<<blocks, kBlock>>>(static_cast<const half*>(table.device_ptr()),
                                                   d_ids, dptr(out), n, hidden);
    else
        embedding_kernel<float><<<blocks, kBlock>>>(dptr(table), d_ids, dptr(out), n, hidden);
    launch_check("embedding_kernel");
    cudaFree(d_ids);
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
                                                         H, seq, D, pos_offset);
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
    if (g_cuda_force_naive_attn) {
        attention_kernel<<<grid1d(H * sq, kBlock), kBlock>>>(
            dptr(q), dptr(k), dptr(v), dptr(out), H, sq, sk, D, causal ? 1 : 0, query_offset, scale);
        launch_check("attention_kernel");
    } else if (sq > 1 && g_cuda_use_tiled_attn) {
        // Opt-in: shared-memory K/V tiling (G5f-tiled). A block of 8 warps shares each K/V tile, so
        // grid is (query-tiles per head, H). Default off — see g_cuda_use_tiled_attn. Decode (sq=1)
        // has no query reuse anyway, so it always takes the non-tiled path below.
        const int threads = 256, warpsPerBlock = threads / 32;  // 8 queries/block
        const dim3 grid(static_cast<unsigned>((sq + warpsPerBlock - 1) / warpsPerBlock),
                        static_cast<unsigned>(H));
        const size_t smemBytes = 2u * kBc * static_cast<size_t>(D) * sizeof(float);
        attention_warp_tiled_kernel<kBc><<<grid, threads, smemBytes>>>(
            dptr(q), dptr(k), dptr(v), dptr(out), H, sq, sk, D, causal ? 1 : 0, query_offset, scale);
        launch_check("attention_warp_tiled_kernel");
    } else if (g_cuda_use_split_attn && split_count(H * sq, sk) >= 2) {
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
    ensure_capacity(end);  // idempotent across layers (length_ advances once per forward)

    // Upload the (small) block table for the two kernels.
    const int64_t nbt = static_cast<int64_t>(block_table_.size());
    int64_t* d_bt = nullptr;
    cuda_check(cudaMalloc(&d_bt, nbt * sizeof(int64_t)), "paged block_table malloc");
    cuda_check(cudaMemcpy(d_bt, block_table_.data(), nbt * sizeof(int64_t), cudaMemcpyHostToDevice),
               "paged block_table H2D");

    float* Kbase = pool_->k_base() + layer * pool_->layer_stride();
    float* Vbase = pool_->v_base() + layer * pool_->layer_stride();

    paged_write_kernel<<<grid1d(n_kv * t * hd, kBlock), kBlock>>>(dptr(k), dptr(v), Kbase, Vbase,
                                                                 d_bt, t, n_kv, hd, bs, start);
    launch_check("paged_write_kernel");

    const int64_t n_heads = q.size(0), sq = q.size(1), D = q.size(2);
    if (D > kMaxHeadDim)
        throw std::runtime_error("CudaPagedKVCache::attend: head_dim > 128 not supported (G4b)");
    Tensor out = device_alloc({n_heads, sq, D});
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    if (g_cuda_force_naive_attn) {
        paged_attention_kernel<<<grid1d(n_heads * sq, kBlock), kBlock>>>(
            dptr(q), Kbase, Vbase, d_bt, dptr(out), n_heads, sq, D, n_kv, n_rep, bs, causal ? 1 : 0,
            query_offset, end, scale);
        launch_check("paged_attention_kernel");
    } else if (g_cuda_use_split_attn && split_count(n_heads * sq, end) >= 2) {
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
        paged_attention_warp_kernel<<<blocks, threads>>>(dptr(q), Kbase, Vbase, d_bt, dptr(out),
                                                         n_heads, sq, D, n_kv, n_rep, bs,
                                                         causal ? 1 : 0, query_offset, end, scale);
        launch_check("paged_attention_warp_kernel");
    }

    cudaFree(d_bt);
    return out;
}

}  // namespace ni
