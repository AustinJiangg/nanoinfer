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
    const size_t elem = (dt == DType::F16) ? sizeof(half) : sizeof(float);
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

// Weight loaders that fold the dtype into the read, so one templated kernel serves both fp32
// and fp16 weights (G5d). ldf returns float (for the fp32-accumulate GEMV/tiled paths); ldh
// returns half (for the wmma staging). On a float* both are the identity / the same round the
// fp32 kernels already did, so the fp32 instantiations stay bit-identical to before.
__device__ inline float ldf(const float* w, size_t i) { return w[i]; }
__device__ inline float ldf(const half* w, size_t i) { return __half2float(w[i]); }
__device__ inline half ldh(const float* w, size_t i) { return __float2half(w[i]); }
__device__ inline half ldh(const half* w, size_t i) { return w[i]; }

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
template <int BM, int BN, int BK, int TM, int TN>
__global__ void linear_tiled_vec_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ bias, float* __restrict__ y, int m,
                                        int n, int k) {
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
            float4 t = *reinterpret_cast<const float4*>(&w[static_cast<size_t>(go) * k + gk]);
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

// out[r,c] = table[ids[r], c]. One thread per output element; ids already on device.
__global__ void embedding_kernel(const float* __restrict__ table, const int64_t* __restrict__ ids,
                                 float* __restrict__ out, int64_t n, int64_t hidden) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n * hidden) return;
    const int64_t r = idx / hidden, c = idx % hidden;
    out[idx] = table[ids[r] * hidden + c];
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
// Opt-in switch (see cuda_backend.hpp) to run the prefill GEMM on the tensor cores (G5d).
bool g_cuda_use_wmma = false;
// Opt-in switch (see cuda_backend.hpp) to upload the layer weights as fp16 (G5d).
bool g_cuda_fp16_weights = false;

Tensor CudaBackend::linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    const int64_t m = x.size(0), k = x.size(1), n = weight.size(0);
    Tensor y = device_alloc({m, n});
    const float* xp = dptr(x);
    const float* wp = dptr(weight);
    const float* bp = bias ? dptr(*bias) : nullptr;
    float* yp = dptr(y);
    const int mi = static_cast<int>(m), ni = static_cast<int>(n), ki = static_cast<int>(k);
    // Pick the kernel. fp16 weights (G5d) read half directly: GEMV for decode, tensor cores
    // for prefill (wmma then streams ½ the weight bytes with no convert — the path that wins).
    // Otherwise the bench knob forces the naive GEMM (the A/B baseline); else small m (decode)
    // is memory-bound → warp-GEMV (G5b), large m (prefill) is compute-bound → tiled GEMM
    // (G5c). Each output row is independent in all paths, so batching never changes a row.
    constexpr int64_t kGemvMaxM = 16;
    if (weight.dtype() == DType::F16) {
        const half* wh = static_cast<const half*>(weight.device_ptr());
        if (m <= kGemvMaxM) {
            const int threads = 128;
            const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
            linear_gemv_kernel<half><<<blocks, threads>>>(xp, wh, bp, yp, mi, ni, ki);
        } else {
            constexpr int BM = 64, BN = 64;
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
        constexpr int BM = 64, BN = 64;  // 2×2 warps = 128 threads compute a 64×64 tile
        const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                        (static_cast<unsigned>(m) + BM - 1) / BM);
        linear_wmma_kernel<float><<<grid, 128>>>(xp, wp, bp, yp, mi, ni, ki);
    } else if (n % 128 == 0 && k % 16 == 0) {
        // G5c+: float4-vectorized tiled GEMM (transposed-As smem, 128-bit loads/stores). Two tile
        // sizes, picked by output width n — every real Qwen linear meets the n%128, k%16 gate, and
        // only the m (row) dimension is ever ragged (bounds-checked inside the kernel):
        //   • huge n (lm_head, n=151936): a 128×128 / 8×8 tile — maximum per-thread reuse, and it
        //     still launches >1000 blocks, so occupancy is fine.
        //   • narrow n (the projections): a 64×64 / 4×4 tile — at m=128 a 128-wide tile would launch
        //     only n/128 blocks (n=896 → 7) and starve a 56-SM GPU; the 64-wide tile keeps ~4× more
        //     blocks resident. BK=16 so 256 threads still load one float4 of x and w apiece per pass.
        if (n >= 8192) {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_tiled_vec_kernel<BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp, mi, ni, ki);
        } else {
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_tiled_vec_kernel<BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp, mi, ni, ki);
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
    Tensor out = device_alloc({n, hidden});
    int64_t* d_ids = nullptr;
    cuda_check(cudaMalloc(&d_ids, n * sizeof(int64_t)), "embedding ids malloc");
    cuda_check(cudaMemcpy(d_ids, ids.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice),
               "embedding ids H2D");
    embedding_kernel<<<grid1d(n * hidden, kBlock), kBlock>>>(dptr(table), d_ids, dptr(out), n, hidden);
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
    attention_kernel<<<grid1d(H * sq, kBlock), kBlock>>>(dptr(q), dptr(k), dptr(v), dptr(out), H, sq,
                                                         sk, D, causal ? 1 : 0, query_offset, scale);
    launch_check("attention_kernel");
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
    paged_attention_kernel<<<grid1d(n_heads * sq, kBlock), kBlock>>>(
        dptr(q), Kbase, Vbase, d_bt, dptr(out), n_heads, sq, D, n_kv, n_rep, bs, causal ? 1 : 0,
        query_offset, end, scale);
    launch_check("paged_attention_kernel");

    cudaFree(d_bt);
    return out;
}

}  // namespace ni
