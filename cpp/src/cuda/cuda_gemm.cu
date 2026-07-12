// CUDA backend: the dense GEMM kernels + CudaBackend::linear (R4 — split out of cuda_backend.cu).
//
// All eight fp32/fp16/bf16 matmul kernels (naive, warp-GEMV, scalar/vectorized/double-buffered
// tiled, warp-tiled, and the two wmma tensor-core variants) and the m/dtype dispatch that picks
// among them. Launched only by CudaBackend::linear, so they live together in one TU (whole-program
// compilation: a kernel must be launched in the TU it's defined in). Shared pool/loaders come from
// cuda_internal.cuh.
#include "cuda/cuda_backend.hpp"   // CudaBackend, cuda_policy()
#include "cuda/cuda_internal.cuh"  // device_alloc, dptr, launch_check, ldf/ldh/load4, kGemvMaxM, kBlock

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>
#include <vector>

#include "tensor.hpp"

namespace ni {
namespace {
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
// valid output. WT (B1): fp32 or a half dtype read through ldf — the weight converts to fp32
// at staging, so the tile math (and its error) is unchanged; only the global read narrows.
template <typename WT, int BM, int BN, int BK, int TM, int TN>
__global__ void linear_tiled_kernel(const float* __restrict__ x, const WT* __restrict__ w,
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
            Bs[innerColB * BN + o] =
                (go < n && gk < k) ? ldf(w, static_cast<size_t>(go) * k + gk) : 0.0f;
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

// Prefill on tensor cores (G5d; B2 templates the staging dtype): the same tiled GEMM, but each
// warp's 16×16×16 multiply runs on the Ada tensor cores via the wmma API. Each BM×BK / BK×BN tile
// is converted to ST (half — G5d — or __nv_bfloat16 — B2, sm_80+) as it is staged into shared
// memory, and the matmul accumulates in fp32 (wmma ST×ST→float). Mixed precision: half inputs,
// fp32 accumulate — far faster compute than the fp32 CUDA-core kernel, but lossy on the staged
// ACTIVATIONS (fp16 rounds to ~3 decimal digits, bf16 to ~2; a weight already stored as ST
// restages exactly). Opt-in via cuda_policy().use_wmma for fp32 weights; the accuracy cost is
// measured (test_cuda / run_cuda_bench). Block = 2×2 warps over a 64×64 tile (each warp a 2×2
// grid of 16×16 frags); n is a multiple of 64 for every Qwen linear, m is bounds-checked at the
// store.
template <typename ST, typename WT>
__global__ void linear_wmma_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                   const float* __restrict__ bias, float* __restrict__ y, int m,
                                   int n, int k) {
    using namespace nvcuda;
    constexpr int BM = 64, BN = 64, BK = 16;
    constexpr int WM = 16, WN = 16, WK = 16;  // wmma fragment shape
    constexpr int numThreads = 128;           // 2×2 warps
    __shared__ ST As[BM * BK];                // x tile [BM][BK], row-major
    __shared__ ST Bs[BK * BN];                // w tile [BK][BN] = [k][o], row-major
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
            As[r * BK + c] = (gr < m && gc < k) ? from_f32<ST>(x[gr * k + gc]) : from_f32<ST>(0.0f);
        }
        for (int idx = tid; idx < BK * BN; idx += numThreads) {  // stage w (k fastest: coalesced)
            const int o = idx / BK, c = idx % BK, go = colBase + o, gc = kk + c;
            Bs[c * BN + o] = (go < n && gc < k) ? from_f32<ST>(ldf(w, static_cast<size_t>(go) * k + gc))
                                                : from_f32<ST>(0.0f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WM, WN, WK, ST, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, ST, wmma::row_major> bFrag;
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
// linear_tiled_db_kernel, a tie on this model). x (activations) is always fp32, staged to ST (half
// or __nv_bfloat16 — B2); w is fp32/fp16/bf16 (WT) read via ldf and restaged to ST. ST inputs, fp32
// accumulate — the accumulator has been fp32 since G5d (the B2 record-correction: it never was
// fp16), so the accuracy cost is the ST rounding of the staged operands. ACC is a BENCH-ONLY knob
// (cuda_policy().wmma_fp16_acc → half): it answers "what would fp16 accumulate buy on GeForce?" by
// measurement — never dispatched in the model path, and bf16 inputs mandate float (hardware rule),
// enforced below. n%128==0 and k%16==0 (the dispatch guarantees it); only m is ragged,
// bounds-checked at the store.
template <typename ST, typename WT, typename ACC = float>
__global__ void linear_wmma_tiled_kernel(const float* __restrict__ x, const WT* __restrict__ w,
                                         const float* __restrict__ bias, float* __restrict__ y,
                                         int m, int n, int k) {
    using namespace nvcuda;
    static_assert(!(std::is_same<ST, __nv_bfloat16>::value && std::is_same<ACC, half>::value),
                  "bf16 wmma mandates an fp32 accumulator");
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WARPS_M = 2, WARPS_N = 4, NUM_WARPS = WARPS_M * WARPS_N;  // 8 warps = 256 threads
    constexpr int WMROWS = BM / WARPS_M, WNCOLS = BN / WARPS_N;             // 64×32 per warp
    constexpr int MFRAG = WMROWS / 16, NFRAG = WNCOLS / 16;                 // 4×2 frags per warp
    __shared__ ST As[BM * BK];              // x tile [BM][BK], row-major (M-major)
    __shared__ ST Bs[BK * BN];              // w tile [BK][BN], row-major (K-major)
    __shared__ ACC Cs[NUM_WARPS][16 * 16];  // per-warp 16×16 epilogue staging (bias + bounds)

    const int tid = static_cast<int>(threadIdx.x);
    const int warpId = tid / 32, lane = tid % 32;
    const int warpRow = warpId / WARPS_N, warpCol = warpId % WARPS_N;  // 2×4 warp grid
    const int rowBase = static_cast<int>(blockIdx.y) * BM;
    const int colBase = static_cast<int>(blockIdx.x) * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, ACC> acc[MFRAG][NFRAG];
    for (int i = 0; i < MFRAG; ++i)
        for (int j = 0; j < NFRAG; ++j) wmma::fill_fragment(acc[i][j], from_f32<ACC>(0.0f));

    for (int kk = 0; kk < k; kk += BK) {
        for (int idx = tid; idx < BM * BK; idx += NUM_WARPS * 32) {  // stage x (k fastest: coalesced)
            const int r = idx / BK, c = idx % BK, gr = rowBase + r;
            As[r * BK + c] = (gr < m) ? from_f32<ST>(x[static_cast<size_t>(gr) * k + kk + c])
                                      : from_f32<ST>(0.0f);
        }
        for (int idx = tid; idx < BK * BN; idx += NUM_WARPS * 32) {  // stage w (k fastest: coalesced)
            const int o = idx / BK, c = idx % BK, go = colBase + o;
            Bs[c * BN + o] = (go < n) ? from_f32<ST>(ldf(w, static_cast<size_t>(go) * k + kk + c))
                                      : from_f32<ST>(0.0f);
        }
        __syncthreads();
        // Reload this warp's A/B fragments once, then 8 MMAs reuse them (the warp-tile leverage).
        wmma::fragment<wmma::matrix_a, 16, 16, 16, ST, wmma::row_major> aFrag[MFRAG];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, ST, wmma::row_major> bFrag[NFRAG];
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
                    y[static_cast<size_t>(gr) * n + gc] =
                        to_f32(Cs[warpId][e]) + (bias ? bias[gc] : 0.0f);
            }
            __syncwarp();
        }
}

}  // namespace

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
            if (cuda_policy().wmma_fp16_acc)  // BENCH-ONLY (B2): measure what fp16 accumulate buys
                linear_wmma_tiled_kernel<half, half, half><<<grid, 256>>>(xp, wh, bp, yp, mi, ni, ki);
            else
                linear_wmma_tiled_kernel<half, half><<<grid, 256>>>(xp, wh, bp, yp, mi, ni, ki);
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
            linear_wmma_kernel<half, half><<<grid, 128>>>(xp, wh, bp, yp, mi, ni, ki);
        }
    } else if (weight.dtype() == DType::BF16) {
        // bf16 weights (B1): the same ½-byte storage win as fp16, read through the SAME dtype-
        // templated kernels — GEMV for decode, the CUDA-core float4 tiled GEMM for aligned prefill
        // (weights convert to fp32 in-register; the compute and its error model are fp32, so the
        // only cost vs fp32 weights is the bf16 rounding of the weight itself — which is ZERO for
        // a bf16-shipped checkpoint). B2 adds the lm_head tensor-core path below (bf16 inputs,
        // fp32 accumulate — the hardware mandate).
        const __nv_bfloat16* wb = static_cast<const __nv_bfloat16*>(weight.device_ptr());
        if (m <= kGemvMaxM) {
            const int threads = 128;
            const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
            linear_gemv_kernel<__nv_bfloat16><<<blocks, threads>>>(xp, wb, bp, yp, mi, ni, ki);
        } else if (n >= 8192 && n % 128 == 0 && k % 16 == 0) {
            // lm_head (B2): the 128² warp-tiled tensor cores reading bf16 weight storage — the
            // exact mirror of the fp16 wmma-h path. The weights restage exactly (already bf16);
            // only the ACTIVATIONS newly round to bf16's 8-bit mantissa, so the error sits ~8×
            // wmma-h's. Feeds argmax — the golden gate (run_cuda_parity bf16 section) decides.
            constexpr int BM = 128, BN = 128;
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_tiled_kernel<__nv_bfloat16, __nv_bfloat16>
                <<<grid, 256>>>(xp, wb, bp, yp, mi, ni, ki);
        } else if (n % 128 == 0 && k % 16 == 0) {
            constexpr int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_tiled_vec_kernel<__nv_bfloat16, BM, BN, BK, TM, TN>
                <<<grid, threads>>>(xp, wb, bp, yp, mi, ni, ki);
        } else {
            // Ragged fallback: the scalar tiled kernel (bounds-checks every edge), templated on
            // the weight dtype — fp32 compute, so it is TIGHTER than fp16's wmma-h fallback.
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            constexpr int threads = (BM / TM) * (BN / TN);  // 256
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_tiled_kernel<__nv_bfloat16, BM, BN, BK, TM, TN>
                <<<grid, threads>>>(xp, wb, bp, yp, mi, ni, ki);
        }
    } else if (cuda_policy().force_naive_gemm) {
        const dim3 block(16, 16);
        const dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
                        (static_cast<unsigned>(m) + block.y - 1) / block.y);
        linear_kernel<<<grid, block>>>(xp, wp, bp, yp, mi, ni, ki);
    } else if (m <= kGemvMaxM) {
        const int threads = 128;  // 4 warps/block, one output channel per warp
        const int blocks = static_cast<int>((n + threads / 32 - 1) / (threads / 32));
        linear_gemv_kernel<float><<<blocks, threads>>>(xp, wp, bp, yp, mi, ni, ki);
    } else if (cuda_policy().use_wmma) {
        if (n >= 8192 && n % 128 == 0 && k % 16 == 0) {  // large n: 128² warp-tiled tensor cores
            constexpr int BM = 128, BN = 128;
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            if (cuda_policy().wmma_fp16_acc)  // BENCH-ONLY (B2): the fp16-accumulate datapoint
                linear_wmma_tiled_kernel<half, float, half><<<grid, 256>>>(xp, wp, bp, yp, mi, ni, ki);
            else
                linear_wmma_tiled_kernel<half, float><<<grid, 256>>>(xp, wp, bp, yp, mi, ni, ki);
        } else {
            constexpr int BM = 64, BN = 64;  // 2×2 warps = 128 threads compute a 64×64 tile
            const dim3 grid((static_cast<unsigned>(n) + BN - 1) / BN,
                            (static_cast<unsigned>(m) + BM - 1) / BM);
            linear_wmma_kernel<half, float><<<grid, 128>>>(xp, wp, bp, yp, mi, ni, ki);
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
            if (cuda_policy().use_dbuf)
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
        linear_tiled_kernel<float, BM, BN, BK, TM, TN><<<grid, threads>>>(xp, wp, bp, yp, mi, ni, ki);
    }
    launch_check("linear_kernel");
    return y;
}

}  // namespace ni
