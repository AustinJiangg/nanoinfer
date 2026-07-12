// CUDA backend: device-memory pool + grid/split helpers (R4 — split out of cuda_backend.cu).
//
// The ONE DevicePool instance lives here and is shared by every CUDA TU through device_alloc()
// (declared in cuda_internal.cuh): a caching allocator so steady-state forwards do no per-op
// cudaMalloc. The grid math (grid1d / sm_count / split_count) and cat_seq live here too — the
// host-side runtime support the kernels lean on. Definitions only; declarations are in the header.
#include "cuda/cuda_internal.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace ni {

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
namespace {
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
    // Return every FREE-list buffer to the driver (real cudaFree) — the one place the pool
    // shrinks. In-use buffers (held by live Tensors, so absent from free_) are untouched, so
    // this changes no computation; it only reclaims retained VRAM. Off the hot path: it exists
    // so a test can free one Model's weights before building the next format (run_cuda_parity
    // builds five), since otherwise the leaked buffers sum across formats and OOM a big model.
    void trim() {
        for (auto& [nbytes, bufs] : free_) {
            for (void* p : bufs) {
                cudaFree(p);
                size_of_.erase(p);
            }
        }
        free_.clear();
    }
};
DevicePool& pool() {
    static DevicePool* p = new DevicePool();  // leaked on purpose (see above)
    return *p;
}
}  // namespace

// Reclaim pooled-but-free device memory to the driver (real cudaFree). Off any hot path — it
// lets a test free one Model's buffers before building the next so the caching pool doesn't
// accumulate every weight format's weights and OOM a large model. Correctness-neutral.
void device_pool_trim() { pool().trim(); }

// Allocate a device tensor of `shape` (+ dtype) from the pool and hand the Tensor a deleter
// that returns the buffer to the pool (not cudaFree) — so steady-state forwards do no
// cudaMalloc. F16/BF16 buffers (half-storage weights, G5d/B1) are half the bytes; the pool
// keys on bytes.
Tensor device_alloc(const std::vector<int64_t>& shape, DType dt) {
    Tensor d(shape, Device::CUDA);
    d.set_dtype(dt);
    const size_t elem = (dt == DType::F16 || dt == DType::BF16) ? 2
                        : (dt == DType::I8)                     ? sizeof(int8_t)
                                                                : sizeof(float);
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

}  // namespace ni
