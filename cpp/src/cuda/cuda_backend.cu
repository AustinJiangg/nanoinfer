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

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

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

// Allocate a device tensor of `shape`: cudaMalloc the bytes and hand ownership to the
// Tensor as a shared_ptr whose deleter is cudaFree (RAII over the C allocator).
Tensor device_alloc(const std::vector<int64_t>& shape) {
    Tensor d(shape, Device::CUDA);
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, static_cast<size_t>(d.numel()) * sizeof(float)), "cudaMalloc");
    d.set_device_ptr(std::shared_ptr<void>(p, [](void* q) { cudaFree(q); }));
    return d;
}

// Round a flat element count up to a whole number of `block`-sized 1-D grid blocks.
int grid1d(int64_t total, int block) { return static_cast<int>((total + block - 1) / block); }
constexpr int kBlock = 256;

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

Device CudaBackend::device() const { return Device::CUDA; }

// Launch a kernel, then check both the launch (bad config) and the run (in-kernel fault).
static void sync_check(const char* what) {
    cuda_check(cudaGetLastError(), what);
    cuda_check(cudaDeviceSynchronize(), what);
}

Tensor CudaBackend::linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    const int64_t m = x.size(0), k = x.size(1), n = weight.size(0);
    Tensor y = device_alloc({m, n});
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
                    (static_cast<unsigned>(m) + block.y - 1) / block.y);
    linear_kernel<<<grid, block>>>(dptr(x), dptr(weight), bias ? dptr(*bias) : nullptr, dptr(y),
                                   static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    sync_check("linear_kernel");
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
    sync_check("embedding_kernel");
    cudaFree(d_ids);
    return out;
}

Tensor CudaBackend::rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    const int64_t d = x.size(x.ndim() - 1), rows = x.numel() / d;
    Tensor out = device_alloc(x.shape());
    rmsnorm_kernel<<<grid1d(rows, kBlock), kBlock>>>(dptr(x), dptr(weight), dptr(out), rows, d, eps);
    sync_check("rmsnorm_kernel");
    return out;
}

Tensor CudaBackend::silu(const Tensor& x) {
    Tensor out = device_alloc(x.shape());
    silu_kernel<<<grid1d(x.numel(), kBlock), kBlock>>>(dptr(x), dptr(out), x.numel());
    sync_check("silu_kernel");
    return out;
}

Tensor CudaBackend::mul(const Tensor& a, const Tensor& b) {
    Tensor out = device_alloc(a.shape());
    mul_kernel<<<grid1d(a.numel(), kBlock), kBlock>>>(dptr(a), dptr(b), dptr(out), a.numel());
    sync_check("mul_kernel");
    return out;
}

Tensor CudaBackend::add(const Tensor& a, const Tensor& b) {
    Tensor out = device_alloc(a.shape());
    add_kernel<<<grid1d(a.numel(), kBlock), kBlock>>>(dptr(a), dptr(b), dptr(out), a.numel());
    sync_check("add_kernel");
    return out;
}

Tensor CudaBackend::split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) {
    const int64_t seq = x.size(0);
    Tensor out = device_alloc({n_heads, seq, head_dim});
    split_heads_kernel<<<grid1d(n_heads * seq * head_dim, kBlock), kBlock>>>(
        dptr(x), dptr(out), seq, n_heads, head_dim);
    sync_check("split_heads_kernel");
    return out;
}

Tensor CudaBackend::merge_heads(const Tensor& x) {
    const int64_t H = x.size(0), seq = x.size(1), D = x.size(2);
    Tensor out = device_alloc({seq, H * D});
    merge_heads_kernel<<<grid1d(H * seq * D, kBlock), kBlock>>>(dptr(x), dptr(out), H, seq, D);
    sync_check("merge_heads_kernel");
    return out;
}

Tensor CudaBackend::repeat_kv(const Tensor& x, int64_t n_rep) {
    const int64_t kv = x.size(0), seq = x.size(1), D = x.size(2), out_heads = kv * n_rep;
    Tensor out = device_alloc({out_heads, seq, D});
    repeat_kv_kernel<<<grid1d(out_heads * seq * D, kBlock), kBlock>>>(
        dptr(x), dptr(out), n_rep, seq, D, out_heads);
    sync_check("repeat_kv_kernel");
    return out;
}

Tensor CudaBackend::apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                               int64_t pos_offset) {
    const int64_t H = x.size(0), seq = x.size(1), D = x.size(2);
    Tensor out = device_alloc({H, seq, D});
    rope_kernel<<<grid1d(H * seq * D, kBlock), kBlock>>>(dptr(x), dptr(cos), dptr(sin), dptr(out),
                                                         H, seq, D, pos_offset);
    sync_check("rope_kernel");
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
    sync_check("attention_kernel");
    return out;
}

}  // namespace ni
