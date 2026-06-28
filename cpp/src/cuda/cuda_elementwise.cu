// CUDA backend: embedding + elementwise/shape kernels and their methods (R4 — split from cuda_backend.cu).
//
// The small per-element ops: the token embedding gather (fp32/fp16 table), rmsnorm, silu, mul, add,
// split/merge heads, repeat_kv (GQA), and RoPE — plus the CudaBackend methods that launch them and the
// batched-decode row plumbing (alloc/extract_row/place_row). f32_to_f16 (paired with to_device_f16) and
// attention stay in cuda_backend.cu. Reads the graph-decode globals (g_cuda_graph_pos/token) for the
// captured decode path; shared pool/loaders come from cuda_internal.cuh.
#include "cuda/cuda_backend.hpp"   // CudaBackend
#include "cuda/cuda_internal.cuh"  // device_alloc, dptr, launch_check, grid1d, ldf, kBlock, g_cuda_graph_*

#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace ni {
namespace {
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
}  // namespace

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
}  // namespace ni
