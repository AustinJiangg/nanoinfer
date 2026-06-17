// CUDA backend kernels + device memory plumbing (C++ stage G1).
//
// This is the only file compiled by nvcc. It holds: the device<->host transfer helpers
// (cuda.hpp), the CudaBackend methods (cuda_backend.hpp), and the kernels they launch.
// Everything the rest of the engine sees is plain C++ (Tensor in, Tensor out) — the
// CUDA surface is contained here.
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "tensor.hpp"

namespace ni {

namespace {

// Turn a non-success CUDA status into a C++ exception (with the call site + message),
// so GPU failures surface the same way as the rest of the engine's errors.
void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(e));
}

// Allocate a device tensor of `shape`: cudaMalloc the bytes and hand ownership to the
// Tensor as a shared_ptr<void> whose deleter is cudaFree — so the GPU buffer is freed
// automatically when the last Tensor referencing it dies (RAII over a C API).
Tensor device_alloc(const std::vector<int64_t>& shape) {
    Tensor d(shape, Device::CUDA);
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, static_cast<size_t>(d.numel()) * sizeof(float)), "cudaMalloc");
    d.set_device_ptr(std::shared_ptr<void>(p, [](void* q) { cudaFree(q); }));
    return d;
}

// Naive GEMM: y[i,o] = sum_j x[i,j] * w[o,j] + (bias ? bias[o] : 0). One thread per
// output element (i,o); each thread walks the length-k dot product itself. This mirrors
// ops.cpp's linear (nn.Linear's y = x @ wᵀ + bias — w stores one row per output
// feature, so no transpose). The accumulator is float, summed front-to-back: that
// differs from the CPU's double-accumulated, two-lane SIMD dot, which is exactly why
// the GPU result drifts from the CPU oracle by ~1e-4 (see CLAUDE.md "GPU parity").
__global__ void linear_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ bias, float* __restrict__ y,
                              int m, int n, int k) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;  // output feature (column of y)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of x / y
    if (i >= m || o >= n) return;                          // edge threads past the matrix
    const float* xr = x + static_cast<size_t>(i) * k;     // row i of x
    const float* wr = w + static_cast<size_t>(o) * k;     // row o of w (output feature o)
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += xr[j] * wr[j];
    if (bias) acc += bias[o];
    y[static_cast<size_t>(i) * n + o] = acc;
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

Tensor CudaBackend::linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    // x [m,k], weight [n,k] (one row per output feature), bias [n] or null -> y [m,n].
    const int64_t m = x.size(0), k = x.size(1), n = weight.size(0);
    Tensor y = device_alloc({m, n});

    // 2-D launch: a 16x16 block of threads, tiled over the (n columns, m rows) output.
    // The grid rounds up, so it covers matrices whose sizes aren't multiples of 16; the
    // overhang threads early-out on the bounds check at the top of the kernel.
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(n) + block.x - 1) / block.x,
                    (static_cast<unsigned>(m) + block.y - 1) / block.y);
    linear_kernel<<<grid, block>>>(
        static_cast<const float*>(x.device_ptr()),
        static_cast<const float*>(weight.device_ptr()),
        bias ? static_cast<const float*>(bias->device_ptr()) : nullptr,
        static_cast<float*>(y.device_ptr()),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
    cuda_check(cudaGetLastError(), "linear_kernel launch");   // catches bad launch config
    cuda_check(cudaDeviceSynchronize(), "linear_kernel sync");  // and any in-kernel fault
    return y;
}

// --- G2 fills these in; until then they fail loudly so a half-ported forward can't
// silently run on a partially-implemented backend. ---
namespace {
[[noreturn]] Tensor not_impl(const char* op) {
    throw std::runtime_error(std::string("CudaBackend::") + op + " not implemented yet (G2)");
}
}  // namespace

Tensor CudaBackend::embedding(const Tensor&, const std::vector<int64_t>&) { return not_impl("embedding"); }
Tensor CudaBackend::rmsnorm(const Tensor&, const Tensor&, float) { return not_impl("rmsnorm"); }
Tensor CudaBackend::silu(const Tensor&) { return not_impl("silu"); }
Tensor CudaBackend::mul(const Tensor&, const Tensor&) { return not_impl("mul"); }
Tensor CudaBackend::add(const Tensor&, const Tensor&) { return not_impl("add"); }
Tensor CudaBackend::split_heads(const Tensor&, int64_t, int64_t) { return not_impl("split_heads"); }
Tensor CudaBackend::merge_heads(const Tensor&) { return not_impl("merge_heads"); }
Tensor CudaBackend::repeat_kv(const Tensor&, int64_t) { return not_impl("repeat_kv"); }
Tensor CudaBackend::apply_rope(const Tensor&, const Tensor&, const Tensor&, int64_t) { return not_impl("apply_rope"); }
Tensor CudaBackend::attention(const Tensor&, const Tensor&, const Tensor&, bool, int64_t) { return not_impl("attention"); }

}  // namespace ni
