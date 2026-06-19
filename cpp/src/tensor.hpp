// Tensor: a row-major, contiguous, owning float32 array (C++ stage C0; device-aware G1).
//
// Deliberately minimal — just enough to carry activations through the ops and to
// teach the mechanics real engines lean on: an explicit shape, the row-major
// strides derived from it, and one contiguous block of memory. CPU storage is a
// std::vector<float> (RAII, no manual new/delete). For accelerators (G1+) the data
// instead lives in a device buffer (e.g. cudaMalloc'd), held type-erased behind a
// shared_ptr<void> so THIS header needs no CUDA include — the CPU-only build never
// pulls in nvcc. A Device tag says where the data lives; the model forward is written
// once and a Backend dispatches on the device (the ggml-style approach).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace ni {

// Where a Tensor's data lives. CPU is the default; CUDA arrives in G1, METAL with the
// cross-platform backend. A Backend operates on tensors resident on its own device.
enum class Device { CPU, CUDA, METAL };

// Element type of a tensor's data. F32 is the default and the only host type; F16 exists
// only on device (a half buffer) — fp16 weights for the tensor-core path (G5d). The host
// vector is always float; dtype() tells a backend how to read the device buffer.
enum class DType { F32, F16 };

class Tensor {
public:
    Tensor() = default;
    // Host (CPU), zero-filled — the common case, unchanged since C0.
    explicit Tensor(std::vector<int64_t> shape) : Tensor(std::move(shape), Device::CPU) {}
    // Shape on a given device. CPU allocates a zero-filled host vector; a non-CPU
    // device sets up shape/strides only and leaves the buffer empty — the backend
    // allocates device memory and attaches it with set_device_ptr().
    Tensor(std::vector<int64_t> shape, Device device);

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    int64_t size(int64_t dim) const { return shape_[dim]; }
    Device device() const { return device_; }
    DType dtype() const { return dtype_; }
    void set_dtype(DType d) { dtype_ = d; }

    // Host storage. Valid for CPU tensors only (a device tensor's vector is empty).
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Raw device buffer (e.g. a cudaMalloc'd float*); null for host tensors. The
    // backend casts this to the concrete pointer type on the device side, so the
    // header itself stays CUDA-agnostic. set_device_ptr takes ownership via a
    // shared_ptr whose deleter frees the device memory (RAII over cudaFree).
    void* device_ptr() const { return ddata_.get(); }
    void set_device_ptr(std::shared_ptr<void> p) { ddata_ = std::move(p); }

    // Flat (already-linearized) indexing into the contiguous block. Host-only.
    float& operator[](int64_t i) { return data_[i]; }
    float operator[](int64_t i) const { return data_[i]; }

    // 2-D / 3-D convenience for the [rows, cols] and [heads, seq, head_dim]
    // tensors the ops work on. (Attention is batch-1 here, so 3-D is enough.) Host-only.
    float& at(int64_t i, int64_t j);
    float at(int64_t i, int64_t j) const;
    float& at(int64_t i, int64_t j, int64_t k);
    float at(int64_t i, int64_t j, int64_t k) const;

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;       // host storage (CPU tensors)
    int64_t numel_ = 0;
    Device device_ = Device::CPU;
    DType dtype_ = DType::F32;       // element type; F16 only for device weight buffers (G5d)
    std::shared_ptr<void> ddata_;   // device storage (CUDA/Metal); RAII-freed, type-erased
};

}  // namespace ni
