#include "tensor.hpp"

#include <cassert>

namespace ni {

Tensor::Tensor(std::vector<int64_t> shape, Device device)
    : shape_(std::move(shape)), device_(device) {
    // numel = product of dims (an empty shape is a scalar -> 1).
    numel_ = 1;
    for (int64_t d : shape_) numel_ *= d;

    // Row-major strides: the last axis is contiguous (stride 1), each axis to its
    // left strides by the product of all axes to its right.
    strides_.resize(shape_.size());
    int64_t acc = 1;
    for (int64_t ax = ndim() - 1; ax >= 0; --ax) {
        strides_[ax] = acc;
        acc *= shape_[ax];
    }

    // CPU tensors own a zero-filled host buffer. Device tensors carry only the shape
    // metadata here; the backend attaches the device buffer via set_device_ptr()
    // (cudaMalloc on the .cu side), keeping this file free of any CUDA dependency.
    if (device_ == Device::CPU) data_.assign(static_cast<size_t>(numel_), 0.0f);
}

float& Tensor::at(int64_t i, int64_t j) {
    assert(ndim() == 2 && device_ == Device::CPU);
    return data_[i * strides_[0] + j * strides_[1]];
}

float Tensor::at(int64_t i, int64_t j) const {
    assert(ndim() == 2 && device_ == Device::CPU);
    return data_[i * strides_[0] + j * strides_[1]];
}

float& Tensor::at(int64_t i, int64_t j, int64_t k) {
    assert(ndim() == 3 && device_ == Device::CPU);
    return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

float Tensor::at(int64_t i, int64_t j, int64_t k) const {
    assert(ndim() == 3 && device_ == Device::CPU);
    return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
}

}  // namespace ni
