#include "tensor.hpp"

#include <cassert>

namespace ni {

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
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

    data_.assign(static_cast<size_t>(numel_), 0.0f);
}

float& Tensor::at(int64_t i, int64_t j) {
    assert(ndim() == 2);
    return data_[i * strides_[0] + j * strides_[1]];
}

float Tensor::at(int64_t i, int64_t j) const {
    assert(ndim() == 2);
    return data_[i * strides_[0] + j * strides_[1]];
}

}  // namespace ni
