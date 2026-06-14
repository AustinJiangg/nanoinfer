// Tensor: a row-major, contiguous, owning float32 array (C++ stage C0).
//
// Deliberately minimal — just enough to carry activations through the ops and to
// teach the mechanics real engines lean on: an explicit shape, the row-major
// strides derived from it, and one contiguous block of memory. Storage is a
// std::vector<float> (RAII, no manual new/delete) on purpose; we only drop to
// manual / aligned allocation later, when SIMD (stage C5) actually needs the
// alignment. Views/transpose are not here yet — every Tensor owns contiguous data.
#pragma once

#include <cstdint>
#include <vector>

namespace ni {

class Tensor {
public:
    Tensor() = default;
    // Allocate a zero-filled tensor of the given shape.
    explicit Tensor(std::vector<int64_t> shape);

    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const { return numel_; }
    int64_t size(int64_t dim) const { return shape_[dim]; }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Flat (already-linearized) indexing into the contiguous block.
    float& operator[](int64_t i) { return data_[i]; }
    float operator[](int64_t i) const { return data_[i]; }

    // 2-D convenience for the [rows, cols] tensors the ops work on.
    float& at(int64_t i, int64_t j);
    float at(int64_t i, int64_t j) const;

private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    std::vector<float> data_;
    int64_t numel_ = 0;
};

}  // namespace ni
