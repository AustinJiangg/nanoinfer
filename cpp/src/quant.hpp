// Weight-only quantization (C++ stage C4).
//
// Symmetric, per-output-channel int8 (Q8): a weight matrix W [out, in] is stored
// as int8 codes plus one fp32 scale per output row. Activations stay fp32; the
// weight is dequantized inside the inner product:
//
//   scale_o = max_j|W[o,j]| / 127      (largest magnitude -> +-127, zero -> 0)
//   q[o,j]  = round(W[o,j] / scale_o)   in [-127, 127]
//   W[o,j] ~= q[o,j] * scale_o          (error <= scale_o / 2 per weight)
//
// linear factors the scale out of the sum: y[i,o] = scale_o * sum_j x[i,j]*q[o,j].
// Storing int8 instead of fp32 is a 4x shrink of the linear weights (the bulk of
// the model).
#pragma once

#include <cstdint>
#include <vector>

#include "tensor.hpp"

namespace ni {

// An int8-quantized [out, in] weight: codes row-major, one scale per output row.
struct QTensor {
    std::vector<int8_t> q;     // [out * in]
    std::vector<float> scale;  // [out]
    int64_t out = 0;
    int64_t in = 0;
};

// Quantize a 2-D [out, in] weight to symmetric per-channel int8.
QTensor quantize_q8(const Tensor& w);

// Reconstruct the approximate fp32 weight (q[o,j] * scale_o).
Tensor dequantize_q8(const QTensor& w);

// y = x @ dequant(w)^T + bias. x is [m, in]; returns [m, out]. The inner product
// runs over the int8 codes; the per-row scale is applied once at the end.
Tensor linear_q8(const Tensor& x, const QTensor& w, const Tensor* bias = nullptr);

}  // namespace ni
