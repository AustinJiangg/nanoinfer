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
// the model). This is WEIGHT-ONLY quant: it saves memory, not compute — the inner
// product still runs in floating point (each int8 code is promoted), so it is no
// faster than fp32 linear. A true int8 x int8 -> int32 GEMM with one final scale
// would be the compute win; that's deferred.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor.hpp"

namespace ni {

// Weight quantization mode for the model. None = fp32; Q8/Q4 are per-channel;
// Q4G is group-wise int4 (one scale per K-element block — accurate enough to use).
enum class QuantMode { None, Q8, Q4, Q4G };

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

// An int4-quantized [out, in] weight (Q4): codes in [-7, 7] packed two per byte
// (each stored as code+8, a nibble in [1, 15]), per row, plus one fp32 scale per
// output row. ~8x smaller than fp32; ~18x coarser per weight than Q8. Like Q8
// this is weight-only — it saves memory, not compute (still an fp inner product).
//
// Q4 is exactly Q4G (below) with group == in. It's kept as a separate, simpler
// path on purpose: as a baseline that *demonstrates* per-channel int4 breaking on
// real weights (one row scale can't absorb an outlier) — the motivation for Q4G.
struct Q4Tensor {
    std::vector<uint8_t> q;    // [out * ceil(in/2)] packed nibbles
    std::vector<float> scale;  // [out]
    int64_t out = 0;
    int64_t in = 0;
};

Q4Tensor quantize_q4(const Tensor& w);
Tensor dequantize_q4(const Q4Tensor& w);
Tensor linear_q4(const Tensor& x, const Q4Tensor& w, const Tensor* bias = nullptr);

// Group-wise int4 (Q4_0-style): codes packed two per byte like Q4, but with one
// fp32 scale per `group` consecutive weights instead of per row. A local scale
// can't be blown out by a distant outlier, so the model stays usable at ~6.4x
// smaller (5 bits/weight). Still weight-only (memory, not compute).
struct Q4GTensor {
    std::vector<uint8_t> q;    // [out * ceil(in/2)] packed nibbles (same as Q4)
    std::vector<float> scale;  // [out * ceil(in/group)] — one per block
    int64_t out = 0;
    int64_t in = 0;
    int64_t group = 0;         // block size K
};

// group defaults to 32 — llama.cpp's Q4_0 block size (the memory/accuracy knob).
Q4GTensor quantize_q4g(const Tensor& w, int64_t group = 32);
Tensor dequantize_q4g(const Q4GTensor& w);
Tensor linear_q4g(const Tensor& x, const Q4GTensor& w, const Tensor* bias = nullptr);

// Polymorphic wrapper so the model can hold any quant mode in one map and add a
// new mode as one subclass + one factory branch (no N-map / N-way-if growth).
class QuantizedWeight {
public:
    virtual ~QuantizedWeight() = default;
    virtual Tensor linear(const Tensor& x, const Tensor* bias) const = 0;
    virtual int64_t bytes() const = 0;       // actual storage
    virtual int64_t fp32_bytes() const = 0;  // out*in*4 (the unquantized size)
};

// Quantize `w` into a QuantizedWeight of the given mode (nullptr for None).
std::unique_ptr<QuantizedWeight> make_quantized(const Tensor& w, QuantMode mode);

}  // namespace ni
