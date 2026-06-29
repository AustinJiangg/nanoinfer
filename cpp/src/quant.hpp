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

// Weight quantization mode for the model. None = fp32; Q8/Q4 are per-channel weight-only
// (int8/int4 weight × fp32 activation — saves memory, not compute); Q4G is group-wise int4
// (one scale per K-element block — accurate enough to use). W8A8 quantizes the ACTIVATIONS to
// int8 too (per row, dynamically), so the inner product is an int8×int8→int32 integer dot — the
// form a GPU DP4A / CPU VNNI runs at int8 throughput: the compute win, not just memory.
enum class QuantMode { None, Q8, Q4, Q4G, W8A8 };

// R5: the single weight-REPRESENTATION tag, shared by every backend (CPU/CUDA, and Metal ahead).
// Every Weight reports format() so storage diagnostics and a backend's weight dispatch reason about
// the representation uniformly — the symmetric counterpart to the Backend's op dispatch (R3's lesson:
// the weight format is what varies most across devices). QuantMode is the construction-time *selector*
// (what to build); Format is the constructed weight's runtime *tag* (what it turned out to be). They
// are NOT the same set: None builds no Weight (so it has no Format), and F32/F16 are real weights
// (DenseWeight, distinguished by Tensor dtype) that no QuantMode names. Q8/Q4/Q4G/W8A8 line up.
enum class Format { F32, F16, Q8, Q4, Q4G, W8A8 };

// R5: the shared Format's human-readable name — the ONE place that maps the representation tag to a
// string, so a storage diagnostic on either backend names a weight's format without a switch of its
// own (the symmetric counterpart to the shared enum). Pure metadata; never on the forward path.
const char* format_name(Format f);

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

// Gather rows of an int8 (Q8) embedding table by id, dequantizing each:
//   out[r, :] = q[ids[r], :] * scale[ids[r]].
// The weight-only int8 analogue of ops.cpp's embedding(). The table is the SAME tied
// [vocab, hidden] QTensor the output projection reads via linear_q8 (out = vocab), so the
// token embedding and the lm_head stay one quantized weight — G5d's int8 for the biggest
// single weight, with fp32 activations (the gather output) feeding the rest of the model.
Tensor embedding_q8(const QTensor& table, const std::vector<int64_t>& ids);

// W8A8: the SAME int8 weight (a QTensor from quantize_q8), but the activations are also quantized
// to int8 — dynamically, per row: a_scale[i] = max_j|x[i,j]|/127. The inner product is then an
// int8×int8→int32 integer dot (simd::dot_qq), dequantized by both scales at the end:
//   y[i,o] = (sum_j xq[i,j]·wq[o,j]) · a_scale[i] · w_scale[o] + bias[o].
// Both operands int8 ⇒ the dot runs at int8 throughput (the compute win Q8 can't get); the int32
// sum is exact, so a GPU DP4A kernel reproduces this integer core identically (G5d, the oracle).
Tensor linear_w8a8(const Tensor& x, const QTensor& w, const Tensor* bias = nullptr);

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
class Weight {
public:
    virtual ~Weight() = default;
    virtual Tensor linear(const Tensor& x, const Tensor* bias) const = 0;
    // Embedding gather (R3b): out[r,:] = weight[ids[r],:]. Only the embedding weight implements it;
    // the default (quant.cpp) throws so a projection weight gathered by mistake fails loudly.
    // DenseWeight (backend.hpp) and the int8-embed weights override it.
    virtual Tensor gather(const std::vector<int64_t>& ids) const;
    // R5: this weight's storage representation, so a caller reads the format without RTTI or knowing
    // the concrete subclass — the one Format enum every backend shares. Pure: every weight declares it.
    virtual Format format() const = 0;
    virtual int64_t bytes() const = 0;       // actual storage
    virtual int64_t fp32_bytes() const = 0;  // out*in*4 (the unquantized size)
};

// Quantize `w` into a Weight of the given mode (nullptr for None).
std::unique_ptr<Weight> make_quantized(const Tensor& w, QuantMode mode);

// R3b: the tied token-embedding / lm_head as a weight-only int8 Weight (the biggest single weight).
// gather() dequantizes a looked-up row (embedding_q8); linear() runs linear_q8 — fp32 activations
// into argmax. The CPU int8-embed Weight; the CUDA mirror is make_cuda_q8_embed (cuda_backend.hpp).
std::unique_ptr<Weight> make_q8_embed(const Tensor& w);

}  // namespace ni
