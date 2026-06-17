#include "backend.hpp"

#include "ops.hpp"

namespace ni {

// Each method is a thin forwarder to the free op in ops.cpp. The `ni::` qualifier is
// load-bearing: an unqualified call would resolve to the member (infinite recursion).

Tensor CpuBackend::embedding(const Tensor& table, const std::vector<int64_t>& ids) {
    return ni::embedding(table, ids);
}
Tensor CpuBackend::linear(const Tensor& x, const Tensor& weight, const Tensor* bias) {
    return ni::linear(x, weight, bias);
}
Tensor CpuBackend::rmsnorm(const Tensor& x, const Tensor& weight, float eps) {
    return ni::rmsnorm(x, weight, eps);
}
Tensor CpuBackend::silu(const Tensor& x) { return ni::silu(x); }
Tensor CpuBackend::mul(const Tensor& a, const Tensor& b) { return ni::mul(a, b); }
Tensor CpuBackend::add(const Tensor& a, const Tensor& b) { return ni::add(a, b); }
Tensor CpuBackend::split_heads(const Tensor& x, int64_t n_heads, int64_t head_dim) {
    return ni::split_heads(x, n_heads, head_dim);
}
Tensor CpuBackend::merge_heads(const Tensor& x) { return ni::merge_heads(x); }
Tensor CpuBackend::repeat_kv(const Tensor& x, int64_t n_rep) {
    return ni::repeat_kv(x, n_rep);
}
Tensor CpuBackend::apply_rope(const Tensor& x, const Tensor& cos, const Tensor& sin,
                              int64_t pos_offset) {
    return ni::apply_rope(x, cos, sin, pos_offset);
}
Tensor CpuBackend::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                             int64_t query_offset) {
    return ni::attention(q, k, v, causal, query_offset);
}

// Batched-decode row plumbing (the former model.cpp row_to_heads / write_row, now a
// backend op so forward_batch is device-agnostic). On CPU these are plain host loops.
Tensor CpuBackend::alloc(const std::vector<int64_t>& shape) { return Tensor(shape); }

Tensor CpuBackend::extract_row(const Tensor& x, int64_t s, int64_t heads, int64_t dim) {
    Tensor out({heads, 1, dim});
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t d = 0; d < dim; ++d) out.at(h, 0, d) = x.at(s, h * dim + d);
    return out;
}

void CpuBackend::place_row(Tensor& dst, int64_t s, const Tensor& row) {
    const int64_t width = dst.size(1);
    for (int64_t c = 0; c < width; ++c) dst.at(s, c) = row.at(0, c);
}

}  // namespace ni
