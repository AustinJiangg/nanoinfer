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

}  // namespace ni
