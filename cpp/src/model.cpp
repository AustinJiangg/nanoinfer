#include "model.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include "serialize.hpp"

#ifdef NI_CUDA
#include "cuda/cuda.hpp"
#include "cuda/cuda_backend.hpp"
#endif

namespace fs = std::filesystem;

namespace ni {

namespace {
// The per-layer projection weights (q/k/v/o/gate/up/down) — "layers.N.*_proj.weight".
// These are the bulk of the model and what we quantize; embed/lm_head/norms stay fp32.
bool is_layer_proj(const std::string& name) {
    const std::string suffix = "_proj.weight";
    return name.rfind("layers.", 0) == 0 && name.size() > suffix.size() &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// forward_batch helpers: move one token between the batched [n, heads*dim]
// projection layout and the [heads, 1, dim] per-token layout the attention ops use.
// Row s of a projection -> [heads, 1, dim] (split_heads specialized to one token).
Tensor row_to_heads(const Tensor& x, int64_t s, int64_t heads, int64_t dim) {
    Tensor out({heads, 1, dim});
    for (int64_t h = 0; h < heads; ++h)
        for (int64_t d = 0; d < dim; ++d) out.at(h, 0, d) = x.at(s, h * dim + d);
    return out;
}
// Write a merged-head [1, width] row back into row s of a [n, width] tensor.
void write_row(Tensor& dst, int64_t s, const Tensor& row) {
    const int64_t width = dst.size(1);
    for (int64_t c = 0; c < width; ++c) dst.at(s, c) = row.at(0, c);
}
}  // namespace

Model::Model(const std::string& weights_dir, QuantMode mode, Device device) {
    // The Backend is the device-dispatch seam (G0): forward() below is written once
    // against it. CPU is the only backend wired in G0; CUDA/Metal throw here until
    // their backends land (G1+), so callers fail loudly rather than silently on CPU.
    if (device == Device::CPU) {
        backend_ = std::make_unique<CpuBackend>();
#ifdef NI_CUDA
    } else if (device == Device::CUDA) {
        backend_ = std::make_unique<CudaBackend>();
#endif
    } else {
        throw std::runtime_error(
            "Model: backend for this device is not built — rebuild with -DNI_CUDA=ON");
    }

    cfg_ = load_config(weights_dir + "/config.txt");
    for (const auto& entry : fs::directory_iterator(weights_dir)) {
        const fs::path& p = entry.path();
        if (p.extension() == ".bin") {
            // stem() drops the trailing ".bin" but keeps the dotted name, e.g.
            // "layers.0.self_attn.q_proj.weight".
            w_.emplace(p.stem().string(), load_bin(p.string()));
        }
    }
    if (w_.empty()) throw std::runtime_error("Model: no .bin weights in " + weights_dir);

    // Build the RoPE tables once for the full context; forward slices them.
    rope_ = build_rope_cache(cfg_.max_position_embeddings, cfg_.head_dim, cfg_.rope_theta);

    if (mode != QuantMode::None) {
        std::vector<std::string> names;
        for (const auto& kv : w_)
            if (is_layer_proj(kv.first)) names.push_back(kv.first);
        for (const std::string& n : names) {
            qweights_.emplace(n, make_quantized(w_.at(n), mode));
            w_.erase(n);  // free the fp32 copy — the quantized weight is live now
        }
    }

#ifdef NI_CUDA
    // CUDA path: upload every resident weight + the RoPE tables to the GPU once, here,
    // so the forward never copies a weight again — H2D only at load, the GPU analogue of
    // C5's "stream each weight once". After this W() returns device tensors and every op
    // in forward() runs on the GPU. (Quant stays on CPU, so this path uses fp32 weights.)
    if (device == Device::CUDA) {
        for (auto& kv : w_) kv.second = to_device(kv.second);
        rope_.cos = to_device(rope_.cos);
        rope_.sin = to_device(rope_.sin);
    }
#endif
}

KVCache Model::make_cache(int64_t max_seq) const {
    return KVCache(cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
}

std::unique_ptr<KVCacheBase> Model::make_kv_cache(int64_t max_seq) const {
#ifdef NI_CUDA
    if (backend_->device() == Device::CUDA)
        return std::make_unique<CudaKVCache>(backend_.get(), cfg_.num_layers, cfg_.num_kv_heads,
                                             cfg_.head_dim);
#endif
    return std::make_unique<KVCache>(cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
}

Tensor Model::project(const Tensor& x, const std::string& name, const Tensor* bias) const {
    // Quantized projections keep their own (CPU) path for now — GPU quant is post-G5.
    // The fp32 path goes through the backend, so it runs on whatever device backend_ is.
    auto it = qweights_.find(name);
    if (it != qweights_.end()) return it->second->linear(x, bias);
    return backend_->linear(x, W(name), bias);
}

std::pair<int64_t, int64_t> Model::weight_bytes() const {
    int64_t actual = 0, fp32 = 0;
    for (const auto& kv : w_) {
        const int64_t n = kv.second.numel();
        actual += n * 4;
        fp32 += n * 4;
    }
    for (const auto& kv : qweights_) {
        actual += kv.second->bytes();
        fp32 += kv.second->fp32_bytes();
    }
    return {actual, fp32};
}

const Tensor& Model::W(const std::string& name) const {
    auto it = w_.find(name);
    if (it == w_.end()) throw std::runtime_error("Model: missing weight " + name);
    return it->second;
}

Tensor Model::forward(const std::vector<int64_t>& ids, KVCacheBase* cache) const {
    const int64_t seq = static_cast<int64_t>(ids.size());
    const float eps = cfg_.rms_norm_eps;

    // With a cache, the new tokens sit at positions start_pos.. (start_pos is the
    // cache length before this pass — the same for every layer; advanced once at
    // the end). Without a cache they run 0..seq-1.
    const int64_t start_pos = cache ? cache->length() : 0;
    if (start_pos + seq > rope_.cos.size(0))
        throw std::runtime_error("forward: position " + std::to_string(start_pos + seq) +
                                 " exceeds context length " + std::to_string(rope_.cos.size(0)));

    Tensor x = backend_->embedding(W("embed_tokens.weight"), ids);  // [seq, hidden]

    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string L = "layers." + std::to_string(i) + ".";

        // --- attention (pre-norm + residual) ---
        Tensor h = backend_->rmsnorm(x, W(L + "input_layernorm.weight"), eps);
        Tensor q = project(h, L + "self_attn.q_proj.weight", &W(L + "self_attn.q_proj.bias"));
        Tensor k = project(h, L + "self_attn.k_proj.weight", &W(L + "self_attn.k_proj.bias"));
        Tensor v = project(h, L + "self_attn.v_proj.weight", &W(L + "self_attn.v_proj.bias"));

        q = backend_->split_heads(q, cfg_.num_attention_heads, cfg_.head_dim);
        k = backend_->split_heads(k, cfg_.num_kv_heads, cfg_.head_dim);
        v = backend_->split_heads(v, cfg_.num_kv_heads, cfg_.head_dim);

        // RoPE at absolute positions; cached keys are stored already rotated.
        q = backend_->apply_rope(q, rope_.cos, rope_.sin, start_pos);
        k = backend_->apply_rope(k, rope_.cos, rope_.sin, start_pos);

        // Cached: the cache appends K/V and attends over its whole history, folding
        // in GQA (the paged cache reads blocks directly; the contiguous one gathers).
        // Uncached (stage-0 full recompute): no cache object, so expand the KV heads
        // and attend over the fresh K/V right here, at offset 0.
        Tensor a;
        if (cache) {
            a = cache->attend(i, q, k, v, cfg_.n_rep(), /*causal=*/true,
                              /*query_offset=*/start_pos);
        } else {
            Tensor kk = backend_->repeat_kv(k, cfg_.n_rep());
            Tensor vv = backend_->repeat_kv(v, cfg_.n_rep());
            a = backend_->attention(q, kk, vv, /*causal=*/true, /*query_offset=*/0);
        }
        a = backend_->merge_heads(a);                            // [seq, n_heads*head_dim]
        a = project(a, L + "self_attn.o_proj.weight", nullptr);  // o_proj has no bias
        x = backend_->add(x, a);

        // --- SwiGLU MLP (pre-norm + residual) ---
        Tensor hm = backend_->rmsnorm(x, W(L + "post_attention_layernorm.weight"), eps);
        Tensor gate = backend_->silu(project(hm, L + "mlp.gate_proj.weight", nullptr));
        Tensor up = project(hm, L + "mlp.up_proj.weight", nullptr);
        Tensor down = project(backend_->mul(gate, up), L + "mlp.down_proj.weight", nullptr);
        x = backend_->add(x, down);
    }

    if (cache) cache->advance(seq);

    x = backend_->rmsnorm(x, W("norm.weight"), eps);
    // Tied models share the embedding as the output projection, so the exporter
    // skips the duplicate lm_head.weight; fall back to embed_tokens here.
    const std::string lm = cfg_.tie_word_embeddings ? "embed_tokens.weight" : "lm_head.weight";
    Tensor logits = backend_->linear(x, W(lm), nullptr);  // [seq, vocab]
#ifdef NI_CUDA
    if (logits.device() == Device::CUDA) logits = to_host(logits);  // D2H only at the edge
#endif
    return logits;
}

Tensor Model::forward_batch(const std::vector<int64_t>& tokens,
                            const std::vector<KVCacheBase*>& caches) const {
    const int64_t n = static_cast<int64_t>(tokens.size());
    if (static_cast<int64_t>(caches.size()) != n)
        throw std::invalid_argument("forward_batch: tokens and caches differ in size");
    if (n == 0) return Tensor({0, cfg_.vocab_size});
    const float eps = cfg_.rms_norm_eps;
    const int64_t H = cfg_.num_attention_heads, KV = cfg_.num_kv_heads, D = cfg_.head_dim;

    // Each sequence's new token sits at its own cache position. Capture every
    // cache's pre-step length once: it is the RoPE position and the attention
    // offset for that sequence this step, and (like single forward) the length is
    // advanced only after all layers have written their slice.
    std::vector<int64_t> pos(static_cast<size_t>(n));
    for (int64_t s = 0; s < n; ++s) {
        if (caches[s] == nullptr) throw std::invalid_argument("forward_batch: null cache");
        pos[static_cast<size_t>(s)] = caches[s]->length();
        if (pos[static_cast<size_t>(s)] + 1 > rope_.cos.size(0))
            throw std::runtime_error("forward_batch: position " +
                                     std::to_string(pos[static_cast<size_t>(s)] + 1) +
                                     " exceeds context length " +
                                     std::to_string(rope_.cos.size(0)));
    }

    Tensor x = backend_->embedding(W("embed_tokens.weight"), tokens);  // [n, hidden]

    for (int64_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string L = "layers." + std::to_string(i) + ".";

        // --- attention (pre-norm + residual) ---
        Tensor h = backend_->rmsnorm(x, W(L + "input_layernorm.weight"), eps);  // [n, hidden]
        // Batched projections: one matmul over all n rows, weights streamed once.
        Tensor q = project(h, L + "self_attn.q_proj.weight", &W(L + "self_attn.q_proj.bias"));
        Tensor k = project(h, L + "self_attn.k_proj.weight", &W(L + "self_attn.k_proj.bias"));
        Tensor v = project(h, L + "self_attn.v_proj.weight", &W(L + "self_attn.v_proj.bias"));

        // Per-sequence attention: each token has its own cache, length, and RoPE
        // position, so this is a loop, not a batched matmul. Each iteration reuses
        // the same single-token ops as forward(), guaranteeing row-for-row parity.
        Tensor attn({n, H * D});
        for (int64_t s = 0; s < n; ++s) {
            Tensor qs = row_to_heads(q, s, H, D);   // [H, 1, D]
            Tensor ks = row_to_heads(k, s, KV, D);  // [KV, 1, D]
            Tensor vs = row_to_heads(v, s, KV, D);
            qs = backend_->apply_rope(qs, rope_.cos, rope_.sin, pos[static_cast<size_t>(s)]);
            ks = backend_->apply_rope(ks, rope_.cos, rope_.sin, pos[static_cast<size_t>(s)]);
            // Append this token's K/V and attend over the sequence's own history.
            Tensor as = caches[s]->attend(i, qs, ks, vs, cfg_.n_rep(), /*causal=*/true,
                                          /*query_offset=*/pos[static_cast<size_t>(s)]);  // [H,1,D]
            write_row(attn, s, backend_->merge_heads(as));  // [1, H*D] -> row s
        }
        Tensor a = project(attn, L + "self_attn.o_proj.weight", nullptr);  // o_proj: no bias
        x = backend_->add(x, a);

        // --- SwiGLU MLP (pre-norm + residual), all batched over the n rows ---
        Tensor hm = backend_->rmsnorm(x, W(L + "post_attention_layernorm.weight"), eps);
        Tensor gate = backend_->silu(project(hm, L + "mlp.gate_proj.weight", nullptr));
        Tensor up = project(hm, L + "mlp.up_proj.weight", nullptr);
        Tensor down = project(backend_->mul(gate, up), L + "mlp.down_proj.weight", nullptr);
        x = backend_->add(x, down);
    }

    for (int64_t s = 0; s < n; ++s) caches[s]->advance(1);

    x = backend_->rmsnorm(x, W("norm.weight"), eps);
    const std::string lm = cfg_.tie_word_embeddings ? "embed_tokens.weight" : "lm_head.weight";
    Tensor logits = backend_->linear(x, W(lm), nullptr);  // [n, vocab]
#ifdef NI_CUDA
    if (logits.device() == Device::CUDA) logits = to_host(logits);
#endif
    return logits;
}

}  // namespace ni
