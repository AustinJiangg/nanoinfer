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

// Weights uploaded as fp16 when g_cuda_fp16_weights is on (G5d): the layer projections PLUS the
// big token embedding / output projection. embed_tokens (vocab×hidden) is the single largest
// weight — ~136M params, ~544 MB in fp32, and tied as the lm_head — so storing it as half is the
// biggest single memory win; fp32-accumulated, it keeps the golden tokens. "lm_head.weight"
// covers an untied checkpoint. Norms/biases stay fp32: tiny, and precision-sensitive.
bool is_fp16_weight(const std::string& name) {
    return is_layer_proj(name) || name == "embed_tokens.weight" || name == "lm_head.weight";
}
}  // namespace

// Default off; a caller (test/bench) sets it before constructing a Model. See model.hpp.
bool g_quantize_embed = false;

Model::Model(const std::string& weights_dir, QuantMode mode, Device device) {
    // The Backend is the device-dispatch seam (G0): forward() below is written once against it.
    // make_backend (R1) is the one place that maps a Device to its concrete backend — it throws
    // for a device whose backend wasn't compiled in, so callers fail loudly rather than silently.
    backend_ = make_backend(device);

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
#ifdef NI_CUDA
            // CUDA + W8A8: a device-resident int8 weight, so the projection runs int8×int8 DP4A on
            // the GPU (the compute win). W8A8 is the GPU quant path; the other modes on CUDA aren't
            // GPU-wired (their CPU linear can't take a device tensor). CPU uses make_quantized.
            if (device == Device::CUDA && mode == QuantMode::W8A8)
                weights_.emplace(n, make_cuda_w8a8(w_.at(n)));
            else
#endif
                weights_.emplace(n, make_quantized(w_.at(n), mode));
            w_.erase(n);  // free the fp32 copy — the quantized weight is live now
        }
    }

    // G5d: weight-only int8 for the tied token-embedding / output-projection (the biggest single
    // weight). The gather dequantizes a row; the lm_head runs linear_q8 — fp32 activations into
    // argmax, so only the weight rounds. On CPU the int8 lives in embed_q8_ (a host QTensor); on CUDA
    // the codes+scale go to device buffers driving cuda_embedding_q8/cuda_linear_q8. Runs BEFORE the
    // CUDA upload loop so the fp32 embed is erased here and not uploaded again below.
    if (g_quantize_embed) {
        // Build the tied embed/lm_head as a weight-only int8 Weight (the biggest single weight): the
        // gather dequantizes a row, the lm_head runs linear_q8 — fp32 activations into argmax, so only
        // the weight rounds. The CPU/CUDA int8-embed BUILDERS are the only #ifdef left here (R3c moves
        // it behind a backend factory); the gather/linear DISPATCH below is already #ifdef-free. Runs
        // before to_resident so the fp32 embed is consumed here, not uploaded again.
#ifdef NI_CUDA
        if (device == Device::CUDA) {
            embed_ = make_cuda_q8_embed(W("embed_tokens.weight"));
            w_.erase("embed_tokens.weight");
            // Untied lm_head on CUDA isn't wired (Qwen2.5 is tied); the CPU branch handles untied.
        } else
#endif
        {
            embed_ = make_q8_embed(W("embed_tokens.weight"));
            w_.erase("embed_tokens.weight");
            if (!cfg_.tie_word_embeddings && w_.count("lm_head.weight")) {
                lm_head_ = make_q8_embed(W("lm_head.weight"));
                w_.erase("lm_head.weight");
            }
        }
    }

    // R3: make every remaining weight + the RoPE tables resident on the compute device, once at
    // load (CPU: identity; CUDA: H2D, fp16 for the big eligible weights — the GPU analogue of C5's
    // "stream each weight once"). The device knowledge and the g_cuda_fp16_weights read moved into
    // CudaBackend::to_resident, so this loop carries no #ifdef and names no device global.
    for (auto& kv : w_)
        kv.second = backend_->to_resident(std::move(kv.second), is_fp16_weight(kv.first));
    rope_.cos = backend_->to_resident(std::move(rope_.cos), /*fp16_eligible=*/false);
    rope_.sin = backend_->to_resident(std::move(rope_.sin), /*fp16_eligible=*/false);

    // R3: wrap the dense (non-quant) projection weights — now device-resident — as DenseWeight in
    // the same weights_ map a quant mode already filled above, so Model::project drives every
    // projection (dense or quantized) through one Weight pointer with no branch. (For a quant mode
    // the projections were placed above and are gone from w_, so this finds none.)
    {
        std::vector<std::string> proj;
        for (const auto& kv : w_)
            if (is_layer_proj(kv.first)) proj.push_back(kv.first);
        for (const std::string& n : proj) {
            weights_.emplace(n, std::make_unique<DenseWeight>(backend_.get(), std::move(w_.at(n))));
            w_.erase(n);
        }
    }

    // R3b: wrap the dense (non-int8) embedding / output projection — now device-resident — as a
    // DenseWeight, so embed_tokens()/lm_head() dispatch through the Weight seam with no #ifdef. A
    // tied model has only embed_tokens.weight (the exporter drops the duplicate lm_head): lm_head_
    // stays null and lm_head() reuses embed_. An untied checkpoint gets its own lm_head_. (When
    // g_quantize_embed already built an int8 embed_/lm_head_ above, these are skipped.)
    if (!embed_) {
        embed_ = std::make_unique<DenseWeight>(backend_.get(),
                                               std::move(w_.at("embed_tokens.weight")));
        w_.erase("embed_tokens.weight");
    }
    if (!lm_head_ && !cfg_.tie_word_embeddings && w_.count("lm_head.weight")) {
        lm_head_ = std::make_unique<DenseWeight>(backend_.get(), std::move(w_.at("lm_head.weight")));
        w_.erase("lm_head.weight");
    }
}

KVCache Model::make_cache(int64_t max_seq) const {
    return KVCache(cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
}

std::unique_ptr<KVCacheBase> Model::make_kv_cache(int64_t max_seq) const {
    // The backend returns its native cache (CPU KVCache / device CudaKVCache) through the base
    // pointer the forward already drives — the #ifdef moved into the backend (R1).
    return backend_->make_kv_cache(cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
}

Tensor Model::project(const Tensor& x, const std::string& name, const Tensor* bias) const {
    // R3: every projection is a Weight (DenseWeight for fp32/fp16, a quant weight otherwise) in one
    // map — no device or format branch here; the Weight owns its own kernel.
    auto it = weights_.find(name);
    if (it == weights_.end()) throw std::runtime_error("Model: missing projection weight " + name);
    return it->second->linear(x, bias);
}

Tensor Model::embed_tokens(const std::vector<int64_t>& ids) const {
    // R3b: one Weight::gather — DenseWeight (backend gather, fp32/fp16) or an int8-embed Weight
    // (embedding_q8 / cuda_embedding_q8). No device or quant branch here.
    return embed_->gather(ids);
}

Tensor Model::lm_head(const Tensor& x) const {
    // R3b: a tied model reuses the embedding weight as the output projection (the exporter drops the
    // duplicate lm_head.weight); an untied checkpoint has its own lm_head_. Either way one
    // Weight::linear — dense (fp32/fp16) or weight-only int8 — with no device/format branch.
    const Weight& lm = lm_head_ ? *lm_head_ : *embed_;
    return lm.linear(x, nullptr);
}

std::pair<int64_t, int64_t> Model::weight_bytes() const {
    int64_t actual = 0, fp32 = 0;
    for (const auto& kv : w_) {
        const int64_t n = kv.second.numel();
        // fp16 device weights (G5d) take 2 bytes/elem, not 4 — so the report reflects the fp16
        // storage win, not just quantization's. CPU weights are always F32, so this is a no-op there.
        actual += n * (kv.second.dtype() == DType::F16 ? 2 : 4);
        fp32 += n * 4;
    }
    for (const auto& kv : weights_) {  // dense (DenseWeight) + quant projection weights (R3)
        actual += kv.second->bytes();
        fp32 += kv.second->fp32_bytes();
    }
    // R3b: the embedding + (untied) output projection report their own bytes through the Weight seam
    // — DenseWeight (fp16/fp32) or an int8-embed Weight — so the old #ifdef'd device-int8 accounting
    // is gone. A tied model has lm_head_ == null (embed_ doubles as the lm_head, counted once).
    actual += embed_->bytes();
    fp32 += embed_->fp32_bytes();
    if (lm_head_) {
        actual += lm_head_->bytes();
        fp32 += lm_head_->fp32_bytes();
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

    Tensor x = embed_tokens(ids);  // [seq, hidden]

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
    Tensor logits = lm_head(x);  // [seq, vocab]
    // Land the result on the host for return (CUDA D2H, unless a graph driver kept it on device;
    // CPU is identity) — the device edge-copy lives behind the Backend now (R1).
    return backend_->finalize_logits(std::move(logits));
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

    Tensor x = embed_tokens(tokens);  // [n, hidden]

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
        Tensor attn = backend_->alloc({n, H * D});
        for (int64_t s = 0; s < n; ++s) {
            Tensor qs = backend_->extract_row(q, s, H, D);   // [H, 1, D]
            Tensor ks = backend_->extract_row(k, s, KV, D);  // [KV, 1, D]
            Tensor vs = backend_->extract_row(v, s, KV, D);
            qs = backend_->apply_rope(qs, rope_.cos, rope_.sin, pos[static_cast<size_t>(s)]);
            ks = backend_->apply_rope(ks, rope_.cos, rope_.sin, pos[static_cast<size_t>(s)]);
            // Append this token's K/V and attend over the sequence's own history.
            Tensor as = caches[s]->attend(i, qs, ks, vs, cfg_.n_rep(), /*causal=*/true,
                                          /*query_offset=*/pos[static_cast<size_t>(s)]);  // [H,1,D]
            backend_->place_row(attn, s, backend_->merge_heads(as));  // [1, H*D] -> row s
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
    Tensor logits = lm_head(x);  // [n, vocab]
    return backend_->finalize_logits(std::move(logits));
}

}  // namespace ni
