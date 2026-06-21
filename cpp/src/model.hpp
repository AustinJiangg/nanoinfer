// The full Qwen2.5 decoder: embedding -> N transformer blocks -> norm -> lm_head.
// Loads NIT0 weights + config.txt from a directory (the export_weights.py output)
// and reproduces nanoinfer's forward pass. Batch-1, full-recompute (no KV cache
// yet — that's C3).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "cache.hpp"
#include "config.hpp"
#include "ops.hpp"
#include "quant.hpp"
#include "tensor.hpp"

namespace ni {

// Opt-in (G5d): quantize the tied token-embedding / lm_head weight to weight-only int8 — the
// biggest single weight (~544 MB fp32 on Qwen2.5-0.5B). Set before constructing a Model. It is
// orthogonal to the layer-projection QuantMode and to g_cuda_fp16_weights: the embed/lm_head get
// int8, the layer projections whatever `mode` says. Like g_cuda_fp16_weights, a load-time global so
// a test can toggle it around one constructor. CPU is wired now; the CUDA device path lands next.
extern bool g_quantize_embed;

class Model {
public:
    // Load config.txt and every <name>.bin in `weights_dir`. With a quant mode,
    // the per-layer projection weights are stored as Q8 (int8, ~4x) or Q4 (int4,
    // ~8x); the embedding, lm_head, norms, and biases stay fp32. `device` selects
    // the compute backend (G0): CPU is wired now, CUDA/Metal land in G1+.
    explicit Model(const std::string& weights_dir, QuantMode mode = QuantMode::None,
                   Device device = Device::CPU);

    // Token ids -> logits [seq, vocab]. Without a cache (C1) the whole sequence is
    // recomputed; with a cache (C3) `ids` is just the new token(s), placed at
    // positions cache.length()..; the cache is advanced by t after the pass.
    Tensor forward(const std::vector<int64_t>& ids, KVCacheBase* cache = nullptr) const;

    // Batched single-token decode (F8a): N sequences, one new token each, at each
    // sequence's own cache position. The projection GEMMs (q/k/v/o/gate/up/down) are
    // fused over the N rows — every weight row is streamed once and reused across
    // all N tokens (the throughput lever; the same row-inner reuse that makes prefill
    // compute-bound). Attention stays a per-sequence loop: each token attends only
    // its own cache, at its own RoPE position. Row s of the result equals a
    // standalone forward({tokens[s]}, caches[s]) bit-for-bit (the rows are
    // independent), so the serving layer can batch decode without changing outputs.
    // Returns logits [N, vocab]; every cache advances by 1.
    Tensor forward_batch(const std::vector<int64_t>& tokens,
                         const std::vector<KVCacheBase*>& caches) const;

    // Allocate a KV cache sized to this model.
    KVCache make_cache(int64_t max_seq) const;

    // Allocate a KV cache matching this model's device — the contiguous CPU cache, or a
    // device-resident cache (G3) for the CUDA backend — returned through the base so the
    // caller drives either via one pointer (the polymorphism forward() already uses).
    std::unique_ptr<KVCacheBase> make_kv_cache(int64_t max_seq) const;

    const Config& config() const { return cfg_; }

    // The device this model computes on (CPU, or CUDA after a -DNI_CUDA build).
    Device device() const { return backend_->device(); }

    // {actual weight bytes, bytes if everything were fp32} — the storage-savings report
    // (quantization, and fp16 device weights under g_cuda_fp16_weights).
    std::pair<int64_t, int64_t> weight_bytes() const;

private:
    const Tensor& W(const std::string& name) const;
    // A linear projection that dispatches to the Q8 path when `name` is quantized.
    Tensor project(const Tensor& x, const std::string& name, const Tensor* bias) const;
    // The token-embedding gather and the output projection to logits — each weight-only int8
    // (embedding_q8 / linear_q8) when g_quantize_embed quantized that tied weight, else the
    // fp32/device backend path. One place each so forward() and forward_batch() share the routing.
    Tensor embed_tokens(const std::vector<int64_t>& ids) const;
    Tensor lm_head(const Tensor& x) const;

    // Device-dispatch seam (G0): forward() runs every op through this. CpuBackend
    // wraps the free ops in ops.cpp; CUDA/Metal backends override them (G1+).
    std::unique_ptr<Backend> backend_;
    Config cfg_;
    std::unordered_map<std::string, Tensor> w_;  // fp32 weights
    // Quantized layer-projection weights (any mode), via the polymorphic wrapper.
    std::unordered_map<std::string, std::unique_ptr<QuantizedWeight>> qweights_;
    // Weight-only int8 (Q8) for the tied token-embedding / output-projection — the biggest single
    // weight. Built when g_quantize_embed is set (CPU; CUDA next). The gather dequantizes a row
    // (embedding_q8); the tied lm_head runs linear_q8. lmhead_q8_ is non-null only for an untied
    // checkpoint (otherwise the lm_head reuses embed_q8_).
    std::unique_ptr<QTensor> embed_q8_;
    std::unique_ptr<QTensor> lmhead_q8_;
#ifdef NI_CUDA
    // The CUDA mirror of embed_q8_ (G5d): int8 codes [vocab, hidden] + fp32 per-row scale, resident on
    // the device. The gather (cuda_embedding_q8) and the tied lm_head (cuda_linear_q8) read these.
    // unique_ptr so "unset" is just null (no Tensor default ctor needed). Untied-on-CUDA isn't wired.
    std::unique_ptr<Tensor> embed_q8_codes_, embed_q8_scale_;
#endif
    RopeCache rope_;  // built once for max_position_embeddings, sliced per forward
};

}  // namespace ni
