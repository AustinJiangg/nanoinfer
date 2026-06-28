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
    // Raw tensors that aren't projected/gathered through a Weight: the norms and the q/k/v biases
    // (and, until R3b, the embed/lm_head). Device-resident on a CUDA model (R3 to_resident).
    std::unordered_map<std::string, Tensor> w_;
    // R3: every layer-projection weight (q/k/v/o/gate/up/down) as a polymorphic Weight — DenseWeight
    // for fp32/fp16, a quant weight for Q8/Q4/Q4G/W8A8 — so Model::project dispatches through one
    // pointer with no device/format branch (the seam that makes the Backend abstraction complete).
    std::unordered_map<std::string, std::unique_ptr<Weight>> weights_;
    // R3b: the token-embedding and output-projection as Weights, so embed_tokens()/lm_head() dispatch
    // through one pointer with no #ifdef. embed_ does BOTH gather() (the embedding) and linear() (the
    // tied lm_head); it is a DenseWeight (fp32/fp16) normally, or an int8-embed Weight under
    // g_quantize_embed (EmbedQ8Weight on CPU, the device mirror on CUDA — the biggest single weight).
    // lm_head_ is non-null only for an UNTIED checkpoint; a tied model leaves it null and reuses embed_.
    std::unique_ptr<Weight> embed_;
    std::unique_ptr<Weight> lm_head_;
    RopeCache rope_;  // built once for max_position_embeddings, sliced per forward
};

}  // namespace ni
