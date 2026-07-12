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

class Model {
public:
    // Load config.txt and every <name>.bin in `weights_dir`. With a quant mode,
    // the per-layer projection weights are stored as Q8 (int8, ~4x) or Q4 (int4,
    // ~8x); the embedding, lm_head, norms, and biases stay fp32. `device` selects
    // the compute backend (G0): CPU is wired now, CUDA/Metal land in G1+.
    explicit Model(const std::string& weights_dir, QuantMode mode = QuantMode::None,
                   Device device = Device::CPU, BackendConfig cfg = {});

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

    // Ragged batched verify for speculative decoding (S3b). N sequences, sequence s
    // contributing counts[s] = k_s+1 query rows (its [cur_s, d_s..] verify block);
    // `tokens` is the flat concatenation of every sequence's rows (length M = sum(counts)).
    // The projection GEMMs (q/k/v/o/gate/up/down) fuse over all M rows — every weight
    // streamed once (the F8a lever, now over the whole verify batch, not one token each);
    // attention is a per-sequence loop over that sequence's CONTIGUOUS query block, each
    // block a multi-query causal attend at the sequence's cache offset (the S0 verify
    // primitive). Rows [row_start_s, row_start_s+counts[s]) of the result equal a standalone
    // forward([cur_s, d_s..], caches[s]) bit-for-bit; each cache advances by counts[s].
    // forward_batch is the all-counts==1 special case. Returns logits [M, vocab].
    Tensor forward_spec_batch(const std::vector<int64_t>& tokens,
                              const std::vector<int64_t>& counts,
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
    // (quantization, and fp16 device weights under the backend's fp16_weights config).
    std::pair<int64_t, int64_t> weight_bytes() const;

    // R5: bytes per storage Format over the polymorphic Weights (the projections + the embedding /
    // lm_head), so a diagnostic can show WHICH representations the model holds, not just the total —
    // the shared Format tag (Weight::format) put to use, the same call on a CPU or CUDA model. The raw
    // w_ tensors (norms, q/k/v biases) are fp32 and aren't Weights, so they're omitted: this reports
    // exactly what a quant mode / fp16 varies. Sorted by Format; only the formats present appear.
    std::vector<std::pair<Format, int64_t>> weight_format_breakdown() const;

private:
    const Tensor& W(const std::string& name) const;
    // The bias tensor `name`, or nullptr when this arch has no q/k/v bias (A0's
    // qkv_bias flag: Qwen2.5 carries one, Qwen3/Llama don't). The exporter simply
    // omits the bias .bin when qkv_bias is off, so this must not look it up then.
    const Tensor* qkv_bias_ptr(const std::string& name) const;
    // QK-Norm (A1 Qwen3): a per-head RMSNorm over head_dim applied to Q and K
    // after the head split and before RoPE (matching the Python oracle). Given
    // [heads, *, head_dim] tensors, rmsnorm normalizes each (head, position) row
    // over its last dim. A no-op — and looks up no weight — when cfg.qk_norm is
    // off (Qwen2.5), keeping that path bit-identical. Shared by all three forwards.
    void apply_qk_norm(Tensor& q, Tensor& k, const std::string& layer_prefix) const;
    // A linear projection that dispatches to the Q8 path when `name` is quantized.
    Tensor project(const Tensor& x, const std::string& name, const Tensor* bias) const;
    // The token-embedding gather and the output projection to logits — each weight-only int8
    // (embedding_q8 / linear_q8) when cfg.quantize_embed quantized that tied weight, else the
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
    // cfg.quantize_embed (EmbedQ8Weight on CPU, the device mirror on CUDA — the biggest single weight).
    // lm_head_ is non-null only for an UNTIED checkpoint; a tied model leaves it null and reuses embed_.
    std::unique_ptr<Weight> embed_;
    std::unique_ptr<Weight> lm_head_;
    RopeCache rope_;  // built once for max_position_embeddings, sliced per forward
};

}  // namespace ni
