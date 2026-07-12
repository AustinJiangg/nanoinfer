# Qwen3-0.6B baseline — the A1 new-architecture rung (2026-07-12)

The first arch beyond Qwen2.5. Qwen3 is the **smallest delta** the A-track ladder
starts on, and A0's feature-flag layer paid off: the port was config flags + one
`apply_qk_norm()` call, **no new kernels and no CUDA change**. The three deltas:

1. **QK-Norm** — a per-head RMSNorm over `head_dim` on Q and K, after the head
   split and before RoPE (V is never normed). Implemented once as
   `Model::apply_qk_norm` (model.cpp) and dropped into all three forward variants
   (forward / forward_batch / forward_spec_batch); the Python oracle mirrors it in
   `Attention`. Because `rmsnorm` already treats `[heads, *, head_dim]` as rows over
   its last dim — on **both** the CPU op and `CudaBackend::rmsnorm` — the Backend
   seam carried QK-Norm to the GPU for free.
2. **No QKV bias** — `attention_bias=false` → A0's `qkv_bias` flag off; the exporter
   omits the bias `.bin`s and `qkv_bias_ptr` returns null.
3. **Explicit `head_dim=128`, non-square q_proj** — `16 heads × 128 = 2048 ≠ hidden
   1024`, so q_proj is `1024→2048`. This flushed any hidden `n_heads·head_dim ==
   hidden` assumption; none surfaced (every dim is read from config).

Everything else — GQA 16/8, tied embeddings, `rope_theta 1e6`, SwiGLU, the caches,
scheduler, spec, HTTP — is unchanged from Qwen2.5.

This file is the A1 analogue of `BASELINE.md` (0.5B) / `BASELINE-1.5b.md`.
Captured on: RTX 4070 SUPER (12 GB), `weights/qwen3-0.6b`, build-cuda (`-DNI_CUDA=ON`, sm_89).
Prompt `"The capital of France is"` → `ref_ids = 785 6722 315 9625 374`. Golden greedy (12):
`12095 13 576 6722 315 15344 374 21718 13 576 6722 315` → *" Paris. The capital of Italy
is Rome. The capital of"*.

## Config (read from HF, written to config.txt — the A0 v2 header)

| field | Qwen2.5-0.5B | Qwen3-0.6B |
|---|---|---|
| hidden_size | 896 | 1024 |
| intermediate_size | 4864 | **3072** |
| num_layers | 24 | **28** |
| num_attention_heads | 14 | **16** |
| num_kv_heads | 2 | **8** (GQA, n_rep=2) |
| **head_dim** | 64 | **128** (non-square: 16·128 ≠ 1024) |
| max_position_embeddings | 32768 | **40960** |
| rope_theta | 1e6 | 1e6 |
| tie_word_embeddings | 1 | 1 |
| vocab_size | 151936 | 151936 (same tokenizer) |
| **qkv_bias** (A0) | 1 | **0** |
| **qk_norm** (A0/A1) | 0 | **1** |
| params / fp32 weights | ~0.5B / 1979 MB | ~0.6B / **2387 MB** |

## Python oracle (gate step 1 — HF → Python, `tests/test_qwen3.py`)

| check | invariant |
|---|---|
| `test_qwen3_config_flags` | A0 reads qk_norm=1, qkv_bias=0, head_dim=128, rope_theta=1e6, rope_scaling=None off HF |
| `test_qwen3_forward_matches_hf` | logits `allclose(atol=1e-3, rtol=1e-3)` vs HF |
| `test_qwen3_greedy_matches_hf` | greedy 10 tokens == HF `generate(do_sample=False)` |

Fast QK-Norm seam tests (`tests/test_layers.py`, no download): `test_qk_norm_default_off`
(Qwen2.5 builds no q_norm/k_norm), `test_qk_norm_builds_and_is_live` (head_dim-sized, changes output).

## CPU oracle (gate step 2 — the parity spine, vs the dumped goldens)

| check | invariant |
|---|---|
| `run_parity` | logits `max\|diff\|=2.00272e-05`, `mean\|diff\|=3.2381e-06`, argmax 0/5, next-token `12095` MATCH |
| `run_generate` | greedy == golden, **0/12** mismatches |
| `run_cache` | cached vs uncached `max\|diff\|=0` over 5 positions (split=2 +chunk) **[=0]** |
| `run_batch` | batched == standalone `max\|diff\|=0`, bit-identical **[=0]**; ~2.5× decode @ batch 16 |
| `run_paged` | paged == contiguous `max\|diff\|=0`; S1 rollback (truncate+replay == decode-only) `max\|diff\|=0` **[=0]** |

## CPU quantization (`run_quant <dir> <mode>`)

| mode | weights | ratio | logits `max\|diff\|` | greedy | next-token | formats |
|---|---|---|---|---|---|---|
| none | 2387.2 MB | 1.00× | 2.00272e-05 | 12/12 | preserved | f32=2383.9 |
| q8   | 1067.4 MB | **2.24×** | 6.31351 | **12/12** | preserved | f32=622.3 q8=441.8 |

The QK-Norm weights are not `*_proj.weight`, so `is_layer_proj` leaves them fp32 (unquantized) —
only the projections quantize, and q8 still lands **12/12** greedy (vs 0.5B's 11/12): the extra
QK-Norm renormalizes Q/K after the quantized projection, damping the int8 error before attention.

## CUDA backend (gate step 3 — vs the CPU oracle + golden; no CUDA code change)

`run_cuda_parity`:

| path | result | weights |
|---|---|---|
| fp32 GPU forward | `max\|diff\|=2.86102e-05`, `mean\|diff\|=4.2789e-06`, argmax 0/5, greedy **0/12** MATCH | 2387 MB |
| fp16 weights     | greedy **0/12** MATCH, `max\|diff\|=3.14713e-05` | 1195 MB (2.00×) |
| W8A8 (int8 DP4A) | greedy **0/12** differ, next-token preserved | 1067 MB |
| int8 embed/lm_head alone | greedy **6/12** differ, `max\|diff\|=0.179823`, next-token preserved | 1921 MB |
| full int8 (W8A8 + int8 embed) | greedy **0/12** differ, next-token preserved | 601 MB (**3.97×**); q8=156.2 w8a8=441.8 |

`run_cuda_cache`: incremental greedy 0/12, flash-decoding split-on == split-off MATCH (ctx 515).
`run_cuda_paged`: paged tiled == contiguous / == non-tiled paged `max|diff|=0`; S1 rollback `max|diff|=0` **[=0]**.
`run_cuda_batch`: 4 seqs bit-identical **[=0]**; batched decode ~2.1× @ batch 32 (119.6 tok/s).

Note the full-int8 greedy is **0/12** here (vs 1.5B's 11/12): at 0.6B the argmax is robust to the
combined int8-activation + int8-embed error on this prompt — same honest "next-token preserved" bar.

## Serving (gate step 4 — the Python layer, unchanged from Qwen2.5)

| gate | result |
|---|---|
| `run_serve` | F7 per-seq / F8a batched / F8b paged / prefix-sharing all MATCH standalone at mb 1/2/8 |
| `run_spec_serve` (0.6B draft==target) | S3a/S3b (ragged verify) / S3c-d paged / S3e prefix, mixed draft+lookup batch — all MATCH standalone spec AND plain greedy |
| `run_http_serve --device cuda` | single non-stream MATCH; concurrent×6 mixed SSE/plain MATCH (ttft_p50 60.5 ms, tpot_p50 24.2 ms); disconnect→cancel; 75.3 tok/s aggregate |

## Regression — Qwen2.5-0.5B (the frozen oracle) stays bit-identical

The QK-Norm branch is a true no-op when `qk_norm=0`: `apply_qk_norm` returns early and builds no
weights, the Python `Attention` builds no `q_norm`/`k_norm`. Confirmed by the full local gate on
Qwen2.5-0.5B: `ctest -L weights` **13/13** (CPU+CUDA), `ctest -L nomodel` **9/9**, and every
`BASELINE.md` number digit-identical (`run_parity max|diff|=4.24385e-05`, golden 0/12).

## Reproduce

```bash
# gate steps 1–5, from repo root
pytest tests/test_qwen3.py -v                                  # (1) HF → Python
python cpp/tools/export_weights.py cpp/weights/qwen3-0.6b Qwen/Qwen3-0.6B
python cpp/tools/dump_reference.py cpp/weights/qwen3-0.6b "The capital of France is" Qwen/Qwen3-0.6B
cd cpp && cmake --build build -j
for t in run_parity run_generate run_cache run_batch run_paged; do ./build/$t weights/qwen3-0.6b; done   # (2)
for t in run_cuda_parity run_cuda_cache run_cuda_paged run_cuda_batch; do ./build/$t weights/qwen3-0.6b; done   # (3)
python tests/python/run_serve.py weights/qwen3-0.6b                                    # (4)
python tests/python/run_spec_serve.py weights/qwen3-0.6b weights/qwen3-0.6b
python tests/python/run_http_serve.py weights/qwen3-0.6b --device cuda
```

## Verdict (A1)

The A0 architecture-description layer did its job: **a new Llama-family arch cost one config flag
(`qk_norm`) already read by A0, one `apply_qk_norm` helper wired into three forwards, and zero new
kernels — CUDA inherited QK-Norm through the existing `rmsnorm` op.** Qwen3-0.6B is now parity-locked
CPU + CUDA (fp32/fp16/int8) and green across the serving + spec + HTTP gates, while Qwen2.5 stays
bit-identical. The "feature flags, not a class hierarchy" bet (the llama.cpp shape) is validated on
the first rung. The Qwen3-1.7B target + the 0.6B/1.7B speculative pair (r≈0.35, a second data point
for S2's "r is the binding constraint") are the remaining A1 items.
