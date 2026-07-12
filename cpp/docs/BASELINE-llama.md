# Llama-3.2-1B baseline — the A2 config-heavy rung (2026-07-12)

The second A-track arch. Llama-3.2 is the "config-heavy, code-light" rung: the only
genuinely new behavior is **llama3 RoPE scaling**, and it lands the same way A1's
QK-Norm did — resolved once at load, no hot-path change, no CUDA kernel change.

The delta:
1. **llama3 RoPE scaling** — the base `inv_freq` is rescaled in three frequency
   bands at table-build time: long wavelengths (low freq) divided by `factor`, short
   ones kept, a smooth blend between (mirrors HF's `_compute_llama3_parameters`;
   `attention_factor=1`, so cos/sin aren't further scaled). Implemented in
   `_apply_rope_scaling` (Python) / `llama3_scale_inv_freq` + `RopeScalingParams`
   (C++ `ops.cpp`), applied inside `build_rope_cache`. The rotation kernel
   (`apply_rope`) is UNTOUCHED — config-time resolution, the rope_theta lesson. And
   because the scaled cos/sin tables are built on the host and `to_resident`-uploaded,
   **CUDA inherited llama3 scaling for free** (apply_rope reads the same tables).
2. **tiktoken-family BPE**, vocab 128256 — different space/byte handling than Qwen.
   The incremental detokenizer must prove tokenizer-generic (see the detok gate).
3. No QKV bias, no QK-Norm, tied embeddings, head_dim 64, GQA 32/8 — all already
   handled by A0's config layer.

`meta-llama/Llama-3.2-1B` is gated; we use the ungated **`unsloth/Llama-3.2-1B`**
mirror (identical weights + config) for local parity. **Llama Community License:**
don't redistribute the weights or the goldens' decoded text — this file lists token
ids only.

Captured on: RTX 4070 SUPER (12 GB), `weights/llama-3.2-1b`, build-cuda (sm_89).
Prompt `"The capital of France is"` → 6 ids (the tiktoken tokenizer prepends BOS).
Golden greedy (12 ids): `12366 13 1102 374 279 1455 95551 3363 304 9822 323 279`.

## Config (read from HF, written to config.txt)

| field | Qwen2.5-0.5B | Llama-3.2-1B |
|---|---|---|
| hidden_size | 896 | 2048 |
| intermediate_size | 4864 | 8192 |
| num_layers | 24 | **16** |
| num_attention_heads | 14 | **32** |
| num_kv_heads | 2 | **8** (GQA, n_rep=4) |
| head_dim | 64 | 64 |
| rope_theta | 1e6 | **5e5** |
| **rope_scaling** | none | **llama3** (factor 32, low 1, high 4, orig_max_pos 8192) |
| vocab_size | 151936 | **128256** (tiktoken BPE) |
| tie_word_embeddings | 1 | 1 |
| qkv_bias / qk_norm | 1 / 0 | 0 / 0 |
| params / fp32 weights | ~0.5B / 1979 MB | ~1.24B / **4946 MB** |

## Python oracle (gate step 1 — `tests/test_llama.py`)

`test_llama_config_flags` (rope_scaling llama3, rope_theta 5e5, head_dim 64, GQA 32/8,
no bias/qk-norm, tied), `test_llama_forward_matches_hf` (logits `allclose(1e-3)`),
`test_llama_greedy_matches_hf` (greedy == HF). Fast band tests in `test_layers.py`
(`test_rope_scaling_none_is_identity`, `test_llama3_rope_scaling_bands`, no download).

## CPU oracle (gate step 2 — vs the dumped goldens)

| check | invariant |
|---|---|
| `run_parity` | logits `max\|diff\|=1.28746e-05`, `mean\|diff\|=1.64285e-06`, argmax 0/6, next-token MATCH |
| `run_generate` | greedy == golden, **0/12** |
| `run_cache` | cached vs uncached `max\|diff\|=0` over 6 positions (split=3 +chunk) **[=0]** |
| `run_batch` | batched == standalone `max\|diff\|=0` **[=0]** |
| `run_paged` | paged == contiguous + S1 rollback `max\|diff\|=0` **[=0]** |

## CPU quantization (`run_quant <dir> <mode>`)

| mode | weights | ratio | logits `max\|diff\|` | greedy | next-token | formats |
|---|---|---|---|---|---|---|
| none | 4946.3 MB | 1.00× | 1.28746e-05 | 12/12 | preserved | f32=4943.0 |
| q8   | 2028.6 MB | **2.44×** | 0.469442 | **12/12** | preserved | f32=1050.7 q8=974.6 |

## CUDA backend (gate step 3 — vs the CPU oracle + golden; no CUDA code change)

| path | result | weights |
|---|---|---|
| fp32 GPU forward | `max\|diff\|=1.32322e-05`, argmax 0/6, greedy **0/12** MATCH | 4946 MB |
| fp16 weights | greedy **0/12** MATCH, `max\|diff\|=1.23978e-05` | 2475 MB (2.00×) |
| W8A8 (int8 DP4A) | greedy 10/12 differ, next-token preserved | 2029 MB |
| int8 embed/lm_head alone | greedy **0/12** differ, `max\|diff\|=0.206013` | 4159 MB |
| full int8 (W8A8 + int8 embed) | greedy **0/12** differ, next-token preserved | 1241 MB (**3.99×**) |

`run_cuda_cache`: incremental greedy 0/12, flash-decoding split MATCH (ctx 516).
`run_cuda_paged`: paged == contiguous + S1 rollback `max|diff|=0` **[=0]**.
`run_cuda_batch`: bit-identical **[=0]**.

W8A8 diverges more here (10/12) — int8 *activation* error over the wider hidden (2048),
same shape as the 1.5B W8A8 5/12; full-int8 stays 0/12 (argmax robust). All next-token-preserved.

## Serving + detok (gate step 4)

| gate | result |
|---|---|
| `run_detok` (NEW) | IncrementalDetokenizer **tokenizer-generic**: Llama tiktoken AND Qwen BPE, adversarial UTF-8 (emoji/CJK/combining marks/space boundaries) — every case: streamed deltas concat == one-shot decode, no mid-stream `�` leaked (emoji runs correctly HELD until complete) |
| `run_serve` | F7 / F8a / F8b / prefix-sharing all MATCH standalone at mb 1/2/8 |
| `run_http_serve --device cuda` | single non-stream MATCH; concurrent×6 mixed SSE/plain MATCH (ttft_p50 74 ms, tpot_p50 34 ms); disconnect→cancel; 62.9 tok/s aggregate |

The `run_detok` gate is the A2 detok deliverable: V1's sliding-window algorithm was
byte-stub-tested; this proves it on the real tiktoken BPE, whose "Ġ" space handling and
byte splits differ from Qwen's — and it holds unchanged, because the algorithm only ever
calls `tokenizer.decode()` (tokenizer-agnostic by construction).

## Regression — Qwen2.5-0.5B (frozen oracle) bit-identical

`build_rope_cache` gained a `RopeScalingParams` argument defaulting to `enabled=false`,
so the no-scaling path (Qwen2.5/Qwen3) leaves `inv_freq` untouched. Confirmed:
`ctest -L weights` **13/13** (CPU+CUDA), `ctest -L nomodel` **9/9**, full slow `pytest`
**44 passed** (Qwen2.5 + Qwen3 + Llama), every `BASELINE.md` number digit-identical.

## Reproduce

```bash
pytest tests/test_llama.py -v                                                   # (1)
python cpp/tools/export_weights.py cpp/weights/llama-3.2-1b unsloth/Llama-3.2-1B
python cpp/tools/dump_reference.py cpp/weights/llama-3.2-1b "The capital of France is" unsloth/Llama-3.2-1B
cd cpp && cmake --build build -j
for t in run_parity run_generate run_cache run_batch run_paged; do ./build/$t weights/llama-3.2-1b; done          # (2)
for t in run_cuda_parity run_cuda_cache run_cuda_paged run_cuda_batch; do ./build/$t weights/llama-3.2-1b; done   # (3)
python tests/python/run_detok.py                                                # (4) tokenizer-generic detok
python tests/python/run_serve.py weights/llama-3.2-1b
python tests/python/run_http_serve.py weights/llama-3.2-1b --device cuda
```

## Verdict (A2)

Llama-3.2 cost **one build-time frequency rescale** — `_apply_rope_scaling` /
`llama3_scale_inv_freq` inside `build_rope_cache`, gated on the A0 `rope_scaling` flag —
and nothing else: no kernel touched (Python or CUDA), no forward change, the config
plumbing already in place from A0. The RoPE-scaling lesson is the rope_theta lesson
generalized: **resolve the frequencies at load, keep the rotation kernel dumb.** The
tiktoken tokenizer needed no engine change either — the incremental detokenizer was
already tokenizer-generic, and the new `run_detok` gate proves it. Two arches in
(Qwen3, Llama-3.2), the A0 feature-flag bet keeps paying off: each new Llama-family
model is config + a tiny, local, kernel-free change.
