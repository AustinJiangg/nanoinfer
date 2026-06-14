# Roadmap

Each stage is independently runnable and testable. Land one fully (code + tests
green) before starting the next. Check the box when done.

## Stage 0 — Naive forward pass + greedy loop  ✅ landed
The minimal correct engine. No cache, no batching. Recomputes the full sequence
every step. Slow but easy to verify.
- [x] ModelConfig from HF config
- [x] Weight loading with name remapping
- [x] RMSNorm, RoPE, GQA attention, SwiGLU, transformer block
- [x] Model.forward(input_ids) -> logits
- [x] Greedy generation loop + CLI
- [x] Shape tests + greedy end-to-end test
**Done when:** `python -m nanoinfer.generate --prompt "..."` produces coherent,
deterministic text that matches HF greedy decoding token-for-token.

## Stage 1 — KV cache  ✅ landed
Stop recomputing. Cache K and V per layer; each step processes only the new
token. This is the single biggest correctness-preserving speedup.
- [x] Per-layer KV cache structure (preallocated to max length) — `cache.py`
- [x] Attention path for the single-token (decode) case
- [x] Split prefill (process prompt) from decode (one token at a time)
- [x] Test: cached output == stage-0 uncached output, token-for-token
- [x] Benchmark: tokens/sec before vs after (~2.4× on Qwen2.5-0.5B, CPU, 60 tok)
**Done when:** `generate(..., use_cache=True)` matches the uncached path
token-for-token and HF greedy decoding, and is measurably faster. The stage-0
full-recompute path stays reachable via `--no-cache` for A/B.

## Stage 2 — Sampling strategies  ✅ landed
- [x] temperature, top-k, top-p (nucleus), repetition penalty — `sampling.py`
- [x] Seeded RNG for reproducible sampling (`torch.Generator`)
- [x] `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--seed` CLI flags
- [x] Tests: temperature=0 == greedy; top-k=1 == greedy; distribution sanity
**Done when:** `SamplingParams()` defaults to greedy (so prior behavior is
unchanged), the warpers compose in HF order, and a seed makes sampling
reproducible. temperature 0 short-circuits to argmax; top-k 1 collapses to greedy.

## Stage 3 — Continuous batching
Serve multiple prompts at once; admit/evict sequences as they finish, don't wait
for the whole batch.
- [ ] Batched forward with right-padding + attention mask
- [ ] Scheduler: per-sequence state, dynamic add/remove
- [ ] Simple request queue + async-ish driver loop
- [ ] Test: batched results match single-sequence results

## Stage 4 — Paged attention
The vLLM idea: KV cache in fixed-size blocks, a block table per sequence, no
contiguous-memory requirement. Enables high memory utilization.
- [ ] Block allocator + per-sequence block table
- [ ] Paged attention read path
- [ ] Test: paged output matches contiguous-cache output

## Stage 5 — Quantization / custom kernels (stretch)
- [ ] int8/int4 weight-only quant for the linear layers
- [ ] Optional: a fused kernel (Triton) for attention or RMSNorm
- [ ] Benchmark memory + throughput tradeoffs

---

# C++ track (`cpp/`)

A second engine, written from scratch in C++, for the low-level / kernel side.
The **Python engine is frozen at stage 2 and serves as the reference oracle** —
every C++ stage is parity-tested against it (and numpy), the same discipline as
Python-vs-HuggingFace. Target model is the same: **Qwen2.5-0.5B**.

The Python roadmap (stages 3–4 above) is the *serving* layer (vLLM direction);
the C++ track is the *compute/kernel* layer (llama.cpp direction). They are two
layers, not a fork — they merge in the fusion stages below.

Build sequence: **pure C++ core first** (no Python in the hot path), then expose
it via pybind11 and rebuild the serving layer in Python on top of our own kernels
— the vLLM shape (Python orchestration + C++ kernels). Keep the C++ core's public
API simple/C-style so the later binding is cheap.

## Pure C++ core
- [x] **C0** — Tensor + ops (matmul/rmsnorm/softmax/add), CMake, numpy parity
- [x] **C1** — Qwen2.5 forward pass; NIT0 weight export, logit parity vs nanoinfer (~4e-5)
- [ ] **C2** — sampling (temperature / top-k / top-p / repetition penalty)
- [ ] **C3** — KV cache (prefill / decode split)
- [ ] **C4** — quantization (Q8/Q4 weight-only, dequant on the fly) ← new math, derive first
- [ ] **C5** — SIMD (AVX2/NEON) + multithreading (OpenMP) ← the performance well

## Fusion (→ mini-vLLM)
- [ ] **F6** — pybind11: expose the C++ core to Python
- [ ] **F7** — Python serving layer: scheduler + continuous batching over C++ kernels
- [ ] **F8** — paged attention (C++ kernel + Python block scheduler) — the merge point

## Reference reading
- PagedAttention / vLLM paper — KV memory management
- RadixAttention / SGLang — prefix sharing
- FlashAttention — IO-aware exact attention
- The annotated Llama / nanoGPT — minimal architectures
- Karpathy's llama2.c — single-file C Llama2 inference (blueprint for C0–C2)
- ggml / llama.cpp — CPU kernels, quantization formats (GGUF)
