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

Beyond the fusion the engine goes **multi-backend** behind a `Backend` abstraction
(`backend.hpp`) over a device-aware `Tensor` (ggml-style — one Tensor type, one model
forward, a `Device` tag picks where it runs). `CpuBackend` wraps the existing ops; a
`CudaBackend` (RTX 4070S) and `MetalBackend` / NEON (Mac M4) slot in behind the same
interface, so the same Python serving layer runs on CPU, CUDA, or Metal. The CPU
backend (HF-parity-locked) is the oracle for the accelerator backends. See the
**Multi-backend (G-track)** stages below.

## Pure C++ core
- [x] **C0** — Tensor + ops (matmul/rmsnorm/softmax/add), CMake, numpy parity
- [x] **C1** — Qwen2.5 forward pass; NIT0 weight export, logit parity vs nanoinfer (~4e-5)
- [x] **C2** — sampling + generate loop; greedy generation matches nanoinfer token-for-token
- [x] **C3** — KV cache (prefill / decode); bit-identical to full recompute, ~7× faster
- [x] **C4** — weight quantization (weight-only): per-channel int8 (q8) & int4 (q4),
      group-wise int4 (q4g, Q4_0-style 32-blocks — recovers accuracy vs per-channel),
      behind a QuantizedWeight interface. (Optional: quantize embedding/lm_head too.)
- [x] **C5** — SIMD (AVX2/FMA) + multithreading (OpenMP) + once-per-call weight
      streaming. Double-accum dot products (`simd.hpp`, scalar fallback), output-channel/
      head parallelism, and a row-inner loop so each weight is streamed once not per-row —
      all parity-preserving (logits unchanged: 4.24e-5 vs nanoinfer, greedy
      token-for-token). ~16× prefill, ~2.1× decode (memory-bound) on 20 cores;
      `run_bench` reports tok/s. NEON is a labeled `#elif` extension point. ← the
      performance well

## Fusion (→ mini-vLLM)
- [x] **F6** — pybind11: expose the C++ core to Python. The `nicpp` module binds
      Model / KVCache / SamplingParams / Config / QuantMode; `forward()` returns
      logits as a numpy array and `generate()` returns token ids — the heavy calls
      drop the GIL. Parity-tested *through the binding* (`tests/run_binding.py`:
      logits 4.24e-5 vs nanoinfer, greedy token-for-token, cached==uncached) and a
      text demo (`tools/generate.py`: HF tokenize → our C++ kernels → HF decode) —
      the first end-to-end fusion, the vLLM shape in miniature. These are the exact
      primitives F7 builds a scheduler on (per-sequence KVCache + forward).
- [x] **F7** — Python serving layer: a continuous-batching `Scheduler`
      (`cpp/python/scheduler.py`) over the F6 kernels. Each request gets its own KV
      cache; each step decodes the running set by one token, evicts the finished, and
      admits queued requests into the freed slots (iteration-level batching — no
      head-of-line blocking, evict+admit overlap in one step). Token selection is in
      Python (numpy), same warpers/order as nanoinfer/sampling.py. Tested
      (`tests/run_serve.py`): interleaved greedy output is token-identical to
      standalone generate at every batch size, incl. a repetition-penalty request
      (per-sequence context tracked independently). Honest scope: the kernels are
      still batch-1 (one forward per sequence per step), so this is the *scheduling*
      win (latency/utilization), not yet batched-GEMM throughput — a batched C++
      forward slots in under the same scheduler (pairs with F8).
- [x] **F8a** — batched decode kernel: `Model::forward_batch` runs N sequences'
      one-token decode in a single pass — the per-sequence projection GEMMs fuse over
      the N rows (each weight streamed once, reused across tokens; the same lever that
      makes prefill compute-bound), while attention stays a per-sequence loop (each
      token attends only its own cache). Bit-identical to N standalone forwards
      (`tests/run_batch.cpp`, `max|diff|=0`), wired under the F7 scheduler
      (`batched=True`, parity re-checked in `tests/run_serve.py`); ~2.5× aggregate
      decode tok/s at batch 16 on 20 cores — the throughput lever F7 was missing.
- [x] **F8b** — paged attention: a shared `BlockPool` of fixed-size KV blocks + a
      per-sequence block table (`PagedKVCache`), behind a `KVCacheBase::attend()`
      interface so `forward`/`forward_batch` drive contiguous or paged caches through
      one pointer. The paged `attend()` is a true paged-attention kernel — it indexes
      K/V straight from the blocks via the block table, folding GQA into the read, so
      there is no contiguous gather and no repeat_kv expansion. Mirrors the attention
      op's arithmetic exactly, so it stays bit-identical to the contiguous cache
      (`tests/run_paged.cpp`, `max|diff|=0`, single + batched) while skipping the
      per-step copies — ~1.5× decode at context 128. The Python block scheduler
      (`Scheduler(block_size=...)`) gives each sequence a PagedKVCache, gates admission
      on KV blocks (conservative worst-case reservation), and frees blocks on finish
      for reuse — token-identical to standalone (`tests/run_serve.py`), with no
      per-sequence max_seq preallocation. The vLLM merge point.
- [x] **F8c** — prefix sharing (RadixAttention): KV blocks are reference-counted, and
      a `PrefixCache` (`python/scheduler.py`) keyed by the block-aligned token prefix
      lets a request reuse a cached prefix's blocks (`PagedKVCache.share_prefix`)
      instead of recomputing them — prefilling only its suffix
      (`Scheduler(prefix_sharing=True)`). The shared KV is bit-identical to recomputing
      it (causal: a token's KV depends only on its prefix), so output stays
      token-identical to standalone (`tests/run_serve.py`, `tests/run_paged.cpp`); the
      win is skipped prefill + shared blocks. A shared block frees only when no holder
      (sequence or cache) remains.

## Multi-backend (G-track → CPU / CUDA / Metal)

A `Backend` abstraction lets one model forward run on CPU, CUDA, or Metal. The CPU
backend is the parity oracle; GPU reductions reorder float adds, so the bar becomes
**CUDA ≈ CPU within ~1e-3..1e-4 + golden tokens**, not bit-identical (see CLAUDE.md).
Direction set 2026-06-17: learn the GPU deeply first (4070S), then cross-platform (M4).
The Python serving layer (`cpp/python/scheduler.py`) and the oracle (`nanoinfer/`) stay.

- [x] **G0** — Backend seam + device-aware Tensor. `backend.hpp`/`backend.cpp`
      (`Backend` + `CpuBackend` wrapping the free ops), a `Device` tag on `Tensor`,
      and `Model` routing every op through `backend_`. Pure refactor: CPU stays
      bit-identical (logits 4.24e-5 vs nanoinfer, every self-parity `max|diff|=0`,
      binding + scheduler MATCH). `-DNI_CUDA` CMake option (OFF) reserved.
- [x] **G1** — CudaBackend skeleton: device alloc/free, H2D/D2H, a naive `__global__`
      GEMM; cuBLAS linked as a yardstick / cross-check. Single linear ≈ CpuBackend (~1e-3).
- [x] **G2** — full forward on GPU; every weight uploaded once and resident on device.
      `run_cuda_parity.cpp`: logits ≈ CPU backend + golden greedy tokens.
- [x] **G3** — KV cache + single-stream decode on GPU (inject the Backend into the
      cache, deferred from G0). cached == uncached within tolerance.
- [x] **G4** — batched decode + a paged-attention CUDA kernel; drive the existing
      Python scheduler with the CUDA-backed Model (G4a batched, G4b paged kernel, G4c
      contiguous scheduler, G4d paged + prefix-sharing). Throughput tok/s × batch.
- [ ] **G5** — optimize the GEMM, closing the gap to cuBLAS. `linear` dominates the
      forward (168 projection matmuls + the 896×151936 lm_head), so it's the kernel to
      sharpen first; `attention` (FlashAttention-style: online softmax, shared-mem K/V
      tiles, warp-per-query) is second, once long context / large batch make it matter.
      The naive kernel is one-thread-per-output: ~0.25 FLOP/byte intensity vs a
      ~70 FLOP/byte roofline ridge (4070S), uncoalesced stride-k weight reads, and a
      `cudaDeviceSynchronize` after every launch. Staged:
  - [x] **G5a** — measurement scaffold (`run_cuda_bench`): time linear() against a cuBLAS
        sgemm baseline (newly linked — G1 never actually wired it in) with CUDA events on
        the real shapes. Found the naive kernel at 3-12% of cuBLAS on prefill, 14-50% of
        bandwidth on decode; per-op alloc is only 0-4% (and cudaMalloc already syncs the
        device), so the sync/alloc plumbing is NOT the bottleneck — deferred until the
        kernels are fast enough to expose it (re-measure with the scaffold then).
  - [x] **G5b** — decode GEMV: one warp per output channel, coalesced strided weight reads,
        `__shfl` reduction; m>16 (prefill) stays on the naive kernel. Decode's big ops jump
        to 59-86% of bandwidth — lm_head 217 GFLOP/s = 86% BW / 97% of cuBLAS — and batched
        decode (m=16) ~2.2×. Bit-identical row-wise (run_cuda_batch max|diff|=0), golden
        tokens unchanged. float4 loads unneeded (already near the BW wall). End-to-end
        (`run_cuda_decode_bench`, A/B vs forced-naive) the decode win is only ~1.2×:
        Amdahl-capped, because the matmul is now a minority of the decode step — per-op
        CUDA overhead (cudaMalloc + cudaDeviceSynchronize × ~360 ops/token) and the
        contiguous cache's growing per-step copy dominate. That plumbing (parked in G5a) is
        now the decode bottleneck — re-measure once it's pooled.
  - [x] **G5-pool** — the decode-overhead pull-in (run_cuda_decode_bench pointed here, not
        G5c): a caching device-memory pool (reuse freed buffers — no per-op cudaMalloc) plus
        dropping the per-launch cudaDeviceSynchronize (the one default stream already orders
        kernels). Decode 21→46 tok/s at 128 ctx (2.15×, bigger than the kernel's own 1.2×);
        prefill ~1.5× too; every golden / bit-identical gate unchanged. The remaining decode
        drag at long context is the contiguous cache's O(ctx) cat_seq copy — paged territory.
  - [ ] **G5c** — prefill GEMM (compute-bound, m=seq): shared-memory tiling → register
        blocking (TM×TN micro-tiles) → double-buffered float4 — lifts intensity over the
        roofline ridge. The "close the gap to cuBLAS" arc.
  - [ ] **G5d** — low precision (stretch): fp16 tensor cores (wmma) → an int8×int8→int32
        tensor-core GEMM wired to the C4 quantized weights → quantize lm_head. The lever
        past the fp32 decode bandwidth wall (fewer bytes streamed).

## Cross-platform (portability proof, after the GPU is learned)
- [ ] **NEON** — fill `simd.hpp`'s `#elif` NEON path so `CpuBackend` runs on Apple ARM
      (M4 on CPU — the cheapest path to running on the Mac).
- [ ] **Metal** — a `MetalBackend` on the M4 GPU; unified memory removes most H2D. The
      same Python serving layer on a third backend = the Backend boundary proven real.

## Backlog (pull in when the moment fits)
Open candidates, not a closed/deferred-forever list — fold one into a stage when it's
the right moment:
- int8×int8→int32 GEMM — C4 quant is weight-only today (saves memory, not compute); fits G5d.
- quantize embedding / lm_head — the biggest single weight; fits G5d.
- batched sampling — the token draw is still per-sequence in Python; fits the G4/G5 decode path.
- SIMD nibble-unpack for q4/q4g — helps compute-bound q4 prefill.
- NEON `simd.hpp` `#elif` path — the cross-platform leg above.
- float32-accumulation in the SIMD dot — **has a real cost**: it trades away the
  bit-for-bit CPU parity oracle (the project's correctness spine). Only as a conscious
  tolerance change, never silent.

## Reference reading
- PagedAttention / vLLM paper — KV memory management
- RadixAttention / SGLang — prefix sharing
- FlashAttention — IO-aware exact attention
- The annotated Llama / nanoGPT — minimal architectures
- Karpathy's llama2.c — single-file C Llama2 inference (blueprint for C0–C2)
- ggml / llama.cpp — CPU kernels, quantization formats (GGUF)
