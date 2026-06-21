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
  - [x] **G5c** — prefill tiled GEMM: a thread block stages BM×BK / BK×BN tiles of x/w into
        shared memory, each thread computes a TM×TN (4×4) register micro-tile (config
        64×64×8), ragged m/n bounds-checked. Prefill matmul 3% → 20-68% of cuBLAS (gate/up
        10.9 TFLOP/s = 68%); end-to-end prefill 493 → 2192 tok/s (4.4×, A/B vs naive in
        run_cuda_decode_bench). Verified vs the CPU oracle incl. ragged m (test_cuda),
        golden tokens unchanged. Further gains pursued in G5c+ below.
  - [x] **G5c+** — vectorize the tiled GEMM (Boehm "kernel 6"): float4 (128-bit) global
        loads/stores, x staged TRANSPOSED in shared memory so the inner-loop register reads are
        float4 too (and bank-conflict-light), and a size-aware tile dispatch — the huge lm_head
        (n=151936) takes a 128×128 / 8×8 tile (max reuse, still launches >1000 blocks), the
        narrower projections a 64×64 / 4×4 tile, since at m=128 a 128-wide tile launches only
        n/128 blocks (n=896 → 7) and starves the 56-SM GPU. Prefill matmul vs cuBLAS: gate/up
        61% → **89%** (13.1 TFLOP/s — the dominant matmul, past the ~85% target), lm_head
        26% → 51%, q/o 36% → 48%, down 22% → 28%; end-to-end prefill 1915 → 2330 tok/s (1.22×
        over the G5c scalar tile, 4.8× over naive — A/B in run_cuda_decode_bench). Bit-identical
        to the scalar tiled kernel (a given output still sums its products in ascending-k order,
        tile shape aside), so golden tokens are unchanged; both tiles parity-tested vs the CPU
        oracle incl. ragged m (test_cuda, large-n cases added). Second lever — **warp-tiling**
        (Boehm "kernel 10") for the big lm_head matmul: a warp tile sits between the block and
        per-thread tiles, so each warp's shared-memory slice feeds WMITER·WNITER register subtiles
        (more FMAs per smem word — the ratio a compute-bound GEMM lives on). lm_head 51% → **90%**
        of cuBLAS (13.9 TFLOP/s), still bit-identical (test_cuda n=8192 cases). But end-to-end
        prefill stays ~flat: lm_head is one op of ~360, and at prefill=128 the step is only ~15ms
        of matmul vs ~55ms total — the naive G2 attention (×24) + ~360 per-op launches dominate now
        (same Amdahl shape as G5b's decode). The narrow projections can't take warp-tiling (they're
        occupancy-bound at m=128, not reuse-bound) and gate/up is already ~89%, so lm_head is its
        only home. Still ahead: cp.async double-buffering (another lm_head-local micro-gain); the
        real end-to-end prefill lever is now attention (FlashAttention — G5's named "second" kernel)
        and per-op launch overhead.
  - [x] **G5d** — low precision (fp16 + int8 W8A8; lm_head int8 deferred to backlog). **fp16 wmma landed** (linear_wmma_kernel, opt-in
        g_cuda_use_wmma): a 64×64 / 2×2-warp tensor-core kernel, correct (fp16 cost ~1-2% vs
        the oracle — test_cuda) but the naive version is SLOWER than the tuned fp32 tiled GEMM
        (gate/up 7.9 vs 10.9 TFLOP/s). The lesson: tensor-core FLOPS are huge (~142 TFLOP/s on
        Ada), but a kernel that reads fp32 + converts per tile, with a small tile and no
        double-buffering, starves them (~5% utilization). To win: fp16 weight STORAGE (½ the
        DRAM bytes + no convert), 128×128 tiles, cp.async double-buffering. **fp16 weight
        storage then landed** (DType on Tensor + to_device_f16 + g_cuda_fp16_weights, layer
        projections only): end-to-end greedy still MATCHES golden (logits 3.6e-5 vs fp32 — real
        weights are small, so fp16 is ~free) and it halves the layer-weight device memory. But
        it did NOT make wmma win (wmma-h ≈ wmma-fp32 ≈ 2.6 < tiled 4.1 TFLOP/s — the kernel is
        FEEDING-bound, not byte-bound) and decode is only ~1.07× (overhead-bound). Real value:
        ~free memory + the dtype path int8 builds on. **fp16 embed/lm_head then landed**
        (is_fp16_weight extends the opt-in to embed_tokens — the single largest weight, ~544 MB,
        tied as the lm_head; a templated embedding gather reads the fp16 table, and the tied
        lm_head linear already took F16): greedy still MATCHES golden with the logits themselves
        produced in fp16 (max|diff| 3.6e-5, ~unchanged from the fp32-GPU 3.5e-5 — fp16 lm_head is
        ~free even feeding argmax), and total device weight storage halves 1979→991 MB (2.00×,
        run_cuda_parity). **128² warp-tiled wmma then landed** for lm_head
        (linear_wmma_tiled_kernel, n≥8192): 8 wmma frags/warp over a 128×128 tile, >1000 blocks so
        cross-block occupancy hides the load-sync stall. The decisive finding is that wmma only wins
        with fp16 STORAGE — lm_head fp16 128² hits 17.6k GFLOP/s (1.32× the fp32 warp-tiled 13.3k,
        105% of cuBLAS sgemm), while the fp32-weight wmma path (4-byte read + per-tile convert) is a
        LOSER at 11.2k < fp32-tiled. BUT end-to-end fp16 prefill still REGRESSES (3460→2529 tok/s,
        run_cuda_decode_bench): the layer projections run the 64² wmma-h kernel, which loses to the
        fp32 tiled GEMM (q/o 2610<5425, down 3062<5271 — small n + wmma overhead), and they are 168
        of the 169 prefill matmuls. Decode wins (gemv-h, ½ bytes: 83→101 tok/s, 1.22×) and memory is
        2×. **fp16 projection tiled-h then landed**
        (linear_tiled_vec_kernel templated on weight dtype + a load4 helper): the projections route to
        the CUDA-core float4 tiled GEMM reading ½ the weight bytes (convert to fp32 in-register, fp32
        compute), NOT wmma — wmma's fragment overhead loses at small n. tiled-h lifts the projections
        from ~0.5× (wmma-h) to ~0.95× of fp32 tiled (they're compute-bound, so ½ bytes doesn't speed
        them and the convert costs ~5%), but that stops swamping the lm_head wmma win, so end-to-end
        fp16 prefill now BEATS fp32: 3253→3424 tok/s (1.05×, run_cuda_decode_bench). fp16 weights now
        win on every axis — memory 2×, decode 1.21× (gemv-h), prefill 1.05× — golden tokens MATCH,
        tiled-h error 8e-3 (fp16-weight-only + fp32 compute, tighter than wmma's fp16-accumulate). The
        fp16 sub-track is done. **int8 W8A8 then began** — the FLOP-reduction (compute) lever, vs
        fp16's byte-reduction. **CPU W8A8 oracle landed** (QuantMode::W8A8 + linear_w8a8 +
        simd::dot_qq): the C4 int8 weight (a QTensor) plus dynamic per-row int8 activations, an
        int8×int8→int32 integer dot (exact — AVX2 madd == scalar == a GPU DP4A int32 accumulate),
        dual-scale dequant. run_quant w8a8 on Qwen2.5-0.5B: same 2.18× memory as Q8 (same weight),
        logit err 11.2 vs Q8's 3.8 (the added activation quant), but next-token preserved and 11/12
        greedy == Q8 — accuracy is fine; the point is the compute. **GPU DP4A int8 GEMM then landed**
        (cuda_linear_w8a8: device per-row activation quant + a 64² __dp4a tile, int32 accumulate,
        dual-scale dequant; DType::I8 device buffers): parity vs the CPU oracle is max|diff| 3.8e-6 —
        the integer core is identical (DP4A int32 == dot_qq), only the float dequant drifts. The
        compute win fp16 lacked — int8 beats the float4 fp32 tiled on the compute-bound matmuls:
        gate/up 1.11×, down 1.14×, lm_head 1.36× (run_cuda_bench m=128); the tiny q/o,k/v lose 0.88×
        (basic tile + the activation-quant pass dominate at small n). Even a basic DP4A tile beating
        the tuned fp32 shows the 4:1-MAC lever. **W8A8 then wired into the GPU model forward**
        (model.cpp routes CUDA+W8A8 layer projections to a device CudaW8A8Weight in qweights_, so
        Model::project drives int8 DP4A through the same QuantizedWeight interface — no forward
        change; embedding/lm_head/norms stay fp32): run_cuda_parity runs W8A8 end-to-end on the GPU —
        greedy differs from fp32 at 1/12, next-token preserved, weights 907 MB — IDENTICAL to the CPU
        W8A8 oracle (argmax is robust to the float dequant). **G5d done**: fp16 (memory 2× + decode
        1.21× + prefill 1.05×) and int8 W8A8 (the compute win on the compute-bound projections fp16
        could only tie) both land, CPU-oracle-locked at every step. The one open low-precision item —
        int8-quantizing the embedding/lm_head (the biggest weight) — is in the backlog: it feeds
        argmax directly, so it needs a token guard, and fp16 already halves it ~losslessly.
  - [x] **G5e** — attention, the GEMM's successor on the critical path. Once G5c+ made the
        matmuls fast, a prefill=128 step was only ~15ms of matmul out of ~55ms — the naive G2
        attention dominated. It ran one THREAD per (head,query): just H·sq = 1792 threads (~3% of
        the GPU), each grinding a serial key loop. The fix is **warp-per-query**: a whole warp per
        (head,query), its 32 lanes striding the keys — 32× the threads, 32× less serial work each.
        Same two-pass max-subtract as the CPU op (pass 1 warp-reduces the global max — bit-exact,
        max doesn't round; pass 2 re-scores and warp-reduces e and e·V), so only the per-key sums
        reorder: max|diff| ~1e-7 vs the oracle (test_cuda, incl. sq>32 and decode), golden tokens
        unchanged. Isolated A/B (NI_NAIVE_ATTN, GEMM held fixed): **prefill 2395 → 3597 tok/s
        (1.50×), decode 38.7 → 86.9 tok/s (2.25×)**. The decode win is the big one — the naive
        kernel walked the whole growing KV serially per layer, so it, not the GEMV, was the decode
        bottleneck (which is why G5's "attention matters once long context / large batch make it"
        undersold it: under-parallelism bit immediately at ctx 128). Still ahead, the actual
        FlashAttention levers (for long context, where K/V outgrow L2): online softmax (one pass,
        no K re-read) + shared-mem K/V tiling. The paged attention kernel got the same
        warp-per-query treatment (paged stays bit-identical to contiguous, run_cuda_paged
        max|diff|=0, golden MATCH), so both KV paths are now warp-parallel.

## Cross-platform (portability proof, after the GPU is learned)
- [x] **NEON** — `simd.hpp`'s `#elif` path now carries the three inner products on aarch64
      (Apple M-series), so `CpuBackend` runs on Apple ARM. `dot_f32`/`dot_qf32` widen each
      float to double and `vfmaq_f64` (the float64x2 NEON is aarch64-only) — the same
      double-accum as AVX2, so the cast back to float lands on the scalar value; `dot_qq`
      does `vmull_s8` + `vpadalq_s16` into an int32 accumulator — integer-exact, so it
      equals the scalar loop AND the AVX2 madd AND the GPU DP4A (the W8A8 integer core is
      one number on every backend). Cross-compiled with a checked-in toolchain file
      (`cmake/aarch64-linux-gnu.cmake`) and run under qemu-user: test_simd (the dot helpers
      vs a scalar double/int32 reference across lengths 0..40 — every SIMD tail), test_ops,
      test_quant (Q8 + W8A8 through the NEON dots), and ops_parity (linear/rmsnorm/softmax/
      attention == numpy) all pass with `simd target: neon`; the whole `nicore` core also
      cross-builds for ARM64. The x86 AVX2/scalar paths are untouched (native test_simd/
      test_ops/test_quant still green, `avx2+fma`). **Honest scope:** qemu proves
      CORRECTNESS (NEON == the scalar oracle), not speed — it emulates the ISA, not M4
      timing, so tok/s and the weight-level run_parity wait for the real M4. Op-level
      parity is the right-sized gate for a change to the inner-product primitives (it's the
      same gate that validated the C5 AVX2 SIMD).
- [ ] **Metal** — a `MetalBackend` on the M4 GPU; unified memory removes most H2D. The
      same Python serving layer on a third backend = the Backend boundary proven real.

## Backlog (pull in when the moment fits)
Open candidates, not a closed/deferred-forever list — fold one into a stage when it's
the right moment:
- int8×int8→int32 GEMM — DONE (G5d W8A8: CPU oracle linear_w8a8 + simd::dot_qq, GPU cuda_linear_w8a8
  DP4A, model-integrated; the compute win — gate/up 1.11×, down 1.14×, lm_head 1.36×).
- int8-quantize embedding / lm_head — **DONE** (CPU + GPU; weight-only int8, the biggest single
  weight). A per-vocab-row Q8 of the tied embed_tokens, shared by the gather (embedding_q8 /
  cuda_embedding_q8: dequant the looked-up row) and the lm_head (linear_q8 / cuda_linear_q8) — fp32
  activations into argmax, the lowest-risk place for int8 next to a token decision. Opt-in
  `g_quantize_embed` (orthogonal to the layer QuantMode + g_cuda_fp16_weights); on CUDA the codes+scale
  are device-resident (a dedicated I8 gather kernel + a tiled weight-only int8 GEMM). GPU == CPU oracle:
  the int8 embed/lm_head ALONE gives the SAME logits max|diff|=0.333 + 11/12 greedy on both, next-token
  preserved (`run_quant <dir> none embed`, run_cuda_parity); test_cuda parity gather max|diff|=0,
  linear_q8 ~1e-4 vs the CPU oracle. The full int8 GPU model (W8A8 DP4A layers + int8 embed) is 3.97×
  smaller (1979→499 MB), next-token preserved. fp16 storage of the same weight also remains (G5d,
  ~lossless). Open follow-ups: a decode GEMV / warp-tiling for the int8 lm_head (the kernel is a correct
  tiled GEMM, not yet speed-tuned), and a W8A8 lm_head (the compute win) once the memory win is banked.
- batched sampling — the token draw is still per-sequence in Python; fits the G4/G5 decode path.
- SIMD nibble-unpack for q4/q4g — helps compute-bound q4 prefill.
- NEON `simd.hpp` `#elif` path — DONE (the Cross-platform NEON stage above: NEON
  dot_f32/dot_qf32/dot_qq, cross-compiled + qemu-parity-tested vs the scalar oracle).
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
