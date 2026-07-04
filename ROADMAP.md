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
        only home. Double-buffering (the "cp.async" micro-gain) was then explored in G5h below — a TIE on
        this model (the projections already hide the load latency at m=128), confirming this call: the real
        end-to-end prefill lever is now attention (FlashAttention — G5's named "second" kernel) and per-op
        launch overhead, not the GEMM.
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
  - [x] **G5f** — FlashAttention (IO-aware): the two levers G5e named "still ahead" — **online
        softmax (one pass)** + **shared-mem K/V tiling**. Honest scope first: for Qwen2.5-0.5B the
        per-layer KV is ~130 KB at ctx 128 and ~8 MB even at ctx 8k, both inside the 4070S's large
        L2, so FlashAttention's textbook HBM→SRAM win is *muted* (L2 already catches the reuse). The
        clean measurable win here is the one-pass online softmax; tiling is the structurally-correct
        FlashAttention that matters more as context/batch grow and on smaller-L2 GPUs.
    - [x] **G5f-online** — one-pass online softmax in the warp-per-query kernel. G5e read each key
          TWICE (pass 1 for the global max, pass 2 to re-score + accumulate). Now each lane keeps a
          running (m, l, acc) over its strided keys and rescales acc/l by exp(m_old−m_new) when the
          max moves; the 32 lane-partials then merge by the associative FlashAttention combine. Same
          D-vector op count as two-pass but the score dot is computed ONCE → **K read once not twice**,
          the win growing with context (decode walks the whole KV each step). Sharp edge found & fixed:
          a query with limit<32 leaves lanes limit..31 empty (m=−inf), and the both-empty combine does
          exp(−inf−(−inf))=exp(NaN)=NaN — which poisoned the model's logits (every prefill has a
          query 0 with limit 1). The isolated attention test *passed* anyway because its max-diff used
          std::max(x,NaN)=x, silently swallowing the NaN; fixed the combine (empty segment ⇒ correction
          0) AND made the test NaN-aware (+ a tiny-causal case). Parity restored: test_cuda attn
          ~1e-7, run_cuda_parity logits 3.5e-5→3.77e-5 (the extra online rescaling, golden tokens
          MATCH), run_cuda_paged still max|diff|=0 vs contiguous (the paged kernel runs the same body
          verbatim), run_cuda_cache / run_cuda_batch max|diff|=0. Decode A/B (real GEMM held fixed, vs
          the recorded G5e two-pass): **77.2 vs 70.1 tok/s @ctx128 (1.10×), 56.4 vs 48.4 @ctx512
          (1.17×)** — and 7× over naive attention at ctx 1024 (45.2 vs 6.5). The online body is also
          the *prerequisite* for tiling: you can only stream K in blocks because the running max is
          rescaled as later blocks arrive. The isolated kernel (run_cuda_attn_bench, no full-step
          dilution) shows online's true size vs naive: **4.9× @sq128, 2.6× @sq512, and 20× at decode**
          (sq=1, sk=4096: 1.68 vs 34.7 ms — the naive kernel walks the whole KV with H=14 threads).
    - [x] **G5f-tiled** — shared-memory K/V tiling (the IO-aware lever), opt-in (`g_cuda_use_tiled_attn`,
          default OFF). A thread block handles one head + a tile of 8 queries and stages each 32-key K/V
          slab into shared memory ONCE, so the block's 8 queries read it from SRAM (8× fewer global
          reads). Designed to stay parity-clean: with tile width 32 and lane l ↔ key (base+l), every
          lane processes the same keys in the same order as the non-tiled kernel, so the tiled output is
          **bit-identical** (run_cuda_attn_bench tile==notile max|diff|=0 at every shape; run_cuda_paged
          stays max|diff|=0 with only the contiguous path tiled — it equals the non-tiled paged kernel
          bit-for-bit; run_cuda_parity logits unchanged to the last digit, 3.76701e-05). **Honest
          result: tiling does NOT win on this model.** Isolated A/B (run_cuda_attn_bench): tiled/notile =
          0.89× @sq128 (the smem staging + 2 syncs/tile LOSE when reuse is small), 1.00–1.01× at sq
          512–2048 — a tie. Cause, as predicted: Qwen2.5-0.5B's per-layer KV is ~130 KB at ctx 128 and
          ~8 MB even at ctx 8k, both inside the 4070S's ~48 MB L2, so the global→smem reuse the tiling
          buys is ALREADY served by L2 (the textbook HBM→SRAM win needs K/V to outgrow L2 — bigger
          model/batch, longer context, or a smaller-L2 GPU). So it's kept opt-in as the correct
          FlashAttention structure (same call as G5d keeping the losing wmma path behind a flag), while
          the **non-tiled online kernel is the default** — it carries the real G5f win. The genuine
          long-context decode lever left is Flash-Decoding (split-K over the KV: more blocks per query,
          combine partials) — backlog.
  - [x] **G5g** — Flash-Decoding (split-KV), the long-context DECODE lever G5f named "still ahead". G5e
        gave each (head,query) a warp and G5f-online halved its K reads, but at decode (sq=1) the kernel
        still launches only H·sq = 14 warps — ~3% of the 56-SM GPU — each walking the WHOLE KV serially,
        so at long context attention (not the GEMV) is the decode bottleneck. The win G5f-online couldn't
        add was *parallelism*. Flash-Decoding splits the KV across `num_splits` warps per (head,query):
        **pass 1** (`attention_split_kv_kernel`) runs the SAME one-pass online softmax over each key
        chunk and writes an UNNORMALIZED partial (m, l, acc) to a pooled scratch; **pass 2**
        (`attention_combine_kernel`) merges the splits with the associative FlashAttention rule (M=max
        m_s; rescale l_s,acc_s by exp(m_s−M); divide). H·sq·num_splits warps fill the GPU and each warp's
        serial key-walk shrinks num_splits×. `num_splits` is sized from the device SM count (`split_count`:
        ~8 warps/SM, ≥128 keys/split, and =1 for short context or prefill — where it degrades to the plain
        warp kernel, bit-identical). The chunks tile [0,sk) disjointly and causal clamps each to `limit`,
        so the merge is the exact softmax over [0,limit) — only the reduction ORDER differs (chunked, not
        lane-strided-over-all), so it is **not bit-identical** to the non-split kernel but matches the CPU
        oracle to ~1e-8 (test_cuda sk=1024/4096), and split-on == split-off greedy over a 515-token
        context (run_cuda_cache, the decode-path token gate). Opt-in (`g_cuda_use_split_attn`,
        `NI_SPLIT_ATTN`), default OFF — run_cuda_parity unchanged. Isolated kernel A/B
        (run_cuda_attn_bench): decode **6.8× @ctx1024, 23× @ctx4096, 13× @ctx16384** over the non-split
        online kernel; prefill ties (split_count=1 → no-op, bit-identical — a built-in control). End-to-end
        decode (run_cuda_decode_bench): **1.30× @ctx~512, 2.15× @ctx~2048** — the win grows with context as
        attention's share of the step grows, but is Amdahl-diluted by the contiguous cache's O(ctx) cat_seq
        copy + ~360 per-op launches (paged territory). **Paged path also done** (`paged_attention_split_kv_kernel`,
        sharing the same `attention_combine_kernel`): bit-identical to the contiguous split path
        (run_cuda_paged max|diff|=0, ctx 515), so paging stays pure indexing under split too. The paged cache
        has no cat_seq copy, so it's where the win lands undiluted — at ctx~2048 paged+split reaches ~81 tok/s
        (the fastest of contiguous/paged × split-off/on; split ~2.8× on paged vs ~2.4× on contiguous), and at
        ctx 4096 the contiguous cache OOMs (its non-reusing cat_seq buffers) while paged+split runs at ~74 tok/s.
  - [x] **G5h** — double-buffered projection GEMM, the "cp.async double-buffering" micro-gain G5c+ named
        "still ahead". `linear_tiled_db_kernel` software-pipelines `linear_tiled_vec_kernel`: two smem stages
        ping-pong, the loop prefetches K-tile kt+1 into registers (LDGs issued early) while it computes tile
        kt, hiding the ~400-cycle DRAM latency a single-buffered tile eats at each `__syncthreads`. Aimed at
        the LOW-OCCUPANCY projections — down (n=896,k=4864) and q/o launch only 28 blocks on the 56-SM 4070S,
        so there's no second resident block to hide the stall behind (the lm_head's >1000 blocks and gate/up's
        152 already do, which is why it's scoped to the narrow-n path, not lm_head). **Deliberately NOT
        cp.async**: cp.async (LDGSTS) copies a CONTIGUOUS gmem chunk to a CONTIGUOUS smem chunk, but this
        kernel stages As/Bs TRANSPOSED ([BK][BM]) so the inner reads are conflict-free float4 (the G5c+ lever);
        a cp.async version needs the natural [BM][BK] layout, whose inner reads stride by BK·TN ≡ 0 (mod 32
        banks) → a 16-way bank conflict — reconciling the two needs smem swizzling, out of scope for a
        micro-gain. Register-prefetch hits the SAME load/compute overlap while keeping the transpose, so it's
        **BIT-IDENTICAL** to the tiled kernel (only the load TIMING moves): run_cuda_bench A/B `dbuf-tiled`
        max|diff|=0 on every projection shape, and the full 24-layer forward gives logits identical to the last
        printed digit with dbuf on vs off (run_cuda_parity NI_DBUF=1, 24-token prompt → m>16 fires it, golden
        MATCH). **Honest result: a TIE.** With robust min-of-8 interleaved timing (peak-clock, killing the
        boost-clock drift that made a naive 30-iter A/B swing 0.86×–1.19×), every projection shape is within
        1–2% (q/o 1.01×, k/v 1.02×, gate/up 1.00×, down 1.02×) — at m=128 these microsecond ops already hide
        the load latency via ILP + warp occupancy, so the explicit double-buffer's overlap buys nothing and its
        2× smem just trades occupancy back. Same shape as G5f-tiled: the structurally-correct latency-hiding
        lever, kept opt-in (`g_cuda_use_dbuf`, default OFF) for when the projections are big enough / occupancy-
        starved enough that the global load is genuinely exposed. End-to-end it's Amdahl-nil (down is ~20% of
        prefill matmul, matmul ~30% of the step), confirming G5c+'s call that this is a minor micro-gain and the
        real prefill levers are attention + per-op launch overhead.
- [x] **G6** — CUDA graphs: fold a decode step's ~360 per-op kernel launches into ONE `cudaGraphLaunch`,
      the per-op-launch lever G5 kept naming. **Measurement first (the G5a discipline), and it corrected a
      wrong hypothesis** — the instructive part. nsys on a decode run showed `cudaMemcpy` at 77% of CPU API
      time (the contiguous cache's `cat_seq` + `repeat_kv` per step) and `cudaLaunchKernel` at only 17%, so
      "cat_seq dominates" looked obvious — but the wall-clock A/B (`NI_PAGED`) said paged ≈ contiguous decode
      (84 vs 84 tok/s). The nsys "cudaMemcpy time" was SYNC-WAIT folded into the call, not bandwidth — the
      bench is the oracle, not the profiler's CPU-API attribution. The real picture: 0.5B fp32 decode ~84
      tok/s = ~12 ms/step is ~3.5× over its ~3.4 ms memory-bandwidth floor and cache-INDEPENDENT, i.e.
      bounded by path-independent per-op overhead — exactly what graphs collapse. Built on the PAGED cache
      (contiguous can't be captured: sync `cat_seq` memcpy + growing buffers), staged:
  - [x] **per-thread default stream** (`--default-stream per-thread`) so the implicit `<<<>>>` launches land
        on a CAPTURABLE stream (legacy stream 0 can't be); single-threaded, so ordering/golden unchanged.
  - [x] **device-resident block table** — `CudaPagedKVCache` keeps its block table in a STABLE device buffer
        refreshed incrementally (no-op, no CUDA call, when nothing grew → safe inside a capture), replacing
        the per-call `cudaMalloc`+sync H2D+`cudaFree` every paged attend did. Parity-clean (run_cuda_paged
        max|diff|=0) AND an independent eager win: **paged decode 1.15×** (ctx8 84→97, ctx256 68→78 tok/s) by
        dropping those per-attend sync points.
  - [x] **device-side decode inputs** — the per-step values a single captured graph must span are made
        device-resident: the position/length (read by rope, paged_write, paged_attention via an optional
        `d_pos` pointer) and the token id (the embedding gathers from `g_cuda_graph_token`, skipping the sync
        H2D id-upload a capture can't record). Decode keeps sq=1, so every kernel GRID is fixed — only these
        loop-bound/offset values move, now from device — so ONE graph replays at every length. Eager passes
        nullptr → the host args, bit-identical (run_cuda_paged max|diff|=0, run_cuda_parity golden MATCH).
  - [x] **`CudaGraphDecoder`** (capture/replay) — first decode step runs EAGER (warms the device pool so the
        capture does no `cudaMalloc`); step 2 captures the forward (host bookkeeping runs, kernels only
        record — so its `advance()` is rolled back via `set_length`, `keep_device_logits` leaves the logits on
        device for the driver's own D2H), instantiates, and replays thereafter — each step just writes
        d_pos/d_token, launches, and D2Hs the logits. The pool's capture-time addresses are baked into the
        graph, so the invariant is "no pool alloc during the replay loop" (the driver's per-step work uses
        dedicated buffers, not the pool). Opt-in `NI_GRAPH` in run_cuda_decode_bench.
  - **Honest result: correct + a bounded win, biggest at short context.** Graph greedy tokens are
        bit-identical to eager paged decode at every context (run_cuda_decode_bench `NI_GRAPH=1`, golden ==
        over 256/200 tokens, across many block-boundary crossings), and decode is **1.06–1.07× @ctx≤72, 1.04×
        @ctx≥128** — the launch overhead the graph recovers is the EXPOSED fraction (not already hidden behind
        the async-pipelined GEMV execution), which shrinks as the memory-bound attention/lm_head grow with
        context. Same shape as G5h/G5f-tiled: the structurally-correct lever (every production engine graphs
        the decode step; it wins big in the launch-bound regimes — smaller models, larger batch, more/smaller
        kernels), kept opt-in, honestly bounded on this 0.5B batch-1 model. Still ahead (backlog): batched /
        paged-split graphs (the split grid grows with context — needs the max-split fixed-grid trick), and
        the int8-embed (`NI_QEMBED`) decode under graph (the gather'd need the same device-token path).

## Speculative decoding (S-track) — the two-model track

**Direction updated 2026-07-04: Metal is deferred** (parked below — *not* dropped; the
R-track already paid its structural prerequisite, so it stays a clean pickup later). The
lever now is the freshly-landed **Qwen2.5-1.5B**, which moves the roadmap two independent
ways:

1. **A draft/target pair.** 0.5B (fast, rough) + 1.5B (slow, accurate) share an *identical*
   tokenizer (both `vocab_size` 151936) → a textbook speculative-decoding pair: the biggest
   decode win we haven't built, and one that genuinely *needs* two models.
2. **Big enough to un-mute the 0.5B ties** (P-track). head_dim 64→128, matrices ~3×, KV
   ~2.3×/token — several G-track levers that honestly tied on 0.5B *and named the model as the
   cause* now have real headroom.

They both add tok/s — **S cuts the *number* of target forwards; P makes *each* target forward
faster** — but S2 measured that they **add, they don't multiply**: P speeds the target verify AND
the plain baseline (and the draft) together, so it lifts absolute tok/s without changing S's
speedup *ratio*, which is bound by `r = t_draft/t_target` (S2). Built in the Python serving layer
over the C++ kernels (the F6–F8 shape), CPU-oracle-locked: greedy speculative decode is
**token-identical to plain 1.5B greedy** (the accept rule guarantees it), so the golden-token gate
extends unchanged (1.5B fits the ~1.7B fp32 CPU-oracle RAM ceiling).

- [x] **S0** ✅ landed — draft/target harness + the greedy accept test, CPU-oracle-locked.
      **Key finding: the verify primitive already existed.** `forward([cur, d₀..d_{K-1}], cache)`
      onto a POPULATED cache is bit-identical to K+1 sequential single-token forwards (max|diff|=0,
      `run_spec.py` foundation check) — the attention op's `causal + query_offset` masking already
      handles the `(length>0, t>1)` corner, so "prefill K+1 onto an existing cache" needed ZERO new
      kernel code; it was only the untested *combination* of prefill's multi-query mask and decode's
      offset. Rollback is `KVCache::truncate(L)` (contiguous: move the length pointer — stale slots
      are overwritten before the next read, so truncate+replay is bit-identical, max|diff|=0). The
      greedy loop (`cpp/python/speculative.py`): draft proposes k from cur (k+1 feeds, so the draft
      cache stays lock-step with the target and ONE `truncate(L+a+1)` rolls back BOTH — no
      all-accepted special case), target verifies `[cur, d₀..d_{k-1}]` in one forward, accept the
      longest prefix the target's argmax agrees with, emit accepted + the correction/bonus.
      Token-identical to plain target greedy by construction (every emitted token is the target's
      own argmax; the primitive makes the batched verify logits equal sequential decode). Tested
      (`tests/run_spec.py`): draft==target (0.5B) → 100% accept + output==plain greedy at k∈{1,2,4,8};
      draft=0.5B/target=1.5B → output==plain 1.5B greedy at k∈{2,4,8}, real accept **65%@k=2 (2.4
      tok/verify)** falling to 48%@k=8 (the draft drifts the longer it runs — the k tradeoff S2 tunes).
      No regression (test_cache, run_serve MATCH). Sampling-parity accept (rejection sampling) is the
      open S0 tail, folds into S2. Already efficient on the contiguous cache; S1 promotes `truncate`
      to a tested `KVCacheBase` method + the paged impl (free blocks) + the dedicated rollback gate.
- [x] **S1** ✅ landed — KV cache rollback, promoted to a `truncate()` override on ALL FOUR caches
      (S0 shipped only the CPU-contiguous one, behind a base-class default-throw so any cache missing
      it fails loudly). The verify forward writes K+1 tentative K/V; `truncate(L)` discards the
      rejected tail, and each cache does its own surgery: **CPU-contiguous** (preallocated) just moves
      the length pointer — the stale slots past L are overwritten before the next read; **paged (CPU +
      GPU)** frees every block holding only rejected positions (keep `ceil(L/bs)` — the block
      straddling L is kept, its stale tail overwritten before read), returning them to the pool where
      the next forward re-allocates and reuses them (LIFO free-list → it hands back the just-freed
      tail blocks); **GPU-contiguous** `CudaKVCache` has NO preallocation (it grows by `cat_seq`), so
      the "move a pointer" trick doesn't exist — rollback slices each layer's `[n_kv, len, hd]` device
      history to its first L rows per head (the same per-head device-to-device copy `cat_seq` does for
      its "old" side, so the retained K/V is byte-for-byte unchanged). **Sharp edge (GPU paged):** the
      device block table only ever *appends* (`sync` uploads `[d_bt_count_, size)`), so after freeing
      tail blocks `d_bt_count_` is rolled back to the kept count — else a re-allocated slot handed a
      DIFFERENT physical block id would keep the stale device entry and read wrong K/V. No new kernel
      code and no binding change (`truncate` was already virtual on `KVCacheBase`) — pure cache
      bookkeeping. **Done-when gate** (`run_paged` CPU + `run_cuda_paged` GPU): `truncate(L)` +
      replay-the-accepted-tail is **bit-identical** (max|diff|=0 on every cache, both backends) to a
      cache that only ever decoded the accepted tokens — a tail longer than the accepted continuation
      at block_size 4 forces the free+reuse path, and the pool fully recovers its blocks afterward.
      Regressions green (run_cache, run_cuda_cache, run_cuda_parity golden, run_serve, run_cuda_serve,
      and S0's run_spec — 0.5B/0.5B 100% accept, 0.5B→1.5B output == plain 1.5B greedy).
- [x] **S2** ✅ landed — measured the speedup + swept K on the clock (the G5a "no number until
      measured" discipline), and the honest result is a **wash-to-modest win that the draft/target
      COST RATIO caps, not the accept rate.** Instrumented the accept-length DISTRIBUTION
      (`SpecStats.accept_histogram` — the mean hides a bimodal shape) and built `tests/bench_spec.py`:
      times plain target greedy (the baseline), plain draft greedy (to measure `r = t_draft/t_target`),
      then sweeps K reporting accept%, tok/verify, the accept histogram, and WALL-CLOCK spec tok/s vs a
      roofline prediction — every K correctness-gated (token-identical to plain greedy, the S0
      invariant; `bench_spec: ok` = all matched, across 5 built-in diverse prompts). Measured on the
      4070S, fp32, both models resident (~8 GB), max_tokens 128:
      - **r ≈ 0.45** — the 0.5B draft forward costs ~45% of a 1.5B target forward (only ~2.2× cheaper;
        plain target ~40 tok/s, plain draft ~91). This is the binding constraint: even at 100% accept
        the ceiling is (k+1)/(1+k·r) ≈ 1.4–2.0×, and the target VERIFY forward over k+1 tokens is a
        mini-prefill (costs more than one decode step), pulling realized speedup to ~1.2×.
      - **Accept rate is text-dependent:** repetitive/predictable text (france 82–97%, code 83–97%,
        list 67–95%), reasoning 63–88%, open-ended story ~36%.
      - **Speedup range 0.91×–1.24×** (best K per prompt): code 1.24×, france 1.22×, list 1.13×, reason
        1.11×, **story 0.91× — a NET LOSS** (k=8 → 0.64×): when the draft misses, its wasted forwards
        (0.45× each) + the fatter verify exceed the tokens saved. Predicted (roofline) ran ~0.3–0.4×
        above measured — the gap is the fatter k+1-token verify + Python's k sequential draft calls,
        each D2H-ing a full 151936-vocab logit row (device-side token selection is the lever, S3).
      **Verdict — below the roadmap's *hoped* 1.5–2.5×, and the diagnosis says why:** that target
      assumes a 7B/70B-style gap (r ~0.1), not 0.5B/1.5B (r ~0.45); the 0.5B isn't cheap enough. Two
      honest consequences: **P-track can't rescue the *ratio*** (fp16/int8 speeds the target AND the
      draft together, so tok/s rises but the speedup ratio is ~unchanged — S and P add tok/s
      independently, they don't multiply here), and **S4 (prompt-lookup, r→0) is the real ratio
      lever** — a free draft can't regress and its ceiling is k+1. No universal best K (accept falls
      with k while the verify fattens): k=3–4 balances upside against the low-accept regression.
      Tuning K is a knob; lowering r is the fix.
- **S3** — batched speculative decode + scheduler integration: fold S into the serving layer
      (each sequence carries its own target cache + proposer; the running set is driven together).
      Staged the way this repo has always split scheduling from kernels (F7 → F8a):
  - [x] **S3a** ✅ landed — the **scheduling** integration (the F7-analog), **pure Python over the
        existing kernels, no rebuild.** `SpecScheduler` (`cpp/python/spec_scheduler.py`) mirrors the
        continuous-batching `Scheduler` shape — admit / evict / dynamic batch — around the S0 spec
        loop, but a step emits a VARIABLE number of tokens per sequence (0..k+1), so it can't reuse
        `Scheduler` directly. Each running `SpecSequence` owns its target KV cache AND its proposer (a
        `DraftModelProposer` keeps a per-sequence draft cache lock-step with the target;
        `PromptLookupProposer` is stateless), and the step runs three phases: **propose** (each
        proposer guesses k_s, k_s may be 0 on a lookup miss → that sequence degrades to a plain greedy
        forward), **verify** (S3a: one target `forward([cur_s, d_s..])` PER sequence — the single point
        S3b swaps for a ragged batched forward), **commit** (per sequence: the shared `accept_prefix`
        rule → `truncate` BOTH caches to the confirmed prefix → emit → stop on eos/max_tokens). The
        accept rule was extracted to ONE helper (`speculative.accept_prefix`) shared by the
        single-sequence loop and the scheduler, so the token-identity invariant is proved once; the
        scheduler's `_emit` mirrors `_greedy_spec_core.emit` (eos convention included) exactly, making
        a scheduled sequence provably identical to standalone `greedy_speculative` /
        `greedy_prompt_lookup`. **Done-when gate** (`tests/run_spec_serve.py`, the run_serve.py analog):
        every request's output is token-identical to BOTH standalone spec AND plain greedy (the S0
        invariant), at every batch size (mb 1/2/8) and for every proposer — including a **MIXED
        draft+lookup batch** (the real test of per-sequence proposer independence). Green on 0.5B/0.5B
        (100% accept, all-accepted+bonus path every step) and the real 1.5B/0.5B pair (genuine 36.8%
        accept, peak_batch 6, batch-invariant: verifies + output identical across mb). Honest scope,
        same as F7's: this is the **scheduling** win (no head-of-line blocking, dynamic admit/evict,
        one place to serve draft-model AND prompt-lookup requests together) — the verify is still one
        forward per sequence, so the batched-GEMM **throughput** win is S3b.
  - [x] **S3b** ✅ landed — the **batched ragged verify kernel** (the F8a-analog, C++).
        `Model::forward_spec_batch(tokens, counts, caches)` generalizes `forward_batch` (1 query row
        per sequence) to a RAGGED batch: sequence s contributes `counts[s]=k_s+1` query rows and
        `tokens` is the flat concatenation of every sequence's `[cur_s, d_s..]` (M = Σcounts). The
        projection GEMMs (q/k/v/o/gate/up/down) fuse over ALL M rows — every weight streamed once (the
        F8a lever, now across the whole verify batch, not one token per sequence) — while attention
        stays a per-sequence loop over that sequence's CONTIGUOUS query block: a multi-query causal
        attend at the sequence's cache offset (the S0 verify primitive, one block at a time).
        **The key simplification that kept it small: extract the contiguous `[cnt, width]` block then
        `split_heads` — the SAME op sequence `forward()` runs for one sequence — so NO head transpose is
        needed.** (`extract_row`'s single-row→`[heads,1,dim]` reshape doesn't generalize: a multi-row
        block would transpose heads↔rows, but `split_heads` already does exactly that.) So the only new
        backend ops are `extract_rows`/`place_rows` — contiguous block copies (CPU `memcpy` / one CUDA
        D2D), the block generalization of `extract_row`/`place_row`, `count==1` recovering them. Wired
        under the phase-2 `_verify` seam (`SpecScheduler(batched=True)`, the default; `batched=False`
        keeps the S3a per-seq loop for A/B). **Gate** (`tests/run_spec_serve.py`): `forward_spec_batch`
        is **bit-identical** to the per-sequence forward on the CPU oracle (`max|diff|=0`, sequences at
        different cache lengths + block sizes incl. the k=0 1-row shape — the ragged pos/row_start
        bookkeeping), and every request token-identical to standalone spec at every batch size under
        BOTH backings (0.5B/0.5B 100% accept + the real 1.5B/0.5B pair 36.8%). On CUDA the batched vs
        per-seq row count can pick a different GEMM kernel (tiled vs GEMV) → the bar is the CLAUDE.md
        GPU rule (within ~1e-3 + identical greedy tokens); here M≤16 so both hit GEMV and it's `0.0`
        anyway. **Honest scope:** S3 makes speculative decode *composable with continuous batching* and
        fuses the verify's projection GEMMs — the throughput *lever*. It does NOT change spec's
        per-sequence economics: the realized tok/s win is still bound by S2's r-cap (draft/target cost
        ratio) per sequence; batching amortizes the *weight streaming* across sequences, which pays when
        many sequences verify together (server load), the same win F8a gives plain decode.
  - [x] **S3c** ✅ landed — **paged spec cache** (the F8b analog, pure Python — no rebuild).
        `SpecScheduler(block_size=..., num_blocks=...)` draws each sequence's TARGET cache from one
        shared `BlockPool` (`CudaBlockPool` on GPU) instead of preallocating a contiguous cache to its
        worst-case length, and gates admission on KV blocks (FCFS: peek the head, reserve its
        worst-case, or wait — no starvation). The reservation is the key correctness point: it includes
        the **tentative-verify tail** (`cap = prompt + max_tokens + max_k + 1`), because one
        `forward_spec_batch` writes all N sequences' k_s+1 tentative K/V *before* any `truncate` rolls
        the rejected part back — so all N tentative peaks coexist in the pool at once; reserving each
        sequence's worst-case (≥ its tentative peak) means the sum can't over-commit mid-verify. **No
        new kernel:** `forward_spec_batch` already drives the paged cache through the same polymorphic
        `attend()` (block-indexed write, tested for multi-query since prefill) and `_commit` rolls back
        via the same `truncate()` (paged frees the rejected tail blocks, S1) — so paging is bit-exact
        and output token-identical. **Gate** (`tests/run_spec_serve.py`, the run_serve.py paged analog):
        every request token-identical to standalone spec under a paged target cache, on CPU + CUDA; a
        TIGHT pool (bs=8, nb=8) forces block-aware queueing + reuse (peak_batch 6→2) and every block
        returns to the pool once all sequences finish (`pool_free == num_blocks`, no leak). The draft
        proposer's cache stays contiguous (the smaller 0.5B model; paging it needs a second draft-dim
        pool — the remaining refinement, plus prefix-sharing across spec requests, both mechanical).
- [x] **S4** ✅ landed — draft *without* a second model: **prompt-lookup / n-gram decoding**, the r→0
      ratio lever S2 named as the point (the 0.5B/1.5B pair's r ≈ 0.45 caps it at 1.24×; a free draft
      doesn't). **Same verify + rollback machinery, a different proposer** — realized literally: the S0
      loop is refactored into a proposer-parameterized `_greedy_spec_core`, and `greedy_speculative`
      (draft model) + `greedy_prompt_lookup` (n-gram) are thin wrappers differing ONLY in a `Proposer`
      (`DraftModelProposer` vs `PromptLookupProposer`). The n-gram proposer (`prompt_lookup`) matches
      the last `ngram` tokens of the context against the most-recent earlier occurrence and copies up
      to k tokens that followed — **no model, no draft cache** (so rollback is a no-op; only the
      target's `truncate` runs). A miss returns [], degrading that step to a plain greedy forward([cur]),
      the loop's k=0 corner (verify over 1 token, keep=L+1).
      **Correctness (S0 invariant, proposer-independent):** token-identical to plain target greedy at
      every (ngram, k) — a wrong copy is rejected by the verify, never emitted. `run_spec.py` §C (0.5B +
      1.5B) + a `prompt_lookup` matcher unit test; every config MATCH, and the accept path is exercised
      (the France prompt's greedy self-repeats so proposals actually land).
      **Measured (`tests/bench_lookup.py`, 4070S, 1.5B fp32, max_tokens 128, every config token-gated):
      the win is COPY-DEPENDENT, not a flat ~1.2×.** france **3.35×**, copy(Fibonacci passage)
      **2.80×** — the target's OWN greedy quotes earlier context (hit 52–74%, accept 82–97%, tok/verify
      up to 5.57); reason 1.21×, code 1.17× (some structure to copy); story 1.04×, list 1.00× (open-
      ended → no matches → degrades to plain greedy). Range **1.00×–3.35×**.
      **Sharp edge / the honest nuance: the PROPOSER is free (r→0) but the VERIFY is NOT** — a k+1-token
      verify is a mini-prefill, so on rarely-matching text a *large* k can mildly net-lose (list @k=8:
      **0.86×**): the few matched-but-rejected steps pay a fat verify for ~1 token. This is far milder
      than the draft model's story @k=8: 0.64× (which paid k draft forwards EVERY step at r=0.45) — here
      a miss costs only the free lookup, so the floor is ~1× not 0.64×. Tuning: small k when hits are
      rare (k=4 keeps list ≥0.93×), large k when copies are long (france/copy peak at k=16); higher
      ngram → fewer but higher-accept matches (france ng=3 accept 95–97% vs ng=2 82%).
      **Verdict — S4 delivers the lever S2 named:** on copy-heavy text (RAG / summarize / code-edit /
      repetitive) it reaches 2.8–3.4×, well past the draft pair's 1.24× ceiling, *because the draft cost
      is gone* — and it needs no second model and no extra memory. Not universal (open-ended is a ~1×
      no-op), but it never meaningfully regresses. The open S0 tail (sampling-parity rejection-sampling
      accept) is unchanged; batched integration is S3.

## Perf retune for 1.5B (P-track) — harvest the un-muted levers
Not new algorithms — **collect** existing G-track levers on a model that finally pays for them.
Pulled in alongside S whenever the target (1.5B) forward is the bottleneck. Same parity spine
(CPU-oracle `max|diff|` + golden tokens).
- [ ] **P0** — re-bench the opt-in flags on 1.5B; flip the defaults that now win.
      `g_cuda_use_tiled_attn` (head_dim 64→128 doubles per-tile K/V reuse — the real lever, not
      KV-outgrows-L2), `g_cuda_use_dbuf` (bigger GEMMs → more blocks → the occupancy-bound
      down/q-o unstick), wmma (bigger tiles feed the tensor cores), CUDA graphs (28 vs 24 layers).
      The `head_dim>=128` prefill-attention flip (commit e76b69a) was the first — find the rest.
- [ ] **P1** — quantization ROI on 1.5B. fp32 is 5.8 GB; on the 12 GB 4070S, keeping target +
      draft resident (S3) + longer context needs int8 W8A8 / fp16 — now the *enabler*, not just
      speed. The 4:1-MAC and ½-byte levers pay more on the ~3× matrices. Measure memory headroom +
      prefill/decode deltas at 1.5B scale.
- [ ] **P2** — long context on 1.5B. KV ~2.3×/token → Flash-Decoding (G5g) + paged wins grow and
      the "muted by L2" ceiling lifts; re-run the ctx sweep, find where paged+split now dominates.

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
- [ ] **Metal** — *deferred 2026-07-04: parked behind the S/P tracks (not dropped — the
      R-track already paid its structural prerequisite, so it's a clean pickup later).* A
      `MetalBackend` on the M4 GPU; unified memory removes most H2D. The same Python serving
      layer on a third backend = the Backend boundary proven real.

## Refactor track (R-track)
The feature roadmap is ~90% done; the GPU optimization arc left exploration-debt (a
quant-dispatch hole in the Backend abstraction, ~11 global dispatch flags, `#ifdef
NI_CUDA` scattered through `model.cpp`, a 2272-line `cuda_backend.cu`). **[REFACTOR.md](REFACTOR.md)**
stages the paydown (R0–R5) — parity-locked, the same `max|diff|=0` + golden-token
discipline as the feature stages — *before* Metal, since Metal on today's structure
would replicate the debt a third time. The refactor is Metal's prerequisite.

## Backlog (pull in when the moment fits)
Open candidates, not a closed/deferred-forever list — fold one into a stage when it's
the right moment:
- Flash-Decoding (split-K attention) — **DONE** (G5g above: split the KV across num_splits warps per
  (head,query), one-pass online softmax per chunk → partial (m,l,acc), then an associative combine.
  Isolated decode 6.8–23×, end-to-end decode 1.30× @ctx512 / 2.15× @ctx2048, oracle ~1e-8,
  split-on==split-off greedy). Both follow-ups now DONE: (a) the PAGED attention kernel got the same
  split treatment (`paged_attention_split_kv_kernel`, sharing `attention_combine_kernel`) — bit-identical
  to the contiguous split path (run_cuda_paged max|diff|=0); (b) paired with the paged cache (no cat_seq),
  the win lands undiluted — paged+split ~81 tok/s @ctx2048 (fastest of all four), and runs at ctx4096
  where the contiguous cache OOMs. Remaining: Flash-Decoding combines partials in a separate kernel pass
  — could fuse, and the split kernel could share the G5f-tiled smem staging (low priority on this model).
- shared-mem tiling for the PAGED attention kernel — G5f tiled only the contiguous path (enough to
  characterize the muted-by-L2 result). Mirroring it into paged_attention_warp_kernel (gather blocks
  into smem) is mechanical and would stay bit-identical; do it if/when tiling starts winning.
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
  ~lossless). **Decode GEMV — DONE**: cuda_linear_q8 ran the prefill-tiled GEMM at every m, so at decode
  (m=1) its 64-row tile left ~63/64 of the warps idle yet still streamed the whole int8 weight — the huge
  lm_head ran at only 6% of bandwidth. `linear_q8_gemv_kernel` gives it the G5b warp-per-output treatment
  (one warp per output channel, 32 lanes coalesce-striding the int8 codes — ¼ the fp32 bytes — a __shfl
  reduction, the per-channel dequant scale folded in once at the store, the SAME accumulate-then-scale
  order as the tiled kernel and the CPU oracle). Routed by the shared kGemvMaxM=16 split (decode→GEMV,
  prefill→tiled), parity-tested vs the CPU oracle at the true m=1 shape (test_cuda linear_q8 ~5e-6, incl.
  +bias and ragged k) and golden tokens unchanged (run_cuda_parity 1/12 + 0/12, IDENTICAL to the tiled
  path — the warp-reduce only reorders the sum). Isolated A/B (run_cuda_bench, g_cuda_force_tiled_q8):
  the lm_head decode op **6% → 82% of bandwidth, 13.5× @m=1** (2.0× @m=16 batched decode); end-to-end
  (run_cuda_decode_bench NI_QEMBED=1, lm_head kernel toggled, layers held on the GEMV) **decode 1.40× @ctx
  32/64, 1.45× @ctx 128**, prefill control flat — Amdahl-diluted (the lm_head is one op of ~360/step, but
  the biggest single weight, ~27% of decode traffic). Remaining follow-up: a W8A8 lm_head (the int8 COMPUTE
  win at prefill, where lm_head is compute- not memory-bound) once that prefill matmul matters — it feeds
  argmax, so it needs a token guard.
- batched sampling — **DONE**. forward_batch already returned [N, vocab] in one pass, but the token
  draw was a Python loop calling sample_token per row — each copying the whole vocab row even for greedy.
  `sample_batch` (scheduler.py) selects the running set's tokens together: rows that are plain greedy
  (temperature 0, no repetition penalty — the default) are decided with one vectorized `argmax(axis=1)`
  over the batch (no per-row copy; numpy's first-max == sample_token's argmax, so bit-identical), while
  sampled / repetition-penalty rows keep their per-sequence pipeline (their temperature/top-k/top-p/RNG/
  context don't vectorize). The stop logic factored into `_accept`, shared by the batched decode path and
  the single-sequence prefill-admission `_emit`. Token-identical to standalone on both backends
  (run_serve.py + run_cuda_serve.py MATCH at every batch size, incl. a rep-penalty request — the mixed
  greedy+penalized batch); isolated sampling step ~1.6–2× (the saved per-row copies). Honest scope: a
  scheduling/cleanliness win, not a headline number — sampling is <1% of the decode step (the forward
  dominates), so the value is the architecture (batched forward → batched selection) + removed copies.
- SIMD nibble-unpack for q4/q4g — **DONE**. The Q4/Q4G linears were the one quant format whose inner
  loop was still scalar (a `q4_code()` nibble-extract per element, redone per activation row).
  `simd::unpack_q4` (AVX2 + NEON + scalar) unpacks a packed weight row to int8 codes once (low nibble
  → col 2i, high → col 2i+1, code = nibble−8), then the existing SIMD `dot_qf32` runs over the buffer,
  reused across all m rows — so the nibble work is vectorized AND amortized, and the dot is SIMD. The
  unpack only pays once it amortizes, so it's gated on `m ≥ kQ4UnpackMinRows (8)`: prefill takes the
  unpacked path, decode (m=1) keeps the fused scalar dot (materializing the whole weight for one dot is
  net slower — measured a decode regression without the gate). Parity-safe (the dot re-associates in
  double like every C5 SIMD path): `test_quant` `linear_q4/q4g == linear(dequant)` within 1e-4 on BOTH
  the scalar and unpacked paths, and `unpack_q4` is bit-exact vs the scalar `q4_code` ref
  (`test_simd`), both on AVX2 (native) and NEON (qemu). End-to-end prefill (`run_bench`, 20 cores):
  **q4 28.5 → 66 tok/s (2.3×), q4g 26.9 → 67 tok/s (2.5×)**; decode unchanged (identical scalar path at
  m=1). q4/q4g model outputs are the same as before (only the dot order changed, not the values).
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
