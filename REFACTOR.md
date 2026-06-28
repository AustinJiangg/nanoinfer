# Refactor track (R-track)

The feature roadmap is ~90% done: Python oracle (stages 0–2), C++ core (C0–C5),
Fusion (F6–F8c, the mini-vLLM), CUDA (G0–G6), and NEON all landed. What remains is
one big feature — **Metal** — plus a few backlog micro-items (see ROADMAP.md).

But the GPU optimization arc (G5a→G6, ~25 commits) left real, locatable
**exploration-debt**, and that debt sits directly in front of Metal. This track pays
it down *before* Metal, because a Metal backend on today's structure would replicate
the debt a third time (another `#ifdef NI_METAL`, another quant bypass, another batch
of `g_metal_*` flags). The refactor is Metal's prerequisite, not a detour.

**The safety net is the project's spine.** Every R-stage keeps the CPU oracle
bit-identical (`max|diff|=0`) and the CUDA golden tokens token-for-token. The same
discipline that gated every feature stage gates every refactor stage: small,
reviewable, parity-locked. R0 freezes the baseline numbers; nothing after it may move
them (except where a stage explicitly re-verifies the same value before/after).

## The four debts (with evidence)

1. **A "quant hole" in the Backend abstraction.** `forward()` is written once against
   `backend_->op()` and is device-agnostic — the G0 win. But quantization bypasses it:
   Q8/Q4/W8A8 projections go through `qweights_[name]->linear()` and the int8
   embed/lm_head through `#ifdef NI_CUDA` free functions (`cuda_embedding_q8`,
   `cuda_linear_q8`, `cuda_linear_w8a8`). Two parallel dispatch paths. The abstraction
   abstracts *compute* but not the thing that varies most — the **weight
   representation**.

2. **~11 mutable global dispatch flags.** `g_cuda_use_wmma`, `g_cuda_use_tiled_attn`,
   `g_cuda_use_split_attn`, `g_cuda_use_dbuf`, `g_cuda_fp16_weights`,
   `g_cuda_force_naive_{gemm,attn}`, `g_cuda_force_tiled_q8`, `g_quantize_embed`,
   `g_cuda_keep_device_logits`, `g_cuda_graph_pos/token` — plus ~10 `NI_*` env vars
   toggling them via `getenv` in tests. Process-global mutable state: not reentrant
   (can't run two policies in one process), not composable, A/B-by-mutation. Scaffolding
   from the optimization journey that has become a tangle.

3. **`#ifdef NI_CUDA` scattered through `model.cpp`** (8+ sites: constructor,
   `embed_tokens`, `lm_head`, `make_kv_cache`, the `to_host` at the end of `forward`,
   `forward_batch`, `weight_bytes`). A class that is supposed to be device-agnostic,
   shot through with device conditionals.

4. **`cuda_backend.cu` is one 2272-line file** holding 8 GEMM kernel variants, 3 quant
   GEMMs, 6 attention kernels, all elementwise/shape ops, H2D/D2H, two KV caches
   (contiguous + paged + block pool), and the CUDA graph decoder.

## Stages

Ordered by dependency. Semantic changes (where the risk is) first, each isolated and
parity-gated; mechanical moves (zero parity risk) last, landing on stabilized code.

### R0 — freeze the oracle as one command  ⬜
- **Change:** make every parity/golden test runnable as a single `ctest` gate. Record
  the current baseline numbers — logits `max|diff|` vs the CPU oracle, each golden
  token sequence, `weight_bytes` per quant mode — as the refactor's invariant.
- **Done when:** `ctest` is one green command; the baseline is on disk and is the diff
  target for every stage below.

### R1 — backend factory + kill the cache/logits `#ifdef`s  ⬜
- **Change:** `make_backend(Device, const BackendConfig&)` replaces the constructor's
  `if(CPU)…#ifdef…else if(CUDA)`. KV-cache creation moves onto the backend
  (`backend_->make_kv_cache(...)` / `make_paged_cache(...)`), killing the `#ifdef` in
  `Model::make_kv_cache`. Logits landing becomes `backend_->finalize_logits(t)` (CPU
  returns as-is, CUDA does the D2H under the keep-device flag), killing the inline
  `#ifdef` at the end of `forward`/`forward_batch`.
- **Done when:** `model.cpp`'s `#ifdef` count roughly halves; every parity number
  unchanged vs R0. Pure plumbing, no math.

### R2 — policy object: de-globalize the flags  ⬜
- **Change:** replace the 11 globals with a **construction-time, immutable** config:
  ```cpp
  enum class GemmVariant { Auto, Naive, Tiled, TiledVec, TiledDB, WarpTiled, Wmma };
  enum class AttnVariant { Auto, Naive, WarpOnline, Tiled, Split };
  struct CudaPolicy {
      GemmVariant gemm = GemmVariant::Auto;   // Auto = the existing m<=16 GEMV / prefill tiled heuristic
      AttnVariant attn = AttnVariant::Auto;
      bool fp16_weights   = false;
      bool quantize_embed = false;
  };
  ```
  The `force_naive_*` bench switches become explicit policy values (`gemm = Naive`).
  Graph state (`g_cuda_graph_pos/token`, `keep_device_logits`) is **not policy** — it
  is per-decode-call state; it moves into a `DecodeContext` / onto `CudaGraphDecoder`,
  threaded through the call, not global. Tests migrate from `getenv("NI_DBUF")` to
  `cfg.gemm = GemmVariant::TiledDB` at backend construction.
- **Bonus:** the backend becomes reentrant — two policies can run side by side in one
  process for A/B (the globals forbid this today).
- **Done when:** `grep g_cuda_` is empty (bar the migrated graph state); `ctest`
  numbers unchanged vs R0.

### R3 — the weight seam (keystone; highest value, highest risk)  ⬜
- **Change:** unify fp32 / fp16 / Q8 / Q4 / W8A8 / int8-embed behind ONE call. Make the
  **weight** polymorphic, not the call site:
  ```cpp
  // before: two paths + #ifdef in model.cpp
  auto it = qweights_.find(name);
  if (it != qweights_.end()) return it->second->linear(x, bias);   // quant bypass
  return backend_->linear(x, W(name), bias);                       // fp32/fp16 path
  // (+ a third cuda_linear_q8 bypass under #ifdef for the int8 lm_head)

  // after: one path; the backend dispatches on weight.format internally
  return backend_->linear(x, weight(name), bias);   // weight carries {device, format, tensors}
  ```
  `Weight` is a light handle/variant: fp32/fp16 is just a `Tensor` (already carries
  `Device` + `DType`); quant packs codes/scale/group. **The backend owns every kernel**;
  the weight is tagged data. Each backend implements the formats it supports and throws
  on the rest. This collapses the existing `QuantizedWeight` polymorphism, the `cuda_*`
  free functions, and the embed/lm_head `#ifdef` routing into one seam. `Model` holds
  one `unordered_map<string, Weight>`; `qweights_` / `embed_q8_*` / the device
  `embed_q8_codes_/scale_` all dissolve.
- **Result:** **zero `#ifdef NI_CUDA` in `model.cpp`** — the model is genuinely
  device-agnostic, and the Backend abstraction is complete for the first time (it now
  abstracts the weight representation, the thing that varied most).
- **Done when:** int8 / W8A8 / fp16 / int8-embed golden tokens AND `weight_bytes` are
  byte-identical before/after (this stage is gated hardest); CPU `max|diff|=0`.

### R4 — split the monolith (mechanical, zero parity risk)  ⬜
- **Change:** carve `cuda_backend.cu` into translation units by concern:
  `cuda_runtime.cu` (alloc/pool/H2D/D2H/probe/sm_count), `cuda_gemm.cu`,
  `cuda_quant.cu`, `cuda_attention.cu` (incl. split/combine), `cuda_elementwise.cu`,
  `cuda_cache.cu`, `cuda_graph.cu`; `cuda_backend.cu` keeps only the method bodies
  wiring them. Split the non-semantic parts (runtime/elementwise/cache) out first to
  shrink the file, then the dispatch units (gemm/quant/attention) once R2/R3 have
  stabilized them. Stand up an empty `metal/` skeleton mirroring the layout to prove it
  is reproducible.
- **Done when:** pure code movement, parity by construction.

### R5 — (optional) tighten the CPU side symmetrically  ⬜
- **Change:** push the same weight-seam / format-enum down into `ops.cpp` / `quant.cpp`
  so CPU and CUDA share one `Format` enum and one dispatch contract. Lower priority
  than R0–R4; after it, CPU / CUDA / Metal share a single dispatch contract.

## The red line

R2 must **not delete the A/B capability — only retype it.** The force-naive /
tiled-on-off / split-on-off comparisons are how this project produces its honest
"this ties on this model, here's the roofline reason" findings (G5h, G5f-tiled, G6).
The globals become typed policy values; the measurement discipline that is the soul of
the project is preserved, not removed. After R2 the A/B is *better*: two policies can
run in one process instead of mutating a shared global.

## Sequencing with the feature work

The refactor enables the features; it does not replace them. After R0–R4:

1. **Metal backend** — now "implement the Backend methods + a `MetalPolicy` subset";
   the `cuda_*.cu` split is the template for `metal_*.mm`. The third backend is what
   finally proves the `Backend` boundary is real.
2. **A bigger model (Qwen2.5-1.5B / 7B)** — the highest single-step learning value. The
   engine is validated only on 0.5B, where G5h / G5f-tiled / G6 / wmma all *tied*
   precisely because the model is small / KV fits in L2 / batch-1. A 7B re-activates
   those parked optimizations as real wins and tests whether the "ties on this model"
   calls were right. R2's policy object makes per-model tuning clean, so this pairs with
   the refactor.
3. **Backlog micro-items** — paged-attention shared-mem tiling, W8A8 lm_head,
   batched/paged-split CUDA graphs, int8-embed under graph (see ROADMAP.md Backlog).
