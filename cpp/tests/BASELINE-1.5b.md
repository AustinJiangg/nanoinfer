# Qwen2.5-1.5B baseline — the bigger-model port (2026-07-03)

The engine ported from Qwen2.5-0.5B to **Qwen2.5-1.5B** with **zero model-code changes** —
every dimension is read from `config.txt`, so the port was export + parity. The only code
change was a device-memory `device_pool_trim()` (cuda_runtime.cu), needed because the five
weight formats in `run_cuda_parity` sum past the 4070S's 12 GB VRAM at 1.5B (fp32 alone ~6 GB);
it is parity-neutral (the 0.5B `BASELINE.md` numbers stay digit-identical — verified).

This file is the 1.5B analogue of `BASELINE.md` (which stays the 0.5B R-track invariant).
Captured on: RTX 4070 SUPER (12 GB), `weights/qwen2.5-1.5b`, build-cuda (`-DNI_CUDA=ON`, sm_89).
Prompt `"The capital of France is"` → `ref_ids = 785 6722 315 9625 374`. Golden greedy (12):
`12095 13 576 6722 315 9625 374 1083 279 6722 315 279` → *"Paris. The capital of France is
also the capital of the"*.

## Config (read from HF, written to config.txt)

| field | 0.5B | 1.5B |
|---|---|---|
| hidden_size | 896 | **1536** |
| intermediate_size | 4864 | **8960** |
| num_layers | 24 | **28** |
| num_attention_heads | 14 | **12** |
| num_kv_heads | 2 | 2 (GQA) |
| **head_dim** | 64 | **128** (== `kMaxHeadDim` ceiling) |
| max_position_embeddings | 32768 | **131072** |
| rope_theta | 1e6 | 1e6 |
| tie_word_embeddings | 1 | 1 |
| vocab_size | 151936 | 151936 |
| params / fp32 weights | ~0.5B / 1979 MB | ~1.54B / **6178 MB** |

## CPU oracle (the parity spine)

| check | invariant |
|---|---|
| `run_parity` | logits `max\|diff\|=4.06504e-05`, `mean\|diff\|=4.38066e-06`, argmax 0/5, next-token `12095` MATCH |
| `run_generate` | greedy == golden, **0/12** mismatches |
| `run_cache` | cached vs uncached `max\|diff\|=0` over 5 positions (split=2 +chunk) **[=0]** |
| `run_batch` | batched == standalone `max\|diff\|=0`, bit-identical **[=0]** |
| `run_paged` | paged == contiguous `max\|diff\|=0`; prefix-share last-row `max\|diff\|=0`, refcount=2 **[=0]** |

## CPU quantization (`run_quant <dir> <mode>`)

| mode | weights | ratio | logits `max\|diff\|` | greedy | next-token | formats |
|---|---|---|---|---|---|---|
| none        | 6177.9 MB | 1.00× | 4.06504e-05 | 12/12 | preserved | f32=6174.3 |
| q8          | 2249.9 MB | **2.75×** | 1.96335 | 11/12 | preserved | f32=933.5 q8=1312.8 |
| q4          | 1594.8 MB | 3.87× | 16.1184 | 0/12 | CHANGED (lossy) | f32=933.5 q4=657.7 |
| q4g         | 1756.0 MB | 3.52× | 19.3023 | 2/12 | preserved | f32=933.5 q4g=818.9 |
| none+embed8 | 5478.4 MB | 1.13× | 0.297134 | 12/12 | preserved | f32=5240.8 q8=234.0 |

Note the q8 ratio is **better than 0.5B's 2.18×** (→ 2.75×): the embedding is a smaller share of a
bigger model (933 MB / 6178 = 15% here vs 27% at 0.5B), so quantizing the projections buys more.

## CUDA backend (vs the CPU oracle + golden)

`run_cuda_parity`:

| path | result | weights |
|---|---|---|
| fp32 GPU forward | `max\|diff\|=3.83854e-05`, `mean\|diff\|=4.09449e-06`, argmax 0/5, greedy **0/12** MATCH | 6178 MB |
| fp16 weights     | greedy **0/12** MATCH, `max\|diff\|=3.8147e-05` | 3091 MB (2.00×) |
| W8A8 (int8 DP4A) | greedy **5/12** differ, next-token preserved | 2250 MB |
| int8 embed/lm_head alone | greedy **0/12** differ, `max\|diff\|=0.297132` | 5478 MB |
| full int8 (W8A8 + int8 embed) | greedy 11/12 differ, next-token preserved | 1550 MB (**3.98×**); q8=234.0 w8a8=1312.8 |

`run_cuda_cache` / `run_cuda_paged`: `max|diff|=0` **[=0]**, greedy 0/12, flash-decoding split MATCH.
`run_cuda_batch`: 4 seqs × 16 steps `max|diff|=0` bit-identical **[=0]**. Batched decode tok/s:
1→39.1, 2→63.4, 4→90.5, 8→112.9, 16→124.6, 32→**187.4** (4.8×).

Quant divergence vs 0.5B: **W8A8 drifts more** (5/12 vs 1/12) — int8 *activation* error compounds over
28 layers × wider hidden; **int8 embed is *more* robust** (0/12 vs 1/12) — a wider hidden dim averages
the per-channel int8 error out before argmax. Both stay next-token-preserved.

## Re-measuring the parked optimizations (did the 0.5B "ties" wake up at 1.5B?)

The point of going bigger (ROADMAP): the G5 arc parked several optimizations as *ties* on 0.5B and
predicted they'd win on a bigger model / longer context / larger batch. Verdict at 1.5B:

**End-to-end decode** (`run_cuda_decode_bench weights/qwen2.5-1.5b`):

| lever | 0.5B | 1.5B | verdict |
|---|---|---|---|
| decode GEMV (G5b) | ~1.4× | **1.73×** (19.6→33.9 tok/s) | holds, bigger |
| fp16 weights decode (G5d) | 1.21× | **1.32×** (33.9→44.8) | GREW — decode is more memory-bound at 1.5B |
| paged vs contiguous | ~1.0× @short | 1.06× @ctx256 | holds (grows w/ ctx) |
| Flash-Decoding split-KV (G5g) | 1.30× @ctx512 | **1.32×** @ctx~896 | holds, grows w/ ctx |
| CUDA graphs (G6) | 1.06× | 1.03× | still small (kernels bigger → launch a smaller %) |
| double-buffered GEMM (G5h) | tie | **0.91× prefill** (1563→1427) | tie/loss HOLDS — projections still occupancy-hidden |

**Isolated attention kernel** (`run_cuda_attn_bench <H> <D>` — parameterized by model this port;
synthetic K/V, so it isolates the kernel from the full step). Tiled-vs-non-tiled (`t/notile`) and
split-vs-non-tiled (`notl/spl`), 30-iter timings:

| sq | sk | `t/notile` 0.5B (D=64) | `t/notile` 1.5B (D=128) | `notl/spl` 0.5B | `notl/spl` 1.5B |
|---|---|---|---|---|---|
| 128 | 128 | 0.92× | **1.26×** | 0.95× | 1.07× |
| 512 | 512 | 1.02× | **1.09×** | 1.05× | 0.99× |
| 1024 | 1024 | 0.97× | **1.07×** | 0.98× | 1.03× |
| 2048 | 2048 | 1.01× | **1.03×** | 1.01× | 1.00× |
| 1 | 1024 | 0.99× | 1.01× | **7.16×** | **5.68×** |
| 1 | 4096 | 1.01× | 1.00× | **23.20×** | **14.54×** |
| 1 | 16384 | 0.99× | 1.00× | **12.63×** | **13.48×** |

- **Shared-mem tiling (G5f) — the tie WOKE UP.** At prefill (sq>1) it straddled 1.0 on 0.5B (a tie,
  the roadmap's honest result: KV fits L2, so L2 already serves the reuse) but is a **consistent
  1.03–1.26× win on 1.5B**. head_dim 64→128 doubles the bytes each key stages into shared memory, so
  the reuse tiling buys is worth more relative to the L2-served baseline — the roadmap predicted
  exactly this ("winning only when K/V outgrow L2 — bigger model/batch, longer context"). Still
  **bit-identical** (`tile==notile max|diff|=0`) at D=128, so the win is free. The win is largest at
  small sq (1.26× @128) and tapers as the O(sq·sk) compute grows.
- **Split-KV / Flash-Decoding (G5g)** stays a large **decode** win on both models (5.7–23×, sq=1) —
  it's parallelism (H·splits warps vs H), independent of head_dim; the isolated win is ~unchanged.
- Decode rows (sq=1) tile-tie exactly on both (the dispatch sends sq=1 to the non-tiled kernel).
- **Tiling landed as the head_dim-gated default.** End-to-end (`run_cuda_decode_bench`) the isolated
  win is Amdahl-diluted to a consistent **~2.7% prefill** gain on 1.5B (prefill 256/512/1024/2048:
  1733→1779, 1668→1714, 1453→1494, 992→1017 tok/s), decode-neutral. Small but free (bit-identical)
  and consistent, so tiling is now the DEFAULT at `head_dim >= kTileMinHeadDim` (=128, cuda_backend.cu)
  — `run_cuda_decode_bench` baseline prefill jumped 1668→**1790** with no flag. `use_tiled_attn` still
  forces it on at any D; **`no_tiled_attn`** forces it off (the R2-red-line A/B baseline — how the bench
  measures the non-tiled column now that tiling is the default). 0.5B (D=64) stays non-tiled, unchanged
  (ctest 23/23); 1.5B logits bit-identical (`3.83854e-05`), all golden/max|diff|=0 gates hold.

### Verdict

The port validated the G5 "ties on this model" calls precisely: **tiling** (the one gated on L2
pressure) turned into a real win at head_dim=128, while **dbuf** and **graphs** (gated on occupancy
and launch-overhead-fraction, which the bigger model doesn't relieve) stayed ties. **fp16** and
**split-KV** grew as predicted. Every parked lever behaved as the roadmap's roofline reasoning said
it would — the honest-tie discipline paid off. And tiling flipped from parked-opt-in to the
`head_dim >= 128` **default** — the first G5 lever a bigger model promoted from tie to shipped win.

