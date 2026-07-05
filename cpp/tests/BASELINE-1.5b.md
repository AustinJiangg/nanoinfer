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

### wmma tensor cores (P0, `run_cuda_bench 1.5b` — the GEMM analog of the attn bench's `<H> <D>`)

The one P0 lever the port hadn't re-benched. `run_cuda_bench` now takes a model arg (`0.5b` default /
`1.5b`), sweeping that model's projection shapes, so the wmma-h (fp16-weight tensor-core) kernel can be
A/B'd against the fp32 tiled default on 1.5B's ~3× matrices. **Isolated GEMM, m=128 prefill (GFLOP/s):**

| shape (n×k) | wmma-h | tiled | wmma/tiled | note |
|---|---|---|---|---|
| q/o (1536×1536)         |  8991 |  9297 | **0.97×** | loses (narrow n) |
| k/v (256×1536)          |  1255 |  1304 | ~1.0× | tiny |
| **gate/up (8960×1536)** | 16252 | 10110 | **1.61×** | wins big (103% of cuBLAS) |
| **down (1536×8960)**    |  8566 |  6851 | **1.25×** | wins |
| lm_head (151936×1536)   | 12108 | 11847 | ~1.02× | tie (tiled already fast) |

**The tie WOKE UP on the wide MLP GEMMs.** At 0.5B wmma-h lost on every projection (feeding-bound — small
n starves the tensor cores, G5d). At 1.5B the two 8960-dim GEMMs (gate/up, down — the prefill FLOP bulk)
win 1.25–1.61×, exactly the roadmap's "bigger tiles feed the tensor cores." wmma-h is lossy (fp16-
accumulate, max|diff| ~1–3e-2 vs tiled's ~1e-4) and needs fp16 weight STORAGE (fp32-weight wmma with a
per-tile convert still loses, G5d).

**But e2e prefill is a WASH** (`run_cuda_decode_bench`, `NI_FP16W` vs `NI_FP16W NI_WMMA` — isolating wmma
on top of fp16): prefill 128 ~1675→1638 tok/s (**0.98×**), prefill 512 ~1910→1885 (**0.99×**). The isolated
1.61× on gate/up is Amdahl-diluted to a slight regression — the projection GEMMs are a minority of the
prefill step (attention ×28 tiled + lm_head + ~per-op launch dominate), and wmma loses on q/o. **Same shape
as G5c+ warp-tiling** (lm_head 90% of cuBLAS yet e2e prefill flat) and G5h dbuf: an isolated kernel win the
full step never exposes. **So wmma-h stays OPT-IN (`use_wmma`) — no default flip.**

**dbuf re-benched on 1.5B** (`run_cuda_bench 1.5b`, bit-identical, min-of-8): q/o 1.04×, k/v 1.07×, down
1.01× — small isolated wins on the narrow projections (up from 0.5B's tie; gate/up is now excluded as
wide-n), but the baseline's e2e **0.91×** (the occupancy trade in the full forward) holds, so dbuf stays
opt-in too.

### Verdict

The port validated the G5 "ties on this model" calls precisely: **tiling** (the one gated on L2
pressure) turned into a real win at head_dim=128, while **dbuf** and **graphs** (gated on occupancy
and launch-overhead-fraction, which the bigger model doesn't relieve) stayed ties. **fp16** and
**split-KV** grew as predicted. **wmma** (P0, newly re-benched) joins the honest-tie set: it WINS
isolated on 1.5B's wide MLP GEMMs (gate/up 1.61×, down 1.25× — the prediction confirmed) but WASHES
OUT e2e prefill (~0.98×, Amdahl), so it stays opt-in. Every parked lever behaved as the roadmap's
roofline reasoning said it would — the honest-tie discipline paid off. **P0 conclusion: tiling was the
one G5 lever a bigger model promoted from tie to a shipped default; wmma/dbuf/graphs all win-or-tie
isolated but not end-to-end, so they remain opt-in.**

## P1 — quantization ROI (memory ENABLER + e2e speed, 2026-07-05)

At 1.5B, quant flips from a speed knob to the **enabler** of the S3 two-model scenario: keeping the
1.5B target + 0.5B draft BOTH resident on the 12 GB 4070S (+ KV for real context) is where fp32 runs
out of room. Measured e2e (`run_cuda_decode_bench weights/qwen2.5-1.5b 128 64`, warm phase — `NI_FP16W`
/ `NI_W8A8` / `NI_W8A8 NI_QEMBED`) + the resident pair (scratch `resident_pair.py`, `nvidia-smi`, ONE
mode per process — the caching device pool doesn't release freed buffers, so modes can't share one).

### e2e speed (1.5B, warm prefill 128 / decode 64)

| mode | prefill tok/s | vs fp32 | decode tok/s | vs fp32 | weights | golden |
|---|---|---|---|---|---|---|
| fp32       | 1598 | 1.00× | 37.3 | 1.00× | 6178 MB | 0/12 |
| **fp16**   | 1670 | 1.05× | **48.7** | **1.31×** | 3091 MB (2.0×) | 0/12 (lossless) |
| W8A8       | **1761** | **1.10×** | 23.8 | **0.64×** | 2250 MB (2.75×) | 5/12, next-tok kept |
| full-int8  | 1561 | 0.98× | 24.4 | 0.65× | 1550 MB (3.98×) | 11/12, next-tok kept |

- **fp16 is the sweet spot** — the only mode that wins BOTH phases (prefill 1.05×, decode **1.31×**) and
  is lossless (golden 0/12). Decode is memory-bound, so halving the weight bytes is a direct 1.31×.
- **W8A8's int8 COMPUTE lever delivers the e2e PREFILL win (1.10×) that wmma couldn't** (P0: wmma's byte
  lever washed to 0.98×). Cutting the MACs 4:1 beats cutting the bytes when prefill is compute-bound —
  the clean P0→P1 contrast (byte lever vs compute lever).
- **But W8A8 DECODE regresses to 0.64×**: there is no int8×int8 decode-GEMV, so at m=1 it runs the
  prefill-tuned DP4A tile (~63/64 warps idle) PLUS a per-row activation-quant every step. Decode dominates
  generation, so W8A8 is a net throughput LOSS despite the prefill win — its value is memory + prefill-
  heavy (long-prompt / short-gen) loads. **The fix is a W8A8 decode-GEMV** (the analog of the shipped
  weight-only q8 one, `cuda_linear_q8` → `linear_q8_gemv_kernel`) — the identified lever, backlog.

### resident pair (1.5B target + 0.5B draft on the 12 GB card — the S3 scenario)

WSL2/display holds ~1.4–1.8 GB at idle, so the real budget is ~10.5 GB. KV is fp32 (the oracle
discipline): 56 KB/tok (1.5B) + 24 KB/tok (0.5B) = **80 KB/tok** for the pair.

| pair mode | weights (tgt+drf) | VRAM free | pair-KV budget | e.g. serving |
|---|---|---|---|---|
| fp32      | 6178+1979 = 8157 MB | **2132 MB** (measured) | ~27k tok | batch 16 @ 1.7k ctx (TIGHT) |
| fp16      | 3091+990 = 4081 MB  | ~6.5 GB (calc)         | ~83k tok | batch 16 @ ~5.2k ctx |
| W8A8      | 2250+907 = 3157 MB  | **6924 MB** (measured) | ~89k tok | batch 16 @ 5.5k ctx |
| full-int8 | 1550+499 = 2049 MB  | ~8.5 GB (calc)         | ~109k tok | batch 16 @ ~6.8k ctx |

The fp32 pair FITS but leaves only ~2 GB — fine for single-sequence short context, but a server
(batch × context) runs out fast. **Quant roughly TRIPLES the KV budget** (fp16/W8A8 ~83–89k vs fp32's
27k pair-tokens), and fp16 does it while winning decode 1.31× losslessly.

### Verdict (P1)

**fp16 is the 1.5B default of choice for the two-model resident scenario:** lossless, wins both phases,
halves the memory — strictly better than fp32 on every axis that matters here. The int8 modes are the
memory-extreme enablers (2.75–3.98×), valuable when VRAM is the hard wall (more concurrent sequences /
longer context) or for prefill-heavy loads, but their decode regresses until a W8A8 decode-GEMV lands.
The **P0→P1 lesson**: on memory-bound **decode**, cut BYTES (fp16); on compute-bound **prefill**, cut
FLOPs (int8) — wmma was a byte lever aimed at compute-bound prefill (wrong tool → washed out), while
W8A8 is the FLOP lever that actually moves prefill.

