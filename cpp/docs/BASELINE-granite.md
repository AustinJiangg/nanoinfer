# Granite-3.1-1B-A400M baseline — the A3 MoE rung (2026-07-14)

The third A-track arch, and the phase's biggest structural delta: the dense FFN
becomes a **sparse mixture-of-experts** (32 experts, top-8, SwiGLU experts at width
512 — 1.3B total / ~400M active params), plus the **muP scalar set** A0 described.
MoE replaces the FFN ONLY — attention, all four KV caches, the schedulers,
speculative decode, and HTTP serving ran on day one, which is exactly what the A3
plan predicted and the gates below prove.

The delta:
1. **MoE FFN** — router linear `[hidden→32]`, top-8, softmax over ONLY the selected
   8 logits (locked by parity, not docs — Mixtral renormalizes differently), output
   = Σ gate·expert(x). HF stores experts FUSED (`input_linear [E, 2·ffn, hidden]` =
   gate rows then up rows; `output_linear [E, hidden, ffn]` = down); the
   `granitemoe` entry in the A0 name-map registry (now one-to-many) unfuses to
   per-expert `experts.{e}.{gate,up,down}_proj.weight`. Both engines group tokens
   **by expert** (ascending id): gather rows → one SwiGLU per active expert (each
   expert weight streamed once, the C5/F8a lever) → gate-scaled scatter-add.
   Decode (1 row) degrades to its 8 experts, one row each — same code.
2. **muP scalars, resolved into the weights at load** (config-time resolution, the
   rope_theta lesson): `q_proj ×= attention_multiplier·√head_dim` (HF's multiplier
   REPLACES 1/√d; on Granite the fold factor is 0.125, a power of two → bit-exact;
   guarded against qk_norm, which would normalize the fold away),
   `o_proj`/`down_proj ×= residual_multiplier`, `embed ×= embedding_multiplier`,
   `norm.weight ÷= logits_scaling·embedding_multiplier` (the tied lm_head picked up
   the embed fold; logits_scaling DIVIDES in HF). Granite therefore runs the exact
   per-token op sequence of a dense model — no new kernels, no scale parameter
   threaded through 4 caches and every attention kernel, nothing added to the
   launch-bound CUDA decode path (the G6 lesson). The Python oracle applies the
   scalars at runtime instead (readable reference; parity confirms both).
3. **New Backend MoE primitives** — `to_host_copy` (router-logits D2H; the routing
   decision is host-side), `zeros`, `gather_rows`, `scatter_add_rows` (CPU loops /
   CUDA one-thread-per-element kernels; scatter needs no atomics — top-k indices
   are distinct, so rows are unique per call). Defaults throw loudly (the S1
   pattern). Sharp edge hit: naming the D2H virtual `to_host` SHADOWED the free
   `ni::to_host` inside `CudaBackend::finalize_logits` → identity instead of D2H →
   segfault; renamed + `ni::` qualified, lesson recorded in comments both places.
4. Everything else was already config: no QKV bias, no QK-Norm, head_dim 64, GQA
   16/8, rope_theta 1.5e6 (no scaling), vocab 49152 (GPT-2-style BPE), tied
   embeddings, `attention_dropout` (eval no-op). `moe_intermediate_size` reads HF's
   `intermediate_size` — GraniteMoe has no separate field.

Weights **Apache 2.0** (`ibm-granite/granite-3.1-1b-a400m-base`), ~2.6 GB bf16
download; the bf16 NIT1 export is 2669 MB (all 2474 tensors pass the lossless gate —
the checkpoint is fully bf16).

Captured on: RTX 4070 SUPER (12 GB), `weights/granite-3.1-1b-a400m`, build-cuda (sm_89).
Prompt `"The capital of France is"` → 5 ids.
Golden greedy (12 ids): `2716 297 32 203 203 1318 18926 432 44796 438 20859 1800`.

## Config (read from HF, written to config.txt)

| field | Qwen2.5-0.5B | Granite-3.1-1B-A400M |
|---|---|---|
| hidden_size | 896 | 1024 |
| intermediate_size | 4864 | **512** (= the expert width) |
| num_layers | 24 | 24 |
| num_attention_heads | 14 | 16 |
| num_kv_heads | 2 | **8** (GQA, n_rep=2) |
| head_dim | 64 | 64 |
| rope_theta | 1e6 | **1.5e6** |
| vocab_size | 151936 | **49152** |
| tie_word_embeddings | 1 | 1 |
| qkv_bias / qk_norm | 1 / 0 | 0 / 0 |
| **n_experts / top_k / moe_ffn** | 0 (dense) | **32 / 8 / 512** |
| **embedding_multiplier** | 1 | **12.0** |
| **attention_multiplier** | 1 (→ 1/√64 = 0.125) | **0.015625** (= 1/64) |
| **residual_multiplier** | 1 | **0.22** |
| **logits_scaling** | 1 | **6.0** (divides) |
| params / fp32 weights | ~0.5B / 1979 MB | **1.33B total, ~0.4B active** / 5339 MB |

## Python oracle (gate step 1 — `tests/test_granite.py`)

`test_granite_config_flags` (experts 32/8/512, the four muP scalars, rope 1.5e6,
GQA 16/8, tied, vocab 49152), `test_granite_moe_layer_matches_hf` (the MoE module
alone vs HF's `block_sparse_moe` on layer 0, `allclose(1e-5)` — isolates
router+experts from the rest of the forward), `test_granite_forward_matches_hf`
(logits `allclose(1e-3)`), `test_granite_greedy_matches_hf` (greedy == HF).
Full `pytest`: **48 passed** (Qwen2.5 + Qwen3 + Llama + Granite).

## CPU oracle (gate step 2 — vs the dumped goldens)

| check | invariant |
|---|---|
| `run_parity` | logits `max\|diff\|=7.34329e-05`, `mean\|diff\|=4.90642e-06`, argmax 0/5, next-token MATCH |
| `run_generate` | greedy == golden, **0/12** |
| `run_cache` | cached vs uncached `max\|diff\|=0` over 5 positions (split=2 +chunk) **[=0]** |
| `run_batch` | batched == standalone `max\|diff\|=0`, 4 seqs × 16 steps **[=0]** |
| `run_paged` | paged == contiguous + prefix-share + S1 rollback `max\|diff\|=0` **[=0]** |

`run_parity`'s max|diff| is ~5× the dense models' (~1.3e-5): the residual /
embedding / logits folds multiply weights by non-powers-of-two (0.22, 12, 1/72),
costing ~1 ulp per weight element vs the oracle's runtime scaling — bounded,
expected, and far under the 0.1 gate. The q_proj fold (0.125) is exact.

**`run_batch`'s bit-identity doubles as the ROADMAP's "expert-grouped == per-token"
gate**: batched decode groups 4 sequences' rows by expert TOGETHER, standalone
groups each row alone — `max|diff|=0` proves the grouping leaves every row's math
untouched (row-independence of the expert GEMMs, the same reason F8a holds).
Router top-k determinism rides every golden gate (ties break to the lower index,
torch.topk's order, in both engines).

## CPU quantization (`run_quant <dir> <mode>`)

| mode | logits `max\|diff\|` | greedy | next-token |
|---|---|---|---|
| q8 | 9.76 (argmax 1/5) | **12/12** | preserved |
| w8a8 | 26.8 (argmax 1/5) | **12/12** | preserved |

The per-expert projections quantize through the untouched `*_proj.weight`
machinery — 2304 expert weights + router untouched (fp32, tiny). Absolute logit
diffs look large because a 49k-vocab logit row under int8 activations drifts more
per position, but greedy holds 12/12 on both modes.

## CUDA backend (gate step 3 — vs the CPU oracle + golden)

| path | result | weights |
|---|---|---|
| fp32 GPU forward | `max\|diff\|=2.95639e-05`, argmax 0/5, greedy **0/12** MATCH | 5339 MB |
| fp16 weights | greedy **0/12** MATCH, `max\|diff\|=0.054` | 2672 MB (2.00×) |
| bf16 weights (B1) | greedy **0/12** MATCH, `max\|diff\|=0.239` | 2672 MB (2.00×) |
| W8A8 (int8 DP4A) | greedy **0/12** differ, next-token preserved | 1496 MB |
| int8 embed/lm_head alone | greedy **0/12** differ, `max\|diff\|=0.450` | 5189 MB |
| full int8 (W8A8 + int8 embed) | greedy **0/12** differ, next-token preserved | 1345 MB (**3.97×**) |

Every format inherited the experts for free (`*_proj.weight` naming). bf16's
larger diff vs fp16 is the muP folds: folded weights are no longer bf16-exact
(the fold multiplies by 12/0.22/…), so B1's "byte-exact to the checkpoint"
property doesn't apply to Granite — tolerance + golden is the bar, and it holds.

`run_cuda_cache`: cached==uncached `max|diff|=0`, incremental greedy 0/12,
flash-decoding split-on==off MATCH (ctx 515). `run_cuda_paged`: paged==contiguous
`max|diff|=0`, golden MATCH, paged-split==contiguous-split, tiled BIT-IDENTICAL,
S1 rollback BIT-IDENTICAL. `run_cuda_batch`: bit-identical; aggregate decode
26.5 → 119.4 tok/s at batch 32 (**4.5×**).

## Performance snapshot (honest scope: correctness-first CUDA MoE)

| engine | prefill 64 | decode 64 |
|---|---|---|
| CPU (20 threads, fp32) | 55.7 tok/s | 6.5 tok/s |
| CUDA fp32 (contiguous, GEMV) | 435 tok/s | **23.8 tok/s** |

CUDA decode is per-op-overhead-bound, much harder than the dense models (0.5B
decodes ~84 tok/s with 25% MORE active params): each layer adds a router D2H
sync (24/token) plus up to 8 experts × (gather + 3 small GEMVs + scatter) — ~5×
the launches of a dense FFN, plus per-call cudaMalloc for the index uploads.
The named next levers, deliberately NOT built before measuring (the G5a
discipline): pool the index buffers, device-side routing (top-k kernel → no D2H),
and batching the 8 selected experts' GEMVs into one grouped launch. The CPU gap
(6.5 vs 0.5B's ~11) is the same shape, milder.

## Serving + spec + detok (gate step 4)

| gate | result |
|---|---|
| `run_serve` | F7/F8a/F8b/prefix-sharing all MATCH standalone at mb 1/2/8 |
| `run_cuda_serve` | MATCH; aggregate 98.2 tok/s at batch 32 |
| `run_spec_serve` (granite as draft+target) | CPU **and** CUDA: mixed draft/lookup batches token-identical to standalone spec AND plain greedy; S3b ragged verify + S3e prefix-sharing green — `forward_spec_batch`'s M-row expert grouping proven |
| `run_http_serve` | CPU 5.9 tok/s, `--device cuda` 28.1 tok/s aggregate; SSE/cancel/malformed all ok |
| `run_detok` | granite's GPT-2-style BPE joins Llama tiktoken + Qwen BPE: streamed deltas == one-shot decode, no `�` leaked |

## Regression — the frozen oracles stay bit-identical

The muP folds are guarded by `!= 1.0` (identity skips), MoE by `n_experts == 0`,
and the dense FFN branch of `Model::ffn()` is the byte-for-byte pre-A3 sequence.
Confirmed: `ctest -L weights` **13/13** (Qwen2.5 CPU+CUDA golden), `ctest -L
nomodel` **9/9**, full `pytest` **48 passed**, `run_binding`/`run_serve`/
`run_spec`/`run_cuda_serve` on Qwen2.5 all MATCH.

## Reproduce

```bash
pytest tests/test_granite.py -v                                                 # (1)
python cpp/tools/export_weights.py cpp/weights/granite-3.1-1b-a400m ibm-granite/granite-3.1-1b-a400m-base bf16
python cpp/tools/dump_reference.py cpp/weights/granite-3.1-1b-a400m "The capital of France is" ibm-granite/granite-3.1-1b-a400m-base
cd cpp && cmake --build build -j
for t in run_parity run_generate run_cache run_batch run_paged; do ./build/$t weights/granite-3.1-1b-a400m; done        # (2)
./build/run_quant weights/granite-3.1-1b-a400m q8; ./build/run_quant weights/granite-3.1-1b-a400m w8a8
for t in run_cuda_parity run_cuda_cache run_cuda_paged run_cuda_batch; do ./build/$t weights/granite-3.1-1b-a400m; done # (3)
python tests/python/run_serve.py weights/granite-3.1-1b-a400m                   # (4)
python tests/python/run_cuda_serve.py weights/granite-3.1-1b-a400m
python tests/python/run_spec_serve.py weights/granite-3.1-1b-a400m weights/granite-3.1-1b-a400m [cuda]
python tests/python/run_http_serve.py weights/granite-3.1-1b-a400m [--device cuda]
python tests/python/run_detok.py ibm-granite/granite-3.1-1b-a400m-base
```

## Verdict (A3)

The MoE rung cost exactly what the plan scoped: a routed-experts FFN (one Python
module, one C++ `ffn()` helper, four Backend primitives + their CUDA kernels) and
zero change anywhere else — the A0 bet ("MoE swaps only the FFN") held literally.
The two transferable lessons: **fold config scalars into weights at load** when the
algebra allows (the muP set folded to a bit-exact q_proj scale + three ~1-ulp folds,
keeping Granite's op count identical to a dense model), and **expert-grouping is
just the F8a row-fusion lever again** — gather rows per expert so each weight
streams once, with bit-identity guaranteed by row-independence. The honest open
item is CUDA MoE decode throughput (23.8 tok/s, overhead-bound): pool the index
uploads, move routing on-device, group the expert GEMVs — measure first.
