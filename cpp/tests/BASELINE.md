# R0 baseline — the refactor invariant

Frozen reference numbers for the R-track (see `../../REFACTOR.md`). Every refactor
stage (R1–R5) must reproduce **every** number below. The refactor changes dispatch,
ownership, and file layout — it changes **no kernel and no reduction order** — so the
bar is stricter than a feature stage: not just "golden tokens hold," but *every printed
number stays identical to the last digit*, including the GPU tolerance values. A moved
digit means a stage changed math it shouldn't have.

Captured on: RTX 4070 SUPER (12 GB), Qwen2.5-0.5B, `weights/qwen2.5-0.5b`, build-cuda
(`-DNI_CUDA=ON`, sm_89). Prompt: `"The capital of France is"` →
`ref_ids = 785 6722 315 9625 374`. Golden greedy continuation (12 tokens):
`12095 13 1084 374 279 7772 3283 304 4505 323 279 2086`.

Hard gates (`max|diff|=0`, must stay EXACTLY zero) are marked **[=0]**; GPU-vs-oracle
tolerance values are soft guards (the golden tokens are the real signal) but, for this
refactor, must also stay digit-identical since no kernel changes.

## How to run the gate

```bash
# 1. self-contained unit + op-parity tests (10 tests)
cd cpp/build-cuda && ctest --output-on-failure

# 2. weight-dependent parity/golden binaries (need the export + ref dump already in the dir)
cd cpp
for b in run_parity run_generate run_cache run_batch run_paged; do ./build-cuda/$b weights/qwen2.5-0.5b; done
for m in none q8 q4 q4g; do ./build-cuda/run_quant weights/qwen2.5-0.5b $m; done
./build-cuda/run_quant weights/qwen2.5-0.5b none embed
for b in run_cuda_parity run_cuda_cache run_cuda_paged run_cuda_batch; do ./build-cuda/$b weights/qwen2.5-0.5b; done
```

(R1 should wire the weight-dependent binaries into `ctest` as a labelled `weights`
suite so this whole list collapses to one `ctest` invocation, per REFACTOR.md R0/R1.)

## ctest (self-contained)

`100% tests passed, 0 tests failed out of 10` — test_tensor, test_ops, test_io,
test_sampling, test_cache, test_quant, test_simd, gen_fixtures, ops_parity, test_cuda.

## CPU oracle (the parity spine)

| check | invariant |
|---|---|
| `run_parity` | logits `max\|diff\|=4.24385e-05`, `mean\|diff\|=4.79201e-06`, argmax 0/5, next-token `12095` MATCH |
| `run_generate` | greedy == golden, **0/12** mismatches |
| `run_cache` | cached vs uncached `max\|diff\|=0` over 5 positions (split=2 +chunk) **[=0]** |
| `run_batch` | batched == standalone, parity ok |
| `run_paged` | paged == contiguous; prefix-sharing last-row `max\|diff\|=0`, shared-block refcount=2, BIT-IDENTICAL **[=0]** |

## CPU quantization (`run_quant <dir> <mode>`)

| mode | weights | ratio | logits `max\|diff\|` vs fp32 | greedy match | next-token |
|---|---|---|---|---|---|
| none        | 1979.2 MB | 1.00× | 4.24385e-05 | 12/12 | preserved |
| q8          |  906.9 MB | 2.18× | 3.78295     | 11/12 | preserved |
| q4          |  728.0 MB | 2.72× | 19.5846     |  0/12 | CHANGED (lossy) |
| q4g         |  771.5 MB | 2.57× | 18.4344     |  5/12 | preserved |
| none+embed8 | 1571.4 MB | 1.26× | 0.332643    | 11/12 | preserved |

(q8 greedy diverges only at the last token: `…323 279 4843` vs golden `…323 279 2086`.
q4g at `…279 10723 315 279 8585 3033 323 279`. These exact sequences are part of the
invariant.)

## CUDA backend (vs the CPU oracle + golden)

`run_cuda_parity`:

| path | result | weights |
|---|---|---|
| fp32 GPU forward | logits `max\|diff\|=3.76701e-05`, `mean\|diff\|=5.39619e-06`, argmax 0/5, greedy **0/12** MATCH | 1979 MB |
| fp16 weights     | greedy **0/12** MATCH, logits `max\|diff\|=4.1008e-05` | 991 MB (2.00×) |
| W8A8 (int8 DP4A) | greedy 1/12 differ, next-token preserved | 907 MB |
| int8 embed/lm_head alone | greedy 1/12 differ, logits `max\|diff\|=0.332645` | 1571 MB |
| full int8 (W8A8 + int8 embed) | greedy **0/12** differ, next-token preserved | 499 MB (3.97×) |

`run_cuda_cache`:
- cached vs uncached `max|diff|=0` over 5 positions (split=2 +chunk) **[=0]**
- incremental greedy decode == golden, **0/12** mismatches
- flash-decoding: split-on vs split-off greedy MATCH, 0/16 mismatches (ctx=515)

`run_cuda_paged`:
- paged vs contiguous `max|diff|=0` over 12 decode steps **[=0]**
- greedy vs golden MATCH, 0/12 mismatches
- flash-decoding: paged-split == contiguous-split `max|diff|=0` over 8 steps (ctx=515) **[=0]**

`run_cuda_batch`:
- 4 seqs × 16 steps `max|diff|=0`, token mismatches 0, BIT-IDENTICAL **[=0]**
