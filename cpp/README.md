# nanoinfer-cpp

A from-scratch CPU inference engine in C++ — the low-level track. The Python
`nanoinfer` engine in the parent repo is the **reference oracle**: every C++ stage
is parity-tested against it (and against numpy), the same way the Python engine is
parity-tested against HuggingFace.

Long-term shape: build the pure C++ compute core first (no Python in the hot
path), then — once it's solid — expose it via pybind11 and rebuild the serving
layer (continuous batching, paged attention) in Python on top of our own kernels.
That fused architecture is the vLLM shape: Python orchestration, C++ kernels. See
the parent `ROADMAP.md` for the staged plan.

## Build & test

```bash
cd cpp
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires a C++17 compiler, CMake ≥ 3.16, and Python 3 + numpy (only to generate
the parity fixtures — the engine itself has no Python dependency).

The build targets the host CPU (`-march=native`, enabling the AVX2/FMA kernels)
and uses OpenMP if found. Both are optional and degrade to the same numbers:
`-DNI_NATIVE=OFF` falls back to the scalar inner products, and a toolchain
without OpenMP runs single-threaded. Cap threads with `OMP_NUM_THREADS=N`.

If `pybind11` is importable (`pip install pybind11`), the build also produces the
`nicpp` Python module (stage F6) in `build/`; it's skipped cleanly otherwise, the
same graceful degrade. See "Use from Python" below.

## Layout

```
cpp/
  src/
    tensor.{hpp,cpp}     row-major contiguous float32 Tensor (shape/stride/data)
    ops.{hpp,cpp}        matmul, rmsnorm, softmax, add — naive, readable
    serialize.{hpp,cpp}  read/write the "NIT0" tensor format (the parity bridge)
    simd.hpp             C5: AVX2/FMA dot products (double-accum), scalar fallback
    parallel.hpp         C5: OpenMP thread-count knobs for the kernels
  tests/
    test_util.hpp        dependency-free CHECK / CHECK_CLOSE / compare_tensors
    test_tensor.cpp      shape/stride/indexing
    test_ops.cpp         hand-verified numerical cases
    ops_parity.cpp       allclose vs numpy fixtures
  tools/
    gen_fixtures.py      numpy reference -> NIT0 fixtures (ctest setup step)
```

## The parity bridge

`tools/gen_fixtures.py` writes inputs and numpy-computed expected outputs in the
little-endian **NIT0** format (`src/serialize.hpp`); the C++ tests read them back
and assert allclose. The same format will carry exported nanoinfer weights into
the C++ engine in the next stage, so both run the *same* Qwen2.5 tensors and the
C++ logits can be compared against the Python engine's directly.

## Run the full Qwen2.5 forward

```bash
# 1. export weights + config from nanoinfer (~2 GB, into cpp/weights/, gitignored)
python tools/export_weights.py weights/qwen2.5-0.5b
# 2. dump nanoinfer's reference logits for a prompt
python tools/dump_reference.py  weights/qwen2.5-0.5b "The capital of France is"
# 3a. compare logits
./build/run_parity weights/qwen2.5-0.5b
# 3b. generate: greedy parity vs nanoinfer + a sampled demo
./build/run_generate weights/qwen2.5-0.5b
# 3c. cached forward == uncached full recompute (bit-identical)
./build/run_cache weights/qwen2.5-0.5b
# 3d. run quantized; report memory + accuracy vs fp32 (q8 | q4 | q4g)
./build/run_quant weights/qwen2.5-0.5b q8
./build/run_quant weights/qwen2.5-0.5b q4
./build/run_quant weights/qwen2.5-0.5b q4g
# 3e. benchmark prefill/decode tok/s; reports SIMD target + thread count
./build/run_bench weights/qwen2.5-0.5b fp32 128 32   # [fp32|q8|q4|q4g] [prefill] [decode]
```

Quantization modes (layer weights only; embedding/lm_head stay fp32), measured
on Qwen2.5-0.5B / "The capital of France is":

| mode | what | memory | 1st next-token | greedy match vs fp32 |
| --- | --- | --- | --- | --- |
| q8  | per-channel int8 | 2.2× | same | 11/12 |
| q4  | per-channel int4 | 2.7× | **changed** | 0/12 |
| q4g | group-wise int4 (32-block scales) | 2.6× | same | 5/12 |

Per-channel int4 breaks the model — a single outlier inflates the whole row's
scale and crushes the rest, so even the first token changes. Group-wise
(llama.cpp's Q4_0 idea: one scale per 32-element block) gives each block a local
scale and recovers the next token; the greedy continuation still drifts (4-bit is
lossy), but far less than per-channel.

The C++ logits match nanoinfer to ~4e-5 max abs diff (float accumulation order),
argmax token-for-token. Greedy generation matches nanoinfer exactly — e.g. "The
capital of France is" → " Paris. It is the largest city in Europe and the second".
Generation uses a KV cache (prefill + decode); cached output is bit-identical to
the full recompute and ~7× faster.

## Use from Python (F6: the pybind11 binding)

The `nicpp` module is the pivot from "pure C++ engine" to the vLLM shape — Python
orchestration over our own C++ kernels. It stays thin: it hands Python the same
objects the C++ drivers use (`Model`, `KVCache`, `SamplingParams`), with
`forward()` returning logits as a numpy array and `generate()` returning token ids.
The kernels run with the GIL released. Tokenization stays in Python (HF, allowed by
the golden rule); the core works purely on token ids.

```python
import sys; sys.path.insert(0, "build")     # the .so lands in build/
import nicpp

model = nicpp.Model("weights/qwen2.5-0.5b")          # or nicpp.quant_mode("q8")
ids = [785, 6722, 315, 9625, 374]                     # "The capital of France is"
logits = model.forward(ids)                           # numpy [seq, vocab]

cache = model.make_cache(len(ids) + 32)               # per-sequence KV cache
model.forward(ids, cache)                             # prefill; cache.length advances
out = model.generate(ids, max_tokens=20)              # greedy ids (uses a KV cache)
```

`forward(ids, cache)` + `make_cache` are exactly the per-sequence primitives the F7
scheduler will drive (one cache per sequence, batched orchestration in Python).

```bash
# parity through the binding: logits + greedy match nanoinfer, cached==uncached
python tests/run_binding.py weights/qwen2.5-0.5b
# text in/out on the C++ engine (HF tokenize -> our kernels -> HF decode)
python tools/generate.py --prompt "The capital of France is" --max-tokens 20
python tools/generate.py --prompt "Once upon a time" --temperature 0.8 --top-p 0.95 --seed 1234
python tools/generate.py --prompt "..." --quant q8     # run quantized weights
```

`run_binding.py` is the F6 analogue of `run_parity` / `run_generate` — the same
MATCH bar, driven from Python. Like them it needs the weight export + reference
dump first, so it isn't wired into `ctest`. Greedy is deterministic and matches
nanoinfer; sampled output is not token-identical (the C++ RNG differs from torch).

### Continuous batching (F7) + batched decode (F8a)

`python/scheduler.py` is the serving layer: a `Scheduler` that runs many requests
at once over the F6 kernels. Each request gets its own KV cache; every step decodes
the running set by one token, evicts finished sequences, and admits queued ones into
the freed slots — *continuous* batching, so a short request never waits behind a
long one and slots stay full while there's work. Token selection is in Python
(numpy), the same warpers/order as nanoinfer/sampling.py.

By default (`batched=True`) the scheduler decodes the whole running set in one
`Model.forward_batch(tokens, caches)` call (stage **F8a**): the per-sequence
projection GEMMs (q/k/v/o/gate/up/down) fuse into one matmul over the N rows, so each
weight is streamed once and reused across all N tokens — the same row-reuse that
makes prefill compute-bound, now applied to decode. Attention stays a per-sequence
loop (each token attends only its own cache, at its own position). Row s is
bit-identical to a standalone `forward([tokens[s]], caches[s])`, so output is
unchanged — `batched=False` keeps the per-sequence `forward()` loop (F7) for A/B.

```python
import sys; sys.path.insert(0, "build"); sys.path.insert(0, "python")
import nicpp
from scheduler import Request, Scheduler

model = nicpp.Model("weights/qwen2.5-0.5b")
sched = Scheduler(model, max_batch=4)
sched.add(Request("a", [785, 6722, 315, 9625, 374], max_tokens=16))   # one cache each
sched.add(Request("b", [785, 6722, 315], max_tokens=8))
outputs = sched.run()        # {request_id: output_ids}; or call sched.step() yourself
```

```bash
# F8a kernel: batched decode == N standalone forwards (bit-identical) + a tok/s sweep
./build/run_batch weights/qwen2.5-0.5b
# greedy output is token-identical to standalone generate at every batch size and on
# both decode paths (batched / per-sequence), incl. a repetition-penalty request
python tests/run_serve.py weights/qwen2.5-0.5b
# serve several prompts at once (HF tokenize -> scheduler -> HF decode)
python tools/serve.py --max-batch 2 --max-tokens 16 \
    --prompt "The capital of France is" --prompt "Once upon a time" --prompt "Water boils at"
```

Batched decode throughput (`run_batch`, Qwen2.5-0.5B, 20-core CPU, aggregate tok/s):

| batch | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| decode tok/s | 21.7 | 33.6 | 45.9 | 49.6 | 53.4 |
| speedup | 1.0× | 1.5× | 2.1× | 2.3× | 2.5× |

The win comes from weight reuse: batch-1 decode streams every weight to emit one
token (memory-bound), while batched decode streams it once for N tokens. It tapers
as the fused GEMMs turn the step compute-bound — the same wall prefill hits.

### Paged attention (F8b)

`Scheduler(block_size=...)` switches the KV cache from contiguous to **paged** (the
vLLM idea, `src/paged.hpp`): one shared `BlockPool` of fixed-size blocks, a
per-sequence block table (`PagedKVCache`). Both implement a `KVCacheBase` interface,
so `forward`/`forward_batch` drive either through one pointer — the paged `update()`
gathers the filled prefix into the same contiguous view the attention kernel takes,
so output is **bit-identical** to the contiguous cache (`run_paged.cpp`, `max|diff|=0`).

The point is memory: no per-sequence `[max_seq]` preallocation. Blocks are drawn from
the shared pool as a sequence grows and returned on finish, so memory tracks actual
lengths and the pool is reused across requests. The Python block scheduler admits a
request only when the pool can reserve its worst case (so lazily-allocated blocks
can't over-commit), and frees its blocks when it finishes — token-identical to
standalone, with admission now gated on KV blocks rather than just slots.

The paged `attend()` is a true paged-attention kernel: each query head reads K/V
straight from the blocks via the block table, mapping query head → KV head inline so
GQA needs no `repeat_kv` copy and no contiguous gather. It mirrors the attention op's
arithmetic exactly (same simd dot, double-accumulated softmax and value sum, same key
order), so it stays bit-identical to the contiguous path while moving far less memory
per step — `run_paged` measures ~1.5× decode at context 128, a gap that widens with
context since the skipped gather/repeat copies scale with it.

```python
sched = Scheduler(model, max_batch=8, block_size=16, num_blocks=256)  # paged
sched.add(Request("a", [785, 6722, 315, 9625, 374], max_tokens=64))
outputs = sched.run()   # PagedKVCache per sequence; blocks recycled through the pool
```

```bash
# paged forward == contiguous forward (bit-identical, single + batched) + pool reuse
./build/run_paged weights/qwen2.5-0.5b
# serve with a paged KV cache (tight pool -> admission gated on blocks)
python tools/serve.py --block-size 16 --num-blocks 64 --max-batch 4 \
    --prompt "The capital of France is" --prompt "Once upon a time"
```

### Prefix sharing (F8c: RadixAttention)

`Scheduler(prefix_sharing=True)` lets requests reuse a shared prompt prefix's KV
instead of recomputing it. KV blocks are reference-counted (`BlockPool`), and a
`PrefixCache` keyed by the block-aligned token prefix maps prefixes to their physical
blocks. On admit, the scheduler matches the longest cached prefix, seeds the sequence
with those blocks (`PagedKVCache.share_prefix`, increfing them), and prefills **only
the suffix**; on prefill it registers the sequence's complete blocks for later reuse.
A block frees only when no holder (sequence or cache) remains.

It's exact: a token's K/V depends only on the tokens up to it (causal attention), so a
shared prefix's KV equals recomputing it — output is token-identical to standalone.
The win is skipped prefill and shared memory: in `run_serve.py`, three requests with a
24-token common prefix skip 48 prefill tokens and share 6 blocks, output unchanged.

```bash
python tools/serve.py --block-size 4 --num-blocks 128 --prefix-sharing --max-tokens 8 \
    --prompt "The capital of France is the city of" \
    --prompt "The capital of France is famous for"   # 2nd reuses the shared prefix
```

With F8c the Fusion track is a mini-vLLM: Python orchestration (continuous batching +
paged block scheduler + prefix sharing) over our own C++ kernels (batched decode +
paged attention), parity-locked end to end. Deliberately left for later: batched
sampling (the draw is still per-sequence in Python) and a true int8×int8→int32 GEMM
(C4 quant is still weight-only, so it saves memory, not compute).

## Performance (C5: SIMD + threads)

The hot kernels (`linear`, the quant linears, attention) reduce through the AVX2/FMA
dot products in `simd.hpp`, parallelize over output channels / heads with OpenMP, and
keep the **row loop innermost** so each weight row is loaded once and reused across all
prefill rows — streamed once per call, not once per row. All three are deliberately
**parity-preserving**: the SIMD path accumulates in `double` (it only re-associates the
sum) and threading splits *output channels*, so each output is one thread's complete
reduction. The logits are therefore unchanged — still `max|diff| 4.24e-5` vs nanoinfer,
greedy still token-for-token — while the engine gets much faster. `run_bench`,
Qwen2.5-0.5B, 20-core CPU:

| build (fp32, prefill 128 / decode 32)   | prefill tok/s | decode tok/s |
| --------------------------------------- | ------------- | ------------ |
| scalar + serial (pre-C5)                | 4.9           | 4.0          |
| C5: AVX2/FMA + OpenMP + weight reuse     | 81   (≈16×)   | 8.6  (2.1×)  |

Three compounding levers act on **prefill**: SIMD (~2.4× single-thread), cores, and —
the big one — streaming each weight *once* instead of once-per-row. The last shows up
in the per-token cost: before the weight-reuse fix, prefill got *more* expensive per
token as the prompt grew (≈30→34 ms/tok at 64→256 tokens, because every row re-read the
whole weight matrix); after, it *drops* and flattens (≈14→12 ms/tok), the weight read
amortized across rows. Prefill is now compute-bound.

**Decode stays memory-bound** and gets only the SIMD + threads ≈2.1×: it processes one
new token, so every weight is streamed once and used once — there is no row dimension to
amortize over, and extra cores just hit the memory-bandwidth wall. That wall is exactly
what weight quantization (C4) attacks: q8 streams a quarter of the bytes, so q8 decode
(10.0 tok/s) overtakes fp32 (8.6) — even though q8 *prefill* is slower, since the
int8→float widening adds compute the now-compute-bound prefill can't hide (weight-only
quant saves memory, not compute).

Deliberately left out: a NEON path (slots into `simd.hpp` at the marked `#elif`),
float32-accumulation (the further, lossy step llama.cpp takes — it would break the
parity floor), and a SIMD nibble-unpack for q4/q4g (would only help compute-bound q4
prefill, an uncommon path).

## Status

- [x] **C0** — Tensor + ops (matmul/rmsnorm/softmax/add), CMake, numpy parity
- [x] **C1** — Qwen2.5 forward pass; NIT0 weight export, logit parity vs nanoinfer
- [x] **C2** — sampling + generate loop; greedy generation matches nanoinfer token-for-token
- [x] **C3** — KV cache (prefill/decode); bit-identical to full recompute, ~7× faster
- [x] **C4** — weight quantization (q8 / q4 / q4g): per-channel int8 & int4 plus
      group-wise int4 (Q4_0-style), behind a QuantizedWeight interface. (Optional
      extension: quantize the embedding/lm_head too, for more overall savings.)
- [x] **C5** — SIMD (AVX2/FMA double-accum dot, scalar fallback) + OpenMP threads
      over output channels/heads + once-per-call weight streaming (row loop inner).
      Parity-preserving: logits unchanged (4.24e-5 vs nanoinfer, greedy
      token-for-token); ~16× prefill, ~2.1× decode (memory-bound) on 20 cores.
- [x] **F6** — pybind11 binding (`nicpp`): Model / KVCache / SamplingParams exposed
      to Python, `forward()` → numpy logits, `generate()` → ids, GIL dropped during
      the kernels. Parity-tested through the binding and a text demo (HF tokenize →
      C++ kernels → HF decode). See "Use from Python" below. The F7 substrate.
- [x] **F7** — Python serving layer: a continuous-batching `Scheduler`
      (`python/scheduler.py`) over the F6 kernels — per-sequence KV cache, one-token
      decode per step, dynamic admit/evict (no head-of-line blocking). Greedy output
      is token-identical to standalone generate at every batch size
      (`tests/run_serve.py`); `tools/serve.py` serves several prompts at once. The
      *scheduling* win; batched-GEMM throughput is the next kernel step.
- [x] **F8a** — batched decode kernel: `Model::forward_batch` decodes N sequences in
      one pass, fusing the per-sequence projection GEMMs over the N rows (attention
      stays a per-sequence loop). Bit-identical to N standalone forwards
      (`tests/run_batch.cpp`), driven under the scheduler (`batched=True`); ~2.5×
      aggregate decode tok/s at batch 16. The throughput lever F7 was missing.
- [x] **F8b** — paged attention: a shared `BlockPool` of fixed-size KV blocks + a
      per-sequence block table (`PagedKVCache`), behind a `KVCacheBase::attend()`
      interface so `forward`/`forward_batch` drive either cache. The paged kernel
      indexes K/V straight from the blocks (no gather, no repeat_kv) — bit-identical
      to the contiguous cache (`tests/run_paged.cpp`), ~1.5× decode at context 128.
      The Python block scheduler (`block_size=...`) gates admission on KV blocks and
      frees them on finish — no per-sequence max_seq preallocation. The vLLM merge point.
- [x] **F8c** — prefix sharing (RadixAttention): refcounted KV blocks + a `PrefixCache`
      keyed by the block-aligned token prefix. A request reuses a cached prefix's
      blocks (`PagedKVCache.share_prefix`) and prefills only its suffix
      (`Scheduler(prefix_sharing=True)`) — bit-identical to standalone, the win being
      skipped prefill + shared blocks (`tests/run_serve.py`, `tests/run_paged.cpp`).
