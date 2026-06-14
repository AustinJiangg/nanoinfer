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

## Performance (C5: SIMD + threads)

The hot kernels (`linear`, the quant linears, attention) reduce through the AVX2/FMA
dot products in `simd.hpp` and parallelize over output channels / heads with OpenMP.
Both choices are deliberately **parity-preserving**: the SIMD path accumulates in
`double` (it only re-associates the sum), and threading splits *output channels* so
each output is one thread's complete reduction. The logits are therefore unchanged —
still `max|diff| 4.24e-5` vs nanoinfer, greedy still token-for-token — while the
engine gets much faster. `run_bench`, Qwen2.5-0.5B, 20-core CPU, prefill 128 / decode 32:

| build (fp32)                | prefill tok/s | decode tok/s |
| --------------------------- | ------------- | ------------ |
| scalar + serial (pre-C5)    | 4.9           | 4.0          |
| AVX2+FMA, 1 thread          | 11.9  (2.4×)  | 6.4  (1.6×)  |
| AVX2+FMA, 20 threads        | 37.3  (7.6×)  | 8.6  (2.1×)  |

The gap between the two columns is the roofline. **Prefill is compute-bound** (the
weight is reused across all 128 rows), so SIMD *and* cores both pay off → 7.6×.
**Decode is memory-bound** (one new token, so every weight is streamed once and used
once), so it saturates memory bandwidth quickly — extra cores can't beat the wall,
and the win is only ~2×. That wall is exactly what weight quantization (C4) attacks:
q8 streams a quarter of the bytes, so q8 decode (10.0 tok/s) overtakes fp32 (8.6),
even though q8 *prefill* is slower (the int8→float widening adds compute that the
reused-weight regime can't hide — weight-only quant saves memory, not compute).

A NEON path slots into `simd.hpp` at the marked `#elif`; float32-accumulation (the
further, lossy step llama.cpp takes) is left out on purpose — it would break the
parity floor this project is built around.

## Status

- [x] **C0** — Tensor + ops (matmul/rmsnorm/softmax/add), CMake, numpy parity
- [x] **C1** — Qwen2.5 forward pass; NIT0 weight export, logit parity vs nanoinfer
- [x] **C2** — sampling + generate loop; greedy generation matches nanoinfer token-for-token
- [x] **C3** — KV cache (prefill/decode); bit-identical to full recompute, ~7× faster
- [x] **C4** — weight quantization (q8 / q4 / q4g): per-channel int8 & int4 plus
      group-wise int4 (Q4_0-style), behind a QuantizedWeight interface. (Optional
      extension: quantize the embedding/lm_head too, for more overall savings.)
- [x] **C5** — SIMD (AVX2/FMA double-accum dot, scalar fallback) + OpenMP threads
      over output channels/heads. Parity-preserving: logits unchanged (4.24e-5 vs
      nanoinfer, greedy token-for-token); ~7.6× prefill, ~2.1× decode on 20 cores.
