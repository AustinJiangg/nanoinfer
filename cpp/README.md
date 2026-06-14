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

## Layout

```
cpp/
  src/
    tensor.{hpp,cpp}     row-major contiguous float32 Tensor (shape/stride/data)
    ops.{hpp,cpp}        matmul, rmsnorm, softmax, add — naive, readable
    serialize.{hpp,cpp}  read/write the "NIT0" tensor format (the parity bridge)
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
# 3d. run quantized; report memory + accuracy vs fp32 (q8 default, or q4)
./build/run_quant weights/qwen2.5-0.5b q8
./build/run_quant weights/qwen2.5-0.5b q4
```

Q8 preserves the greedy next token (11/12 vs fp32). Naive per-channel **Q4 is too
coarse** — one scale per row can't absorb weight outliers, so the model's output
changes (next-token not preserved). That's the motivation for group-wise quant
(llama.cpp's Q4_0 uses per-32-element blocks), the next step.

The C++ logits match nanoinfer to ~4e-5 max abs diff (float accumulation order),
argmax token-for-token. Greedy generation matches nanoinfer exactly — e.g. "The
capital of France is" → " Paris. It is the largest city in Europe and the second".
Generation uses a KV cache (prefill + decode); cached output is bit-identical to
the full recompute and ~7× faster.

## Status

- [x] **C0** — Tensor + ops (matmul/rmsnorm/softmax/add), CMake, numpy parity
- [x] **C1** — Qwen2.5 forward pass; NIT0 weight export, logit parity vs nanoinfer
- [x] **C2** — sampling + generate loop; greedy generation matches nanoinfer token-for-token
- [x] **C3** — KV cache (prefill/decode); bit-identical to full recompute, ~7× faster
- [~] **C4** — quantization (weight-only, per-channel): Q8 int8 (layer weights 4×,
      next-token preserved) + Q4 int4 (8×, but too coarse — motivates group-wise);
      group-wise (Q4_0-style) + embedding quantization next
- [ ] **C5** — SIMD / multithreading
