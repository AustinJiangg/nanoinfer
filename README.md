# nanoinfer

[![CI](https://github.com/AustinJiangg/nanoinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/AustinJiangg/nanoinfer/actions/workflows/ci.yml)

A from-scratch LLM inference engine, built for learning. The forward pass and the
generation loop are written by hand; only weight loading and tokenization lean on
Hugging Face. The goal is to understand inference first, then progressively add the
optimizations real engines use — KV cache, continuous batching, paged attention.

## The golden rule

We never call `model.generate()` or any HF generation helper. The whole point is
that we implement the forward pass and the sampling loop ourselves. Hugging Face is
used **only** for downloading weights, tokenization, and reading the model config.

## Architecture

A Llama-family decoder-only transformer — the current de-facto standard, so what you
learn here transfers to most open models. Dimensions are read from the HF config
rather than hardcoded, so the same code runs on other Llama-style checkpoints.

Components implemented from scratch:

- **RMSNorm** (not LayerNorm)
- **Rotary position embeddings (RoPE)** — neox / half-split convention, applied to Q and K
- **Grouped-query attention (GQA)** — fewer KV heads than Q heads
- **SwiGLU** feed-forward (gate + up projections, SiLU activation)
- **Tied or untied LM head**, depending on the checkpoint

Default model: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) —
CPU-friendly and fast to iterate on.

## Install

Python 3.10+.

```bash
pip install -e ".[dev]"
```

## Run

```bash
python -m nanoinfer.generate --prompt "The capital of France is" --max-tokens 20
```

```
Prompt: The capital of France is
Completion:  Paris. It is the largest city in Europe and the second largest in the world. It is also
```

The first run downloads the model weights (~1 GB) from the Hugging Face Hub.

Decoding defaults to greedy (deterministic). Pass sampling flags to vary output:

```bash
python -m nanoinfer.generate --prompt "Once upon a time" \
    --temperature 0.8 --top-p 0.95 --top-k 50 --seed 1234
```

`--temperature 0` (the default) is greedy; a `--seed` makes sampling
reproducible. `--repetition-penalty 1.3` discourages repeats.

## Test

```bash
pytest                      # everything
pytest -m "not slow"        # fast layer tests only — no model download
pytest tests/test_layers.py -v
```

The correctness guarantee is **parity with Hugging Face**: our forward pass produces
the same logits as HF's (bit-for-bit in float32 on CPU), and greedy decoding matches
HF token-for-token. Get that right and the rest of the engine has a solid floor to
build on.

## Project layout

```
nanoinfer/        the Python engine — frozen as the reference oracle
  config.py       ModelConfig, loaded from the HF config
  weights.py      load an HF checkpoint -> our own state-dict naming
  layers.py       rmsnorm, rope, attention, swiglu, transformer block
  cache.py        per-layer KV cache for incremental decoding (stage 1)
  sampling.py     temperature / top-k / top-p / repetition penalty (stage 2)
  model.py        assembles the blocks; forward(input_ids, cache=None) -> logits
  generate.py     the autoregressive loop + CLI entry point
tests/            pytest for the oracle
  test_layers.py    shape + numerical-parity tests vs HF
  test_cache.py     cached decode == stage-0 full recompute, token-for-token
  test_sampling.py  warpers + greedy-equivalence invariants
  test_generate.py  end-to-end: greedy output matches HF, deterministic
cpp/              the C++/CUDA engine (see cpp/README.md for its layout)
  src/            C++ core + src/cuda/ backend (the `nicore` kernels)
  bindings/       nicpp — the pybind11 seam
  python/ni/      Python orchestration: scheduler, speculative, spec_scheduler,
                  async_engine + server + detok (HTTP serving)
  tools/          weight export, reference dump, generate/serve CLIs
  tests/          C++ parity tests (ctest); tests/python/ = Python parity gates
  bench/          performance benches
  docs/           BASELINE.md / BASELINE-1.5b.md — the frozen gate numbers
ROADMAP.md      the staged plan
CLAUDE.md       how we develop here: the golden rule, conventions, sharp edges
```

## Status & roadmap

The **Python engine** (`nanoinfer/`) is stages 0–2 — forward pass, KV cache, sampling — all
verified bit-for-bit against Hugging Face. It is now **frozen as the reference oracle**. The
work continues in a second, from-scratch **C++/CUDA engine** (`cpp/`): the vLLM shape (Python
orchestration over hand-written kernels), parity-locked to the Python oracle at every step
(the same discipline as Python-vs-HuggingFace). See [ROADMAP.md](ROADMAP.md) for the staged
plan and the "done when" bar for each stage.

Python engine — the oracle:

- [x] **Stage 0** — naive forward pass + greedy decoding (full recompute, no cache)
- [x] **Stage 1** — KV cache (prefill / decode split); ~2.4× faster, same tokens
- [x] **Stage 2** — sampling: temperature, top-k, top-p, repetition penalty (seeded)

C++/CUDA engine — built on our own kernels:

- [x] **C0–C5** — pure C++ core: forward pass, sampling, KV cache, int8/int4 weight quant, AVX2+OpenMP
- [x] **F6–F8c** — mini-vLLM: pybind11 bindings, continuous-batching scheduler, batched decode,
      paged attention, prefix sharing (RadixAttention)
- [x] **G0–G6** — CUDA backend: full GPU forward, paged attention, tuned GEMM / FlashAttention /
      Flash-Decoding kernels, CUDA graphs
- [x] **S0–S5** — speculative decoding: draft-model + prompt-lookup proposers, KV rollback,
      continuous-batching integration, and sampling-parity (rejection sampling)
- [x] **NEON** — the CPU backend cross-compiles and runs on Apple ARM (op-level parity)
- [x] **P-track** — perf retune for Qwen2.5-1.5B (P0 tiling default + wmma re-bench, P1 quant ROI /
      fp16-default, P2 long-context: Flash-Decoding + paging un-muted, paged+split up to 18.9× e2e)
- [x] **V-track** — HTTP serving: an asyncio engine bridge over the schedulers (plain AND spec),
      a hand-rolled HTTP/1.1 + SSE server with incremental detokenization, disconnect→cancel,
      TTFT/TPOT metrics, and a closed-loop load bench (`tools/serve_http.py`)
- [x] **E-track + A0–A2 + B1–B2** — MIT license + 4-job CI; the architecture-description
      config layer; Qwen3-0.6B/1.7B (QK-Norm), Llama-3.2-1B (llama3 RoPE scaling, tiktoken);
      bf16 weight storage (byte-exact to the shipped checkpoints, NIT1 export halves the
      file) + bf16 tensor cores (fp32 accumulate, measured)
- [ ] **Metal** — a third backend on the M4 GPU (deferred; structurally prepped)
- [ ] **Next phase remainder (A3–A4 / B3 / G7)** — Granite-1B-A400M MoE, Gemma-3-1B
      (stretch); half-precision KV cache + activations; batched/split CUDA graphs — see
      the "Next phase" section in ROADMAP.md

## Development

Reading the code or contributing? Three companion docs split the work:

- **[CLAUDE.md](CLAUDE.md)** — how we develop here: the golden rule, conventions,
  and known sharp edges (RoPE, GQA, masking, dtype, config gotchas).
- **[ROADMAP.md](ROADMAP.md)** — the staged plan, with the "done when" bar for each
  stage and reference reading.
- **[REFACTOR.md](REFACTOR.md)** — the R-track: paying down the GPU optimization
  arc's exploration-debt (the quant hole, global flags, `#ifdef` scatter) before
  Metal, parity-locked at every step.

## License

[MIT](LICENSE). The engine code is ours; model weights and tokenizer configs are
downloaded from the Hugging Face Hub under their own licenses (e.g. Qwen2.5 is
Apache-2.0) and are not redistributed here.
