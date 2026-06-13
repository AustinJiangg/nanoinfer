# nanoinfer

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
nanoinfer/
  config.py     ModelConfig, loaded from the HF config
  weights.py    load an HF checkpoint -> our own state-dict naming
  layers.py     rmsnorm, rope, attention, swiglu, transformer block
  model.py      assembles the blocks; forward(input_ids) -> logits
  generate.py   the autoregressive loop + CLI entry point
tests/
  test_layers.py    shape + numerical-parity tests vs HF
  test_generate.py  end-to-end: greedy output matches HF, deterministic
ROADMAP.md      the staged plan
CLAUDE.md       how we develop here: the golden rule, conventions, sharp edges
```

## Status & roadmap

Stage 0 is complete and verified against Hugging Face. See [ROADMAP.md](ROADMAP.md)
for the full plan.

- [x] **Stage 0** — naive forward pass + greedy decoding (full recompute, no cache)
- [ ] **Stage 1** — KV cache (prefill / decode split)
- [ ] **Stage 2** — sampling: temperature, top-k, top-p, repetition penalty
- [ ] **Stage 3** — continuous batching
- [ ] **Stage 4** — paged attention (vLLM-style block KV cache)
- [ ] **Stage 5** — quantization / custom kernels (stretch)

## Development

Reading the code or contributing? Two companion docs split the work:

- **[CLAUDE.md](CLAUDE.md)** — how we develop here: the golden rule, conventions,
  and known sharp edges (RoPE, GQA, masking, dtype, config gotchas).
- **[ROADMAP.md](ROADMAP.md)** — the staged plan, with the "done when" bar for each
  stage and reference reading.
