# nanoinfer — project guide for Claude Code

A from-scratch LLM inference engine, built for learning. The forward pass and
generation loop are written by hand; only weight loading and tokenization lean
on Hugging Face. The goal is to understand inference, then progressively add the
optimizations real engines use (KV cache, batching, paged attention).

## Golden rule

**Do not call `model.generate()` or any HF generation utility.** The whole point
is that we implement the forward pass and sampling loop ourselves. HF is allowed
ONLY for: downloading weights (`AutoModelForCausalLM` to read tensors),
tokenization (`AutoTokenizer`), and config. If you find yourself reaching for an
HF helper that does the thinking for us, stop — that's a signal we're skipping
the part we're here to learn.

## Architecture target

Llama-family decoder-only transformer. This is the current de-facto standard, so
learning it transfers to most open models. Components:

- RMSNorm (not LayerNorm)
- Rotary position embeddings (RoPE), applied to Q and K
- Grouped-query attention (GQA) — fewer KV heads than Q heads
- SwiGLU feed-forward (gate + up projections, SiLU activation)
- Tied or untied LM head, depending on the checkpoint

Default model: `Qwen/Qwen2.5-0.5B` (CPU-friendly, fast iteration, Llama-style).
Keep the code model-agnostic where reasonable — read dims from the HF config,
don't hardcode them.

## Repo layout

```
nanoinfer/
  __init__.py
  config.py        # ModelConfig dataclass, loaded from HF config
  weights.py       # load HF checkpoint -> our own state dict naming
  layers.py        # rmsnorm, rope, attention, swiglu, transformer block
  model.py         # assembles blocks; forward(input_ids) -> logits
  sampler.py       # greedy + temperature/top-k/top-p (stage 2)
  generate.py      # the autoregressive loop; CLI entry point
tests/
  test_layers.py   # shape + numerical sanity (incl. parity vs HF where cheap)
  test_generate.py # end-to-end: greedy output is deterministic & coherent
ROADMAP.md         # staged plan; update as stages land
```

## How we work in this repo

- **One stage at a time.** See ROADMAP.md. Don't pull stage-3 batching into the
  stage-0 forward pass. Land the minimal version, make tests pass, then move on.
- **Test the seams.** Every new layer gets a shape test and, where it's cheap,
  a numerical-parity test against the equivalent HF module (load one HF layer,
  feed both the same input, assert `allclose`). This is the fastest way to catch
  a transposed weight or a wrong RoPE frequency.
- **Small, reviewable diffs.** A layer or a function per change, with its test.
- **Determinism first.** Greedy decoding must be reproducible before we add
  sampling. Tests assume seeded, deterministic behavior.
- **Comment the *why*, not the *what*.** Especially for the fiddly bits: RoPE
  interleaving, GQA head repetition, attention masking, KV cache indexing.

## Commands

```bash
pip install -e ".[dev]"           # install package + dev deps
pytest                            # run all tests
pytest tests/test_layers.py -v    # one file
python -m nanoinfer.generate --prompt "The capital of France is" --max-tokens 20
```

## Known sharp edges (read before debugging)

- **RoPE**: Qwen2/Llama use the "neox/half-split" rotation (split the head dim in
  half), not the interleaved GPT-J style. Mismatch here gives plausible-looking
  but degrading output — a classic time sink. Parity-test against HF's RoPE.
- **GQA**: KV heads must be repeated to match Q heads before attention. Off-by-one
  on `n_rep = n_q_heads // n_kv_heads` silently breaks attention.
- **Causal mask**: in stage 0 (full recompute) we mask the whole `[T, T]` matrix.
  Once the KV cache lands (stage 1) the query is length 1 and attends to all
  cached keys — the masking logic changes, don't copy stage-0 masking blindly.
- **dtype**: do everything in float32 on CPU for sanity/parity. Worry about
  bf16/fp16 only once correctness is locked.
- **Weight naming**: HF Qwen2 uses `model.layers.{i}.self_attn.q_proj` etc.
  weights.py is the one place that knows HF's names; the rest of the code uses
  our naming. Keep that boundary clean.
