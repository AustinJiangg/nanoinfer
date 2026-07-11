# nanoinfer — working guide (Claude Code & contributors)

How we develop in this repo: the rules, the conventions, and the sharp edges.
For the project overview, architecture, and commands see **README.md**; for the
staged plan and what's next see **ROADMAP.md**. This file deliberately does not
repeat them — it's the "how we work here," not the "what this is."

## Golden rule

**Do not call `model.generate()` or any HF generation utility.** The whole point
is that we implement the forward pass and sampling loop ourselves. Hugging Face is
allowed ONLY for: downloading weights (`AutoModelForCausalLM`, to read tensors),
tokenization (`AutoTokenizer`), and config (`AutoConfig`). If you find yourself
reaching for an HF helper that does the thinking for us, stop — that's the signal
we're skipping the part we're here to learn.

## How we work

- **One stage at a time.** See ROADMAP.md. Don't pull stage-3 batching into the
  stage-0 forward pass. Land the minimal version, make tests pass, then move on.
- **Test the seams.** Every new layer gets a shape test and, where it's cheap, a
  numerical-parity test against the equivalent HF module (feed both the same
  input, assert `allclose`). It's the fastest way to catch a transposed weight or
  a wrong RoPE frequency, and it's the correctness floor — trust parity over
  `getattr` config defaults, which fail silently (see sharp edges).
- **Model-agnostic.** Read every dimension from the HF config; don't hardcode.
  The same code should run on any Llama-family checkpoint.
- **Small, reviewable diffs.** A layer or a function per change, with its test.
- **Determinism first.** Greedy decoding must be reproducible before we add
  sampling. Tests assume seeded, deterministic behavior.
- **Comment the *why*, not the *what*.** Especially the fiddly bits: RoPE
  half-split, GQA head repetition, attention masking, KV-cache indexing.

## The tiered gate (what runs where)

Two tiers, split by what they need. The **golden-token** gates need the ~1 GB weight
export + a reference dump (`cpp/tools/export_weights.py` + `cpp/tools/dump_reference.py`),
and the CUDA ones a GPU — none of which the free CI runners have. So the gate is split,
not weakened:

- **CI tier** (`.github/workflows/ci.yml`, model-free + GPU-free) — unit tests and
  numpy-fixture parity, the seams that don't need a checkpoint. Four jobs:
  `pytest -m "not slow"` (the Python oracle's fast layer tests, no download);
  `ctest -L nomodel` (the C++ core's self-contained unit tests + the numpy-parity
  `ops_parity`); the aarch64/qemu **NEON** parity (the checked-in cross-compile
  recipe); and a compile-only `nicore` build with `NI_CUDA=ON` in a CUDA container
  (no GPU → build-breakage detection is the ceiling). Catches build breakage and math
  regressions in the model-free code — a **backstop, not the floor**.
- **Local pre-commit tier** (needs the weight export + ref dump; the CUDA gates need
  the GPU) — the golden-token gates that prove **token-for-token identity to the
  oracle**, the actual correctness floor: `ctest -L weights` (run_parity / run_generate
  / run_cache / run_batch / run_paged / run_quant_*), the CUDA gates (run_cuda_parity /
  cache / paged / batch), the Python serving parity (`cpp/tests/python/run_serve.py`,
  `run_spec*.py`, `run_http_serve.py`, …), and the full `pytest` (incl. the slow
  HF-parity tests). `ctest -L nomodel` reproduces the CI C++ subset locally (use
  `-L nomodel`, not `-LE weights` — the latter also pulls the GPU-only `test_cuda`,
  which is unlabelled, in a `NI_CUDA=ON` build).

**The rule:** run the full local gate before committing anything that touches the
forward pass, a kernel, a cache, or the scheduler — CI cannot see those regress (it has
no weights). CI green ≠ safe to ship; it means "didn't break the build or the model-free
math." The `nomodel` / `weights` ctest labels are exactly this split.

## Known sharp edges (read before debugging)

- **RoPE rotation**: Qwen2/Llama use the "neox / half-split" rotation (split the
  head dim in half), not the interleaved GPT-J style. A mismatch gives
  plausible-looking but degrading output — a classic time sink. Parity-test
  against HF's RoPE.
- **RoPE theta location**: transformers 5.x moved `rope_theta` out of the
  top-level config into `config.rope_parameters["rope_theta"]`. A bare
  `getattr(hf_config, "rope_theta", 10000.0)` silently returns the 10000 default
  instead of Qwen2.5's 1e6 → wrong frequencies → the degrading output above.
  `config.py` reads both layouts (`_read_rope_theta`). Treat any
  `getattr(hf_config, …, default)` as suspect across transformers versions.
- **GQA**: KV heads must be repeated to match Q heads before attention. An
  off-by-one on `n_rep = n_q_heads // n_kv_heads` silently breaks attention.
- **Causal mask**: stage 0 (full recompute) masks the whole `[T, T]` matrix. Once
  the KV cache lands (stage 1) the query is length 1 and attends to all cached
  keys — the masking logic changes; don't copy stage-0 masking blindly.
- **dtype**: do everything in float32 on CPU for sanity/parity. Worry about
  bf16/fp16 only once correctness is locked.
- **Weight naming**: HF Qwen2 uses `model.layers.{i}.self_attn.q_proj` etc.
  `weights.py` is the one place that knows HF's names; the rest of the code uses
  our naming. Keep that boundary clean.
- **GPU parity is not bit-identical** (G1+, `cpp/` multi-backend): a CUDA/Metal
  backend reorders the float reductions in matmul/attention, so its logits differ
  from the CPU backend by ~1e-4 *even when the math is right* — `max|diff|=0` is a
  CPU-only luxury. The bar for accelerator backends is **tolerance + tokens**: CUDA
  ≈ CPU within ~1e-3..1e-4 AND greedy token-for-token against the golden dump. The
  CPU backend (itself HF-parity-locked) is the oracle. Two companion GPU rules:
  upload each weight to the device **once** at load (never per forward — the GPU
  version of C5's "stream each weight once"), and keep activations resident on
  device (H2D/D2H only at the token-id input and the logits output).
