"""HF-parity tests for Llama-3.2-1B — the A2 architecture.

Llama-3.2 is the config-heavy rung: the only genuinely new behavior is **llama3
RoPE scaling** — the base inv_freq is rescaled in three frequency bands at build
time (long wavelengths divided by `factor`, short ones kept, a smooth blend
between), so the rotation kernel is untouched (the rope_theta lesson applied
forward). Everything else — no QKV bias, no QK-Norm, tied embeddings, head_dim 64,
GQA 32/8 — is already handled. These tests prove the same forward pass, with only
cfg.rope_scaling flipped on, reproduces HF's logits and greedy tokens.

The meta-llama repo is gated; we use the ungated `unsloth/Llama-3.2-1B` mirror
(identical weights + config) for local parity. Downloads ~2.5 GB, so slow.
Llama Community License: don't redistribute the weights or the goldens' text.
"""

import pytest
import torch

MODEL = "unsloth/Llama-3.2-1B"


@pytest.mark.slow
def test_llama_config_flags():
    from transformers import AutoConfig

    from nanoinfer.config import ModelConfig

    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.model_type == "llama"
    assert cfg.rope_scaling is not None
    assert cfg.rope_scaling.get("rope_type") == "llama3"
    assert cfg.rope_theta == 500_000.0
    assert cfg.head_dim == 64
    assert (cfg.num_attention_heads, cfg.num_kv_heads) == (32, 8)
    assert cfg.qkv_bias is False and cfg.qk_norm is False
    assert cfg.tie_word_embeddings is True


@pytest.mark.slow
def test_llama_forward_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoinfer.weights import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    our_model, _ = load_model(MODEL, dtype=torch.float32, device="cpu")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_model.eval()

    with torch.no_grad():
        ours = our_model(input_ids)
        theirs = hf_model(input_ids).logits

    assert ours.shape == theirs.shape
    # A wrong rope-scaling band (or forgetting to scale) degrades logits subtly;
    # the same loose-ish tolerance as the other arches still catches it.
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_llama_greedy_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoinfer.generate import generate
    from nanoinfer.weights import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = "The capital of France is"

    our_model, _ = load_model(MODEL, dtype=torch.float32, device="cpu")
    ours = generate(our_model, tokenizer, prompt, max_tokens=10)

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_completion = tokenizer.decode(
        hf_out[0, input_ids.shape[1]:], skip_special_tokens=True
    )

    assert ours == hf_completion
