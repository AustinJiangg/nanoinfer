"""HF-parity tests for Qwen3-0.6B — the A1 architecture.

Qwen3 is the smallest delta over Qwen2.5: per-head QK-Norm, no QKV bias, and an
explicit head_dim=128 (so q_proj is 1024->2048, NOT square — this flushes any
hidden `n_heads * head_dim == hidden_size` assumption). Nothing else changes.
These tests prove the same forward pass, with only the A0 feature flags flipped
(qk_norm=True, qkv_bias=False), reproduces HF's logits and greedy tokens.

They download Qwen3-0.6B (~1.2 GB), so they're marked slow. Run the fast layer
tests alone with:  pytest -m "not slow"
"""

import pytest
import torch

MODEL = "Qwen/Qwen3-0.6B"


@pytest.mark.slow
def test_qwen3_config_flags():
    # The A0 config reader must flip exactly the Qwen3 flags off the HF config,
    # with no hardcoding — the whole point of the feature-flag layer.
    from transformers import AutoConfig

    from nanoinfer.config import ModelConfig

    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.model_type == "qwen3"
    assert cfg.qk_norm is True          # A1: per-head QK-Norm on
    assert cfg.qkv_bias is False        # A1: no QKV bias
    assert cfg.head_dim == 128
    # The non-square q_proj: 16 heads * 128 != hidden 1024.
    assert cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size
    assert cfg.rope_theta == 1_000_000  # read from rope_parameters, not the default
    assert cfg.rope_scaling is None     # rope_type "default" normalized away


@pytest.mark.slow
def test_qwen3_forward_matches_hf():
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
    # Same loose-ish tolerance as the Qwen2.5 parity test: absorbs SDPA-vs-HF op
    # ordering while still catching a wrong QK-Norm placement or transposed weight.
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_qwen3_greedy_matches_hf():
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
