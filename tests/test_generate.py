"""End-to-end tests for stage 0.

These download a small model, so they're marked `slow`. Run the fast layer tests
alone with:  pytest -m "not slow"

The key test is parity: our forward pass must produce the same logits as HF's for
the same input. If that holds, greedy decoding is automatically correct, because
greedy is a deterministic function of the logits.
"""

import pytest
import torch

MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.mark.slow
def test_forward_matches_hf():
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
    # Float32 on CPU should match closely. Loose-ish atol absorbs op-ordering
    # differences (e.g. SDPA vs HF's attention) while still catching real bugs.
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_greedy_matches_hf():
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


@pytest.mark.slow
def test_greedy_is_deterministic():
    from transformers import AutoTokenizer

    from nanoinfer.generate import generate
    from nanoinfer.weights import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_model(MODEL, dtype=torch.float32, device="cpu")

    a = generate(model, tokenizer, "Hello, my name is", max_tokens=8)
    b = generate(model, tokenizer, "Hello, my name is", max_tokens=8)
    assert a == b
