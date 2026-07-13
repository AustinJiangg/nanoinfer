"""HF-parity tests for Granite-3.1-1B-A400M — the A3 architecture (MoE).

Granite is the phase's biggest structural delta: the dense FFN becomes a sparse
mixture-of-experts (32 experts, top-8, SwiGLU experts at width 512), plus the muP
scalar set the A0 config described — embedding*12.0, attention scale 0.015625
(REPLACES 1/sqrt(head_dim)), residual*0.22, logits/6.0. The router semantics are
locked here by parity, not docs: logits in float32, top-k selects, softmax over
ONLY the selected k. Attention/cache/masking are untouched — MoE swaps the FFN
only, which is exactly what these tests prove end-to-end.

Weights are Apache 2.0 (ibm-granite), ~2.6 GB bf16 download, so slow-marked.
"""

import pytest
import torch

MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


@pytest.mark.slow
def test_granite_config_flags():
    from transformers import AutoConfig

    from nanoinfer.config import ModelConfig

    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    assert cfg.model_type == "granitemoe"
    assert (cfg.n_experts, cfg.moe_top_k) == (32, 8)
    # GraniteMoe keeps the expert width in intermediate_size — from_hf must fall
    # back to it (there is no moe_intermediate_size field to read).
    assert cfg.moe_intermediate_size == 512
    assert cfg.embedding_multiplier == 12.0
    assert cfg.attention_multiplier == 0.015625
    assert cfg.residual_multiplier == 0.22
    assert cfg.logits_scaling == 6.0
    assert cfg.rope_theta == 1_500_000.0 and cfg.rope_scaling is None
    assert cfg.head_dim == 64
    assert (cfg.num_attention_heads, cfg.num_kv_heads) == (16, 8)
    assert cfg.qkv_bias is False and cfg.qk_norm is False
    assert cfg.tie_word_embeddings is True
    assert cfg.vocab_size == 49152


@pytest.mark.slow
def test_granite_forward_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoinfer.weights import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

    our_model, _ = load_model(MODEL, dtype=torch.float32, device="cpu", hf_dtype="auto")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_model.eval()

    with torch.no_grad():
        ours = our_model(input_ids)
        theirs = hf_model(input_ids).logits

    assert ours.shape == theirs.shape
    # A wrong router (gate order, softmax-over-all instead of over-selected) or a
    # misplaced muP scalar shifts logits far beyond this tolerance.
    assert torch.allclose(ours, theirs, atol=1e-3, rtol=1e-3)


@pytest.mark.slow
def test_granite_greedy_matches_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanoinfer.generate import generate
    from nanoinfer.weights import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt = "The capital of France is"

    our_model, _ = load_model(MODEL, dtype=torch.float32, device="cpu", hf_dtype="auto")
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
def test_granite_moe_layer_matches_hf():
    """The MoE module alone vs HF's block_sparse_moe on one real layer.

    Isolates the router+experts from the rest of the forward: feed both the same
    random hidden states. Catches a gate/up swap or a wrong softmax domain with a
    much tighter tolerance than the end-to-end logits can.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from nanoinfer.config import ModelConfig
    from nanoinfer.layers import MoE
    from nanoinfer.weights import _remap_granitemoe

    cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL))
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    hf_layer = hf_model.model.layers[0].block_sparse_moe
    hf_layer.eval()

    ours = MoE(cfg)
    sd = {}
    for name, tensor in hf_model.state_dict().items():
        if name.startswith("model.layers.0.block_sparse_moe."):
            for our_name, t in _remap_granitemoe(name, tensor):
                sd[our_name.removeprefix("layers.0.mlp.")] = t
    ours.load_state_dict(sd)
    ours.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 7, cfg.hidden_size)
    with torch.no_grad():
        got = ours(x)
        want = hf_layer(x)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
