"""Load a Hugging Face Llama-family checkpoint into our Model.

This is the ONLY module that knows HF's parameter naming. Everything else uses
our own module names. HF's Qwen2/Llama naming maps almost 1:1 onto ours because
we named our submodules to match — the main jobs here are loading the config,
handling tied embeddings, and (optionally) loading just one HF layer for parity
tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .config import ModelConfig
from .model import Model

if TYPE_CHECKING:
    # For the return annotation only — keeps the real transformers import lazy
    # (it stays inside load_model) while letting type checkers resolve the name.
    from transformers import AutoTokenizer


def load_model(model_name: str, dtype=torch.float32, device="cpu") -> tuple[Model, AutoTokenizer]:
    """Download/load `model_name`, build our Model, copy weights in.

    Returns (model, tokenizer). The model is in eval mode on `device`.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(model_name)
    cfg = ModelConfig.from_hf(hf_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load HF weights only to read tensors out of them — we never call generate().
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    hf_sd = hf_model.state_dict()

    model = Model(cfg).to(dtype=dtype)
    our_sd = model.state_dict()

    remapped = _remap_state_dict(hf_sd, our_sd, cfg)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # rope_cos/rope_sin are non-persistent buffers, so they show up as "missing"
    # from the checkpoint — that's expected. Anything else is a real problem.
    missing = [m for m in missing if not m.startswith("rope_")]
    if missing:
        raise RuntimeError(f"Missing weights after remap: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected weights after remap: {unexpected}")

    model.to(device).eval()
    return model, tokenizer


def _remap_state_dict(hf_sd: dict, our_sd: dict, cfg: ModelConfig) -> dict:
    """Translate HF parameter names to ours.

    HF Qwen2 layout:
        model.embed_tokens.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.norm.weight
        lm_head.weight                       (absent if embeddings are tied)

    Our layout drops the leading "model." for the body and keeps lm_head at top.
    """
    out: dict[str, torch.Tensor] = {}

    for hf_name, tensor in hf_sd.items():
        if hf_name == "lm_head.weight":
            out["lm_head.weight"] = tensor
        elif hf_name.startswith("model."):
            out[hf_name[len("model."):]] = tensor
        else:
            # Be loud about anything we didn't anticipate rather than silently
            # dropping it.
            raise RuntimeError(f"Unrecognized HF weight name: {hf_name}")

    # Tied embeddings: some checkpoints omit lm_head and reuse the input embedding.
    if cfg.tie_word_embeddings and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["embed_tokens.weight"]

    return out
