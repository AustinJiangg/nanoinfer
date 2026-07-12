"""Load a Hugging Face Llama-family checkpoint into our Model.

This is the ONLY module that knows HF's parameter naming. Everything else uses
our own module names. HF's Qwen2/Llama naming maps almost 1:1 onto ours because
we named our submodules to match — the main jobs here are loading the config,
handling tied embeddings, and (optionally) loading just one HF layer for parity
tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from .config import ModelConfig
from .model import Model

if TYPE_CHECKING:
    # For the return annotation only — keeps the real transformers import lazy
    # (it stays inside load_model) while letting type checkers resolve the name.
    from transformers import AutoTokenizer


def load_model(
    model_name: str, dtype=torch.float32, device="cpu", hf_dtype=None
) -> tuple[Model, AutoTokenizer]:
    """Download/load `model_name`, build our Model, copy weights in.

    Returns (model, tokenizer). The model is in eval mode on `device`.

    `hf_dtype` controls the dtype the HF model is *loaded* at, before its tensors
    are copied into our `dtype` model. It defaults to `dtype` (both fp32 — the
    usual, fully-precise oracle path). Pass `"auto"` to load HF at the checkpoint's
    native dtype (bf16 for the Qwen/Llama checkpoints), which HALVES the peak RAM:
    otherwise the fp32 HF model and our fp32 model are co-resident (~2× the model
    size), which is over this box's ceiling at ~1.7B (the A1 Qwen3-1.7B case).
    The copy into our fp32 model UPCASTS each tensor (bf16/fp32 -> fp32), which is
    lossless — bf16->fp32 zero-extends the mantissa — so the fp32 weights are
    byte-identical to the all-fp32 path. Never pass a dtype that would DOWNCAST a
    checkpoint tensor (that would lose precision); "auto" only ever upcasts.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(model_name)
    cfg = ModelConfig.from_hf(hf_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load HF weights only to read tensors out of them — we never call generate().
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=(hf_dtype if hf_dtype is not None else dtype)
    )
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


def _remap_llama_family(hf_name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """The default HF->ours name map, shared by every Llama-family arch.

    HF Qwen2/Qwen3/Llama layout:
        model.embed_tokens.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
        model.layers.{i}.self_attn.{q,k}_norm.weight   (Qwen3/Gemma3 only)
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.norm.weight
        lm_head.weight                       (absent if embeddings are tied)

    Our layout just drops the leading "model." for the body and keeps lm_head at
    top. The per-tensor shape is untouched — this is a pure rename.
    """
    if hf_name == "lm_head.weight":
        return "lm_head.weight", tensor
    if hf_name.startswith("model."):
        return hf_name[len("model."):], tensor
    # Be loud about anything we didn't anticipate rather than silently dropping it.
    raise RuntimeError(f"Unrecognized HF weight name: {hf_name}")


# Weight-name registry keyed by HF model_type (A0). This module is the ONE place
# that knows HF's names; the registry is the seam where a future arch whose layout
# differs (e.g. A3 Granite's fused MoE experts) plugs in its own remapper without
# touching the rest of the loader. The Llama-family archs (Qwen2, Qwen3, Llama3)
# all share the default map, so the registry is a fallback-to-default lookup.
_NAME_MAP_REGISTRY: dict[str, Callable[[str, torch.Tensor], tuple[str, torch.Tensor]]] = {}


def _name_mapper(model_type: str):
    return _NAME_MAP_REGISTRY.get(model_type, _remap_llama_family)


def _remap_state_dict(hf_sd: dict, our_sd: dict, cfg: ModelConfig) -> dict:
    """Translate HF parameter names to ours, per the model_type registry."""
    remap = _name_mapper(cfg.model_type)
    out: dict[str, torch.Tensor] = {}

    for hf_name, tensor in hf_sd.items():
        our_name, t = remap(hf_name, tensor)
        if our_name is not None:
            out[our_name] = t

    # Tied embeddings: some checkpoints omit lm_head and reuse the input embedding.
    if cfg.tie_word_embeddings and "lm_head.weight" not in out:
        out["lm_head.weight"] = out["embed_tokens.weight"]

    return out
