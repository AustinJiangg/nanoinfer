"""Model configuration.

We read dimensions from the Hugging Face config rather than hardcoding them, so
the same code runs on any Llama-family checkpoint (Qwen2, Llama, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _read_rope_theta(hf_config, default: float = 10000.0) -> float:
    """Read the RoPE base frequency across transformers versions.

    transformers 4.x exposed `rope_theta` as a top-level config field. 5.x moved
    it into the `rope_parameters` dict (and mirrors it in `rope_scaling`), so a
    plain getattr(hf_config, "rope_theta", 10000.0) silently falls back to the
    default and feeds wrong RoPE frequencies into the model — the classic
    "plausible but degrading output" failure. Check the new nested layout too.
    """
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        for attr in ("rope_parameters", "rope_scaling"):
            params = getattr(hf_config, attr, None)
            if isinstance(params, dict) and params.get("rope_theta") is not None:
                theta = params["rope_theta"]
                break
    return float(theta) if theta is not None else default


@dataclass
class ModelConfig:
    """The architecture-description layer (A0).

    The first block is the dimensions every Llama-family model needs. The second
    block is the *feature flags* that let one forward pass serve several archs
    (the llama.cpp shape — flags, not a class hierarchy). Every flag defaults to
    the Qwen2.5 value, so an unqualified ModelConfig(...) is a Qwen2.5 config and
    the whole engine is unchanged; A1–A4 flip individual flags. Only `qkv_bias`
    is consumed by the forward today — the rest are described here and wired in
    by the stage that needs them (named per field), which is why A0 regresses
    Qwen2.5 bit-identical.
    """

    vocab_size: int
    hidden_size: int          # model / embedding dimension
    intermediate_size: int    # FFN inner dimension
    num_layers: int
    num_attention_heads: int  # number of query heads
    num_kv_heads: int         # number of key/value heads (GQA: <= query heads)
    head_dim: int
    rms_norm_eps: float
    rope_theta: float         # RoPE base frequency
    max_position_embeddings: int
    tie_word_embeddings: bool

    # --- A0 architecture-description fields (defaults = Qwen2.5) ---
    model_type: str = "qwen2"       # HF model_type; keys the weight-name registry
    qkv_bias: bool = True           # q/k/v projections carry a bias (Qwen2.5 yes; A1 Qwen3 no)
    qk_norm: bool = False           # per-head RMSNorm on Q,K pre-RoPE (A1 Qwen3, A4 Gemma3)
    act_fn: str = "silu"            # FFN gate activation: "silu" (SwiGLU) | "gelu" (GeGLU, A4)
    # RoPE frequency scaling, resolved at load time (A2 Llama-3.2). None = no scaling;
    # otherwise the HF dict, e.g. {"rope_type": "llama3", "factor": 32, ...}.
    rope_scaling: Optional[dict] = None
    # muP-style scalars, identity (1.0) for Qwen2.5; earn their keep in A3 Granite.
    embedding_multiplier: float = 1.0
    attention_multiplier: float = 1.0   # softmax scale multiplier (A3; roadmap "attention_scale")
    residual_multiplier: float = 1.0
    logits_scaling: float = 1.0
    # Mixture-of-experts (A3 Granite). n_experts == 0 => a dense FFN.
    n_experts: int = 0
    moe_top_k: int = 0
    moe_intermediate_size: int = 0
    # Sliding-window attention (A4 Gemma-3). 0 == full attention. sliding_pattern
    # is the global:local layer interleave (Gemma-3 is 5), 0 == every layer full.
    sliding_window: int = 0
    sliding_pattern: int = 0

    @property
    def n_rep(self) -> int:
        """How many times each KV head is repeated to match the query heads."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        """Build our config from a transformers config object.

        head_dim is usually hidden_size // num_attention_heads, but some configs
        set it explicitly — prefer the explicit value when present.

        The A0 flags are read generously across HF layouts, but only the Qwen2.5
        path is parity-tested here; each later stage locks its own model's reads.
        """
        n_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // n_heads
        )
        model_type = getattr(hf_config, "model_type", "qwen2")

        # qkv bias: Qwen2 hardwires it (no config flag), every other arch exposes
        # `attention_bias` (default False). Read the flag when present, else fall
        # back to "qwen2 has it, nobody else does".
        attn_bias = getattr(hf_config, "attention_bias", None)
        qkv_bias = bool(attn_bias) if attn_bias is not None else (model_type == "qwen2")

        # sliding window is only active when the model opts in (Qwen2.5 sets a
        # window value but use_sliding_window=False → treat as full attention).
        sw = getattr(hf_config, "sliding_window", None)
        if not getattr(hf_config, "use_sliding_window", sw is not None):
            sw = None

        # RoPE scaling. transformers 5.x always populates rope_scaling/rope_parameters
        # (it's where rope_theta now lives), tagging the *no-scaling* case rope_type
        # "default". Only a real scaling type (llama3, A2) counts — normalize the rest
        # to None so Qwen2.5 keeps plain inv_freq and the export writes "none".
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rtype = rope_scaling.get("rope_type") or rope_scaling.get("type")
            if rtype in (None, "default", "linear"):
                rope_scaling = None

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=n_heads,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", n_heads),
            head_dim=head_dim,
            rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            rope_theta=_read_rope_theta(hf_config),
            max_position_embeddings=getattr(
                hf_config, "max_position_embeddings", 4096
            ),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            model_type=model_type,
            qkv_bias=qkv_bias,
            qk_norm=getattr(hf_config, "use_qk_norm", model_type in ("qwen3", "gemma3")),
            act_fn=getattr(hf_config, "hidden_act", "silu"),
            rope_scaling=rope_scaling,
            embedding_multiplier=float(getattr(hf_config, "embedding_multiplier", 1.0)),
            attention_multiplier=float(getattr(hf_config, "attention_multiplier", 1.0)),
            residual_multiplier=float(getattr(hf_config, "residual_multiplier", 1.0)),
            logits_scaling=float(getattr(hf_config, "logits_scaling", 1.0)),
            n_experts=int(getattr(hf_config, "num_local_experts", 0)),
            moe_top_k=int(getattr(hf_config, "num_experts_per_tok", 0)),
            moe_intermediate_size=int(getattr(hf_config, "moe_intermediate_size", 0) or 0),
            sliding_window=int(sw) if sw else 0,
            sliding_pattern=int(getattr(hf_config, "sliding_window_pattern", 0) or 0),
        )
