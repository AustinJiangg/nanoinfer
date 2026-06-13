"""Model configuration.

We read dimensions from the Hugging Face config rather than hardcoding them, so
the same code runs on any Llama-family checkpoint (Qwen2, Llama, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def n_rep(self) -> int:
        """How many times each KV head is repeated to match the query heads."""
        return self.num_attention_heads // self.num_kv_heads

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        """Build our config from a transformers config object.

        head_dim is usually hidden_size // num_attention_heads, but some configs
        set it explicitly — prefer the explicit value when present.
        """
        n_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", None) or (
            hf_config.hidden_size // n_heads
        )
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
        )
