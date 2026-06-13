"""Transformer building blocks, implemented from scratch.

These are deliberately plain PyTorch — readable over fast. Stage 0 recomputes the
full sequence every step (no KV cache); the cache arrives in stage 1.

Sharp edges (see CLAUDE.md):
  - RoPE uses the neox / half-split convention (Llama, Qwen2), NOT GPT-J interleave.
  - GQA repeats KV heads `n_rep` times to match the query heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    """Root-mean-square layer norm. No mean subtraction, no bias.

    x / sqrt(mean(x^2) + eps) * weight, computed in float32 for stability.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary position embeddings.

    Returns two tensors of shape [seq_len, head_dim]. Each frequency is duplicated
    across the two halves of the head dimension to match the half-split rotation.
    """
    # inv_freq: [head_dim/2]
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()  # [seq_len]
    freqs = torch.outer(positions, inv_freq)                  # [seq_len, head_dim/2]
    # Duplicate to full head_dim: [seq_len, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and rotate: [-x2, x1]. (neox convention)"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q and k.

    q, k:    [batch, n_heads, seq, head_dim]
    cos/sin: [seq, head_dim]  -> broadcast over batch and heads
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA).

    x: [batch, n_kv_heads, seq, head_dim] -> [batch, n_kv_heads * n_rep, seq, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)


class Attention(nn.Module):
    """Grouped-query causal self-attention with RoPE.

    Stage 0: processes the whole sequence and applies a full causal mask.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim

        # Qwen2 uses biases on q/k/v projections; output proj has none.
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        b, t, _ = x.shape

        # Project and reshape to [b, n_heads, t, head_dim].
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # GQA: expand KV heads to match Q heads.
        k = repeat_kv(k, self.cfg.n_rep)
        v = repeat_kv(v, self.cfg.n_rep)

        # Scaled dot-product attention with a causal mask.
        # is_causal handles the [t, t] lower-triangular mask for us.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated feed-forward: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: attention and FFN each wrapped in RMSNorm + residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
