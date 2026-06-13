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

from .cache import KVCache
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


def build_decode_mask(t: int, start: int, device) -> torch.Tensor:
    """Boolean attend-mask [t, start+t] for cached attention (start > 0).

    Query i sits at absolute position `start + i` and may attend key j iff
    `j <= start + i`. This covers the case a naive [t, t] lower-triangular mask
    gets wrong: the new queries must also attend ALL past keys (j < start), not
    just the current window. For the common decode step (t == 1) this is a single
    row of all-True — the lone query attends every cached key and itself.

    True = attend (the convention F.scaled_dot_product_attention's bool mask uses).
    """
    q_pos = torch.arange(t, device=device).unsqueeze(1) + start  # [t, 1]
    k_pos = torch.arange(start + t, device=device).unsqueeze(0)   # [1, start+t]
    return k_pos <= q_pos


class Attention(nn.Module):
    """Grouped-query causal self-attention with RoPE.

    Stage 0 (no cache): processes the whole sequence with a full causal mask.
    Stage 1 (cache): writes the new token(s)' K/V into the cache and attends over
    everything cached so far.
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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        # Project and reshape to [b, n_heads, t, head_dim].
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE is applied before caching: keys are stored already rotated, so a
        # cached key keeps the rotation for its absolute position forever.
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            # Append the new K/V and read back everything cached so far.
            k, v = cache.update(layer_idx, k, v)

        # GQA: expand KV heads to match Q heads.
        k = repeat_kv(k, self.cfg.n_rep)
        v = repeat_kv(v, self.cfg.n_rep)

        # Masking (see CLAUDE.md "Causal mask"). The model builds the mask once and
        # shares it across layers. None == the square prefill / stage-0 case: use
        # the optimized is_causal triangle. A non-None mask is the cached non-square
        # case, where the new queries must attend the whole history (a naive [t, t]
        # triangle would wrongly hide the past keys).
        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

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
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pass the cache/mask args by keyword: layer_idx and the mask are easy to
        # transpose positionally, and a swap would silently corrupt the cache slice
        # or the masking (exactly the kind of off-by-one CLAUDE.md warns about).
        x = x + self.self_attn(
            self.input_layernorm(x), cos, sin,
            cache=cache, layer_idx=layer_idx, attn_mask=attn_mask,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
