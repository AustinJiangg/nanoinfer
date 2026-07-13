"""The full decoder-only model: embedding -> N transformer blocks -> norm -> LM head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .cache import KVCache
from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock, build_decode_mask, build_rope_cache


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE tables are the same for every layer; build once, reuse.
        # Registered as a buffer so .to(device) moves them with the model.
        cos, sin = build_rope_cache(
            cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta,
            device="cpu", rope_scaling=cfg.rope_scaling,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        """input_ids: [batch, seq] of token ids -> logits [batch, seq, vocab].

        Without a cache (stage 0): the whole sequence is processed every call,
        positions run 0..t-1.

        With a cache (stage 1): `input_ids` is just the new token(s). They sit at
        absolute positions start_pos..start_pos+t-1, where start_pos is however
        many tokens the cache already holds. Slicing the RoPE tables at start_pos
        — not 0 — is what gives each new token the rotation for its true position;
        getting this wrong is the classic cache bug. The length is advanced once,
        after every layer has written its slice for this pass.
        """
        b, t = input_ids.shape
        x = self.embed_tokens(input_ids)  # [b, t, hidden]
        if self.cfg.embedding_multiplier != 1.0:
            # muP (A3 Granite): the embedding output is scaled (Granite 1B: 12.0).
            x = x * self.cfg.embedding_multiplier

        start_pos = cache.length if cache is not None else 0

        # Guard the context limit explicitly. Slicing the RoPE tables past their
        # end is silent (Python returns fewer rows, not an error), which would
        # otherwise surface as a baffling shape mismatch deep in attention rather
        # than a clear "ran out of positions" — mirror the cache's overflow guard.
        max_pos = self.rope_cos.shape[0]
        if start_pos + t > max_pos:
            raise ValueError(
                f"sequence position {start_pos + t} exceeds the model's context "
                f"length {max_pos} (max_position_embeddings)"
            )

        cos = self.rope_cos[start_pos : start_pos + t]
        sin = self.rope_sin[start_pos : start_pos + t]

        # Build the attention mask once for the whole stack — every layer shares
        # it, just like cos/sin. None is the square prefill / stage-0 case (each
        # block takes the optimized is_causal path); a non-None mask is the cached
        # non-square case where new queries must attend the full history.
        attn_mask = None if start_pos == 0 else build_decode_mask(t, start_pos, x.device)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache=cache, layer_idx=i, attn_mask=attn_mask)

        if cache is not None:
            cache.advance(t)

        x = self.norm(x)
        logits = self.lm_head(x)
        if self.cfg.logits_scaling != 1.0:
            # muP (A3 Granite): the final logits are DIVIDED by logits_scaling
            # (HF's convention — a temperature baked into the checkpoint).
            logits = logits / self.cfg.logits_scaling
        return logits

    def init_cache(self, batch: int, max_seq: int) -> KVCache:
        """Allocate a fresh KV cache sized to this model, on its device/dtype."""
        w = self.embed_tokens.weight
        return KVCache(
            num_layers=self.cfg.num_layers,
            batch=batch,
            n_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            max_seq=max_seq,
            device=w.device,
            dtype=w.dtype,
        )
