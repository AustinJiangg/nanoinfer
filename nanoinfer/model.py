"""The full decoder-only model: embedding -> N transformer blocks -> norm -> LM head."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock, build_rope_cache


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
            device="cpu",
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [batch, seq] of token ids -> logits [batch, seq, vocab].

        Stage 0: the whole sequence is processed every call. We slice the RoPE
        tables to the current length and let each block apply causal masking.
        """
        b, t = input_ids.shape
        x = self.embed_tokens(input_ids)  # [b, t, hidden]

        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        return self.lm_head(x)
