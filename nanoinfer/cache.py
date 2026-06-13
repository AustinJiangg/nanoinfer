"""Key/value cache for incremental decoding (stage 1).

Stage 0 recomputed K and V for the *entire* sequence on every step — O(n^2) work
to emit n tokens, and almost all of it redundant: a token's key and value never
change once computed. The cache stores each layer's K and V once. Prefill fills
them from the prompt in a single forward; each decode step then appends only the
new token's K/V and reads the rest straight back.

Layout: one [batch, n_kv_heads, max_seq, head_dim] buffer for K and one for V,
per layer, preallocated to a fixed max length so decode never reallocates. Note
we cache the *post-RoPE* keys — RoPE is applied once when a token is inserted and
the rotated key is reused unchanged thereafter (values carry no positional info).

`length` is the number of filled positions and is shared across layers: every
layer writes the same slice within one forward pass, so the model advances the
length once, after all layers have run (see Model.forward).
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device,
        dtype=torch.float32,
    ):
        shape = (batch, n_kv_heads, max_seq, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)]
        self.max_seq = max_seq
        self.length = 0  # filled positions; advanced by the model, not by update()

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this layer's new K/V at the current length; return full views.

        k, v: [batch, n_kv_heads, new_len, head_dim] for the new token(s).

        Every layer in a forward pass appends at the same position — `length` is
        advanced once, afterwards, by the model (see Model.forward) — so we read
        it from `self.length` here rather than taking a start argument that all
        callers would have to pass identically (and could get wrong).
        returns the slices [.., :length+new_len, ..] of K and V to attend over.
        """
        start = self.length
        end = start + k.shape[2]
        if end > self.max_seq:
            raise ValueError(
                f"KV cache overflow: need {end} positions, capacity is {self.max_seq}"
            )
        self.k[layer_idx][:, :, start:end] = k
        self.v[layer_idx][:, :, start:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, new_len: int) -> None:
        """Mark `new_len` more positions as filled, once per forward pass."""
        self.length += new_len
