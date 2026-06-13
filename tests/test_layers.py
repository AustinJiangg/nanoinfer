"""Shape and numerical sanity tests for the layers.

The fast tests here construct tiny tensors and need no model download. The one
slow test loads a small HF checkpoint and parity-checks our forward pass against
HF's — that's the real correctness guarantee for stage 0.
"""

import torch

from nanoinfer.config import ModelConfig
from nanoinfer.layers import (
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    apply_rope,
    build_rope_cache,
    repeat_kv,
)


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,        # GQA: 2 kv heads, n_rep = 2
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def test_rmsnorm_shape_and_scale():
    norm = RMSNorm(16, eps=1e-6)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape
    # With unit weights, the RMS of the output should be ~1 per row.
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm():
    # Rotation must not change vector magnitude.
    cos, sin = build_rope_cache(seq_len=10, head_dim=8, theta=10000.0, device="cpu")
    q = torch.randn(1, 4, 10, 8)
    k = torch.randn(1, 4, 10, 8)
    q_r, k_r = apply_rope(q, k, cos, sin)
    assert q_r.shape == q.shape
    assert torch.allclose(q.norm(dim=-1), q_r.norm(dim=-1), atol=1e-4)


def test_rope_position_zero_is_identity():
    # At position 0, cos=1 and sin=0, so RoPE is a no-op.
    cos, sin = build_rope_cache(seq_len=4, head_dim=8, theta=10000.0, device="cpu")
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    q_r, _ = apply_rope(q, k, cos, sin)
    assert torch.allclose(q_r[:, :, 0, :], q[:, :, 0, :], atol=1e-5)


def test_repeat_kv():
    x = torch.randn(1, 2, 5, 8)       # 2 kv heads
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (1, 6, 5, 8)  # repeated to 6 heads
    # Heads 0,1,2 should all equal original head 0.
    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.allclose(out[:, 1], x[:, 0])
    assert torch.allclose(out[:, 2], x[:, 0])
    assert torch.allclose(out[:, 3], x[:, 1])


def test_swiglu_shape():
    cfg = tiny_cfg()
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, cfg.hidden_size)
    assert mlp(x).shape == x.shape


def test_transformer_block_shape():
    cfg = tiny_cfg()
    block = TransformerBlock(cfg)
    cos, sin = build_rope_cache(8, cfg.head_dim, cfg.rope_theta, device="cpu")
    x = torch.randn(2, 8, cfg.hidden_size)
    out = block(x, cos[:8], sin[:8])
    assert out.shape == x.shape


def test_causal_attention_is_causal():
    # A token's output must not depend on tokens after it. Change the last
    # token's input; the first token's output should be unchanged.
    cfg = tiny_cfg()
    block = TransformerBlock(cfg)
    cos, sin = build_rope_cache(8, cfg.head_dim, cfg.rope_theta, device="cpu")
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)
    out1 = block(x, cos[:8], sin[:8])

    x2 = x.clone()
    x2[:, -1, :] = torch.randn(cfg.hidden_size)  # perturb only the last position
    out2 = block(x2, cos[:8], sin[:8])

    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)
