"""Stage-1 KV cache tests.

These run on a tiny random-weight model, so they need no download and stay in the
fast suite. They check the cache *mechanism*, not weight parity: incremental
decoding through the cache must produce the same logits as stage-0 full recompute.
HF parity (correct weights) is covered by the slow tests in test_generate.py,
which now exercise the cache end-to-end.
"""

import torch

from nanoinfer.cache import KVCache
from nanoinfer.config import ModelConfig
from nanoinfer.model import Model


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


def test_cache_update_shapes_and_length():
    cache = KVCache(num_layers=2, batch=1, n_kv_heads=2, head_dim=8, max_seq=16, device="cpu")
    assert cache.length == 0

    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    ck, cv = cache.update(layer_idx=0, k=k, v=v)  # appends at length 0
    # Reads back exactly what was written, sliced to the filled length.
    assert ck.shape == (1, 2, 3, 8)
    assert torch.allclose(ck, k)
    assert torch.allclose(cv, v)

    # update() does not advance length — the model does, once per pass.
    assert cache.length == 0
    cache.advance(3)
    assert cache.length == 3

    # Appending one token at the new length reads back all 4 positions.
    ck2, _ = cache.update(layer_idx=0, k=torch.randn(1, 2, 1, 8), v=torch.randn(1, 2, 1, 8))
    assert ck2.shape == (1, 2, 4, 8)


def test_cache_overflow_raises():
    cache = KVCache(num_layers=1, batch=1, n_kv_heads=2, head_dim=8, max_seq=4, device="cpu")
    k = v = torch.randn(1, 2, 5, 8)
    try:
        cache.update(0, k, v)
        assert False, "expected overflow to raise"
    except ValueError:
        pass


def test_cached_decode_matches_full_recompute():
    """The whole point of stage 1: cache output == stage-0 output, token-for-token.

    Reference = one full-recompute forward over a 6-token sequence. Then replay it
    incrementally: prefill the first 3 tokens, decode the remaining 3 one at a
    time through the cache. Each step's last-position logits must match the
    reference at the same absolute position (atol 1e-5 — same math, same dtype).
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = Model(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    ref = model(ids)  # [1, 6, vocab], no cache — full recompute

    cache = model.init_cache(batch=1, max_seq=16)
    prefill = model(ids[:, :3], cache=cache)          # positions 0,1,2
    assert torch.allclose(prefill, ref[:, :3], atol=1e-5)
    assert cache.length == 3

    for pos in range(3, 6):
        step = model(ids[:, pos : pos + 1], cache=cache)  # one token
        assert step.shape == (1, 1, cfg.vocab_size)
        assert torch.allclose(step[:, -1], ref[:, pos], atol=1e-5)
        assert cache.length == pos + 1


def test_chunked_decode_matches_full_recompute():
    """Decode more than one token at once (t > 1, start_pos > 0).

    The single-token decode path only ever hits build_decode_mask with t == 1 (a
    one-row all-True mask). A multi-token continuation exercises the rectangular
    [t, start+t] mask where `k_pos <= q_pos` actually has to get the triangle
    right against the cached past — the branch a t==1-only suite would let ship
    broken. Prefill 3, then decode the remaining 4 in a single pass.
    """
    torch.manual_seed(2)
    cfg = tiny_cfg()
    model = Model(cfg).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    ref = model(ids)  # full recompute, no cache

    cache = model.init_cache(batch=1, max_seq=16)
    prefill = model(ids[:, :3], cache=cache)          # positions 0,1,2
    chunk = model(ids[:, 3:7], cache=cache)           # positions 3..6, t == 4
    assert chunk.shape == (1, 4, cfg.vocab_size)
    assert torch.allclose(prefill, ref[:, :3], atol=1e-5)
    assert torch.allclose(chunk, ref[:, 3:7], atol=1e-5)
    assert cache.length == 7


def test_forward_past_context_length_raises():
    """Going past max_position_embeddings fails loudly, not with a cryptic shape
    error from slicing the RoPE table past its end."""
    cfg = tiny_cfg()  # max_position_embeddings = 64
    model = Model(cfg).eval()
    cache = model.init_cache(batch=1, max_seq=cfg.max_position_embeddings + 8)

    # Fill the cache right up to the context limit, then try one more token.
    model(torch.randint(0, cfg.vocab_size, (1, cfg.max_position_embeddings)), cache=cache)
    try:
        model(torch.zeros(1, 1, dtype=torch.long), cache=cache)
        assert False, "expected a context-length error"
    except ValueError as e:
        assert "context length" in str(e)


def test_cached_greedy_matches_uncached():
    """Greedy argmax over cached vs uncached logits must pick the same tokens.

    No tokenizer needed: drive the two model paths directly and compare the
    argmax token streams. This is the generate()-level analogue of the parity
    test above and guards the prefill/decode split end-to-end.
    """
    torch.manual_seed(1)
    cfg = tiny_cfg()
    model = Model(cfg).eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    steps = 8

    # Uncached: re-feed the whole running sequence each step.
    seq = prompt.clone()
    uncached = []
    for _ in range(steps):
        nxt = int(model(seq)[:, -1].argmax(-1))
        uncached.append(nxt)
        seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)

    # Cached: prefill then one token at a time.
    cache = model.init_cache(batch=1, max_seq=prompt.shape[1] + steps)
    cur = prompt
    cached = []
    for _ in range(steps):
        nxt = int(model(cur, cache=cache)[:, -1].argmax(-1))
        cached.append(nxt)
        cur = torch.tensor([[nxt]])

    assert cached == uncached
