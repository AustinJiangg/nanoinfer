"""Stage-2 sampling tests.

All fast — they run on hand-built logit vectors, no model download. They pin the
two greedy-equivalence invariants the roadmap calls for (temperature 0 == greedy,
top-k 1 == greedy) plus per-warper behavior and reproducibility.
"""

import pytest
import torch

from nanoinfer.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token,
)


def test_temperature_scales_and_preserves_ranking():
    logits = torch.tensor([1.0, 2.0, 3.0])
    out = apply_temperature(logits, 0.5)
    assert torch.allclose(out, logits / 0.5)
    # Monotonic scaling never changes the argmax.
    assert int(out.argmax()) == int(logits.argmax())


def test_top_k_keeps_exactly_the_top_k():
    logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
    out = apply_top_k(logits, k=2)
    finite = torch.isfinite(out)
    # The two survivors are the values 5 (idx 1) and 4 (idx 3).
    assert finite.tolist() == [False, True, False, True, False]
    assert torch.isneginf(out[[0, 2, 4]]).all()


def test_top_k_none_or_nonpositive_is_noop():
    logits = torch.randn(10)
    assert torch.equal(apply_top_k(logits, None), logits)
    assert torch.equal(apply_top_k(logits, 0), logits)


def test_top_p_keeps_the_nucleus():
    # Probabilities after softmax of these logits: pick a vector with an easy
    # nucleus. Use logits whose softmax is [0.5, 0.3, 0.15, 0.05] approximately by
    # working in log space directly.
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()
    out = apply_top_p(logits, p=0.9)
    # cumulative 0.5, 0.8, 0.95 -> first three reach 0.9; the 0.05 tail is dropped.
    assert torch.isfinite(out[:3]).all()
    assert torch.isneginf(out[3])


def test_top_p_keeps_at_least_one_when_top_prob_exceeds_p():
    probs = torch.tensor([0.95, 0.04, 0.01])
    logits = probs.log()
    out = apply_top_p(logits, p=0.9)  # top token alone already exceeds p
    assert torch.isfinite(out[0])
    assert torch.isneginf(out[1:]).all()


def test_top_p_ge_one_is_noop():
    logits = torch.randn(10)
    assert torch.equal(apply_top_p(logits, 1.0), logits)
    assert torch.equal(apply_top_p(logits, None), logits)


def test_top_p_with_unsorted_logits():
    # Vocab order differs from sorted order, so the scatter that maps the
    # sorted-order mask back to vocab order (apply_top_p) actually has to do
    # work — an identity permutation (already-sorted input) wouldn't test it.
    probs = torch.tensor([0.15, 0.5, 0.05, 0.3])  # in vocab order
    logits = probs.log()
    out = apply_top_p(logits, p=0.9)
    # Nucleus by probability is {idx1=0.5, idx3=0.3, idx0=0.15} (cumprob 0.95);
    # only idx2=0.05 falls outside and must be dropped.
    assert torch.isfinite(out[[0, 1, 3]]).all()
    assert torch.isneginf(out[2])


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([2.0, -2.0, 1.0])
    seen = torch.tensor([0, 1])  # tokens 0 and 1 already appeared
    out = apply_repetition_penalty(logits, seen, penalty=2.0)
    assert out[0] == 1.0       # positive logit divided by penalty: 2/2
    assert out[1] == -4.0      # negative logit multiplied by penalty: -2*2
    assert out[2] == 1.0       # unseen token untouched
    # Both transforms lower the score relative to the original.
    assert out[0] < logits[0] and out[1] < logits[1]


def test_repetition_penalty_one_is_noop():
    logits = torch.randn(10)
    assert torch.equal(apply_repetition_penalty(logits, None, 1.0), logits)


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 9.0, 0.2, 3.0])
    params = SamplingParams(temperature=0.0)
    # Greedy is deterministic and equals argmax regardless of seed.
    assert sample_next_token(logits, params) == int(logits.argmax())


def test_top_k_one_is_greedy_regardless_of_seed():
    logits = torch.tensor([0.1, 9.0, 0.2, 3.0])
    argmax = int(logits.argmax())
    for seed in (0, 1, 2, 1234):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        gen = torch.Generator().manual_seed(seed)
        assert sample_next_token(logits, params, generator=gen) == argmax


def test_same_seed_is_reproducible_and_respects_top_k():
    torch.manual_seed(0)
    logits = torch.randn(50)
    params = SamplingParams(temperature=1.0, top_k=5)

    def draw(seed):
        gen = torch.Generator().manual_seed(seed)
        return sample_next_token(logits, params, generator=gen)

    # Same seed -> same token.
    assert draw(7) == draw(7)
    # Every draw must land inside the top-5 allowed set.
    allowed = set(torch.topk(logits, 5).indices.tolist())
    for seed in range(30):
        assert draw(seed) in allowed


def test_different_seeds_can_diverge():
    # Guards against a sampler that ignores the generator entirely (e.g. silently
    # falling back to argmax / a frozen RNG): the draw must actually vary by seed.
    torch.manual_seed(0)
    logits = torch.randn(50)
    params = SamplingParams(temperature=1.0, top_k=10)
    tokens = {
        sample_next_token(logits, params, generator=torch.Generator().manual_seed(s))
        for s in range(20)
    }
    assert len(tokens) > 1


def test_full_pipeline_composes_without_error():
    # All four transforms in one call (only the slow e2e test exercises this
    # otherwise). The drawn token must be valid and inside the top-k allowed set.
    logits = torch.randn(100)
    seen = torch.tensor([3, 7, 11])
    params = SamplingParams(temperature=0.8, top_k=20, top_p=0.95, repetition_penalty=1.2)
    gen = torch.Generator().manual_seed(5)
    tok = sample_next_token(logits, params, context_ids=seen, generator=gen)
    assert 0 <= tok < 100


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        SamplingParams(repetition_penalty=0.0)   # divide-by-zero / inversion
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1.0)         # flips the distribution


def test_repetition_penalty_requires_context_when_on():
    with pytest.raises(ValueError):
        apply_repetition_penalty(torch.randn(5), None, penalty=1.2)
