"""Sampling strategies for next-token selection (stage 2).

Stages 0/1 always took the argmax (greedy). Here we add the standard sampling
toolkit — temperature, top-k, top-p (nucleus), repetition penalty — as small,
independently testable logit transforms, plus a seeded sampler that composes
them.

They compose in the conventional order (matching HF's logits processors):
repetition penalty on the raw logits, then temperature, then top-k, then top-p,
then softmax + a multinomial draw. Two deliberate properties make the boundary
to greedy clean and testable:

  - `temperature == 0` is greedy (argmax) and short-circuits the whole pipeline,
    so it stays exactly deterministic — and never divides by zero.
  - repetition penalty is applied *before* that short-circuit, because (like HF)
    it is a logits *processor*, not a sampling *warper*: it shapes greedy
    decoding too. temperature/top-k/top-p only matter once we actually sample.

Everything operates on a 1-D `[vocab]` logit vector for the current position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.0          # 0.0 == greedy (argmax); >0 enables sampling
    top_k: int | None = None          # keep the k highest-logit tokens
    top_p: float | None = None        # keep the smallest nucleus with cumprob >= p
    repetition_penalty: float = 1.0   # 1.0 == off; >1 discourages already-seen tokens
    seed: int | None = None           # seeds the multinomial draw for reproducibility

    def __post_init__(self) -> None:
        # Catch the params that would otherwise blow up mid-generation rather than
        # at construction. A negative temperature flips the distribution; a
        # repetition_penalty <= 0 divides a positive logit by zero (-> inf -> NaN
        # in softmax, a multinomial crash) or, in the greedy path, silently selects
        # the token it was meant to suppress.
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be > 0 (use >= 1 to actually discourage "
                f"repeats; values in (0, 1) amplify them), got {self.repetition_penalty}"
            )

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def apply_repetition_penalty(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Push down the logits of tokens already in `token_ids` (CTRL-style).

    For a seen token a positive logit is divided by `penalty` and a negative one
    multiplied by it — both lower the score, making a repeat less likely. The
    asymmetry is why it's not just a subtraction: it keeps the penalty
    scale-relative regardless of the logit's sign. `penalty == 1.0` is a no-op (so
    `token_ids` may be None then). `token_ids` is the running context, prompt +
    generated.
    """
    if penalty == 1.0:
        return logits
    if token_ids is None:
        raise ValueError("repetition_penalty != 1.0 requires token_ids (the context)")
    seen = torch.unique(token_ids)
    scores = logits[seen]
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    logits = logits.clone()
    logits[seen] = scores
    return logits


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature: <1 sharpens the distribution, >1 flattens it.

    Assumes temperature > 0 — the greedy case (temperature == 0) is handled by the
    sampler before this is reached, so there is no divide-by-zero here.
    """
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep only the k highest logits; set the rest to -inf. None/<=0 is a no-op.

    Ties at the k-th value are all kept (a token is dropped only if it is strictly
    below the k-th largest logit), matching HF.
    """
    if k is None or k <= 0:
        return logits
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k).values[-1]  # the k-th largest logit
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose cumulative prob >= p.

    None or p >= 1.0 keeps everything. Always keeps at least the top-1 token, even
    when its own probability already exceeds p.
    """
    if p is None or p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Drop everything past the point where the cumulative probability first
    # reaches p. Shift the mask right by one so the token that *crosses* p is
    # kept, and never drop the very first (highest-prob) token.
    drop = cumprobs > p
    drop[1:] = drop[:-1].clone()
    drop[0] = False
    drop_in_vocab_order = drop.scatter(-1, sorted_idx, drop)
    return logits.masked_fill(drop_in_vocab_order, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    context_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """Choose the next token id from last-position `logits` (`[vocab]`).

    Greedy (argmax) when `params.temperature == 0`; otherwise the full pipeline:
    repetition penalty -> temperature -> top-k -> top-p -> softmax -> multinomial,
    the draw seeded by `generator`. `context_ids` (prompt + generated so far) is
    only read when `repetition_penalty != 1.0`.

    Note: `params.seed` is consumed by `generate()`, which builds the `generator`;
    this function takes the `generator` directly, so seed a draw by passing one.
    """
    # Processor: shapes both greedy and sampled decoding.
    logits = apply_repetition_penalty(logits, context_ids, params.repetition_penalty)

    if params.greedy:
        return int(logits.argmax(dim=-1))

    # Warpers: only relevant once we sample.
    logits = apply_temperature(logits, params.temperature)
    logits = apply_top_k(logits, params.top_k)
    logits = apply_top_p(logits, params.top_p)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
