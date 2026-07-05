"""Speculative decoding (S-track): a cheap *proposer* guesses K tokens, the big *target*
model verifies all K+1 in ONE forward, and we keep the longest prefix the target agrees
with — emitting several tokens per target forward instead of one.

The engine already has everything this needs (verified in S0):

  * `model.forward([t0, t1, ..], cache)` on a POPULATED cache places the tokens at
    positions cache.length.. and returns a logits row per token — bit-identical to
    decoding them one at a time (attention masks query i to keys 0..length+i). That
    single multi-token pass IS the verify step: forward([cur, d0..d_{K-1}], target_cache)
    → K+1 logit rows, the target's own next-token at each drafted position.
  * `cache.truncate(L)` drops the K/V for positions >= L — the rollback that discards
    rejected drafts (S1 gave all four caches this).

The verify + rollback machinery is FIXED; the only thing that varies is the **proposer** —
where the K candidate tokens come from (roadmap: "same verify + rollback, a different
proposer"). Two live proposers:

  * `DraftModelProposer` (S0/S1) — a small draft model runs K forwards. Cost per verify is
    K draft forwards, i.e. r = t_draft/t_target per proposed token; S2 measured r ≈ 0.45 on
    the 0.5B/1.5B pair, which caps the speedup regardless of accept rate.
  * `PromptLookupProposer` (S4) — no model at all: match the recent context against an
    earlier occurrence and copy what followed (prompt-lookup / n-gram decoding). Cost per
    verify is an array scan → r → 0: a MISS costs only the (free) lookup — not k draft
    forwards — so the step degrades to plain greedy with no waste. The one remaining downside
    is the fatter verify (k+1 tokens is a mini-prefill), so a large k on rarely-matching text
    can *mildly* net-lose (S4 measured list @k=8: 0.86×) — far milder than a draft model's
    r·k tax (S2 measured story @k=8: 0.64×). Ceiling K+1 tokens/verify; S4 measured up to
    3.35× on copy-heavy text. S2 named this the real ratio lever the draft/target pair lacks.

Correctness (the S0 gate, proposer-independent): with GREEDY decoding this is *token-
identical* to plain target greedy. Each emitted token is the target's argmax at its
position (an accepted guess equals it by construction; a rejected one is replaced by it),
and the target logits at a proposed position equal what plain sequential decode would
produce. So the proposer only changes SPEED (how many tokens land per verify), never the
output — a wrong guess costs a verify slot, never a wrong token.

CPU-oracle-locked: over the CPU backend, `tokens == model.generate(...)` for the target.
`run_spec.py` is the test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))  # nicpp.*.so
import nicpp  # noqa: E402


def _draw(probs, rng) -> int:
    """Categorical draw from a `[vocab]` probability vector by inverse-CDF. A float64
    cumsum makes it robust to the tiny normalization error a float32 `token_probs` row
    can carry (so it never trips a strict sum-to-1 check), and `searchsorted(side=right)`
    can never land on a masked (zero-prob) token. `rng` is a numpy Generator."""
    c = np.cumsum(np.asarray(probs, dtype=np.float64))
    return int(np.searchsorted(c, rng.random() * c[-1], side="right"))


@dataclass
class SpecStats:
    """Speculation efficiency (for tuning; irrelevant to correctness)."""
    verifies: int = 0        # target forwards (each emits >= 1 token)
    proposals: int = 0       # verifies that actually proposed >=1 token (prompt-lookup can find none)
    drafted: int = 0         # candidate tokens proposed
    accepted: int = 0        # candidate tokens the target accepted
    emitted: int = 0         # tokens actually emitted
    accept_lengths: list[int] = field(default_factory=list)  # accepted count `a` per verify (S2)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def hit_rate(self) -> float:
        # Fraction of verifies that found something to propose. For a draft model this is
        # always 1.0; for prompt-lookup it's the match rate — the r→0 win only lands on hits.
        return self.proposals / self.verifies if self.verifies else 0.0

    @property
    def tokens_per_verify(self) -> float:
        # The speedup proxy: how many tokens each (expensive) target forward yields.
        return self.emitted / self.verifies if self.verifies else 0.0

    def accept_histogram(self, k: int) -> list[int]:
        """Counts of verifies by accepted length a = 0..k — the accept-length DISTRIBUTION
        (S2). The mean (accept_rate) hides the shape: a proposer that either nails a whole
        run or fails at token 0 is bimodal, and that shape is what picks K."""
        h = [0] * (k + 1)
        for a in self.accept_lengths:
            if 0 <= a <= k:
                h[a] += 1
        return h


# ---------------------------------------------------------------------------
# Proposers — the ONLY thing S0 (draft model) and S4 (prompt-lookup) differ by.
# A proposer owns: prefill(prompt, cap), propose(cur, context) -> [tok..], rollback(keep),
# and max_k (its longest proposal, for cache sizing).
# ---------------------------------------------------------------------------


class DraftModelProposer:
    """S0/S1: a small draft model autoregressively proposes k tokens, keeping its own KV
    cache lock-step with the target so ONE truncate(keep) rolls back both."""

    def __init__(self, draft, k: int):
        self.draft = draft
        self.max_k = k
        self.dcache = None

    def prefill(self, prompt_ids: list[int], cap: int) -> None:
        # The draft cache must track the same history so its proposals are conditioned right.
        self.dcache = self.draft.make_cache(cap)
        self.draft.forward(prompt_ids, self.dcache)

    def propose(self, cur: int, context: list[int]) -> list[int]:
        # Feed cur, d0, .., d_{k-2} to collect d0..d_{k-1} (k proposals), then feed d_{k-1}
        # once more (output discarded) so the draft cache ends at L+k+1 — the SAME length the
        # target reaches after verifying k+1 tokens. Lock-step caches let one truncate(L+a+1)
        # roll back BOTH, with no all-accepted special case (else at a==k the draft is short).
        d: list[int] = []
        x = cur
        for _ in range(self.max_k):
            dl = self.draft.forward([x], self.dcache)
            x = int(np.argmax(dl[-1]))
            d.append(x)
        self.draft.forward([d[-1]], self.dcache)   # k+1-th feed; keeps dcache == tcache length
        return d

    def propose_sampling(self, cur: int, context: list[int], params, rng):
        """S5 (sampling): SAMPLE k tokens from the draft's own shaped distribution q_i =
        token_probs(draft_logits_i, params) instead of taking its argmax, and return both
        the tokens and those q distributions (rejection sampling needs q to accept/reject
        against the target's p). Same lock-step cache bookkeeping as `propose` (the k+1-th
        feed keeps the draft cache length == the target's), so ONE truncate rolls back
        both. `ctx` grows with the sampled tokens so the draft's rep-penalty conditions
        the same way plain draft sampling would."""
        d: list[int] = []
        qs: list[np.ndarray] = []
        x = cur
        ctx = list(context)
        for _ in range(self.max_k):
            dl = self.draft.forward([x], self.dcache)
            q = nicpp.token_probs(dl[-1], params, ctx)
            x = _draw(q, rng)
            d.append(x)
            qs.append(q)
            ctx.append(x)
        self.draft.forward([d[-1]], self.dcache)   # k+1-th feed; keeps dcache == tcache length
        return d, qs

    def rollback(self, keep: int) -> None:
        self.dcache.truncate(keep)


class PromptLookupProposer:
    """S4: no model. Match the last `ngram` tokens of the context against an earlier
    occurrence and copy up to `max_k` tokens that followed (prompt-lookup / n-gram
    decoding). Cost per verify ~0 (an array scan) → r → 0. Stateless: no cache to keep,
    so rollback is a no-op."""

    def __init__(self, ngram: int, max_k: int):
        self.ngram = ngram
        self.max_k = max_k

    def prefill(self, prompt_ids: list[int], cap: int) -> None:
        pass

    def propose(self, cur: int, context: list[int]) -> list[int]:
        return prompt_lookup(context, self.ngram, self.max_k)

    def propose_sampling(self, cur: int, context: list[int], params, rng):
        """S5 (sampling): the n-gram copy is DETERMINISTIC, so each proposed token's
        proposal distribution q_i is a point mass (q_i(d_i) = 1). We signal that with
        `None` per token — rejection_accept reads a point mass as accept-prob p(d_i) and,
        on a reject, resamples from p with the proposed token zeroed out. A free draft
        (r->0) with no sampling of its own, exactly as in the greedy path."""
        d = prompt_lookup(context, self.ngram, self.max_k)
        return d, [None] * len(d)

    def rollback(self, keep: int) -> None:
        pass


def accept_prefix(d: list[int], target_argmax) -> tuple[int, list[int]]:
    """The accept rule — the single source of truth for what a verify confirms (S0).

    Given the proposal `d` (k candidate tokens) and the target's own greedy token at each
    of the k+1 verified positions (`target_argmax`, an argmax over the k+1 verify logit
    rows), return `(a, confirmed)`:

      * `a` — the number of accepted drafts: the length of the longest prefix of `d` the
        target's argmax agrees with.
      * `confirmed = d[:a] + [target_argmax[a]]` — the tokens to emit this step. If a < k,
        `target_argmax[a]` is the *correction* replacing the wrong draft d[a]; if a == k,
        every draft matched and it's the *bonus* token after all k (target_argmax has k+1
        rows, so index a is always valid).

    Every confirmed token is the target's own argmax at its position, so the output is
    token-identical to plain target greedy no matter what the proposer guessed — a wrong
    guess only costs a verify slot, never a wrong token. Shared by the single-sequence loop
    (`_greedy_spec_core`) and the batched `SpecScheduler`, so the invariant is proved once.
    """
    k = len(d)
    a = 0
    while a < k and int(target_argmax[a]) == d[a]:
        a += 1
    return a, d[:a] + [int(target_argmax[a])]


def rejection_accept(d: list[int], q_list, p_list, rng) -> tuple[int, list[int]]:
    """The SAMPLING accept rule (S5) — speculative / rejection sampling, the distribution
    analog of `accept_prefix`. It's the single source of truth for what a sampled verify
    confirms, shared by the single-sequence loop and the scheduler.

    Inputs (all over the SAME k+1 verified positions):
      * `d`      — the k proposed tokens.
      * `q_list` — q_list[i] is the distribution `[vocab]` that d[i] was proposed from, or
        `None` for a deterministic point-mass proposer (prompt-lookup: q_i(d_i)=1).
      * `p_list` — p_list[i] is the TARGET's shaped distribution `[vocab]` at position i
        (from `token_probs` on the verify logits); length k+1 (row k is the bonus position).

    Draft token d[i] is accepted with probability min(1, p_i(d_i) / q_i(d_i)); on the first
    rejection we resample a correction from the normalized residual (p_i - q_i)_+ and STOP
    (discard the rest). If all k are accepted we sample a bonus from p_k. Returns
    `(a, emitted)` with `emitted = d[:a] + [one token]` — the same shape as `accept_prefix`.

    The theorem (Leviathan et al. / Chen et al.): every emitted token's marginal is exactly
    p_i, for ANY proposal q — so a sampled speculative decode is *distribution-identical* to
    plain target sampling, the sampling analog of the greedy token-identity. At temperature
    0, `token_probs` is a one-hot argmax, so min(1, p/q) is 0/1 and the residual is a one-hot
    correction: this reduces EXACTLY to `accept_prefix` (the greedy floor is the temp->0
    limit, tested bit-identically)."""
    k = len(d)
    for i in range(k):
        x = d[i]
        p = p_list[i]
        px = float(p[x])
        qx = 1.0 if q_list[i] is None else float(q_list[i][x])
        # Accept with prob min(1, p(x)/q(x)); qx==0 (draft gave x zero mass) -> always accept
        # (p/0 -> inf). rng.random() is in [0,1), so a ratio >= 1 always accepts.
        if qx <= 0.0 or rng.random() < px / qx:
            continue
        # Reject: correction ~ normalized (p - q)_+ , then stop. A point mass just zeros d[i].
        if q_list[i] is None:
            resid = p.astype(np.float64).copy()
            resid[x] = 0.0
        else:
            resid = np.maximum(p.astype(np.float64) - q_list[i].astype(np.float64), 0.0)
        s = resid.sum()
        corr = _draw(resid / s, rng) if s > 0.0 else int(np.argmax(p))
        return i, d[:i] + [corr]
    # All k accepted -> a bonus token sampled from the target dist at the (k+1)-th position.
    return k, d[:k] + [_draw(p_list[k], rng)]


def prompt_lookup(context: list[int], ngram: int, max_k: int) -> list[int]:
    """Find an earlier occurrence of the last `ngram` tokens of `context` and return the
    up-to-`max_k` tokens that followed it. Returns [] when there's no match — a free draft
    that simply degrades that step to plain greedy (no wasted forward, so it can never
    regress; this is S2's r→0 lever).

    Scans most-recent-first so the copy reflects the latest matching context. Any guess is
    safe: the target verify keeps only the prefix it agrees with, so a wrong match costs
    nothing but the (free) lookup — output stays token-identical to plain greedy.
    """
    n = len(context)
    if ngram < 1 or n <= ngram:          # nothing precedes the suffix yet
        return []
    suffix = context[n - ngram:]
    # Positions n-ngram.. are the suffix itself; scan earlier starts, most recent first.
    for i in range(n - ngram - 1, -1, -1):
        if context[i:i + ngram] == suffix:
            return context[i + ngram:i + ngram + max_k]
    return []


# ---------------------------------------------------------------------------
# The greedy speculative loop, parameterized by a proposer.
# ---------------------------------------------------------------------------


def _greedy_spec_core(target, prompt_ids: list[int], max_tokens: int, proposer,
                      eos_id: int) -> tuple[list[int], SpecStats]:
    """Core greedy speculative decode. Returns (generated_ids, stats) with generated_ids
    token-identical to plain target greedy — regardless of what `proposer` returns."""
    # Caches sized for prompt + everything we emit, plus max_k slack for the tentative tail
    # a verify writes before rollback.
    cap = len(prompt_ids) + max_tokens + proposer.max_k + 1
    tcache = target.make_cache(cap)
    proposer.prefill(prompt_ids, cap)

    # Prefill the target on the prompt; its last row gives the first token.
    tlog = target.forward(prompt_ids, tcache)
    cur = int(np.argmax(tlog[-1]))

    generated: list[int] = []
    stats = SpecStats()
    context = list(prompt_ids)     # == prompt_ids + generated; proposer matches against it

    def emit(tok: int) -> bool:
        """Append tok unless it's eos; return False when generation should stop."""
        if eos_id >= 0 and tok == eos_id:
            return False                       # eos: stop without emitting (matches generate.cpp)
        generated.append(tok)
        context.append(tok)                    # keep context == prompt + generated for lookup
        stats.emitted += 1
        return len(generated) < max_tokens

    if not emit(cur):                          # the first token (g_0), eos/limit-checked
        return generated, stats

    while True:
        L = tcache.length                      # confirmed prefix length

        # --- Propose: k candidate tokens (k may be 0 — prompt-lookup with no match, which
        # degrades this step to a plain greedy forward([cur]), never a regression). ---
        d = proposer.propose(cur, context)
        k = len(d)

        # --- Verify: ONE target forward over [cur, d0..d_{k-1}] -> k+1 logit rows. ---
        # Row i is the target's distribution AFTER [cur, d0..d_{i-1}], i.e. its own token
        # for the position the proposer guessed as d_i (row k is the bonus after all drafts).
        tl = target.forward([cur] + d, tcache)
        tv = np.argmax(tl, axis=1).astype(np.int64)   # target's greedy token at each position

        # --- Accept the longest prefix the target agrees with (the shared rule). ---
        a, new = accept_prefix(d, tv)
        stats.verifies += 1
        stats.proposals += 1 if k else 0
        stats.drafted += k
        stats.accepted += a
        stats.accept_lengths.append(a)

        # --- Rollback: keep cur + d[0..a-1] (positions L..L+a); tv[a] becomes the next cur,
        # fed fresh next iteration (its K/V isn't in the cache yet). ---
        keep = L + a + 1
        tcache.truncate(keep)
        proposer.rollback(keep)

        # --- Emit the confirmed tokens, tracking cur as the last one. ---
        for tok in new:
            cur = tok
            if not emit(tok):
                return generated, stats


def greedy_speculative(target, draft, prompt_ids: list[int], max_tokens: int,
                       k: int = 4, eos_id: int = -1) -> tuple[list[int], SpecStats]:
    """Greedy speculative decode with a DRAFT MODEL (S0/S1). Returns (generated_ids, stats),
    with generated_ids token-identical to `target.generate(prompt_ids, max_tokens, greedy)`.

    `target` and `draft` are nicpp.Model handles (pass the same model as both for a
    draft==target self-check: every draft is accepted). k >= 1 is the draft length.
    """
    if k < 1:
        raise ValueError("k (draft length) must be >= 1")
    if max_tokens <= 0 or not prompt_ids:
        return [], SpecStats()
    return _greedy_spec_core(target, prompt_ids, max_tokens, DraftModelProposer(draft, k), eos_id)


def greedy_prompt_lookup(target, prompt_ids: list[int], max_tokens: int,
                         ngram: int = 3, k: int = 10, eos_id: int = -1) -> tuple[list[int], SpecStats]:
    """Greedy speculative decode with PROMPT-LOOKUP (S4) — no draft model. Proposes by
    matching the last `ngram` tokens of the context against an earlier occurrence and
    copying up to `k` tokens that followed. Returns (generated_ids, stats), token-identical
    to plain target greedy. The draft is free (r→0), so it can't regress; k can be larger
    than a draft model's (a long wrong copy costs only a verify slot, not k forwards).
    """
    if ngram < 1:
        raise ValueError("ngram must be >= 1")
    if k < 1:
        raise ValueError("k (max proposal length) must be >= 1")
    if max_tokens <= 0 or not prompt_ids:
        return [], SpecStats()
    return _greedy_spec_core(target, prompt_ids, max_tokens, PromptLookupProposer(ngram, k), eos_id)


# ---------------------------------------------------------------------------
# The SAMPLING speculative loop (S5) — the sibling of _greedy_spec_core.
# Same verify + rollback machinery; the accept rule is rejection_accept (which reduces to
# accept_prefix at temperature 0), and every draw goes through token_probs so the output
# is distribution-identical to plain target sampling.
# ---------------------------------------------------------------------------


def _sample_spec_core(target, prompt_ids: list[int], max_tokens: int, proposer, params,
                      rng, eos_id: int) -> tuple[list[int], SpecStats]:
    """Core SAMPLING speculative decode. Returns (generated_ids, stats). Every emitted
    token is drawn from the target's own shaped distribution p at its position (an accepted
    draft has marginal p by the rejection-sampling theorem, a correction/bonus is drawn from
    p directly), so the output is distribution-identical to plain target sampling — the
    sampling analog of _greedy_spec_core's token-identity. `params` is a nicpp.SamplingParams
    (temperature 0 makes this reduce to the greedy core); `rng` is a numpy Generator."""
    cap = len(prompt_ids) + max_tokens + proposer.max_k + 1
    tcache = target.make_cache(cap)
    proposer.prefill(prompt_ids, cap)

    # Prefill the target; the first token is a draw from its shaped last-row distribution —
    # the SAME conditional plain sampling draws its first token from.
    tlog = target.forward(prompt_ids, tcache)
    cur = _draw(nicpp.token_probs(tlog[-1], params, prompt_ids), rng)

    generated: list[int] = []
    stats = SpecStats()
    context = list(prompt_ids)

    def emit(tok: int) -> bool:
        if eos_id >= 0 and tok == eos_id:
            return False
        generated.append(tok)
        context.append(tok)
        stats.emitted += 1
        return len(generated) < max_tokens

    if not emit(cur):
        return generated, stats

    while True:
        L = tcache.length

        # --- Propose: k candidates + the distribution q each was drawn from (None = point
        # mass for a deterministic proposer). k may be 0 (prompt-lookup miss -> plain draw). ---
        d, qs = proposer.propose_sampling(cur, context, params, rng)
        k = len(d)

        # --- Verify: ONE target forward over [cur, d0..d_{k-1}] -> k+1 logit rows; shape
        # each into the target's distribution p_i at that position (position i conditions on
        # context + d[:i], matching what plain sampling would condition on there). ---
        tl = target.forward([cur] + d, tcache)
        p_list = [nicpp.token_probs(tl[i], params, context + d[:i]) for i in range(k + 1)]

        # --- Accept by rejection sampling (the shared rule); marginal of each emitted token
        # is exactly p_i. ---
        a, new = rejection_accept(d, qs, p_list, rng)
        stats.verifies += 1
        stats.proposals += 1 if k else 0
        stats.drafted += k
        stats.accepted += a
        stats.accept_lengths.append(a)

        keep = L + a + 1
        tcache.truncate(keep)
        proposer.rollback(keep)

        for tok in new:
            cur = tok
            if not emit(tok):
                return generated, stats


def sample_speculative(target, draft, prompt_ids: list[int], max_tokens: int, params,
                       k: int = 4, seed: int = 0, eos_id: int = -1
                       ) -> tuple[list[int], SpecStats]:
    """Speculative decode with SAMPLING and a DRAFT MODEL (S5). The draft samples k tokens
    from its own shaped distribution; the target verifies by rejection sampling. Returns
    (generated_ids, stats), distribution-identical to plain target sampling with `params`.
    At params.temperature == 0 it's token-identical to `greedy_speculative` (the greedy
    floor is the temp->0 limit). `params` is a nicpp.SamplingParams; `seed` seeds the draws."""
    if k < 1:
        raise ValueError("k (draft length) must be >= 1")
    if max_tokens <= 0 or not prompt_ids:
        return [], SpecStats()
    rng = np.random.default_rng(seed)
    return _sample_spec_core(target, prompt_ids, max_tokens, DraftModelProposer(draft, k),
                             params, rng, eos_id)


def sample_prompt_lookup(target, prompt_ids: list[int], max_tokens: int, params,
                         ngram: int = 3, k: int = 10, seed: int = 0, eos_id: int = -1
                         ) -> tuple[list[int], SpecStats]:
    """Speculative decode with SAMPLING and PROMPT-LOOKUP (S5) — no draft model. The n-gram
    copy is a point-mass proposal; the target verifies by rejection sampling. Returns
    (generated_ids, stats), distribution-identical to plain target sampling with `params`,
    and token-identical to `greedy_prompt_lookup` at params.temperature == 0."""
    if ngram < 1:
        raise ValueError("ngram must be >= 1")
    if k < 1:
        raise ValueError("k (max proposal length) must be >= 1")
    if max_tokens <= 0 or not prompt_ids:
        return [], SpecStats()
    rng = np.random.default_rng(seed)
    return _sample_spec_core(target, prompt_ids, max_tokens, PromptLookupProposer(ngram, k),
                             params, rng, eos_id)
