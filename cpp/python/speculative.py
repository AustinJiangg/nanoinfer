"""Speculative decoding (S-track, S0): a small fast *draft* model proposes K tokens,
the big *target* model verifies all K+1 in ONE forward, and we keep the longest prefix
the target agrees with — emitting several tokens per target forward instead of one.

The engine already has everything this needs (verified in S0):

  * `model.forward([t0, t1, ..], cache)` on a POPULATED cache places the tokens at
    positions cache.length.. and returns a logits row per token — bit-identical to
    decoding them one at a time (attention masks query i to keys 0..length+i). That
    single multi-token pass IS the verify step: forward([cur, d0..d_{K-1}], target_cache)
    → K+1 logit rows, the target's own next-token at each drafted position.
  * `cache.truncate(L)` drops the K/V for positions >= L — the rollback that discards
    rejected drafts (the contiguous cache just moves its length pointer).

Correctness (the S0 gate): with GREEDY decoding this is *token-identical* to plain
target greedy. Each emitted token is the target's argmax at its position (an accepted
draft equals it by construction; a rejected one is replaced by it), and the target
logits at a drafted position equal what plain sequential decode would produce. So the
draft only changes SPEED (how many tokens land per verify), never the output.

CPU-oracle-locked: run over the CPU backend, `tokens == model.generate(...)` for the
target. `run_spec.py` is the test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpecStats:
    """Speculation efficiency (for S2's tuning; irrelevant to correctness)."""
    verifies: int = 0        # target forwards (each emits >= 1 token)
    drafted: int = 0         # draft tokens proposed
    accepted: int = 0        # draft tokens the target accepted
    emitted: int = 0         # tokens actually emitted
    accept_lengths: list[int] = field(default_factory=list)  # accepted count `a` per verify (S2)

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_verify(self) -> float:
        # The speedup proxy: how many tokens each (expensive) target forward yields.
        return self.emitted / self.verifies if self.verifies else 0.0

    def accept_histogram(self, k: int) -> list[int]:
        """Counts of verifies by accepted-draft length a = 0..k — the accept-length
        DISTRIBUTION (S2). The mean (accept_rate) hides the shape: a draft that either
        nails a whole run or fails at token 0 is bimodal, and that shape is what picks K."""
        h = [0] * (k + 1)
        for a in self.accept_lengths:
            if 0 <= a <= k:
                h[a] += 1
        return h


def greedy_speculative(target, draft, prompt_ids: list[int], max_tokens: int,
                       k: int = 4, eos_id: int = -1) -> tuple[list[int], SpecStats]:
    """Greedy speculative decode. Returns (generated_ids, stats), with generated_ids
    token-identical to `target.generate(prompt_ids, max_tokens, greedy, eos_id)`.

    `target` and `draft` are nicpp.Model handles (pass the same model as both for a
    draft==target self-check: every draft is accepted). k >= 1 is the draft length.
    """
    if k < 1:
        raise ValueError("k (draft length) must be >= 1")
    if max_tokens <= 0 or not prompt_ids:
        return [], SpecStats()

    # Caches sized for prompt + everything we emit, plus k slack for the tentative
    # tail a verify writes before rollback.
    cap = len(prompt_ids) + max_tokens + k + 1
    tcache = target.make_cache(cap)
    dcache = draft.make_cache(cap)

    # Prefill both on the prompt (the draft cache must track the same history so its
    # proposals are conditioned correctly). The target's last row gives the first token.
    tlog = target.forward(prompt_ids, tcache)
    draft.forward(prompt_ids, dcache)
    cur = int(np.argmax(tlog[-1]))

    generated: list[int] = []
    stats = SpecStats()

    def emit(tok: int) -> bool:
        """Append tok unless it's eos; return False when generation should stop."""
        if eos_id >= 0 and tok == eos_id:
            return False                       # eos: stop without emitting (matches generate.cpp)
        generated.append(tok)
        stats.emitted += 1
        return len(generated) < max_tokens

    if not emit(cur):                          # the first token (g_0), eos/limit-checked
        return generated, stats

    while True:
        L = tcache.length                      # confirmed prefix length (== dcache.length)

        # --- Draft: propose k tokens autoregressively from cur. ---
        # We feed cur, d0, .., d_{k-2} to collect d0..d_{k-1} (k proposals), then feed
        # d_{k-1} once more (its output discarded) so the draft cache ends at L+k+1 —
        # the SAME length the target reaches after verifying k+1 tokens. Keeping the two
        # caches lock-step lets one truncate(L+a+1) roll back BOTH, with no all-accepted
        # special case (when a==k the draft would otherwise be one position short).
        d: list[int] = []
        x = cur
        for _ in range(k):
            dl = draft.forward([x], dcache)
            x = int(np.argmax(dl[-1]))
            d.append(x)
        draft.forward([d[-1]], dcache)         # the k+1-th feed; keeps dcache == tcache length

        # --- Verify: ONE target forward over [cur, d0..d_{k-1}] -> k+1 logit rows. ---
        # Row i is the target's distribution AFTER [cur, d0..d_{i-1}], i.e. its own token
        # for the position draft guessed as d_i (and row k is the bonus after all drafts).
        tl = target.forward([cur] + d, tcache)
        tv = np.argmax(tl, axis=1).astype(np.int64)   # target's greedy token at each position

        # --- Accept the longest prefix the target agrees with. ---
        a = 0
        while a < k and int(tv[a]) == d[a]:
            a += 1
        # Confirmed this step: d[0..a-1] (accepted) + tv[a] (the correction if a<k, else
        # the free bonus token if all k accepted). tv has k+1 rows, so tv[a] is valid.
        new = d[:a] + [int(tv[a])]
        stats.verifies += 1
        stats.drafted += k
        stats.accepted += a
        stats.accept_lengths.append(a)

        # --- Rollback: keep cur + d[0..a-1] (positions L..L+a); tv[a] becomes the next
        # cur, fed fresh next iteration (its K/V isn't in the cache yet). ---
        keep = L + a + 1
        tcache.truncate(keep)
        dcache.truncate(keep)

        # --- Emit the confirmed tokens, tracking cur as the last one. ---
        for tok in new:
            cur = tok
            if not emit(tok):
                return generated, stats
