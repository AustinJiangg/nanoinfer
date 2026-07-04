"""S3 — speculative decoding folded into the continuous-batching serving layer.

`Scheduler` (F7/F8) advances every running sequence by ONE token per step. A speculative
step is structurally different: each sequence *proposes* k candidate tokens, the target
*verifies* all k+1 in one forward, and we keep the longest accepted prefix — so a step
emits a VARIABLE number of tokens per sequence (0..k+1). Rather than bolt that onto the
clean F7/F8 code, `SpecScheduler` mirrors its continuous-batching shape (admit / evict /
dynamic batch) around the S0 spec loop.

Each running sequence carries its own target KV cache AND its own proposer (a draft model
keeps a per-sequence draft cache lock-step with the target; prompt-lookup is stateless).
The scheduler drives the running set in three phases per step:

  1. propose — each sequence's proposer guesses k_s tokens (k_s may be 0: a prompt-lookup
     miss degrades that sequence to a plain greedy forward, never a regression).
  2. verify  — the target confirms each sequence's [cur_s, d_s..]. batched=True (S3b) fuses
     these into ONE ragged forward_spec_batch — the projection GEMMs run over Sigma(k_s+1)
     rows (the F8a-analog: the throughput win); batched=False (S3a) runs one forward() per
     sequence (the F7-analog: the scheduling win alone). The two are token-identical.
  3. commit  — per sequence: accept the longest prefix (the shared `accept_prefix` rule),
     truncate() both caches to discard rejects, emit the confirmed tokens, stop on eos /
     max_tokens.

Correctness (the S0 gate, batch-invariant): with greedy decoding every emitted token is
the target's own argmax at its position, so a sequence's output is token-identical to plain
target greedy whether it runs alone or interleaved with others — exactly the guarantee
`Scheduler` gives for plain decode, and what `tests/run_spec_serve.py` asserts against
standalone generate at every batch size and proposer mix.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "build"))  # nicpp.*.so

import nicpp  # noqa: E402

from speculative import (  # noqa: E402
    DraftModelProposer,
    PromptLookupProposer,
    SpecStats,
    accept_prefix,
)


@dataclass
class SpecRequest:
    """A speculative-decoding request. Greedy only for now — token-identity to plain
    greedy is the S0 invariant this whole track rests on; sampling-parity (rejection
    sampling) is the open S0 tail. `proposer` picks how the k candidates are produced:

      * "draft"  — a draft model runs k forwards (needs a draft Model on the scheduler).
      * "lookup" — prompt-lookup / n-gram: match the recent context against an earlier
        occurrence and copy what followed. No model; `ngram` sets the match length.
    """

    request_id: str
    prompt_ids: list[int]
    max_tokens: int = 32
    eos_id: int = -1          # -1 -> the model's eos_token_id
    proposer: str = "draft"   # "draft" | "lookup"
    k: int = 4                # draft length / max proposal length
    ngram: int = 3            # prompt-lookup match length (ignored for "draft")

    @property
    def max_k(self) -> int:
        # Longest tentative tail a verify writes before rollback — sizes the cache slack.
        return self.k


class State(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class SpecSequence:
    """Per-sequence spec runtime: its target cache, proposer, running context, and the
    `cur` token fed into the next verify (the target's last confirmed argmax, whose K/V is
    NOT yet in the cache — the verify feeds it fresh as [cur, d..], exactly like the
    single-sequence loop)."""

    def __init__(self, req: SpecRequest, tcache, proposer):
        self.req = req
        self.tcache = tcache
        self.proposer = proposer
        self.output_ids: list[int] = []
        self.context: list[int] = list(req.prompt_ids)  # prompt + generated; drives lookup
        self.cur: int = -1                # set from the prefill logits at admission
        self.verify_L: int = 0            # tcache length captured just before a verify
        self.reserved_blocks = 0          # target KV blocks reserved for it (paged mode)
        self.state = State.WAITING


class SpecScheduler:
    """Continuous-batching scheduler for speculative decoding (S3).

    Usage: construct with a target model (and a draft model if any request uses the "draft"
    proposer), add() requests, then run(). Mirrors `Scheduler`: add/evict/admit keep the
    batch full while there's work, and each finished sequence's output is token-identical to
    standalone greedy.
    """

    def __init__(self, target, draft=None, max_batch: int = 8, batched: bool = True,
                 block_size: int = 0, num_blocks: int = 0):
        self.model = target
        self.draft = draft
        self.max_batch = max_batch
        # batched=True (S3b) verifies the whole running set in ONE ragged forward_spec_batch:
        # the per-sequence verify blocks ([cur_s, d_s..], k_s+1 rows each) concatenate into
        # one call, so the projection GEMMs fuse over Sigma(k_s+1) rows (each weight streamed
        # once). False (S3a) keeps the per-sequence forward() loop — same tokens, for A/B and
        # the bit-identity gate. Either way the output is token-identical (paged/contiguous
        # both bit-exact per block), so this is a throughput knob, never an output change.
        self.batched = batched
        # block_size>0 pages the TARGET cache (the F8b analog for the verify cache): each
        # sequence draws a PagedKVCache from one shared BlockPool instead of preallocating a
        # contiguous cache to its worst-case length, and admission is gated on KV blocks
        # (conservatively reserving each sequence's worst case so the pool can't over-commit,
        # even at the tentative-verify peak). block_size=0 keeps the contiguous cache. The
        # target is the memory-dominant 1.5B cache and the one truncate() rolls back; the
        # draft proposer's cache stays contiguous (the smaller 0.5B model — paging it needs a
        # separate draft-dim pool, a follow-up). Paged is bit-exact (S1), so tokens are
        # unchanged; the win is no per-sequence preallocation + block reuse across requests.
        self.paged = block_size > 0
        self.pool = target.make_block_pool(block_size, num_blocks) if self.paged else None
        self.reserved = 0              # target KV blocks reserved by running sequences (paged)
        self.waiting: deque[SpecRequest] = deque()
        self.running: list[SpecSequence] = []
        self.finished: dict[str, list[int]] = {}
        self.steps = 0
        self.peak_batch = 0            # max sequences running in any one step (utilization)
        self.stats = SpecStats()       # aggregate speculation efficiency (tuning only)

    def add(self, req: SpecRequest) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def _make_proposer(self, req: SpecRequest):
        if req.proposer == "draft":
            if self.draft is None:
                raise ValueError(f"request {req.request_id} uses the draft proposer but the "
                                 "scheduler has no draft model")
            return DraftModelProposer(self.draft, req.k)
        if req.proposer == "lookup":
            return PromptLookupProposer(req.ngram, req.k)
        raise ValueError(f"unknown proposer {req.proposer!r} (want 'draft' or 'lookup')")

    def _emit(self, seq: SpecSequence, tok: int) -> bool:
        """Append `tok` to the sequence unless it's eos, then apply the stop rules — the SAME
        order and eos convention as the single-sequence loop (`_greedy_spec_core.emit`): the
        raw request eos_id is used (eos_id < 0 -> no eos stop, only max_tokens), and eos is
        checked *before* the token is kept (so it's excluded). Returns True when the sequence
        is done (so the caller stops feeding it the rest of this step's confirmed tokens).
        This exact mirroring is why a SpecScheduler sequence is token-identical to standalone
        greedy_speculative / greedy_prompt_lookup."""
        eos = seq.req.eos_id
        if eos >= 0 and tok == eos:
            return True                                  # eos: stop without emitting
        seq.output_ids.append(tok)
        seq.context.append(tok)                          # keep context == prompt + generated
        self.stats.emitted += 1
        return len(seq.output_ids) >= seq.req.max_tokens

    def _finish(self, seq: SpecSequence) -> None:
        seq.state = State.FINISHED
        self.finished[seq.req.request_id] = seq.output_ids
        if self.paged:
            # Drop the reservation; the PagedKVCache destructor returns its blocks to the pool.
            self.reserved -= seq.reserved_blocks
        seq.tcache = None            # release the target cache; the draft cache frees with the proposer
        seq.proposer = None

    def _verify(self, seqs: list[SpecSequence], proposals: list[list[int]]) -> list[np.ndarray]:
        """Verify each sequence's [cur_s, d_s..] (k_s+1 tokens) against the target, returning
        the target's own argmax at each of the k_s+1 positions per sequence. Two backings,
        token-identical (S3b is bit-identical to S3a per block):
          * batched (S3b): ONE ragged forward_spec_batch over the concatenated verify blocks —
            the projection GEMMs fuse over Sigma(k_s+1) rows.
          * per-seq (S3a): one forward() per sequence.
        Each sequence's pre-verify cache length (verify_L) is captured FIRST — a forward only
        advances its own cache, so capturing up front is valid for both paths — so commit's
        truncate rolls back to exactly the confirmed prefix."""
        for seq in seqs:
            seq.verify_L = seq.tcache.length            # confirmed prefix length before verify

        if self.batched:
            tokens: list[int] = []
            counts: list[int] = []
            caches = []
            for seq, d in zip(seqs, proposals):
                block = [seq.cur] + d                   # k_s+1 verify tokens
                tokens.extend(block)
                counts.append(len(block))
                caches.append(seq.tcache)
            logits = self.model.forward_spec_batch(tokens, counts, caches)  # [sum(counts), vocab]
            out: list[np.ndarray] = []
            off = 0
            for c in counts:                            # slice each sequence's block back out
                out.append(np.argmax(logits[off:off + c], axis=1).astype(np.int64))
                off += c
            return out

        out = []
        for seq, d in zip(seqs, proposals):
            tl = self.model.forward([seq.cur] + d, seq.tcache)
            out.append(np.argmax(tl, axis=1).astype(np.int64))
        return out

    def _commit(self, seq: SpecSequence, d: list[int], tv: np.ndarray) -> None:
        """Accept the longest agreed prefix, roll BOTH caches back to it, emit the confirmed
        tokens. `tv` has len(d)+1 rows (the verify's per-position argmax)."""
        a, new = accept_prefix(d, tv)
        k = len(d)
        self.stats.verifies += 1
        self.stats.proposals += 1 if k else 0
        self.stats.drafted += k
        self.stats.accepted += a
        self.stats.accept_lengths.append(a)

        # Keep cur + d[0..a-1] (positions verify_L .. verify_L+a); the correction/bonus tv[a]
        # becomes the next cur, fed fresh next step (its K/V isn't in the cache yet). One
        # truncate rolls back the target; the proposer rolls back its own (lock-step) cache.
        keep = seq.verify_L + a + 1
        seq.tcache.truncate(keep)
        seq.proposer.rollback(keep)

        for tok in new:
            seq.cur = tok
            if self._emit(seq, tok):
                self._finish(seq)
                return

    def _admit(self) -> None:
        """Fill free slots with queued requests: prefill the target on the prompt, prime the
        proposer, emit the first token from the prefill logits (so an admitted sequence has
        produced one token by the end of its admission step — same as the single-sequence
        loop and as Scheduler's prefill-then-emit). In paged mode admission is also gated on
        target KV blocks (FCFS: if the head request can't be reserved yet, wait rather than
        skip ahead — no starvation)."""
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting[0]                        # peek: FCFS block gating may make us wait
            if not req.prompt_ids:                       # nothing to condition on
                self.waiting.popleft()
                self.finished[req.request_id] = []
                continue

            # Cache sized for prompt + everything emitted + max_k slack for the tentative tail
            # (the k+1 tentative rows a verify writes before truncate rolls them back).
            cap = len(req.prompt_ids) + req.max_tokens + req.max_k + 1
            reserved = 0
            if self.paged:
                bs = self.pool.block_size
                worst = (cap + bs - 1) // bs             # worst-case blocks (incl. tentative tail)
                if worst > self.pool.num_blocks:
                    raise ValueError(
                        f"request {req.request_id} needs {worst} blocks; pool has "
                        f"{self.pool.num_blocks} (raise num_blocks or block_size)")
                if self.reserved + worst > self.pool.num_blocks:
                    break                                # not enough free blocks now; keep it queued
                reserved = worst
                tcache = self.pool.make_cache()          # PagedKVCache / CudaPagedKVCache by device
                self.reserved += reserved
            else:
                tcache = self.model.make_cache(cap)

            self.waiting.popleft()
            proposer = self._make_proposer(req)
            proposer.prefill(req.prompt_ids, cap)        # draft: prefill its lock-step cache
            seq = SpecSequence(req, tcache, proposer)
            seq.reserved_blocks = reserved

            tlog = self.model.forward(req.prompt_ids, seq.tcache)
            seq.cur = int(np.argmax(tlog[-1]))           # first token (its K/V fed fresh next step)
            seq.state = State.RUNNING
            if req.max_tokens <= 0 or self._emit(seq, seq.cur):
                self._finish(seq)
            else:
                self.running.append(seq)

    def step(self) -> None:
        # 1-3. Advance the running set by a full spec step: propose, verify, commit.
        if self.running:
            proposals = [s.proposer.propose(s.cur, s.context) for s in self.running]
            verifies = self._verify(self.running, proposals)
            for seq, d, tv in zip(self.running, proposals, verifies):
                self._commit(seq, d, tv)
            # 4. Evict the finished, freeing their slots.
            self.running = [s for s in self.running if s.state is State.RUNNING]

        # 5. Admit queued requests into free slots (prefill + first token).
        self._admit()

        self.peak_batch = max(self.peak_batch, len(self.running))
        self.steps += 1

    def run(self) -> dict[str, list[int]]:
        """Drive all queued requests to completion; return {request_id: output_ids}."""
        while self.has_work():
            self.step()
        return self.finished
