"""Python serving layer (stages F7–F8b): continuous (iteration-level) batching over
the C++ kernels exposed in F6 (the `nicpp` module). The vLLM shape: Python
orchestration, C++ compute.

Each request gets its own KV cache and per-sequence state. The scheduler advances
every running sequence by one token per step, evicts the ones that finished, and
admits queued requests into the freed slots — *continuous* batching: the batch
composition changes every iteration, so a short request never waits behind a long
one (no head-of-line blocking) and slots stay full while there's work.

Three knobs select how the kernels back that scheduling, each parity-preserving:
  - batched (F8a, default): decode the whole running set in one forward_batch call —
    the per-sequence projection GEMMs fuse into one matmul. batched=False keeps the
    per-sequence forward() loop (F7) for A/B.
  - block_size>0 (F8b): a paged KV cache — one shared BlockPool, a PagedKVCache per
    sequence (no per-sequence max_seq preallocation), admission gated on KV blocks.
    block_size=0 keeps the contiguous cache.
  - prefix_sharing (paged only): a PrefixCache lets requests reuse a shared prefix's
    KV blocks (RadixAttention) instead of recomputing them — skipping that prefill.

Token selection happens here in Python (numpy) over the logits the C++ forward
returns — greedy by default, with the same repetition-penalty / temperature / top-k
/ top-p warpers, in the same order, as nanoinfer/sampling.py. Greedy is deterministic
and operates on the float32 logits exactly as the C++ sampler does, so a sequence's
output is identical whether it runs alone or interleaved, batched or paged — the
parity test (tests/python/run_serve.py)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass
class Request:
    """A generation request: a tokenized prompt + decoding params. Mirrors the
    knobs of nicpp.SamplingParams / GenerateConfig so a request maps 1:1 onto a
    standalone generate() call (that mapping is what the parity test checks)."""

    request_id: str
    prompt_ids: list[int]
    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 0
    eos_id: int = -1  # -1 -> fall back to the model's eos_token_id


class State(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    """Per-sequence runtime state: its own KV cache, generated ids, sampler RNG."""

    def __init__(self, req: Request, cache):
        self.req = req
        self.cache = cache
        self.output_ids: list[int] = []
        self.state = State.WAITING
        self.reserved_blocks = 0  # KV blocks reserved for it (paged mode)
        self._rng = np.random.default_rng(req.seed)

    @property
    def last_token(self) -> int:
        # What the next decode step feeds: the most recent token, or — right after
        # prefill, before any token is emitted — the prompt's final token.
        return self.output_ids[-1] if self.output_ids else self.req.prompt_ids[-1]

    @property
    def context(self) -> list[int]:
        # Running context for the repetition penalty (prompt + generated so far),
        # excluding the token about to be sampled — the same window generate() uses.
        return self.req.prompt_ids + self.output_ids


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def sample_token(logits: np.ndarray, req: Request, context: list[int], rng) -> int:
    """Pick the next id from a [vocab] logit row — the nanoinfer/sampling.py pipeline:
    repetition penalty (a processor; shapes greedy too) -> greedy short-circuit ->
    temperature -> top-k -> top-p -> softmax -> categorical draw.

    Greedy stays in float32 (the dtype the C++ forward emits and the C++ sampler uses)
    so argmax is bit-identical to the standalone path; only the sampling draw promotes
    to float64 for a stable softmax (and sampled output is not cross-engine identical
    anyway — the RNG differs)."""
    logits = np.array(logits, dtype=np.float32, copy=True)  # don't mutate the caller's row

    if req.repetition_penalty != 1.0:
        seen = np.unique(np.asarray(context, dtype=np.int64))
        seen = seen[(seen >= 0) & (seen < logits.shape[0])]
        s = logits[seen]
        pen = np.float32(req.repetition_penalty)
        logits[seen] = np.where(s > 0, s / pen, s * pen).astype(np.float32)

    if req.temperature == 0.0:  # greedy: argmax (first max, matching C++ / numpy)
        return int(np.argmax(logits))

    logits = logits / np.float32(req.temperature)
    if req.top_k > 0:
        k = min(req.top_k, logits.shape[0])
        kth = np.partition(logits, -k)[-k]  # k-th largest; ties at it are kept
        logits[logits < kth] = -np.inf
    if req.top_p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = _softmax(logits[order].astype(np.float64))
        cum = np.cumsum(probs) - probs  # prefix sum EXCLUDING current (shifted); keeps top-1
        logits[order[cum > req.top_p]] = -np.inf

    probs = _softmax(logits.astype(np.float64))
    return int(rng.choice(logits.shape[0], p=probs))


def sample_batch(logits: np.ndarray, seqs: list["Sequence"]) -> np.ndarray:
    """Select one token per row of a [N, vocab] logit batch — the batched mirror of
    sample_token, for the batched decode step. Rows that are plain greedy (temperature 0
    and no repetition penalty — the default) are decided together with one vectorized
    argmax over the whole batch, skipping the per-row vocab-sized copy sample_token makes;
    numpy's argmax takes the first max along the axis, so each row is bit-identical to
    sample_token's argmax. Rows that sample or carry a repetition penalty keep their own
    per-sequence pipeline (their temperature / top-k / top-p / RNG / context don't share
    across the batch), still via sample_token. Token-identical to calling sample_token on
    every row, so interleaved output is unchanged (run_serve.py asserts it at every batch)."""
    n = len(seqs)
    greedy = np.fromiter(
        (s.req.temperature == 0.0 and s.req.repetition_penalty == 1.0 for s in seqs),
        dtype=bool, count=n)
    if greedy.all():  # the common case: one argmax over the batch, no per-row copy
        return np.argmax(logits, axis=1)
    toks = np.empty(n, dtype=np.int64)
    if greedy.any():  # mixed batch: still batch the greedy rows together...
        toks[greedy] = np.argmax(logits[greedy], axis=1)
    for i, s in enumerate(seqs):  # ...and run the genuinely-sampled rows one at a time
        if not greedy[i]:
            toks[i] = sample_token(logits[i], s.req, s.context, s._rng)
    return toks


class PrefixCache:
    """A radix-style cache of computed KV prefix blocks (the RadixAttention idea, on
    the F8b block table), keyed by the block-aligned token prefix. A new request can
    reuse another's prefix KV instead of recomputing it — skipping that prefill and
    sharing the blocks. The cache holds a reference to each block it stores (incref),
    so blocks survive after the producing sequence finishes; clear() releases them.

    Correctness: a token's K/V depends only on the tokens up to it (causal attention),
    so two requests with an identical block-aligned prefix have identical KV there —
    sharing is exact and output is unchanged (run_serve.py checks it vs standalone)."""

    def __init__(self, pool, block_size: int):
        self.pool = pool
        self.block_size = block_size
        self._blocks: dict[tuple, int] = {}  # prefix-key (tokens) -> physical block id

    @property
    def held_blocks(self) -> int:
        return len(self._blocks)

    def match(self, tokens: list[int]) -> tuple[list[int], int]:
        """Longest run of cached leading blocks for `tokens` -> (block_ids, length).
        length is block-aligned and < len(tokens) (a suffix always remains, since the
        logits need at least the last token recomputed). No incref — the caller's
        share_prefix() takes the borrowing reference."""
        bs = self.block_size
        shared: list[int] = []
        for k in range((len(tokens) - 1) // bs):  # leave >= 1 token for the suffix
            b = self._blocks.get(tuple(tokens[: (k + 1) * bs]))
            if b is None:
                break
            shared.append(b)
        return shared, len(shared) * bs

    def register(self, tokens: list[int], block_table: list[int]) -> None:
        """Cache a sequence's complete prompt blocks for later reuse, increfing each
        newly cached block. A partial trailing block isn't shareable, so skip it."""
        bs = self.block_size
        for k in range(min(len(tokens) // bs, len(block_table))):
            key = tuple(tokens[: (k + 1) * bs])
            if key not in self._blocks:
                self._blocks[key] = block_table[k]
                self.pool.incref(block_table[k])

    def clear(self) -> None:
        for b in self._blocks.values():
            self.pool.free(b)
        self._blocks.clear()


class Scheduler:
    """Continuous-batching scheduler over the C++ kernels (F7/F8a/F8b).

    Usage: add() requests, then run() to drive them to completion, or call step()
    yourself to interleave with other work (an event loop, request arrivals, ...).
    """

    def __init__(self, model, max_batch: int = 8, batched: bool = True,
                 block_size: int = 0, num_blocks: int = 0, prefix_sharing: bool = False):
        self.model = model
        self.max_batch = max_batch
        # batched=True decodes the whole running set in one forward_batch call (F8a:
        # the per-sequence projection GEMMs fuse into one matmul). False keeps the
        # per-sequence forward() loop (F7) — same tokens, for A/B and parity checks.
        self.batched = batched
        # block_size>0 enables the paged KV cache (F8b): one shared BlockPool, a
        # PagedKVCache per sequence (no per-sequence max_seq preallocation). Admission
        # is then gated on KV blocks (conservatively reserving each sequence's
        # worst-case so the pool can't be over-committed). block_size=0 keeps the
        # contiguous cache. Either way the tokens are identical (paged is bit-exact).
        self.paged = block_size > 0
        self.pool = model.make_block_pool(block_size, num_blocks) if self.paged else None
        # prefix_sharing reuses a cached prefix's KV blocks across requests (paged only).
        if prefix_sharing and not self.paged:
            raise ValueError("prefix_sharing requires paged mode (block_size > 0)")
        self.prefix_cache = PrefixCache(self.pool, block_size) if prefix_sharing else None
        self.reserved = 0  # KV blocks reserved by running sequences (paged only)
        self.shared_prefill_tokens = 0  # prompt tokens skipped via prefix sharing
        self.waiting: deque[Request] = deque()
        self.running: list[Sequence] = []
        self.finished: dict[str, list[int]] = {}
        self._eos_default = model.config.eos_token_id
        self.steps = 0
        self.peak_batch = 0  # max sequences running in any one step (utilization)

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    def _finish(self, seq: Sequence) -> None:
        seq.state = State.FINISHED
        self.finished[seq.req.request_id] = seq.output_ids
        if self.paged:
            # Drop the reservation and the cache itself — its PagedKVCache destructor
            # returns the sequence's blocks to the pool for the next request to reuse.
            self.reserved -= seq.reserved_blocks
            seq.cache = None

    def _accept(self, seq: Sequence, tok: int) -> bool:
        """Apply the stop rules to an already-selected token, exactly as generate() does:
        EOS is checked *before* the token is emitted (so it's excluded), then max_tokens.
        Returns True if the sequence is now done. Shared by the single-sequence prefill path
        (_emit) and the batched decode path (which selects tokens together via sample_batch)."""
        eos = seq.req.eos_id if seq.req.eos_id >= 0 else self._eos_default
        if eos >= 0 and tok == eos:
            return True
        seq.output_ids.append(tok)
        return len(seq.output_ids) >= seq.req.max_tokens

    def _emit(self, seq: Sequence, logits_row: np.ndarray) -> bool:
        """Sample one token from a single logits row, then apply the stop rules. The prefill-
        admission path uses this (one sequence at a time); the batched decode step selects the
        whole running set at once via sample_batch, then calls _accept on each."""
        return self._accept(seq, sample_token(logits_row, seq.req, seq.context, seq._rng))

    def step(self) -> None:
        # 1. Decode the running set by one token. Batched: one forward_batch over all
        #    N sequences (fused projections). Else: a forward() per sequence. Either
        #    way row i belongs to running[i], so the emit/stop logic is shared.
        if self.running:
            if self.batched:
                tokens = [s.last_token for s in self.running]
                caches = [s.cache for s in self.running]
                logits = self.model.forward_batch(tokens, caches)  # [N, vocab]
            else:
                logits = np.stack(
                    [self.model.forward([s.last_token], s.cache)[-1] for s in self.running])
            # Select the whole running set's next tokens together (greedy rows vectorized).
            for seq, tok in zip(self.running, sample_batch(logits, self.running)):
                if self._accept(seq, int(tok)):
                    self._finish(seq)

        # 2. Evict the finished, freeing their slots.
        self.running = [s for s in self.running if s.state is State.RUNNING]

        # 3. Admit waiting requests into free slots, prefilling each (full-prompt
        #    forward). The first token is emitted from the prefill logits, so a
        #    newly admitted sequence has produced one token by the end of this step.
        #    In paged mode admission is also gated on KV blocks (FCFS: if the head
        #    request can't be reserved yet, wait rather than skip ahead).
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting[0]
            if not req.prompt_ids:  # nothing to condition on (generate() returns [])
                self.waiting.popleft()
                self.finished[req.request_id] = []
                continue

            reserved = 0
            shared_blocks: list[int] = []
            shared_len = 0
            if self.paged:
                bs = self.pool.block_size
                # Prefix sharing: reuse a cached prefix's blocks if one matches, so we
                # prefill (and reserve) only the suffix past it.
                if self.prefix_cache is not None:
                    shared_blocks, shared_len = self.prefix_cache.match(req.prompt_ids)
                # Reserve the worst case (prompt + all decode tokens), minus the shared
                # blocks (already allocated), so lazy allocation can't exceed the pool.
                worst = (len(req.prompt_ids) + max(req.max_tokens, 1) + bs - 1) // bs
                reserved = worst - len(shared_blocks)
                if worst > self.pool.num_blocks:
                    raise ValueError(
                        f"request {req.request_id} needs {worst} blocks; pool has "
                        f"{self.pool.num_blocks} (raise num_blocks or block_size)")
                # Owned reservations of running seqs + blocks the prefix cache holds
                # must fit the pool (held counted conservatively — never over-commits).
                held = self.prefix_cache.held_blocks if self.prefix_cache else 0
                if self.reserved + held + reserved > self.pool.num_blocks:
                    break  # not enough free KV blocks right now; keep it queued
                cache = self.pool.make_cache()  # PagedKVCache or CudaPagedKVCache by device
                if shared_blocks:
                    cache.share_prefix(shared_blocks, shared_len)
                    self.shared_prefill_tokens += shared_len
                self.reserved += reserved
            else:
                cache = self.model.make_cache(len(req.prompt_ids) + max(req.max_tokens, 1))

            self.waiting.popleft()
            seq = Sequence(req, cache)
            seq.reserved_blocks = reserved
            seq.state = State.RUNNING
            # Prefill only the suffix past the shared prefix (shared_len=0 if none); its
            # last position is the prompt's last token, so the first emitted token is
            # the same as a full prefill.
            logits = self.model.forward(req.prompt_ids[shared_len:], cache)
            if self.prefix_cache is not None:
                self.prefix_cache.register(req.prompt_ids, cache.block_table)
            if req.max_tokens <= 0 or self._emit(seq, logits[-1]):
                self._finish(seq)
            else:
                self.running.append(seq)

        self.peak_batch = max(self.peak_batch, len(self.running))
        self.steps += 1

    def cancel(self, request_id: str) -> bool:
        """Abort a request wherever it is (the serving layer's disconnect path).
        Queued: drop it, finishing it with an empty output. Running: evict it now
        — _finish frees its KV (paged: the blocks return to the pool for the next
        admit) and records the partial output in `finished`. Already finished (or
        unknown): no-op. Returns True if it was live when cancelled."""
        for r in self.waiting:
            if r.request_id == request_id:
                self.waiting.remove(r)
                self.finished[request_id] = []
                return True
        for seq in self.running:
            if seq.req.request_id == request_id:
                self._finish(seq)
                self.running = [s for s in self.running if s.state is State.RUNNING]
                return True
        return False

    def run(self) -> dict[str, list[int]]:
        """Drive all queued requests to completion; return {request_id: output_ids}."""
        while self.has_work():
            self.step()
        return self.finished

    def clear_prefix_cache(self) -> None:
        """Release every block the prefix cache holds, returning them to the pool
        (e.g. between batches, or to reclaim KV memory). No-op without prefix sharing."""
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
