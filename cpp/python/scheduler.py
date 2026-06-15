"""F7 — Python serving layer: continuous (iteration-level) batching over the C++
kernels exposed in F6 (the `nicpp` module). This is the vLLM shape taking form:
Python orchestration, C++ compute.

Each request gets its own KV cache and per-sequence state. The scheduler advances
every running sequence by one token per step, evicts the ones that finished, and
admits queued requests into the freed slots — *continuous* batching: the batch
composition changes every iteration, so a short request never waits behind a long
one (no head-of-line blocking) and slots stay full while there's work.

Token selection happens here in Python (numpy) over the logits the C++ forward
returns — greedy by default, with the same repetition-penalty / temperature / top-k
/ top-p warpers, in the same order, as nanoinfer/sampling.py. Greedy is deterministic
and operates on the float32 logits exactly as the C++ sampler does, so a sequence's
output is identical whether it runs alone or interleaved with others — the parity
test (tests/run_serve.py).

Scope, stated honestly: this is the *scheduling* layer. The kernels are still
batch-1, so step() loops one forward() per running sequence — it does not yet fuse
the per-sequence GEMMs into a single batched matmul (the raw-throughput win). That
needs a batched C++ forward kernel, and it slots in *under* this scheduler without
changing it: the serving layer is decoupled from whether the kernel is batched. It
pairs naturally with the F8 paged-attention read path (per-sequence cache lengths).
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "build"))  # nicpp.*.so

import nicpp  # noqa: E402


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


class Scheduler:
    """Continuous-batching scheduler over the C++ (batch-1) kernels.

    Usage: add() requests, then run() to drive them to completion, or call step()
    yourself to interleave with other work (an event loop, request arrivals, ...).
    """

    def __init__(self, model, max_batch: int = 8):
        self.model = model
        self.max_batch = max_batch
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

    def _emit(self, seq: Sequence, logits_row: np.ndarray) -> bool:
        """Sample one token from the last-position logits and apply the stop rules,
        exactly as generate() does: EOS is checked *before* the token is emitted (so
        it's excluded), then max_tokens. Returns True if the sequence is now done."""
        eos = seq.req.eos_id if seq.req.eos_id >= 0 else self._eos_default
        tok = sample_token(logits_row, seq.req, seq.context, seq._rng)
        if eos >= 0 and tok == eos:
            return True
        seq.output_ids.append(tok)
        return len(seq.output_ids) >= seq.req.max_tokens

    def step(self) -> None:
        # 1. Decode: advance each running sequence by one token (a 1-token forward).
        for seq in self.running:
            logits = self.model.forward([seq.last_token], seq.cache)  # [1, vocab]
            if self._emit(seq, logits[-1]):
                self._finish(seq)

        # 2. Evict the finished, freeing their slots.
        self.running = [s for s in self.running if s.state is State.RUNNING]

        # 3. Admit waiting requests into free slots, prefilling each (full-prompt
        #    forward). The first token is emitted from the prefill logits, so a
        #    newly admitted sequence has produced one token by the end of this step.
        while len(self.running) < self.max_batch and self.waiting:
            req = self.waiting.popleft()
            if not req.prompt_ids:  # nothing to condition on (generate() returns [])
                self.finished[req.request_id] = []
                continue
            cache = self.model.make_cache(len(req.prompt_ids) + max(req.max_tokens, 1))
            seq = Sequence(req, cache)
            seq.state = State.RUNNING
            logits = self.model.forward(req.prompt_ids, cache)  # prefill
            if req.max_tokens <= 0 or self._emit(seq, logits[-1]):
                self._finish(seq)
            else:
                self.running.append(seq)

        self.peak_batch = max(self.peak_batch, len(self.running))
        self.steps += 1

    def run(self) -> dict[str, list[int]]:
        """Drive all queued requests to completion; return {request_id: output_ids}."""
        while self.has_work():
            self.step()
        return self.finished
