"""V0 — the async serving bridge: asyncio requests in, streamed tokens out.

The schedulers (F7 `Scheduler`, S3 `SpecScheduler`) are synchronous step machines:
`add()` requests, call `step()` until `has_work()` is false. That shape is perfect
for parity gates and batch CLIs, but a server needs the opposite interface — many
concurrent callers, each awaiting *its own* token stream while requests keep
arriving. `AsyncEngine` is that bridge, and it is deliberately scheduler-agnostic:
it drives anything with the add/step/has_work/running/finished surface (both
schedulers expose it), so plain and speculative decoding share one serving layer.

Threading model — one engine thread, the event loop stays free:

  * The step loop runs in a dedicated background thread. The heavy C++ calls
    (forward / forward_batch / forward_spec_batch) drop the GIL (F6), so the
    asyncio event loop keeps accepting connections and writing responses while
    the model computes.
  * The event loop talks TO the engine thread through one thread-safe inbox
    queue (submit / cancel / stop). The engine thread talks BACK through
    `loop.call_soon_threadsafe`, pushing events onto each request's own
    `asyncio.Queue` — the only two crossing points, so neither side ever locks
    around scheduler state.
  * Everything else (handles, sent-counts, the scheduler itself) is touched by
    the engine thread only.

Token streaming needs no scheduler hook: after each step the engine *diffs* every
live sequence's `output_ids` against what it already published and emits the tail.
Diffing (rather than an emit callback) is what makes the engine scheduler-agnostic
— a spec step emits 0..k+1 tokens per sequence and the diff picks that up for free
— and it leaves the parity-locked schedulers untouched.

Correctness bar (tests/python/run_async_serve.py): the async path must not change
a token — every streamed request equals the same request run standalone, and the
chunks concatenate to exactly the final output. Same invariant as run_serve.py,
extended through the asyncio seam.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class TokenChunk:
    """New confirmed tokens for one request — one step's emission (1 for plain
    decode, 0..k+1 for speculative)."""

    ids: list[int]


@dataclass
class RequestTiming:
    """Per-request wall-clock marks (engine-thread perf_counter timestamps).

    ttft — time to first token: submit -> the step that produced the first token
    finished. This is the user-facing number, so it INCLUDES queue wait and
    prefill (a request admitted instantly on an idle engine measures ~prefill;
    one stuck behind a full batch measures the queueing policy).
    tpot — time per output token, steady-state: (finish - first token) spread
    over the n-1 inter-token gaps. Needs n >= 2.
    """

    submitted: float
    first_token: float | None = None
    finished: float | None = None
    n_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token is None:
            return None
        return (self.first_token - self.submitted) * 1e3

    @property
    def tpot_ms(self) -> float | None:
        if self.finished is None or self.first_token is None or self.n_tokens < 2:
            return None
        return (self.finished - self.first_token) / (self.n_tokens - 1) * 1e3

    @property
    def total_ms(self) -> float | None:
        if self.finished is None:
            return None
        return (self.finished - self.submitted) * 1e3


@dataclass
class Done:
    """Terminal event: the request's full output and why it stopped.
    finish_reason: "stop" (eos), "length" (max_tokens), or "cancelled"."""

    finish_reason: str
    output_ids: list[int]
    timing: RequestTiming


class RequestHandle:
    """The caller's end of one request: an async iterator of TokenChunk events
    ending with exactly one Done (also kept as `.done`). Consume with
    `async for`, or `await collect()` for the non-streaming case."""

    def __init__(self, engine: "AsyncEngine", req):
        self._engine = engine
        self.req = req
        self.request_id = req.request_id
        self.done: Done | None = None
        self._events: asyncio.Queue = asyncio.Queue()
        self._exhausted = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._exhausted:
            raise StopAsyncIteration
        ev = await self._events.get()
        if isinstance(ev, Done):
            self.done = ev
            self._exhausted = True
        return ev

    async def collect(self) -> Done:
        """Drain the stream and return the terminal Done (the non-stream path)."""
        async for _ in self:
            pass
        return self.done

    def cancel(self) -> None:
        """Ask the engine to abort this request (e.g. the client disconnected).
        The stream still terminates normally, with finish_reason "cancelled" and
        whatever partial output existed; the sequence's KV/blocks are freed."""
        self._engine.cancel(self.request_id)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile over an already-sorted list (q in [0, 1])."""
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


class AsyncEngine:
    """Drive a (Spec)Scheduler from asyncio: submit() -> RequestHandle streams.

    Usage:
        engine = AsyncEngine(Scheduler(model, ...))
        await engine.start()
        handle = engine.submit(Request("r0", ids, max_tokens=32))
        async for ev in handle: ...
        await engine.stop()
    """

    def __init__(self, scheduler, idle_wait_s: float = 0.02, history: int = 1024):
        self.scheduler = scheduler
        # How long the idle engine blocks on the inbox per wait; only affects
        # shutdown latency and idle wake-ups, never a busy engine (which drains
        # the inbox non-blocking between steps).
        self._idle_wait_s = idle_wait_s
        self._inbox: queue.Queue = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Engine-thread-only state.
        self._handles: dict[str, RequestHandle] = {}
        self._sent: dict[str, int] = {}
        self._timings: dict[str, RequestTiming] = {}
        self._cancelled: set[str] = set()
        # Metrics, shared with metrics() readers under a lock.
        self._mlock = threading.Lock()
        self._completed: deque = deque(maxlen=history)
        self._n_finished = 0
        self._n_cancelled = 0
        self._tokens_out = 0
        self._started_at = time.perf_counter()

    # -- event-loop side ----------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="ni-engine", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        """Stop the engine thread. Outstanding requests are terminated with
        finish_reason "cancelled" so no consumer is left awaiting forever."""
        if self._thread is None:
            return
        self._inbox.put(("stop",))
        await asyncio.to_thread(self._thread.join)
        self._thread = None

    def submit(self, req) -> RequestHandle:
        """Enqueue a Request/SpecRequest; returns immediately with its handle.
        Call from the event-loop thread (the handle's queue belongs to it)."""
        if self._thread is None:
            raise RuntimeError("engine not started (await engine.start() first)")
        handle = RequestHandle(self, req)
        self._inbox.put(("add", req, handle))
        return handle

    def cancel(self, request_id: str) -> None:
        self._inbox.put(("cancel", request_id))

    def metrics(self) -> dict:
        """A point-in-time snapshot: request counts, throughput, and TTFT/TPOT
        percentiles over the retained completion history. Reads scheduler
        counters cross-thread — len()/int reads, safe under the GIL, and they
        are gauges anyway (a step racing the read moves them by one)."""
        s = self.scheduler
        with self._mlock:
            done = list(self._completed)
            n_fin, n_can, toks = self._n_finished, self._n_cancelled, self._tokens_out
        uptime = time.perf_counter() - self._started_at
        out = {
            "uptime_s": round(uptime, 3),
            "scheduler": type(s).__name__,
            "requests": {
                "finished": n_fin,
                "cancelled": n_can,
                "running": len(s.running),
                "queued": len(s.waiting),
            },
            "throughput": {
                "tokens_generated": toks,
                "tokens_per_s": round(toks / uptime, 3) if uptime > 0 else 0.0,
            },
            "batch": {"steps": s.steps, "peak": s.peak_batch, "current": len(s.running)},
        }
        for name in ("ttft_ms", "tpot_ms"):
            vals = sorted(v[name] for v in done if v[name] is not None)
            out[name] = (
                {
                    "count": len(vals),
                    "mean": round(sum(vals) / len(vals), 3),
                    "p50": round(_percentile(vals, 0.50), 3),
                    "p95": round(_percentile(vals, 0.95), 3),
                }
                if vals
                else {"count": 0}
            )
        return out

    # -- engine-thread side --------------------------------------------------

    def _run(self) -> None:
        sched = self.scheduler
        while True:
            # Drain the inbox. A busy engine polls it non-blocking between
            # steps; an idle one blocks (briefly) so it doesn't spin.
            block = not sched.has_work()
            while True:
                try:
                    item = (self._inbox.get(timeout=self._idle_wait_s)
                            if block else self._inbox.get_nowait())
                except queue.Empty:
                    break
                block = False
                if item[0] == "stop":
                    self._shutdown()
                    return
                if item[0] == "add":
                    _, req, handle = item
                    rid = req.request_id
                    self._handles[rid] = handle
                    self._sent[rid] = 0
                    self._timings[rid] = RequestTiming(submitted=time.perf_counter())
                    sched.add(req)
                elif item[0] == "cancel":
                    rid = item[1]
                    if rid in self._handles and rid not in self._cancelled:
                        # The scheduler evicts it (freeing KV/blocks) and records
                        # the partial output in `finished`; _publish turns that
                        # into the Done("cancelled") event.
                        if sched.cancel(rid):
                            self._cancelled.add(rid)
            if not sched.has_work():
                # A cancel (or an empty-prompt add) can finish a request without
                # a step — publish so its Done isn't stranded until the next one.
                self._publish()
                continue
            sched.step()
            self._publish()

    def _publish(self) -> None:
        """Diff every live request against the scheduler and emit what's new."""
        sched = self.scheduler
        now = time.perf_counter()
        running = {s.req.request_id: s for s in sched.running}
        finished_rids = []
        for rid, handle in self._handles.items():
            sent = self._sent[rid]
            timing = self._timings[rid]
            if rid in sched.finished:
                out = sched.finished[rid]
                if len(out) > sent:
                    if sent == 0:
                        timing.first_token = now
                    self._emit(handle, TokenChunk(list(out[sent:])))
                timing.finished = now
                timing.n_tokens = len(out)
                if rid in self._cancelled:
                    reason = "cancelled"
                elif len(out) >= handle.req.max_tokens and handle.req.max_tokens > 0:
                    reason = "length"
                else:
                    reason = "stop"  # eos (or an empty prompt)
                self._emit(handle, Done(reason, list(out), timing))
                finished_rids.append(rid)
            elif rid in running:
                out = running[rid].output_ids
                if len(out) > sent:
                    if sent == 0:
                        timing.first_token = now
                    self._emit(handle, TokenChunk(list(out[sent:])))
                    self._sent[rid] = len(out)
        for rid in finished_rids:
            timing = self._timings.pop(rid)
            reason = "cancelled" if rid in self._cancelled else None
            self._handles.pop(rid)
            self._sent.pop(rid)
            self._cancelled.discard(rid)
            # The scheduler keeps every finished output forever (fine for a
            # gate's run(); a leak for a long-lived server) — the engine owns
            # the delivery, so it reclaims the entry once delivered.
            out = sched.finished.pop(rid)
            with self._mlock:
                self._completed.append({"ttft_ms": timing.ttft_ms, "tpot_ms": timing.tpot_ms})
                self._n_finished += 1
                self._n_cancelled += reason == "cancelled"
                self._tokens_out += len(out)

    def _emit(self, handle: RequestHandle, ev) -> None:
        self._loop.call_soon_threadsafe(handle._events.put_nowait, ev)

    def _shutdown(self) -> None:
        """Terminate every outstanding request so no consumer hangs; cancel the
        running ones through the scheduler so KV/blocks are freed."""
        for rid, handle in list(self._handles.items()):
            self.scheduler.cancel(rid)
            out = self.scheduler.finished.pop(rid, [])
            timing = self._timings.pop(rid)
            timing.finished = time.perf_counter()
            timing.n_tokens = len(out)
            self._emit(handle, Done("cancelled", list(out), timing))
        self._handles.clear()
        self._sent.clear()
        self._cancelled.clear()
