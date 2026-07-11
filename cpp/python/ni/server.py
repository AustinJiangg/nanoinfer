"""V1 — the HTTP serving layer: hand-rolled asyncio HTTP/1.1 + SSE over AsyncEngine.

Zero new dependencies, deliberately: the golden-rule analog for serving is that
the plumbing — request parsing, Server-Sent-Events streaming, incremental
detokenization, disconnect→cancel, TTFT/TPOT metrics — IS the thing we're here
to learn, so we don't reach for FastAPI/uvicorn to do it for us. The protocol
subset is small and honest: HTTP/1.1, every response `Connection: close` (no
keep-alive state machine), chunked framing avoided the same way.

Routes:
  GET  /healthz          liveness + what's serving
  GET  /metrics          the engine's snapshot: TTFT/TPOT percentiles, throughput,
                         queue/batch gauges (json)
  POST /v1/completions   one generation request:
    {"prompt": str | "prompt_ids": [int],       one of the two; text needs a tokenizer
     "max_tokens": 32, "temperature": 0.0, "top_k": 0, "top_p": 1.0,
     "repetition_penalty": 1.0, "seed": 0,
     "stream": false,                           true -> SSE token stream
     "ignore_eos": false,                       fixed-length output (benchmarking)
     "id": "...",                               optional; must be unique if given
     -- speculative engine only --
     "proposer": "lookup"|"draft", "k": 4, "ngram": 3}

Streaming responses are SSE: one `data: {json}` event per token chunk (with the
incrementally-detokenized text delta when a tokenizer is loaded), a final event
with `done: true` + finish_reason/usage/timing, then `data: [DONE]`. A client
that disconnects mid-stream cancels its request — the scheduler evicts the
sequence and its KV blocks go back to the pool (the V0 cancel path).

The server is engine-agnostic the same way the engine is scheduler-agnostic:
constructed over a plain-Scheduler engine it serves plain requests; over a
SpecScheduler engine the same endpoint takes proposer/k/ngram knobs. One HTTP
layer, any decode strategy underneath.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from .async_engine import AsyncEngine, TokenChunk
from .detok import IncrementalDetokenizer
from .engine import nicpp
from .scheduler import Request
from .spec_scheduler import SpecRequest, SpecScheduler

_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found",
            405: "Method Not Allowed", 500: "Internal Server Error"}
# An id no real vocab reaches: how a plain Request disables the eos stop, since
# its eos_id=-1 convention means "the model's default", not "none" (SpecRequest
# is the opposite: -1 already means "no eos stop").
_NO_EOS = 2**31 - 1


def _timing_json(t) -> dict:
    r = lambda v: None if v is None else round(v, 2)  # noqa: E731
    return {"ttft_ms": r(t.ttft_ms), "tpot_ms": r(t.tpot_ms), "total_ms": r(t.total_ms)}


class HttpServer:
    """Serve an AsyncEngine over HTTP. `tokenizer` (an HF tokenizer, or anything
    with __call__/decode) is optional: without it the API is ids-in/ids-out —
    which is also what keeps the gate hermetic."""

    def __init__(self, engine: AsyncEngine, tokenizer=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self._spec = isinstance(engine.scheduler, SpecScheduler)
        self._eos = engine.scheduler.model.config.eos_token_id
        self._server: asyncio.Server | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self, host: str = "127.0.0.1", port: int = 8000) -> int:
        """Bind and start accepting. Returns the actual port (pass 0 to let the
        OS pick — the gate does, so parallel runs can't collide)."""
        self._server = await asyncio.start_server(self._handle, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def serve_forever(self) -> None:
        await self._server.serve_forever()

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    # -- request building ------------------------------------------------------

    def _build_request(self, body: dict):
        """JSON body -> (Request | SpecRequest, n_prompt). Raises ValueError with
        a client-facing message on anything malformed."""
        if "prompt_ids" in body:
            ids = body["prompt_ids"]
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ValueError("prompt_ids must be a list of ints")
        elif "prompt" in body:
            if self.tokenizer is None:
                raise ValueError("server has no tokenizer; send prompt_ids instead")
            ids = self.tokenizer(str(body["prompt"]), return_tensors=None)["input_ids"]
        else:
            raise ValueError("need prompt or prompt_ids")

        rid = str(body.get("id") or f"req-{uuid.uuid4().hex[:12]}")
        max_tokens = int(body.get("max_tokens", 32))
        temperature = float(body.get("temperature", 0.0))
        top_k = int(body.get("top_k", 0))
        top_p = float(body.get("top_p", 1.0))
        rep = float(body.get("repetition_penalty", 1.0))
        seed = int(body.get("seed", 0))
        ignore_eos = bool(body.get("ignore_eos", False))

        if self._spec:
            proposer = body.get("proposer") or (
                "draft" if self.engine.scheduler.draft is not None else "lookup")
            # Greedy rides the argmax accept rule; temperature>0 switches the
            # request to speculative SAMPLING (S5, rejection accept).
            params = None if temperature == 0.0 else nicpp.SamplingParams(
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=rep)
            req = SpecRequest(
                rid, ids, max_tokens=max_tokens,
                eos_id=-1 if ignore_eos else self._eos,
                proposer=proposer, k=int(body.get("k", 4)),
                ngram=int(body.get("ngram", 3)), params=params, seed=seed)
        else:
            req = Request(
                rid, ids, max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=rep, seed=seed,
                eos_id=_NO_EOS if ignore_eos else -1)
        return req, len(ids)

    # -- http plumbing ---------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """One connection = one request = one response (Connection: close)."""
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            parts = line.decode("latin-1").split()
            if len(parts) < 2:
                return
            method, path = parts[0], parts[1]
            headers = {}
            while True:
                h = await asyncio.wait_for(reader.readline(), timeout=30)
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            body = b""
            n = int(headers.get("content-length", "0") or 0)
            if n:
                body = await reader.readexactly(n)
            await self._route(method, path, body, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError,
                TimeoutError):
            pass  # client went away mid-parse; nothing to answer
        except Exception as e:  # a broken request must never take the server down
            try:
                self._write_json(writer, 500, {"error": f"{type(e).__name__}: {e}"})
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _route(self, method, path, body, writer) -> None:
        if path == "/healthz" and method == "GET":
            self._write_json(writer, 200, {
                "status": "ok",
                "scheduler": type(self.engine.scheduler).__name__,
                "tokenizer": self.tokenizer is not None})
        elif path == "/metrics" and method == "GET":
            self._write_json(writer, 200, self.engine.metrics())
        elif path == "/v1/completions" and method == "POST":
            await self._completions(body, writer)
        else:
            self._write_json(writer, 404, {"error": f"no route {method} {path}"})
        if not writer.is_closing():  # a disconnected stream already gave up the socket
            await writer.drain()

    def _write_json(self, writer, status: int, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        writer.write(
            f"HTTP/1.1 {status} {_REASONS[status]}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n".encode() + payload)

    # -- the endpoint ----------------------------------------------------------

    async def _completions(self, body: bytes, writer) -> None:
        try:
            parsed = json.loads(body or b"{}")
            if not isinstance(parsed, dict):
                raise ValueError("body must be a json object")
        except (json.JSONDecodeError, ValueError) as e:
            self._write_json(writer, 400, {"error": f"invalid json: {e}"})
            return
        try:
            req, n_prompt = self._build_request(parsed)
        except (ValueError, TypeError) as e:
            self._write_json(writer, 400, {"error": str(e)})
            return

        handle = self.engine.submit(req)
        if parsed.get("stream", False):
            await self._stream(handle, n_prompt, writer)
        else:
            done = await handle.collect()
            out = [int(t) for t in done.output_ids]
            self._write_json(writer, 200, {
                "id": handle.request_id,
                "object": "text_completion",
                "token_ids": out,
                "text": (self.tokenizer.decode(out, skip_special_tokens=True)
                         if self.tokenizer and out else None),
                "finish_reason": done.finish_reason,
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out)},
                "timing": _timing_json(done.timing)})

    async def _stream(self, handle, n_prompt: int, writer) -> None:
        """SSE: one event per token chunk, a final done event, then [DONE].
        A write failing (or the transport closing) means the client is gone:
        cancel the request so its slot and KV free immediately."""
        writer.write(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/event-stream\r\n"
                     b"Cache-Control: no-cache\r\n"
                     b"Connection: close\r\n\r\n")
        detok = (IncrementalDetokenizer(self.tokenizer)
                 if self.tokenizer is not None else None)
        try:
            async for ev in handle:
                if writer.is_closing():
                    raise ConnectionResetError
                if isinstance(ev, TokenChunk):
                    ids = [int(t) for t in ev.ids]
                    payload = {"id": handle.request_id, "token_ids": ids,
                               "text": detok.push(ids) if detok else None,
                               "done": False}
                else:  # Done
                    payload = {"id": handle.request_id, "done": True,
                               "finish_reason": ev.finish_reason,
                               "text": detok.flush() if detok else None,
                               "usage": {"prompt_tokens": n_prompt,
                                         "completion_tokens": len(ev.output_ids)},
                               "timing": _timing_json(ev.timing)}
                writer.write(f"data: {json.dumps(payload)}\n\n".encode())
                await writer.drain()
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError,
                TimeoutError, OSError):
            handle.cancel()
