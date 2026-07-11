"""V1 gate — the HTTP layer must not change a token, end to end.

run_async_serve.py proved the AsyncEngine seam; this gate stacks the real HTTP
server on top and drives it with a real client (httpx) over loopback: JSON in,
tokens out, byte-for-byte through request parsing, SSE framing, and json
round-trips. Hermetic like run_serve.py — no tokenizer is loaded (prompt_ids
in, token_ids out), so it needs only the NIT0 weights export.

Checks: /healthz; a non-streamed completion == standalone; concurrent mixed
stream/non-stream requests all == standalone (HTTP concurrency -> one batched
engine); SSE chunks concatenate exactly with a sane final event; malformed
requests get 4xx (never a hang or a crash); a client disconnect mid-stream
cancels the request (the scheduler evicts it — watched via /metrics) and the
server keeps serving correctly afterwards; and the IncrementalDetokenizer
holds back split UTF-8 sequences (checked with a byte-level stub tokenizer —
no HF download).

    cmake --build build -j                 # build nicpp (F6)
    python tests/python/run_http_serve.py [weights_dir] [--device cuda]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.async_engine import AsyncEngine  # noqa: E402
from ni.detok import IncrementalDetokenizer  # noqa: E402
from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.scheduler import Scheduler  # noqa: E402
from ni.server import HttpServer  # noqa: E402

from gateutil import build_requests, standalone  # noqa: E402


async def sse_events(client, url: str, body: dict) -> list[dict]:
    """POST a streaming completion; return the decoded SSE events (sans [DONE])."""
    events = []
    async with client.stream("POST", url + "/v1/completions", json=body) as r:
        assert r.status_code == 200, r.status_code
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            events.append(json.loads(line[6:]))
    return events


async def check_http(model, base) -> bool:
    reqs = build_requests(base)
    ref = {r.request_id: standalone(model, r) for r in reqs}

    engine = AsyncEngine(Scheduler(model, max_batch=4, batched=True,
                                   block_size=16, num_blocks=256))
    await engine.start()
    server = HttpServer(engine, tokenizer=None)
    port = await server.start("127.0.0.1", 0)
    url = f"http://127.0.0.1:{port}"
    ok = True

    # trust_env=False: this box's no_proxy contains IPv6 entries httpx can't
    # parse (the known hf-download sharp edge) — and loopback needs no proxy.
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        # -- liveness ------------------------------------------------------
        r = await client.get(url + "/healthz")
        ok &= r.status_code == 200 and r.json()["status"] == "ok"

        # -- one non-streamed completion ------------------------------------
        r0 = reqs[0]
        r = await client.post(url + "/v1/completions", json={
            "id": r0.request_id, "prompt_ids": r0.prompt_ids,
            "max_tokens": r0.max_tokens})
        j = r.json()
        single = (r.status_code == 200 and j["token_ids"] == ref["r0"]
                  and j["finish_reason"] == "length"
                  and j["usage"] == {"prompt_tokens": len(r0.prompt_ids),
                                     "completion_tokens": len(ref["r0"])}
                  and j["timing"]["ttft_ms"] > 0)
        ok &= single
        print(f"single non-stream: {'MATCH' if single else 'MISMATCH ' + str(j)}")

        # -- concurrent mixed stream/non-stream over one engine ---------------
        async def one(i, req):
            body = {"id": f"c-{req.request_id}", "prompt_ids": req.prompt_ids,
                    "max_tokens": req.max_tokens,
                    "repetition_penalty": req.repetition_penalty,
                    "seed": req.seed}
            if i % 2 == 0:  # streamed
                evs = await sse_events(client, url, {**body, "stream": True})
                ids = [t for e in evs if not e["done"] for t in e["token_ids"]]
                fin = evs[-1]
                sane = (fin["done"] and fin["finish_reason"] in ("stop", "length")
                        and fin["usage"]["completion_tokens"] == len(ids)
                        and fin["timing"]["ttft_ms"] > 0
                        and len(evs) >= 3)  # genuinely incremental, not one blob
                return ids, sane
            r = await client.post(url + "/v1/completions", json=body)
            return r.json()["token_ids"], r.status_code == 200

        results = await asyncio.gather(*(one(i, rq) for i, rq in enumerate(reqs)))
        conc = all(sane and ids == ref[rq.request_id]
                   for rq, (ids, sane) in zip(reqs, results))
        ok &= conc
        m = engine.metrics()
        print(f"concurrent x{len(reqs)} (mixed SSE/plain): peak_batch="
              f"{m['batch']['peak']} ttft_p50={m['ttft_ms'].get('p50')}ms "
              f"tpot_p50={m['tpot_ms'].get('p50')}ms "
              f"-> {'MATCH' if conc else 'MISMATCH'}")

        # -- malformed requests: 4xx, never a crash or a hang ------------------
        bad = [
            (await client.post(url + "/v1/completions", content=b"{nope")).status_code,
            (await client.post(url + "/v1/completions", json={})).status_code,
            (await client.post(url + "/v1/completions",
                               json={"prompt": "hi"})).status_code,  # no tokenizer
            (await client.post(url + "/v1/completions",
                               json={"prompt_ids": ["x"]})).status_code,
            (await client.get(url + "/nope")).status_code,
        ]
        errs = bad == [400, 400, 400, 400, 404]
        ok &= errs
        print(f"malformed -> {bad} {'ok' if errs else 'FAIL'}")

        # -- disconnect mid-stream cancels the request -------------------------
        got = 0
        async with client.stream("POST", url + "/v1/completions", json={
                "id": "disc0", "prompt_ids": base, "max_tokens": 512,
                "ignore_eos": True, "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    got += 1
                    if got >= 3:
                        break  # leave the context manager: the connection closes
        for _ in range(150):  # the server notices on its next write
            m = (await client.get(url + "/metrics")).json()
            if (m["requests"]["cancelled"] >= 1 and m["requests"]["running"] == 0
                    and m["requests"]["queued"] == 0):
                break
            await asyncio.sleep(0.2)
        disc = m["requests"]["cancelled"] == 1
        pool = engine.scheduler.pool
        freed = pool.free_blocks == pool.num_blocks
        # ...and the server is still healthy and correct afterwards.
        r = await client.post(url + "/v1/completions", json={
            "prompt_ids": reqs[0].prompt_ids, "max_tokens": reqs[0].max_tokens})
        after = r.json()["token_ids"] == ref["r0"]
        ok &= disc and freed and after
        print(f"disconnect: cancelled={m['requests']['cancelled']} "
              f"pool_free={pool.free_blocks}/{pool.num_blocks} "
              f"serves_after={'MATCH' if after else 'MISMATCH'} "
              f"-> {'ok' if disc and freed and after else 'FAIL'}")

        # -- metrics accounting -------------------------------------------------
        m = (await client.get(url + "/metrics")).json()
        acct = (m["requests"]["finished"] == len(reqs) + 3  # +single, +disc, +after
                and m["ttft_ms"]["count"] >= len(reqs)
                and m["throughput"]["tokens_generated"]
                    >= sum(len(v) for v in ref.values()))
        ok &= acct
        print(f"metrics: finished={m['requests']['finished']} "
              f"tokens={m['throughput']['tokens_generated']} "
              f"({m['throughput']['tokens_per_s']} tok/s aggregate) "
              f"-> {'ok' if acct else 'FAIL'}")

    await server.stop()
    await engine.stop()
    return ok


def check_detok() -> bool:
    """IncrementalDetokenizer vs a byte-level stub (each id = one raw UTF-8 byte —
    the worst case: EVERY multi-byte char is split across 'tokens'). Deltas must
    never leak a held-back replacement char, and must concatenate exactly."""

    class ByteTok:
        def decode(self, ids, **kw):
            return bytes(ids).decode("utf-8", errors="replace")

    s = "héllo, wörld — 🚀→中文!"
    d = IncrementalDetokenizer(ByteTok())
    deltas = [d.push([b]) for b in s.encode()]
    out = "".join(deltas) + d.flush()
    ok = out == s and all("�" not in x for x in deltas)

    # A genuinely malformed tail (a dangling UTF-8 lead byte) is held while the
    # stream runs, but flush() must still deliver it — visibly, not dropped.
    d2 = IncrementalDetokenizer(ByteTok())
    held = d2.push(list("caf".encode()) + [0xC3])
    ok &= held == "" and d2.flush() == "caf�"
    print(f"detok (byte-split utf-8): {len(deltas)} pushes, "
          f"held={sum(x == '' for x in deltas)} -> {'ok' if ok else 'FAIL'}")
    return ok


async def main_async() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("weights_dir", nargs="?", default=str(default_weights_dir()))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    model = nicpp.Model(args.weights_dir, device=args.device)
    base = read_ids(Path(args.weights_dir) / "ref_ids.txt")

    ok = check_detok()
    ok &= await check_http(model, base)

    print("\nrun_http_serve:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
