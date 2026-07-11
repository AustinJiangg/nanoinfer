"""Serving load bench — the throughput/latency curve through the real HTTP stack.

Closed-loop load: C concurrent clients, each posting a completion and waiting
for it before posting the next, until N requests drain. Sweeping C traces the
curve every serving paper plots: aggregate tok/s rises with concurrency (the
continuous batch fills — decode GEMMs fuse over more rows) while per-request
latency (TTFT: queue+prefill; TPOT: per-token pace) degrades as sequences share
the step. Where tok/s flattens but TPOT keeps climbing is the knee — added
concurrency is pure queueing past it.

Measures, doesn't gate (bench/ convention). Hermetic: ids in, ids out, no
tokenizer; ignore_eos pins every request to exactly --max-tokens tokens so
levels are comparable. TTFT/TPOT come from the server's own engine timing (the
/v1/completions response), e2e from the client's clock.

    python bench/bench_http.py [weights_dir] --device cuda \
        --concurrency 1,4,8,16 --requests 32 --max-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from ni.async_engine import AsyncEngine  # noqa: E402
from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.scheduler import Scheduler  # noqa: E402
from ni.server import HttpServer  # noqa: E402


def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


async def run_level(url: str, base: list[int], conc: int, n_req: int,
                    max_tokens: int) -> dict:
    jobs: asyncio.Queue = asyncio.Queue()
    for i in range(n_req):
        # Distinct tails so no two in-flight requests are identical; same length
        # so the work per request is constant.
        jobs.put_nowait({"prompt_ids": base + [11 + i], "max_tokens": max_tokens,
                         "ignore_eos": True})
    ttft, tpot, e2e, toks = [], [], [], 0

    async def worker(client):
        nonlocal toks
        while True:
            try:
                body = jobs.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            r = await client.post(url + "/v1/completions", json=body)
            e2e.append((time.perf_counter() - t0) * 1e3)
            j = r.json()
            toks += j["usage"]["completion_tokens"]
            ttft.append(j["timing"]["ttft_ms"])
            if j["timing"]["tpot_ms"] is not None:
                tpot.append(j["timing"]["tpot_ms"])

    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(conc)))
        wall = time.perf_counter() - t0
    return {"conc": conc, "reqs": n_req, "wall": wall, "tok_s": toks / wall,
            "ttft50": pct(ttft, 0.5), "ttft95": pct(ttft, 0.95),
            "tpot50": pct(tpot, 0.5), "tpot95": pct(tpot, 0.95),
            "e2e50": pct(e2e, 0.5)}


async def main_async() -> None:
    p = argparse.ArgumentParser(description="nanoinfer-cpp HTTP serving load bench")
    p.add_argument("weights_dir", nargs="?", default=str(default_weights_dir()))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--concurrency", default="1,4,8,16")
    p.add_argument("--requests", type=int, default=32, help="requests per level")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=512)
    args = p.parse_args()

    model = nicpp.Model(args.weights_dir, device=args.device)
    base = read_ids(Path(args.weights_dir) / "ref_ids.txt")
    engine = AsyncEngine(Scheduler(model, max_batch=args.max_batch, batched=True,
                                   block_size=args.block_size,
                                   num_blocks=args.num_blocks))
    await engine.start()
    server = HttpServer(engine, tokenizer=None)
    port = await server.start("127.0.0.1", 0)
    url = f"http://127.0.0.1:{port}"

    levels = [int(c) for c in args.concurrency.split(",")]
    print(f"bench_http: {args.device}, max_batch={args.max_batch}, "
          f"paged(bs={args.block_size}, {args.num_blocks} blocks), "
          f"{args.requests} reqs/level x {args.max_tokens} tok (ignore_eos), "
          f"prompt_len={len(base) + 1}\n")
    print("conc   tok/s   wall_s   ttft p50/p95 ms    tpot p50/p95 ms   e2e p50 ms")
    # Warm-up (first-touch allocations, page-in) — not counted.
    await run_level(url, base, 1, 2, args.max_tokens)
    for c in levels:
        r = await run_level(url, base, c, args.requests, args.max_tokens)
        print(f"{r['conc']:4d} {r['tok_s']:7.1f} {r['wall']:8.2f}  "
              f"{r['ttft50']:8.1f}/{r['ttft95']:6.1f}   "
              f"{r['tpot50']:8.2f}/{r['tpot95']:6.2f}  {r['e2e50']:9.1f}")

    await server.stop()
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main_async())
