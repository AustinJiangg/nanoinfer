"""V0 gate — the AsyncEngine (the asyncio serving bridge) must not change a token.

run_serve.py proved the schedulers are interleaving-invariant; this gate extends the
same bar through the async seam: every request streamed through AsyncEngine equals
the same request run standalone (nicpp.generate), the streamed chunks concatenate to
exactly the final output, and the serving-only machinery — dynamic arrival, client
cancellation, engine shutdown — never corrupts a neighboring sequence or leaks KV
blocks. The engine is scheduler-agnostic, so the same checks run over the plain
Scheduler AND the SpecScheduler (prompt-lookup proposer: hermetic, no draft model).

    cmake --build build -j                 # build nicpp (F6)
    python tests/python/run_async_serve.py [weights_dir] [--device cuda]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.async_engine import AsyncEngine, TokenChunk  # noqa: E402
from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.scheduler import Request, Scheduler  # noqa: E402
from ni.spec_scheduler import SpecRequest, SpecScheduler  # noqa: E402

from gateutil import build_requests, plain_greedy, standalone  # noqa: E402


async def consume(handle):
    """Stream a handle to completion -> (streamed ids, #chunks, Done)."""
    ids, chunks = [], 0
    async for ev in handle:
        if isinstance(ev, TokenChunk):
            ids.extend(ev.ids)
            chunks += 1
    return ids, chunks, handle.done


async def check_identity(model, reqs, ref, make_sched, name, strict_chunks) -> bool:
    """Submit `reqs` concurrently (half of them arriving late), half consumed as
    streams and half collected, and require every output token-identical to the
    standalone reference. strict_chunks: on the plain scheduler a step emits
    exactly one token per running sequence, so #chunks must equal #tokens — the
    stream really is incremental, not one buffered blob at the end."""
    engine = AsyncEngine(make_sched())
    await engine.start()
    early, late = reqs[: len(reqs) // 2], reqs[len(reqs) // 2:]
    handles = [engine.submit(r) for r in early]
    await asyncio.sleep(0.2)  # dynamic arrival: the engine is mid-flight already
    handles += [engine.submit(r) for r in late]

    ok = True
    for i, (r, h) in enumerate(zip(early + late, handles)):
        if i % 2 == 0:
            ids, chunks, done = await consume(h)
            if ids != done.output_ids:
                ok = False
                print(f"  {r.request_id}: chunks do not concatenate to the output")
            if strict_chunks and chunks != len(ids):
                ok = False
                print(f"  {r.request_id}: {chunks} chunks for {len(ids)} tokens")
        else:
            done = await h.collect()
        if done.output_ids != ref[r.request_id]:
            ok = False
            print(f"  {r.request_id}: MISMATCH {done.output_ids}")
        if done.finish_reason not in ("stop", "length"):
            ok = False
            print(f"  {r.request_id}: unexpected finish_reason {done.finish_reason}")
        t = done.timing
        if t.ttft_ms is None or t.ttft_ms <= 0 or (t.total_ms or 0) < t.ttft_ms:
            ok = False
            print(f"  {r.request_id}: bad timing ttft={t.ttft_ms} total={t.total_ms}")
    m = engine.metrics()
    await engine.stop()
    sched = engine.scheduler
    print(f"{name}: steps={sched.steps} peak_batch={sched.peak_batch} "
          f"ttft_p50={m['ttft_ms'].get('p50')}ms tpot_p50={m['tpot_ms'].get('p50')}ms "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    return ok


async def check_cancel(model, base) -> bool:
    """Cancel a running request mid-stream: it must terminate with reason
    "cancelled" and a PREFIX of its standalone output (greedy is deterministic,
    so the partial tokens are still the right ones), its KV blocks must return
    to the pool, and the sequence sharing the batch must be untouched."""
    long_req = Request("c0", base, max_tokens=64)
    mate_req = Request("c1", base[:3], max_tokens=12)
    ref_long = standalone(model, long_req)
    ref_mate = standalone(model, mate_req)

    sched = Scheduler(model, max_batch=4, batched=True, block_size=4, num_blocks=64)
    engine = AsyncEngine(sched)
    await engine.start()
    h_long, h_mate = engine.submit(long_req), engine.submit(mate_req)

    got: list[int] = []
    async for ev in h_long:
        if isinstance(ev, TokenChunk):
            got.extend(ev.ids)
            if len(got) >= 4:
                h_long.cancel()  # keep consuming; the stream must still terminate
    done = h_long.done
    mate_done = await h_mate.collect()
    # Cancelling twice / cancelling a finished request must be a harmless no-op.
    h_long.cancel()
    await asyncio.sleep(0.1)
    m = engine.metrics()
    await engine.stop()

    cancelled = done.finish_reason == "cancelled"
    partial = 4 <= len(done.output_ids) < 64
    prefix = done.output_ids == ref_long[: len(done.output_ids)]
    mate_ok = mate_done.output_ids == ref_mate
    freed = sched.pool.free_blocks == sched.pool.num_blocks
    counted = m["requests"]["cancelled"] == 1 and m["requests"]["finished"] == 2
    ok = cancelled and partial and prefix and mate_ok and freed and counted
    print(f"cancel: reason={done.finish_reason} got {len(done.output_ids)}/64 tok "
          f"prefix_of_ref={prefix} mate={'MATCH' if mate_ok else 'MISMATCH'} "
          f"pool_free={sched.pool.free_blocks}/{sched.pool.num_blocks} "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


async def check_spec(model, base) -> bool:
    """Scheduler-agnosticism: the SAME engine drives the SpecScheduler (prompt-
    lookup proposer — no draft model, hermetic). A spec step emits 0..k+1 tokens,
    which the engine's output_ids diff must deliver unchanged: every output is
    token-identical to plain greedy (the S0 invariant, through the async seam)."""
    reqs = [
        SpecRequest("s0", base, max_tokens=16, proposer="lookup", k=4, ngram=3),
        SpecRequest("s1", base[:3], max_tokens=10, proposer="lookup", k=4, ngram=2),
        SpecRequest("s2", base + [12095], max_tokens=12, proposer="lookup", k=3, ngram=3),
    ]
    ref = {r.request_id: plain_greedy(model, r.prompt_ids, r.max_tokens) for r in reqs}
    engine = AsyncEngine(SpecScheduler(model, max_batch=2, batched=True))
    await engine.start()
    handles = [engine.submit(r) for r in reqs]
    results = [await consume(h) for h in handles]
    await engine.stop()

    ok = True
    for r, (ids, chunks, done) in zip(reqs, results):
        if done.output_ids != ref[r.request_id] or ids != done.output_ids:
            ok = False
            print(f"  {r.request_id}: MISMATCH {done.output_ids}")
    print(f"spec (lookup): {len(reqs)} reqs, chunks per req "
          f"{[c for _, c, _ in results]} (0..k+1 tok each) "
          f"-> {'MATCH' if ok else 'MISMATCH'}")
    return ok


async def check_shutdown(model, base) -> bool:
    """stop() with a request in flight must terminate its stream (reason
    "cancelled"), not leave the consumer awaiting forever."""
    engine = AsyncEngine(Scheduler(model, max_batch=2, batched=True))
    await engine.start()
    h = engine.submit(Request("z0", base, max_tokens=128))
    async for ev in h:  # wait for the first tokens, then pull the plug
        if isinstance(ev, TokenChunk):
            break
    await engine.stop()
    done = await h.collect()
    ok = done.finish_reason == "cancelled" and len(done.output_ids) < 128
    print(f"shutdown: reason={done.finish_reason} got {len(done.output_ids)} tok "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


async def main_async() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("weights_dir", nargs="?", default=str(default_weights_dir()))
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    wd = Path(args.weights_dir)
    model = nicpp.Model(str(wd), device=args.device)
    base = read_ids(wd / "ref_ids.txt")
    reqs = build_requests(base)
    ref = {r.request_id: standalone(model, r) for r in reqs}

    ok = True
    ok &= await check_identity(
        model, reqs, ref, lambda: Scheduler(model, max_batch=2, batched=True),
        "async contiguous mb=2   ", strict_chunks=True)
    ok &= await check_identity(
        model, reqs, ref,
        lambda: Scheduler(model, max_batch=8, batched=True, block_size=4, num_blocks=64),
        "async paged      bs=4   ", strict_chunks=True)
    ok &= await check_cancel(model, base)
    ok &= await check_spec(model, base)
    ok &= await check_shutdown(model, base)

    print("\nrun_async_serve:", "ok" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
