"""F7 parity check — the continuous-batching scheduler must produce the SAME tokens
as running each request standalone. This is the stage bar ("batched results match
single-sequence results"): greedy is deterministic and each sequence has its own KV
cache, so interleaving, queueing, and dynamic admit/evict must not change any output.

For every request we compute a reference with nicpp.generate (standalone, the F6
path already parity-tested vs nanoinfer), then run all requests through the Scheduler
at a few batch sizes and assert the outputs are token-identical. One request carries a
repetition penalty — a processor that shapes greedy too — to check the scheduler
tracks each sequence's context independently.

    cmake --build build -j                 # build nicpp (F6)
    python tests/run_serve.py weights/qwen2.5-0.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
sys.path.insert(0, str(CPP / "build"))   # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))  # scheduler
sys.path.insert(0, str(CPP / "tools"))   # (unused here, kept parallel to run_binding)

import nicpp  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def standalone(model, req: Request) -> list[int]:
    """The reference: one request via the F6 single-sequence generate()."""
    params = nicpp.SamplingParams(
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    return model.generate(req.prompt_ids, max_tokens=req.max_tokens, params=params,
                          seed=req.seed, eos_id=req.eos_id)


def build_requests(base: list[int]) -> list[Request]:
    # Distinct prompts (varied content + length) and varied max_tokens, so sequences
    # finish at different steps — that staggering is what exercises continuous
    # batching (evict a short one, admit a queued one). eos_id=-1 → fixed lengths,
    # so the comparison is clean. r4 adds a repetition penalty.
    return [
        Request("r0", base, max_tokens=12),
        Request("r1", base[:3], max_tokens=6),
        Request("r2", base + [12095], max_tokens=8),
        Request("r3", base[:2], max_tokens=4),
        Request("r4", base, max_tokens=10, repetition_penalty=1.3),
        Request("r5", base[1:4], max_tokens=7),
    ]


def main() -> int:
    wd = Path(sys.argv[1] if len(sys.argv) > 1 else "weights/qwen2.5-0.5b")
    model = nicpp.Model(str(wd))
    base = read_ids(wd / "ref_ids.txt")
    reqs = build_requests(base)

    # References: each request run on its own.
    ref = {r.request_id: standalone(model, r) for r in reqs}

    ok = True
    # max_batch 1 (fully serial through the scheduler) and 2 and a roomy 8 must all
    # agree with standalone — greedy output is invariant to how sequences are packed.
    for max_batch in (1, 2, 8):
        sched = Scheduler(model, max_batch=max_batch)
        for r in reqs:
            sched.add(r)
        out = sched.run()

        mism = [rid for rid in ref if out.get(rid) != ref[rid]]
        match = not mism
        ok = ok and match and len(out) == len(reqs)
        print(f"max_batch={max_batch}: steps={sched.steps} peak_batch={sched.peak_batch} "
              f"-> {'MATCH' if match else 'MISMATCH ' + str(mism)}")

    # Show one interleaving concretely (the smallest batch that actually queues).
    print("\nper-request greedy output (vs standalone):")
    sched = Scheduler(model, max_batch=2)
    for r in reqs:
        sched.add(r)
    out = sched.run()
    for r in reqs:
        tag = "ok" if out[r.request_id] == ref[r.request_id] else "DIFF"
        extra = f"  rep_penalty={r.repetition_penalty}" if r.repetition_penalty != 1.0 else ""
        print(f"  {r.request_id}: prompt_len={len(r.prompt_ids)} "
              f"max_tokens={r.max_tokens} [{tag}] {out[r.request_id]}{extra}")

    print("\nrun_serve:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
