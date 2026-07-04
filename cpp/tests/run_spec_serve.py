"""S3 parity — the speculative scheduler must produce the SAME tokens as running each
request standalone, at every batch size and for every proposer (draft model / prompt-
lookup / a mix). This is the stage bar, the exact analog of run_serve.py (F7): greedy
speculative decode is token-identical to plain greedy (the S0 invariant), and folding it
into continuous batching — interleaving, queueing, dynamic admit/evict — must not change
any output.

Two reference points per request, both of which the scheduler must match:
  * standalone SPEC (greedy_speculative / greedy_prompt_lookup) — the direct contract:
    SpecScheduler mirrors that loop's accept/emit/eos exactly, so it must agree token-for-
    token even in how the draft interacts with the cache.
  * plain greedy (nicpp.generate) — the ultimate S0 invariant: spec output == plain target
    greedy, so a wrong proposer guess is never a wrong token.

A MIXED batch (draft + lookup requests interleaved) is the real test of per-sequence
proposer state — each sequence carries its own draft cache / n-gram context independently.

    cmake --build build -j
    python tests/run_spec_serve.py weights/qwen2.5-0.5b weights/qwen2.5-1.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
sys.path.insert(0, str(CPP / "build"))   # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))  # spec_scheduler, speculative

import numpy as np  # noqa: E402
import nicpp  # noqa: E402
from spec_scheduler import SpecRequest, SpecScheduler  # noqa: E402
from speculative import greedy_prompt_lookup, greedy_speculative  # noqa: E402


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def check_spec_batch(model, base: list[int], device: str = "cpu") -> bool:
    """S3b foundation: the ragged batched verify forward_spec_batch produces the SAME logits
    as a per-sequence forward for each sequence's block. Sequences at DIFFERENT cache lengths
    and DIFFERENT block sizes (incl. a 1-row block — the k=0 / prompt-lookup-miss shape)
    exercise the ragged row_start / pos bookkeeping.

    On the CPU oracle each output row is an independent dot, so fusing the projection GEMMs
    over all rows cannot change any row -> bit-identical (max|diff|==0), the same reason
    forward_batch is row-for-row exact. On CUDA the batched vs per-seq row counts can pick a
    different GEMM kernel (tiled vs GEMV), reordering the float reductions, so the bar is the
    CLAUDE.md GPU rule: within ~1e-3 AND the greedy token (argmax) identical per row."""
    # (prompt, verify_block); the block tokens are arbitrary (a real draft guesses wrong).
    specs = [
        (base,           [40, 100, 785]),               # 3-row block
        (base[:3],       [11]),                          # 1-row block (k=0 shape)
        (base + [12095], [13, 279, 7772, 304, 5]),       # 5-row block, longer prefix
    ]
    ref = []                                             # per-sequence forward (the reference)
    for prompt, block in specs:
        c = model.make_cache(256)
        model.forward(prompt, c)
        ref.append(model.forward(block, c))              # [count, vocab]

    caches = []                                          # fresh caches, same prefills
    for prompt, _ in specs:
        c = model.make_cache(256)
        model.forward(prompt, c)
        caches.append(c)
    tokens: list[int] = []
    counts: list[int] = []
    for _, block in specs:
        tokens.extend(block)
        counts.append(len(block))
    batched = model.forward_spec_batch(tokens, counts, caches)  # [sum(counts), vocab]

    off, dmax, tok_ok = 0, 0.0, True
    for r, (_, block) in zip(ref, specs):
        cnt = len(block)
        dmax = max(dmax, float(np.max(np.abs(batched[off:off + cnt] - r))))
        tok_ok = tok_ok and np.array_equal(np.argmax(batched[off:off + cnt], axis=1),
                                            np.argmax(r, axis=1))
        off += cnt
    tol = 0.0 if device == "cpu" else 2e-3
    ok = dmax <= tol and tok_ok
    print(f"  spec_batch: forward_spec_batch == per-seq forward  |diff|={dmax:.1e} "
          f"(tol={tol:.0e}) tokens={'==' if tok_ok else '!='} -> {'OK' if ok else 'FAIL'}")
    return ok


def plain_greedy(model, prompt: list[int], max_tokens: int, eos_id: int = -1) -> list[int]:
    """The ultimate reference: plain single-sequence greedy (F6, parity-locked vs nanoinfer)."""
    return model.generate(prompt, max_tokens=max_tokens, params=nicpp.SamplingParams(),
                          seed=0, eos_id=eos_id)


def standalone_spec(target, draft, req: SpecRequest) -> list[int]:
    """The direct contract: this request run on its own through the single-sequence spec loop."""
    if req.proposer == "draft":
        out, _ = greedy_speculative(target, draft, req.prompt_ids, req.max_tokens,
                                    k=req.k, eos_id=req.eos_id)
    else:
        out, _ = greedy_prompt_lookup(target, req.prompt_ids, req.max_tokens,
                                      ngram=req.ngram, k=req.k, eos_id=req.eos_id)
    return out


def build_requests(base: list[int]) -> list[SpecRequest]:
    # Mixed proposers + varied prompts/max_tokens so sequences finish at different steps —
    # that staggering is what exercises continuous batching (evict a short one, admit a
    # queued one), and the draft/lookup mix exercises per-sequence proposer independence.
    # The full-`base` prompts are repetitive geography, so prompt-lookup actually lands.
    return [
        SpecRequest("d0", base,           max_tokens=12, proposer="draft",  k=4),
        SpecRequest("l1", base,           max_tokens=12, proposer="lookup", ngram=3, k=10),
        SpecRequest("d2", base + [12095], max_tokens=8,  proposer="draft",  k=2),
        SpecRequest("l3", base[:6],       max_tokens=6,  proposer="lookup", ngram=2, k=4),
        SpecRequest("d4", base[:3],       max_tokens=10, proposer="draft",  k=8),
        SpecRequest("l5", base,           max_tokens=16, proposer="lookup", ngram=3, k=16),
    ]


def run_configs(label, target, draft, reqs, batch_sizes) -> bool:
    # References computed once (batch-invariant): standalone spec AND plain greedy.
    ref_spec = {r.request_id: standalone_spec(target, draft, r) for r in reqs}
    ref_plain = {r.request_id: plain_greedy(target, r.prompt_ids, r.max_tokens, r.eos_id)
                 for r in reqs}
    # The S0 invariant, checked directly: standalone spec already == plain greedy.
    base_ok = all(ref_spec[r.request_id] == ref_plain[r.request_id] for r in reqs)
    print(f"{label}: standalone spec == plain greedy -> {'OK' if base_ok else 'FAIL'}")

    ok = base_ok
    # Every request token-identical to standalone spec, invariant to batch size AND to the
    # verify backing: S3a (per-seq forward loop) and S3b (one ragged forward_spec_batch).
    for batched in (False, True):
        tag = "S3b batched " if batched else "S3a per-seq "
        for mb in batch_sizes:
            sched = SpecScheduler(target, draft=draft, max_batch=mb, batched=batched)
            for r in reqs:
                sched.add(r)
            out = sched.run()
            mism = [rid for rid in ref_spec if out.get(rid) != ref_spec[rid]]
            match = not mism and len(out) == len(reqs)
            ok = ok and match
            tpv = sched.stats.tokens_per_verify
            print(f"  {tag} mb={mb}: steps={sched.steps} peak_batch={sched.peak_batch} "
                  f"verifies={sched.stats.verifies} accept={sched.stats.accept_rate:5.1%} "
                  f"tok/verify={tpv:.2f} -> {'MATCH' if match else 'MISMATCH ' + str(mism)}")
    return ok


def main() -> int:
    d05 = Path(sys.argv[1] if len(sys.argv) > 1 else "weights/qwen2.5-0.5b")
    d15 = Path(sys.argv[2] if len(sys.argv) > 2 else "weights/qwen2.5-1.5b")
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"  # 'cpu' oracle (bit-identical) or 'cuda'

    m05 = nicpp.Model(str(d05), device=device)
    ok = True
    print(f"device={device}")

    # 0. S3b foundation: the ragged batched verify vs per-seq forward (bit-identical on CPU,
    #    within tolerance + identical tokens on CUDA).
    print("0. ragged batched verify:")
    ok &= check_spec_batch(m05, read_ids(d05 / "ref_ids.txt"), device)

    # A. target == draft (0.5B/0.5B): draft requests hit 100% accept (all-accepted + bonus
    #    path every step), a strong exercise of the verify+rollback machinery under batching.
    print("A. target=draft=0.5B (draft reqs ~100% accept) + prompt-lookup, mixed batch")
    ok &= run_configs("  0.5B/0.5B", m05, m05, build_requests(read_ids(d05 / "ref_ids.txt")),
                      batch_sizes=(1, 2, 8))

    # B. the real pair: target=1.5B, draft=0.5B — genuine (<100%) accept, still token-identical.
    if d15.exists():
        print("\nB. target=1.5B, draft=0.5B (the real pair) + prompt-lookup, mixed batch")
        m15 = nicpp.Model(str(d15), device=device)
        ok &= run_configs("  0.5B->1.5B", m15, m05, build_requests(read_ids(d15 / "ref_ids.txt")),
                          batch_sizes=(1, 2, 8))
    else:
        print(f"\nB. skipped — no 1.5B weights at {d15}")

    # Show one interleaving concretely (mixed proposers, mb=2 so it queues).
    print("\nper-request output (0.5B, mb=2, vs standalone spec):")
    reqs = build_requests(read_ids(d05 / "ref_ids.txt"))
    ref = {r.request_id: standalone_spec(m05, m05, r) for r in reqs}
    sched = SpecScheduler(m05, draft=m05, max_batch=2)
    for r in reqs:
        sched.add(r)
    out = sched.run()
    for r in reqs:
        tag = "ok" if out[r.request_id] == ref[r.request_id] else "DIFF"
        print(f"  {r.request_id}: {r.proposer:6s} prompt_len={len(r.prompt_ids):2d} "
              f"max_tokens={r.max_tokens:2d} [{tag}] {out[r.request_id]}")

    print("\nrun_spec_serve:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
