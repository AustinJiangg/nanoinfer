"""S5 parity: SAMPLING speculative decode preserves the target's distribution.

The greedy S-track rests on token-IDENTITY (every emitted token is the target's argmax).
Sampling can't be token-identical — it's random — so the invariant becomes DISTRIBUTIONAL:
speculative sampling (rejection sampling) emits tokens whose marginal is exactly the
target's own shaped distribution p, for ANY proposer. The gate has four tiers, strongest
to cheapest:

  U1  the accept rule's math, pinned exactly with a scripted RNG (accept / reject-correction
      / all-accept-bonus / point-mass / q(x)=0 short-circuit).
  D-alg  the theorem itself, model-free: over random (q, p) on a tiny vocab, the emitted
      token's empirical marginal == p (TVD -> 0), for a real q (draft) AND a point mass
      (lookup). This is the correctness proof — no model, so cheap and low-variance.
  E1  the greedy floor as the temperature->0 LIMIT: sampling core == greedy core == plain
      greedy, TOKEN-identical (max|diff|=0 on the ids). Anchors D-alg to the real model.
  D2/D3  single-source-of-truth on the real model: plain generate draws its first token from
      token_probs' distribution (D2), and a full draft speculative sample's first accepted
      token matches p (D3) — the algorithm and the C++ sampler agree end to end.

    cmake --build build -j
    python tests/python/run_spec_sample.py weights/qwen2.5-0.5b weights/qwen2.5-1.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.speculative import (  # noqa: E402
    _draw,
    greedy_prompt_lookup,
    greedy_speculative,
    rejection_accept,
    sample_prompt_lookup,
    sample_speculative,
)


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance between two distributions (half the L1)."""
    return 0.5 * float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum())


class ScriptedRNG:
    """A stand-in numpy Generator that returns a fixed sequence from .random(), so the
    accept/reject/draw decisions are deterministic and the emitted tokens can be pinned."""

    def __init__(self, vals):
        self.vals = list(vals)
        self.i = 0

    def random(self) -> float:
        v = self.vals[self.i]
        self.i += 1
        return v


# ---------------------------------------------------------------------------
# U1 — the accept rule, exact. Distributions are chosen so every correction/bonus target
# is ONE-HOT (its draw is independent of the RNG value), leaving only the accept/reject
# comparisons to script. This pins rejection_accept's control flow token-for-token.
# ---------------------------------------------------------------------------


def f(*xs) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def check_accept_rule() -> bool:
    cases = [
        # name, d, q_list, p_list, rng vals, expected (a, emitted)
        # A: point mass, accept both (random < p(x)), bonus from one-hot p2.
        ("pm accept+bonus", [0, 1], [None, None],
         [f(0.7, 0.3, 0.0), f(0.2, 0.5, 0.3), f(0.0, 0.0, 1.0)], [0.5, 0.4, 0.9], (2, [0, 1, 2])),
        # B: point mass, reject at 0 (random >= p(1)); residual zeros token 1 -> one-hot on 0.
        ("pm reject", [1], [None], [f(0.6, 0.4, 0.0)], [0.9, 0.5], (0, [0])),
        # C: draft q, ratio p/q = 1.125 >= 1 -> always accept; bonus from one-hot p1.
        ("q accept(ratio>=1)", [2], [f(0.1, 0.1, 0.8)],
         [f(0.1, 0.0, 0.9), f(1.0, 0.0, 0.0)], [0.99, 0.3], (1, [2, 0])),
        # D: draft q, ratio 0.222 -> reject; residual relu(p-q) is one-hot on 0.
        ("q reject->residual", [2], [f(0.05, 0.05, 0.9)], [f(0.8, 0.0, 0.2)], [0.5, 0.1], (0, [0])),
        # E: q(x)=0 -> short-circuit accept WITHOUT consuming an RNG value; then one-hot bonus.
        ("q(x)=0 short-circuit", [1], [f(0.5, 0.0, 0.5)],
         [f(0.0, 0.0, 0.0), f(0.0, 0.0, 1.0)], [0.7], (1, [1, 2])),
    ]
    ok = True
    for name, d, q, p, vals, exp in cases:
        rng = ScriptedRNG(vals)
        got = rejection_accept(d, q, p, rng)
        # got is (a, list); normalize to a comparable tuple
        got_t = (got[0], list(got[1]))
        consumed = rng.i == len(vals)     # exactly the scripted draws were used (E tests the skip)
        good = got_t == exp and consumed
        ok = ok and good
        print(f"  accept-rule {name:22s}: {'OK' if good else 'FAIL'}  got={got_t} exp={exp}"
              f"{'' if consumed else f'  (consumed {rng.i}/{len(vals)})'}")
    return ok


# ---------------------------------------------------------------------------
# D-alg — the rejection-sampling theorem, model-free. Draw the proposal from q, run the
# accept rule, and the emitted token's empirical marginal must converge to the target p.
# ---------------------------------------------------------------------------


def rand_dist(rng, n) -> np.ndarray:
    x = rng.random(n).astype(np.float32) + 0.05     # keep every token > 0 so ratios are finite
    return x / x.sum()


def check_theorem(n_samples: int = 20000, tol: float = 0.02) -> bool:
    """For random (q, p) on a small vocab, the first emitted token ~ p. Tested for k=1 and
    k=3 (position-0 bookkeeping must survive a longer proposal) and for both proposer kinds:
    a real draft distribution q, and a point mass (lookup). Position 0 depends only on
    (d_0, q_0, p_0), so the first-token marginal isolates the accept math. One RNG stream
    per config (independent draws) keeps it fast."""
    seed = np.random.default_rng(20260705)
    rng = np.random.default_rng(11)
    vocab = 6
    ok = True
    for kind in ("draft-q", "point-mass"):
        for k in (1, 3):
            q0 = rand_dist(seed, vocab)
            q_extra = [rand_dist(seed, vocab) for _ in range(k - 1)]       # for d_1.. (unused at pos 0)
            p_list = [rand_dist(seed, vocab) for _ in range(k + 1)]
            p0 = p_list[0]
            counts = np.zeros(vocab, dtype=np.int64)
            for _ in range(n_samples):
                if kind == "draft-q":
                    d = [_draw(q0, rng)]                                   # d_0 ~ q_0 (the tested pos)
                    d += [int(rng.integers(vocab)) for _ in range(k - 1)]  # d_1.. don't affect pos 0
                    q = [q0] + q_extra
                else:
                    d = [int(rng.integers(vocab)) for _ in range(k)]       # deterministic-per-draw copy
                    q = [None] * k
                _, emitted = rejection_accept(d, q, p_list, rng)
                counts[emitted[0]] += 1
            emp = counts / counts.sum()
            d_tvd = tvd(emp, p0)
            good = d_tvd < tol
            ok = ok and good
            print(f"  theorem {kind:11s} k={k}: TVD(emitted_0, p)={d_tvd:.4f}  "
                  f"{'OK' if good else 'FAIL'} (tol {tol})")
    return ok


# ---------------------------------------------------------------------------
# E1 — the greedy floor as the temperature->0 limit (token-identical, exact).
# ---------------------------------------------------------------------------


def check_temp0_bridge(target, draft, prompt, max_tokens=24, eos_id=-1) -> bool:
    greedy = nicpp.SamplingParams()   # temperature 0
    ref = target.generate(prompt, max_tokens=max_tokens, params=greedy, seed=0, eos_id=eos_id)
    ok = True
    for k in (1, 2, 4, 8):
        gs, _ = greedy_speculative(target, draft, prompt, max_tokens, k=k, eos_id=eos_id)
        ss, _ = sample_speculative(target, draft, prompt, max_tokens, greedy, k=k, seed=0, eos_id=eos_id)
        good = ss == gs == ref
        ok = ok and good
        print(f"  bridge draft k={k}: sample==greedy=={good and 'plain' or '...'}  "
              f"{'OK' if good else 'FAIL'}")
    for ng, k in ((2, 4), (3, 10)):
        gl, _ = greedy_prompt_lookup(target, prompt, max_tokens, ngram=ng, k=k, eos_id=eos_id)
        sl, _ = sample_prompt_lookup(target, prompt, max_tokens, greedy, ngram=ng, k=k, seed=0, eos_id=eos_id)
        good = sl == gl == ref
        ok = ok and good
        print(f"  bridge lookup ng={ng} k={k}: {'OK' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# I1 — draft == target: q == p at every position, so every draft is accepted (100%).
# ---------------------------------------------------------------------------


def check_self_accept(model, prompt, params, max_tokens=24) -> bool:
    ok = True
    for k in (1, 2, 4):
        _, st = sample_speculative(model, model, prompt, max_tokens, params, k=k, seed=k)
        good = st.accept_rate == 1.0
        ok = ok and good
        print(f"  self-accept k={k}: accept={st.accept_rate:6.1%} tok/verify={st.tokens_per_verify:.2f}  "
              f"{'OK' if good else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# D2 — single source of truth: plain C++ generate draws its first token from exactly
# token_probs' distribution (so spec, which samples from token_probs, matches generate).
# ---------------------------------------------------------------------------


def check_generate_parity(model, prompt, params, n=2000, tol=0.06) -> bool:
    cap = len(prompt) + 4
    tcache = model.make_cache(cap)
    p = nicpp.token_probs(model.forward(prompt, tcache)[-1], params, prompt)   # the known dist
    vocab = p.shape[0]
    counts = np.zeros(vocab, dtype=np.int64)
    for s in range(n):
        tok = model.generate(prompt, max_tokens=1, params=params, seed=s, eos_id=-1)[0]
        counts[tok] += 1
    emp = counts / counts.sum()
    d_tvd = tvd(emp, p)
    good = d_tvd < tol
    print(f"  generate-parity: TVD(generate_first_token, token_probs)={d_tvd:.4f}  "
          f"support={int((p > 0).sum())}  {'OK' if good else 'FAIL'} (tol {tol}, n={n})")
    return good


# ---------------------------------------------------------------------------
# D3 — real-model draft path: the first ACCEPTED token of a draft speculative sample has
# marginal p (the target dist after the prefill token). Exercises draft-sample + verify +
# residual together on the model.
# ---------------------------------------------------------------------------


def check_real_draft_marginal(target, draft, prompt, params, n=250, k=4, tol=0.08) -> bool:
    # Prefill once, fix cur = the target's first token (a spread draw would blur p); then the
    # step's first emitted token (loop position 0) must be distributed as p0 = dist after cur.
    # BOTH caches are prefilled ONCE and truncated back each sample (S1 rollback is bit-exact),
    # so every sample is a clean independent draft-sample + verify + accept — no re-prefilling.
    from ni.speculative import DraftModelProposer  # local: only this check needs the raw proposer
    cap = len(prompt) + 8
    tc = target.make_cache(cap)
    cur = int(np.argmax(target.forward(prompt, tc)[-1]))          # fix the conditioning token
    p0 = nicpp.token_probs(target.forward([cur], tc)[-1], params, prompt + [cur])
    tc.truncate(len(prompt))                                      # undo the probe forward
    vocab = p0.shape[0]
    counts = np.zeros(vocab, dtype=np.int64)

    prop = DraftModelProposer(draft, k)
    prop.prefill(prompt, cap)
    prop.draft.forward([cur], prop.dcache)                        # advance draft to "cur emitted"
    draft_L = prop.dcache.length
    for s in range(n):
        rng = np.random.default_rng(s)
        d, qs = prop.propose_sampling(cur, prompt + [cur], params, rng)
        tl = target.forward([cur] + d, tc)
        p_list = [nicpp.token_probs(tl[i], params, prompt + [cur] + d[:i]) for i in range(len(d) + 1)]
        _, emitted = rejection_accept(d, qs, p_list, rng)
        counts[emitted[0]] += 1
        tc.truncate(len(prompt))                                 # roll back both tentative tails
        prop.dcache.truncate(draft_L)
    emp = counts / counts.sum()
    d_tvd = tvd(emp, p0)
    good = d_tvd < tol
    print(f"  real-draft-marginal: TVD(emitted_0, p0)={d_tvd:.4f}  support={int((p0 > 0).sum())}  "
          f"{'OK' if good else 'FAIL'} (tol {tol}, n={n})")
    return good


# ---------------------------------------------------------------------------
# S — the scheduler under sampling: a scheduled sampling sequence must be TOKEN-identical to
# standalone sample_speculative / sample_prompt_lookup at the same seed, invariant to batch
# size and the verify backing (per-seq / ragged batched). The per-sequence RNG is what makes
# the draw stream independent of interleaving; on the CPU oracle forward_spec_batch is
# bit-identical to per-seq forward, so the shaped p is identical and the draws land the same.
# ---------------------------------------------------------------------------


def check_scheduler_sampling(target, draft, base, params) -> bool:
    from ni.spec_scheduler import SpecRequest, SpecScheduler  # local import: only this check

    reqs = [
        SpecRequest("d0", base,     max_tokens=12, proposer="draft",  k=4, params=params, seed=1),
        SpecRequest("l1", base,     max_tokens=12, proposer="lookup", ngram=3, k=10, params=params, seed=2),
        SpecRequest("d2", base[:8], max_tokens=8,  proposer="draft",  k=2, params=params, seed=3),
        SpecRequest("l3", base,     max_tokens=16, proposer="lookup", ngram=2, k=8,  params=params, seed=4),
    ]
    # Standalone references (the direct contract), one per request, at its own seed.
    ref = {}
    for r in reqs:
        if r.proposer == "draft":
            out, _ = sample_speculative(target, draft, r.prompt_ids, r.max_tokens, r.params,
                                        k=r.k, seed=r.seed, eos_id=r.eos_id)
        else:
            out, _ = sample_prompt_lookup(target, r.prompt_ids, r.max_tokens, r.params,
                                          ngram=r.ngram, k=r.k, seed=r.seed, eos_id=r.eos_id)
        ref[r.request_id] = out

    ok = True
    configs = [("per-seq mb=1", dict(max_batch=1, batched=False)),
               ("per-seq mb=8", dict(max_batch=8, batched=False)),
               ("batched mb=2", dict(max_batch=2, batched=True)),
               ("batched mb=8", dict(max_batch=8, batched=True)),
               ("batched paged mb=8", dict(max_batch=8, batched=True, block_size=8, num_blocks=32))]
    for name, kw in configs:
        sched = SpecScheduler(target, draft=draft, **kw)
        for r in reqs:
            sched.add(r)
        out = sched.run()
        mism = [rid for rid in ref if out.get(rid) != ref[rid]]
        good = not mism and len(out) == len(reqs)
        ok = ok and good
        print(f"  scheduler {name:20s}: steps={sched.steps} peak={sched.peak_batch} "
              f"accept={sched.stats.accept_rate:5.1%} -> {'MATCH' if good else 'MISMATCH ' + str(mism)}")
    return ok


def check_prefix_sharing_sampling(target, draft, base, params) -> bool:
    """S3e under SAMPLING — prefix sharing must not perturb the sampled draws. Sharing only moves
    WHERE the prefix KV lives (bit-identical, causal), so token_probs and each sequence's seeded
    draw stream are unchanged: a sampled sequence stays TOKEN-identical to standalone at the same
    seed even when it reuses a shared prefix (the S5 batch-invariance property extended to prefix
    sharing). Mixed draft/lookup share a long prefix; all blocks free on clear (both pools)."""
    from ni.spec_scheduler import SpecRequest, SpecScheduler  # local import: only this check

    shared = [base[i % len(base)] for i in range(24)]        # 24-tok common prefix (6 blocks @ bs=4)
    reqs = [
        SpecRequest("s0", shared + [11],       max_tokens=8, proposer="draft",  k=4, params=params, seed=1),
        SpecRequest("s1", shared + [785],      max_tokens=8, proposer="lookup", ngram=3, k=8, params=params, seed=2),
        SpecRequest("s2", shared + [11, 1933], max_tokens=6, proposer="draft",  k=2, params=params, seed=3),
    ]
    ref = {}                                                 # standalone at each request's own seed
    for r in reqs:
        if r.proposer == "draft":
            out, _ = sample_speculative(target, draft, r.prompt_ids, r.max_tokens, r.params,
                                        k=r.k, seed=r.seed, eos_id=r.eos_id)
        else:
            out, _ = sample_prompt_lookup(target, r.prompt_ids, r.max_tokens, r.params,
                                          ngram=r.ngram, k=r.k, seed=r.seed, eos_id=r.eos_id)
        ref[r.request_id] = out

    sched = SpecScheduler(target, draft=draft, max_batch=4, batched=True,
                          block_size=4, num_blocks=128, prefix_sharing=True)
    for r in reqs:
        sched.add(r)
    out = sched.run()
    mism = [rid for rid in ref if out.get(rid) != ref[rid]]
    shared_tok = sched.shared_prefill_tokens
    sched.clear_prefix_cache()
    freed = (sched.pool.free_blocks == sched.pool.num_blocks
             and (sched.dpool is None or sched.dpool.free_blocks == sched.dpool.num_blocks))
    ok = not mism and len(out) == len(reqs) and shared_tok > 0 and freed
    print(f"  prefix-sharing sampling: shared_prefill={shared_tok} tok, freed={freed} "
          f"-> {'MATCH' if not mism else 'MISMATCH ' + str(mism)}")
    return ok


def main() -> int:
    d05 = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights_dir()
    d15 = Path(sys.argv[2]) if len(sys.argv) > 2 else default_weights_dir("qwen2.5-1.5b")

    ok = True
    print("U1. accept rule (exact, scripted RNG):")
    ok &= check_accept_rule()

    print("\nD-alg. rejection-sampling theorem (model-free, emitted ~ p):")
    ok &= check_theorem()

    m05 = nicpp.Model(str(d05))
    prompt05 = read_ids(d05 / "ref_ids.txt")

    print("\nE1. temperature->0 bridge (sample == greedy == plain, token-identical):")
    ok &= check_temp0_bridge(m05, m05, prompt05)     # draft == target: full-accept + lookup

    sp = nicpp.SamplingParams(temperature=1.0, top_k=40)
    print("\nI1. draft == target under sampling (q == p -> 100% accept):")
    ok &= check_self_accept(m05, prompt05, sp)

    print("\nD2. single source of truth (generate draws from token_probs):")
    ok &= check_generate_parity(m05, prompt05[:12], sp, n=1200)

    print("\nS. scheduler sampling == standalone (token-identical, batch-invariant):")
    ok &= check_scheduler_sampling(m05, m05, prompt05, sp)
    ok &= check_prefix_sharing_sampling(m05, m05, prompt05, sp)  # S3e under sampling

    if d15.exists():
        m15 = nicpp.Model(str(d15))
        prompt15 = read_ids(d15 / "ref_ids.txt")
        print("\nE1b. temperature->0 bridge on the real pair (draft 0.5B, target 1.5B):")
        ok &= check_temp0_bridge(m15, m05, prompt15)
        # A narrower nucleus (top_k=8) concentrates p onto few tokens so the empirical marginal
        # converges at a feasible sample count on the expensive 1.5B verify forward.
        sp_narrow = nicpp.SamplingParams(temperature=1.0, top_k=8)
        print("\nD3. real-model draft path marginal (draft 0.5B -> target 1.5B, emitted ~ p):")
        ok &= check_real_draft_marginal(m15, m05, prompt15, sp_narrow)
    else:
        print(f"\nE1b/D3 skipped — no 1.5B weights at {d15}")

    print("\nrun_spec_sample:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
