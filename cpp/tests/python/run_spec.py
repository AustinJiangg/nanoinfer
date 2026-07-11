"""S0 parity: greedy speculative decode must be TOKEN-IDENTICAL to plain target
greedy. The accept rule guarantees it — every emitted token is the target's own
argmax at that position — so the draft changes only speed, never output. This is the
S0 "done when" bar.

Two checks:
  A. draft == target (0.5B/0.5B): every draft is accepted (accept_rate == 100%,
     ~k+1 tokens per verify), a strong exercise of the verify + rollback machinery
     (the all-accepted + bonus path fires every step). Correctness AND the invariant.
  B. draft=0.5B, target=1.5B (the real pair): output == plain 1.5B greedy at every k,
     with a genuine (< 100%) accept rate — the actual speculative-decoding win to come.

    cmake --build build -j
    python tests/python/run_spec.py weights/qwen2.5-0.5b weights/qwen2.5-1.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.speculative import greedy_speculative, greedy_prompt_lookup, prompt_lookup  # noqa: E402

from gateutil import plain_greedy  # noqa: E402  (sibling: the shared reference path)


def check_lookup_matcher() -> bool:
    """The S4 seam in isolation: prompt_lookup() copies the tokens that followed an earlier
    occurrence of the context's suffix, most-recent-first, capped at max_k, [] on no match.
    Pure logic (no model), so it's cheap to pin exactly — a wrong match is a silent speed
    regression, never a correctness bug (the verify catches it), so this only guards speed."""
    cases = [
        (([1, 2, 3, 1, 2], 2, 4), [3, 1, 2]),   # suffix "1 2" recurs at pos 0, copy what followed
        (([1, 2, 3, 1, 2], 2, 1), [3]),          # capped at max_k
        (([1, 2, 3, 1, 2], 3, 4), []),           # "3 1 2" never occurs earlier
        (([1, 2, 3], 3, 4), []),                 # n<=ngram: nothing precedes the suffix
        (([], 2, 4), []),
        (([9, 5, 9, 7, 9], 1, 2), [7, 9]),       # most-recent-first: copy after the LATEST "9"
    ]
    ok = all(prompt_lookup(ctx, ng, k) == exp for (ctx, ng, k), exp in cases)
    print(f"  matcher: prompt_lookup {len(cases)} cases -> {'OK' if ok else 'FAIL'}")
    return ok


def check_primitive(model, prompt: list[int]) -> bool:
    """The two S0 foundations, isolated and bit-identical (max|diff|==0):

    1. Multi-token forward() onto a POPULATED cache == decoding those tokens one at a
       time. This IS the verify step; arbitrary (even "wrong") tokens, since a draft
       guesses wrong. If this drifted, every verify would be subtly off.
    2. truncate(L) + replay == the first run — rollback restores the exact state, so
       discarding rejected drafts leaves no residue.
    """
    extra = [40, 100, 12095, 785, 11]   # arbitrary continuation (positions, not values)

    ca = model.make_cache(256)
    model.forward(prompt, ca)
    multi = model.forward(extra, ca)                        # one pass
    cb = model.make_cache(256)
    model.forward(prompt, cb)
    seq = np.concatenate([model.forward([t], cb) for t in extra], axis=0)  # one at a time
    d_prim = float(np.max(np.abs(multi - seq)))

    cc = model.make_cache(256)
    model.forward(prompt, cc)
    first = model.forward(extra, cc)
    cc.truncate(len(prompt))                                # roll back the tentative tail
    replay = model.forward(extra, cc)
    d_roll = float(np.max(np.abs(first - replay)))

    ok = d_prim == 0.0 and d_roll == 0.0
    print(f"  primitive: multi==sequential |diff|={d_prim:.1e}  "
          f"truncate+replay |diff|={d_roll:.1e}  -> {'OK' if ok else 'FAIL'}")
    return ok


def run_case(name, target, draft, prompt, max_tokens, ks, *, expect_full_accept=False,
             eos_id=-1) -> bool:
    ref = plain_greedy(target, prompt, max_tokens, eos_id)   # once — same for every k
    ok = True
    for k in ks:
        out, st = greedy_speculative(target, draft, prompt, max_tokens, k=k, eos_id=eos_id)
        match = out == ref
        full_ok = (not expect_full_accept) or st.accept_rate == 1.0
        ok = ok and match and full_ok
        flag = "MATCH" if match else "MISMATCH"
        if not full_ok:
            flag += " (accept!=100%)"
        print(f"  {name}  k={k}: {flag}  accept={st.accept_rate:5.1%}  "
              f"tok/verify={st.tokens_per_verify:.2f}  verifies={st.verifies}")
        if not match:
            print(f"     ref={ref}")
            print(f"     out={out}")
    return ok


def run_lookup_case(name, target, prompt, max_tokens, ngrams, ks, *, eos_id=-1) -> bool:
    """S4: prompt-lookup output must be token-identical to plain target greedy at every
    (ngram, k) — no draft model, so this is purely the verify + rollback machinery driven by
    the n-gram proposer. We also require the accept path to actually FIRE (some verify accepts
    a proposal), else the prompt never repeats and we'd be trivially matching plain greedy
    without exercising the interesting path."""
    ref = plain_greedy(target, prompt, max_tokens, eos_id)   # once — same for every config
    ok, any_accept = True, False
    for ng in ngrams:
        for k in ks:
            out, st = greedy_prompt_lookup(target, prompt, max_tokens, ngram=ng, k=k, eos_id=eos_id)
            match = out == ref
            any_accept = any_accept or st.accepted > 0
            ok = ok and match
            print(f"  {name}  ng={ng} k={k:2d}: {'MATCH' if match else 'MISMATCH'}  "
                  f"hit={st.hit_rate:5.1%}  accept={st.accept_rate:5.1%}  "
                  f"tok/verify={st.tokens_per_verify:.2f}")
            if not match:
                print(f"     ref={ref}")
                print(f"     out={out}")
    if not any_accept:
        print(f"  {name}: WARN — no proposal ever accepted (accept path not exercised)")
    return ok and any_accept


def main() -> int:
    d05 = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights_dir()
    d15 = Path(sys.argv[2]) if len(sys.argv) > 2 else default_weights_dir("qwen2.5-1.5b")

    m05 = nicpp.Model(str(d05))
    ok = True

    print("0. foundations (bit-identical):")
    ok &= check_primitive(m05, read_ids(d05 / "ref_ids.txt"))
    ok &= check_lookup_matcher()

    print("\nA. draft == target (0.5B/0.5B): expect 100% accept, output == plain greedy")
    ok &= run_case("self  0.5B/0.5B", m05, m05, read_ids(d05 / "ref_ids.txt"),
                   max_tokens=24, ks=(1, 2, 4, 8), expect_full_accept=True)

    m15 = None
    if d15.exists():
        print("\nB. draft=0.5B, target=1.5B (the real pair): output == plain 1.5B greedy")
        m15 = nicpp.Model(str(d15))
        ok &= run_case("pair  0.5B->1.5B", m15, m05, read_ids(d15 / "ref_ids.txt"),
                       max_tokens=24, ks=(2, 4, 8))
    else:
        print(f"\nB. skipped — no 1.5B weights at {d15}")

    # S4: prompt-lookup — no draft model. Token-identity is proposer-independent, so 48 tokens
    # off the France prompt (which drifts into repetitive geography) suffice to make the target's
    # OWN greedy quote earlier context and fire the accept path. Run on both models it's present.
    print("\nC. prompt-lookup (S4): no draft model — output == plain target greedy")
    lookup_prompt = read_ids(d05 / "ref_ids.txt")
    ok &= run_lookup_case("lookup 0.5B", m05, lookup_prompt, max_tokens=48,
                          ngrams=(2, 3), ks=(4, 10))
    if m15 is not None:
        ok &= run_lookup_case("lookup 1.5B", m15, read_ids(d15 / "ref_ids.txt"), max_tokens=48,
                              ngrams=(2, 3), ks=(4, 10))

    print("\nrun_spec:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
