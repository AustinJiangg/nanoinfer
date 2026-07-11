"""S4 — measure PROMPT-LOOKUP speculative decode: speedup over plain target greedy with NO
second model, on the clock (the G5a discipline: no speedup number until it's measured).

S2's verdict on the 0.5B/1.5B draft/target pair: the speedup is capped by the cost ratio
r = t_draft/t_target ≈ 0.45 (the 0.5B draft isn't cheap enough), NOT by accept rate — so
even 100% accept only buys ~1.2×, and open-ended text can NET-LOSE. S2 named prompt-lookup
the fix: propose by matching the recent context against an earlier occurrence and copying
what followed — a DRAFT WITH r → 0 (an array scan, no model forward). Removing the r·k
draft-forward tax means a MISS barely costs (the step degrades to plain greedy) and the
ceiling is tokens-per-verify itself (k+1 when a whole copy lands). The one residual cost is
the fatter verify (below), so a large k on rarely-matching text can *mildly* net-lose (S4
measured list @k=8: 0.86×) — far milder than the draft model's per-step r·k tax.

So the win here is a different shape than S0's: not a flat ~1.2× on everything, but a big
win on COPY-HEAVY text (the target's own greedy quotes earlier context — RAG, summarize,
code-edit, repetitive structure) and a ~1× no-op on open-ended text (no matches → every
step degrades to a plain greedy forward). The sweep shows both honestly.

Each config is correctness-gated: its output must be TOKEN-IDENTICAL to plain target greedy
(the S0 invariant), or its speedup is void. `ceil` = tokens/verify is the r→0 ideal speedup
(if a k+1-token verify cost the same as a 1-token decode); measured `speedup` falls below it
by the fatter-verify tax (verifying k+1 tokens is a mini-prefill, > one decode step).

    python bench/bench_lookup.py weights/qwen2.5-1.5b --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from ni.engine import nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.speculative import greedy_prompt_lookup  # noqa: E402

from bench_spec import DIVERSE_PROMPTS, best_time  # noqa: E402  (sibling: the shared prompt set)

# A copy-heavy prompt (Qwen2.5 ids, vocab 151936): a passage whose phrasing recurs verbatim,
# so the target's OWN greedy continuation quotes the earlier text — prompt-lookup's home turf
# (summarize / RAG / code-edit / repetitive structure), where the r→0 ceiling actually shows.
# The text: "The Fibonacci sequence is defined as follows: each number is the sum of the two
# preceding ones, starting from 0 and 1. So the sequence goes 0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
# and so on. In other words, the Fibonacci sequence is defined as follows: each number is the
# sum of the two". Regenerate: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")(text).
COPY_PROMPT = [
    785, 79683, 8500, 374, 4512, 438, 11017, 25, 1817, 1372, 374, 279, 2629, 315, 279, 1378,
    37746, 6174, 11, 5916, 504, 220, 15, 323, 220, 16, 13, 2055, 279, 8500, 5780, 220, 15, 11,
    220, 16, 11, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 20, 11, 220, 23, 11, 220, 16, 18,
    11, 220, 17, 16, 11, 220, 18, 19, 11, 323, 773, 389, 13, 758, 1008, 4244, 11, 279, 79683,
    8500, 374, 4512, 438, 11017, 25, 1817, 1372, 374, 279, 2629, 315, 279, 1378,
]


def main() -> int:
    ap = argparse.ArgumentParser(description="S4: prompt-lookup speedup + (ngram, k) sweep")
    ap.add_argument("target", type=Path, help="target model weights dir (the one to speed up)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ngrams", default="2,3", help="comma-separated match lengths to sweep")
    ap.add_argument("--ks", default="4,8,16", help="comma-separated max proposal lengths to sweep")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per config (best-of-N)")
    ap.add_argument("--prompt-ids", default=None,
                    help="comma-separated files of space-separated token ids. Default: the "
                         "built-in DIVERSE_PROMPTS (realistic self-repetition) + a copy-heavy "
                         "prompt (the r→0 ceiling). Each prompt is swept independently.")
    args = ap.parse_args()
    ngrams = [int(x) for x in args.ngrams.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    N = args.max_tokens
    greedy = nicpp.SamplingParams()

    try:
        target = nicpp.Model(str(args.target), device=args.device)
    except Exception as e:  # no CUDA build / no GPU
        print(f"bench_lookup: target unavailable on {args.device} — skipping ({e})")
        return 0

    if args.prompt_ids:
        prompts = [(Path(p).stem, read_ids(Path(p))) for p in args.prompt_ids.split(",")]
    else:
        prompts = list(DIVERSE_PROMPTS.items()) + [("copy", COPY_PROMPT)]

    def sweep(prompt, label):
        """Full (ngram, k) sweep for one prompt; returns (ok, best_speedup)."""
        def plain():
            return target.generate(prompt, max_tokens=N, params=greedy, seed=0, eos_id=-1)

        # Warm up: pull device-pool growth + first-kernel init out of the timed region.
        target.generate(prompt, max_tokens=8, params=greedy, seed=0, eos_id=-1)

        ref, t_tgt = best_time(plain, args.runs)   # the baseline to beat
        tgt_toks = N / t_tgt

        print(f"\n=== prompt '{label}'  ({len(prompt)} tok) ===")
        print(f"plain target : {tgt_toks:6.1f} tok/s (baseline)")
        print("  ng   K   hit%  accept%  tok/vf   speedup  (ceil)   spec tok/s   accept-len % a=0..K")
        print("  " + "-" * 80)

        ok, best = True, 0.0
        for ng in ngrams:
            for k in ks:
                (out, st), dt = best_time(
                    lambda ng=ng, k=k: greedy_prompt_lookup(target, prompt, N, ngram=ng, k=k,
                                                            eos_id=-1), args.runs)
                spec_toks = N / dt
                match = out == ref                   # S0 invariant: token-identical to plain greedy
                ok = ok and match
                speedup = spec_toks / tgt_toks
                ceil = st.tokens_per_verify           # the r→0 ideal (verify tax aside)
                hist = st.accept_histogram(k)
                tot = max(1, sum(hist))
                dist = " ".join(f"{100 * h // tot:2d}" for h in hist)
                flag = "" if match else "  <- MISMATCH (speedup void)"
                print(f"  {ng:<3} {k:<3} {100 * st.hit_rate:5.0f}  {100 * st.accept_rate:6.0f}  "
                      f"{st.tokens_per_verify:6.2f}  {speedup:6.2f}x  ({ceil:4.2f}x)  {spec_toks:8.1f}   "
                      f"[{dist}]{flag}")
                if match:
                    best = max(best, speedup)
        return ok, best

    print(f"device={args.device}  max_tokens={N}  runs={args.runs}  prompts={len(prompts)}")
    ok, summary = True, []
    for label, ids in prompts:
        pok, best = sweep(ids, label)
        ok = ok and pok
        summary.append((label, best))

    print("\nsummary (best speedup per prompt — prompt-lookup is copy-dependent):")
    for label, best in summary:
        print(f"  {label:10} {best:.2f}x")
    lo, hi = min(b for _, b in summary), max(b for _, b in summary)
    print(f"  range: {lo:.2f}x .. {hi:.2f}x")
    print("\nbench_lookup:", "ok" if ok else "FAIL (a config diverged from plain greedy)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
