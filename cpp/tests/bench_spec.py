"""S2 — measure the speculative-decode speedup and tune the draft length K, on the clock
(the G5a discipline: no speedup number is claimed until it's measured).

Speculative decode's WIN is emitting > 1 token per (expensive) target forward; its COST is
K draft forwards per verify, so there is an optimal K. Per verify, roughly:

    time  ~= K * t_draft + t_target      (the target verify is one forward over K+1 tokens;
                                          decode is memory-bound on weight streaming, so K+1
                                          small tokens ~= one target step)
    yield  = accepted + 1 tokens

so the wall-clock speedup over plain target greedy is about

    speedup ~= tokens_per_verify / (1 + K * r),     r = t_draft / t_target

Bigger K lifts tokens_per_verify but the draft drifts (accept rate falls) and the K*r draft
overhead grows — the sweep finds the knee. Every K is correctness-gated: its output must be
TOKEN-IDENTICAL to plain target greedy (the S0 invariant), or its speedup is void. tok/s here
includes prefill (one target + one draft forward for spec, one target for plain — fair, and
<1% of a long run), so it's the honest end-to-end generate rate.

    python tests/bench_spec.py weights/qwen2.5-0.5b weights/qwen2.5-1.5b --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
sys.path.insert(0, str(CPP / "build"))    # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))   # speculative

import nicpp  # noqa: E402
from speculative import greedy_speculative  # noqa: E402

# Diverse prompts (Qwen2.5 tokenizer ids, both 0.5B/1.5B share vocab 151936). Embedded so the
# honest multi-prompt result is the DEFAULT and reproducible without HF at bench time — accept
# rate is text-dependent, and sweeping only the France ref_ids (which degenerates into repetition)
# would inflate it. Regenerate with: AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")(text).
DIVERSE_PROMPTS = {
    "france": [785, 6722, 315, 9625, 374],                        # "The capital of France is"
    "code":   [750, 3974, 6860, 10939, 982, 262, 421, 2422,       # "def quicksort(arr):\n    if
               10939, 8, 2651, 220, 16, 25],                      #   len(arr) <= 1:"
    "story":  [12522, 5193, 264, 882, 11, 304, 264, 2613, 14126,  # "Once upon a time, in a small
               88677, 1948, 1378, 23501, 11, 1052, 12163],        #   village nestled between two
                                                                  #   mountains, there lived"
    "reason": [1249, 11625, 279, 23606, 220, 18, 87, 488, 220,    # "To solve the equation
               22, 284, 220, 17, 17, 11, 582, 1156],              #   3x + 7 = 22, we first"
    "list":   [8420, 525, 4236, 10414, 369, 19429, 9314, 510,     # "Here are five tips for
               16, 13],                                           #   staying healthy:\n1."
}


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def best_time(fn, runs: int):
    """Run fn() `runs` times; return (result_of_fastest_run, min_seconds). Best-of-N kills
    the GPU boost-clock drift a single timing would eat (the G5h min-of-N discipline)."""
    best_dt, best_res = float("inf"), None
    for _ in range(runs):
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        if dt < best_dt:
            best_dt, best_res = dt, res
    return best_res, best_dt


def main() -> int:
    ap = argparse.ArgumentParser(description="S2: speculative-decode speedup + K sweep")
    ap.add_argument("draft", type=Path, help="draft (small, fast) model weights dir")
    ap.add_argument("target", type=Path, help="target (big, accurate) model weights dir")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ks", default="1,2,3,4,6,8", help="comma-separated draft lengths to sweep")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per config (best-of-N)")
    ap.add_argument("--prompt-ids", default=None,
                    help="comma-separated files of space-separated token ids. Default: the built-in "
                         "DIVERSE_PROMPTS set (accept rate is text-dependent, so an honest number "
                         "needs a range, not just the France ref_ids which degenerates into "
                         "repetition and inflates it). Each prompt is swept independently.")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    N = args.max_tokens
    greedy = nicpp.SamplingParams()

    try:
        draft = nicpp.Model(str(args.draft), device=args.device)
        target = nicpp.Model(str(args.target), device=args.device)
    except Exception as e:  # no CUDA build / no GPU / OOM with both resident
        print(f"bench_spec: models unavailable on {args.device} — skipping ({e})")
        return 0

    if args.prompt_ids:
        prompts = [(Path(p).stem, read_ids(Path(p))) for p in args.prompt_ids.split(",")]
    else:
        prompts = list(DIVERSE_PROMPTS.items())

    def sweep(prompt, label):
        """Full K sweep for one prompt; returns (ok, best_speedup)."""
        def plain(model):
            return model.generate(prompt, max_tokens=N, params=greedy, seed=0, eos_id=-1)

        # Warm up: pull device-pool growth + first-kernel/cuBLAS init out of the timed region.
        draft.generate(prompt, max_tokens=8, params=greedy, seed=0, eos_id=-1)
        target.generate(prompt, max_tokens=8, params=greedy, seed=0, eos_id=-1)

        # Plain baselines: target (the number to beat) + draft (to measure the cost ratio r).
        ref, t_tgt = best_time(lambda: plain(target), args.runs)
        _, t_drf = best_time(lambda: plain(draft), args.runs)
        tgt_toks, drf_toks = N / t_tgt, N / t_drf
        r = tgt_toks / drf_toks   # one draft forward as a fraction of one target forward (< 1)

        print(f"\n=== prompt '{label}'  ({len(prompt)} tok) ===")
        print(f"plain target : {tgt_toks:6.1f} tok/s (baseline)   "
              f"plain draft : {drf_toks:6.1f} tok/s   r = t_draft/t_target = {r:.2f}")
        print("   K  accept%  tok/verify   speedup  (pred)   spec tok/s   accept-len % a=0..K")
        print("  " + "-" * 76)

        ok, best = True, 0.0
        for k in ks:
            (out, st), dt = best_time(
                lambda k=k: greedy_speculative(target, draft, prompt, N, k=k, eos_id=-1), args.runs)
            spec_toks = N / dt
            match = out == ref                       # S0 invariant: token-identical to plain greedy
            ok = ok and match
            speedup = spec_toks / tgt_toks
            pred = st.tokens_per_verify / (1 + k * r)  # the roofline estimate above
            hist = st.accept_histogram(k)
            tot = max(1, sum(hist))
            dist = " ".join(f"{100 * h // tot:2d}" for h in hist)
            flag = "" if match else "  <- MISMATCH (speedup void)"
            print(f"  {k:<3} {100 * st.accept_rate:6.1f}  {st.tokens_per_verify:8.2f}   "
                  f"{speedup:5.2f}x  ({pred:4.2f}x)  {spec_toks:8.1f}   [{dist}]{flag}")
            if match:
                best = max(best, speedup)
        return ok, best

    print(f"device={args.device}  max_tokens={N}  runs={args.runs}  prompts={len(prompts)}")
    ok, summary = True, []
    for label, ids in prompts:
        pok, best = sweep(ids, label)
        ok = ok and pok
        summary.append((label, best))

    if len(summary) > 1:
        print("\nsummary (best speedup per prompt — accept rate is text-dependent):")
        for label, best in summary:
            print(f"  {label:10} {best:.2f}x")
        lo, hi = min(b for _, b in summary), max(b for _, b in summary)
        print(f"  range: {lo:.2f}x .. {hi:.2f}x")
    print("\nbench_spec:", "ok" if ok else "FAIL (a K diverged from plain greedy)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
