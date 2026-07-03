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
    python tests/run_spec.py weights/qwen2.5-0.5b weights/qwen2.5-1.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
sys.path.insert(0, str(CPP / "build"))    # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))   # speculative

import nicpp  # noqa: E402
from speculative import greedy_speculative  # noqa: E402


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


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


def plain_greedy(model, prompt: list[int], max_tokens: int, eos_id: int = -1) -> list[int]:
    """Reference: the F6 single-sequence greedy generate (k-independent)."""
    return model.generate(prompt, max_tokens=max_tokens, params=nicpp.SamplingParams(),
                          seed=0, eos_id=eos_id)


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


def main() -> int:
    d05 = Path(sys.argv[1] if len(sys.argv) > 1 else "weights/qwen2.5-0.5b")
    d15 = Path(sys.argv[2] if len(sys.argv) > 2 else "weights/qwen2.5-1.5b")

    m05 = nicpp.Model(str(d05))
    ok = True

    print("0. foundations (bit-identical):")
    ok &= check_primitive(m05, read_ids(d05 / "ref_ids.txt"))

    print("\nA. draft == target (0.5B/0.5B): expect 100% accept, output == plain greedy")
    ok &= run_case("self  0.5B/0.5B", m05, m05, read_ids(d05 / "ref_ids.txt"),
                   max_tokens=24, ks=(1, 2, 4, 8), expect_full_accept=True)

    if d15.exists():
        print("\nB. draft=0.5B, target=1.5B (the real pair): output == plain 1.5B greedy")
        m15 = nicpp.Model(str(d15))
        ok &= run_case("pair  0.5B->1.5B", m15, m05, read_ids(d15 / "ref_ids.txt"),
                       max_tokens=24, ks=(2, 4, 8))
    else:
        print(f"\nB. skipped — no 1.5B weights at {d15}")

    print("\nrun_spec:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
