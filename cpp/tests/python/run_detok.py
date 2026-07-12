"""Prove IncrementalDetokenizer is tokenizer-GENERIC (A2).

V1's incremental detok (ni/detok.py) is checked hermetically in run_http_serve.py
against a byte-level stub. A2 adds Llama-3.2's tiktoken-family BPE — different space
handling (byte-level "Ġ") and different multi-byte splits than Qwen — so this gate
runs the SAME sliding-window algorithm against the REAL tokenizers and asserts the
invariant on each: the concatenation of every streamed delta (+ flush) equals the
one-shot decode of all ids, and no mid-stream delta leaks a held-back "�".

The inputs are adversarial for byte-level BPE: emoji (4-byte, routinely split across
tokens), CJK, combining marks, and leading/trailing spaces (the "Ġ" boundary case).

Usage:  python tests/python/run_detok.py [tok1 tok2 ...]
Defaults to the two arches whose tokenizers differ most (Llama tiktoken + Qwen BPE).
Downloads the tokenizers (small), so it's a local-tier gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nanoinfer  # noqa: F401  — sanitizes the IPv6 no_proxy that crashes httpx (hub download)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.detok import IncrementalDetokenizer  # noqa: E402

DEFAULT_TOKENIZERS = ["unsloth/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B"]

# Adversarial strings: multi-byte chars that byte-level BPE routinely splits across
# tokens, plus space-boundary cases ("Ġ"/"▁" handling) that a per-token decode botches.
CASES = [
    "The capital of France is Paris.",
    "héllo, wörld — 🚀→中文!",
    "emoji run: 😀😃😄😁 and flags 🇯🇵🇺🇸",
    "  leading and trailing spaces  ",
    "混合 English 和 中文 with 日本語 テキスト",
    "math: ∑ᵢ αᵢ·xᵢ ≥ 0 ∀i ∈ ℝⁿ",
]


def check_one(tok, name: str) -> bool:
    ok = True
    for s in CASES:
        ids = tok.encode(s)
        # The reference is the one-shot decode of these ids (what detok.text() returns);
        # the streamed deltas must reconstruct EXACTLY that, whatever the string round-trips to.
        ref = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        d = IncrementalDetokenizer(tok)
        deltas = [d.push([i]) for i in ids]      # stream one id at a time (worst case)
        streamed = "".join(deltas) + d.flush()

        leaked = [x for x in deltas if "�" in x]  # no held-back replacement char
        held = sum(x == "" for x in deltas)
        case_ok = streamed == ref and not leaked
        ok &= case_ok
        if not case_ok:
            print(f"  [{name}] FAIL on {s!r}: streamed={streamed!r} ref={ref!r} leaked={leaked}")
        else:
            print(f"  [{name}] ok  ({len(ids)} ids, {held} held) {s[:24]!r}...")
    return ok


def main() -> int:
    from transformers import AutoTokenizer

    names = sys.argv[1:] or DEFAULT_TOKENIZERS
    ok = True
    for name in names:
        print(f"tokenizer: {name}")
        tok = AutoTokenizer.from_pretrained(name)
        ok &= check_one(tok, name.split("/")[-1])
    print("\nrun_detok:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
