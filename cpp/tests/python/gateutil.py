"""Shared helpers for the Python parity gates in this directory.

Every gate compares a scheduled / speculative run against the same reference
path: the F6 single-sequence nicpp generate() (itself parity-locked against
nanoinfer). These helpers keep that reference — and the staggered request set
that exercises continuous batching — defined once, so the gates can't drift
apart on what "standalone" means.

The gates run as scripts (python tests/python/run_serve.py), so this module is
imported as a sibling (the script's directory is sys.path[0]).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.engine import nicpp  # noqa: E402
from ni.scheduler import Request  # noqa: E402


def standalone(model, req: Request) -> list[int]:
    """The reference: one request via the F6 single-sequence generate()."""
    params = nicpp.SamplingParams(
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    return model.generate(req.prompt_ids, max_tokens=req.max_tokens, params=params,
                          seed=req.seed, eos_id=req.eos_id)


def plain_greedy(model, prompt: list[int], max_tokens: int, eos_id: int = -1) -> list[int]:
    """The ultimate reference: plain single-sequence greedy (F6, parity-locked vs
    nanoinfer) — what every speculative configuration must be token-identical to."""
    return model.generate(prompt, max_tokens=max_tokens, params=nicpp.SamplingParams(),
                          seed=0, eos_id=eos_id)


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
