"""Serve several prompts at once on the C++ engine via the F7 continuous-batching
scheduler — the multi-prompt counterpart of tools/generate.py. HF tokenizes each
prompt, the scheduler interleaves them over our C++ kernels (one KV cache per
sequence, dynamic admit/evict), HF decodes the results.

    python tools/serve.py --max-batch 2 \
        --prompt "The capital of France is" \
        --prompt "Once upon a time" \
        --prompt "Water boils at"

Greedy (the default) is deterministic and each completion is identical to running
that prompt alone (tests/run_serve.py). With --max-batch < #prompts the extra ones
queue and are admitted as others finish; raise it to run them all concurrently.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "build"))   # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))  # scheduler

import nicpp  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    p = argparse.ArgumentParser(description="nanoinfer-cpp multi-prompt serving (F7)")
    p.add_argument("--prompt", action="append", required=True, dest="prompts",
                   help="repeat to enqueue multiple prompts")
    p.add_argument("--weights-dir", default=str(CPP / "weights/qwen2.5-0.5b"))
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id for the tokenizer only")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--max-batch", type=int, default=4, help="concurrent sequences")
    p.add_argument("--quant", default="fp32", choices=["fp32", "q8", "q4", "q4g"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # HF only for tokenization (golden rule). Import nanoinfer first for its HF Hub
    # proxy shim — see tools/generate.py / nanoinfer/_env.py.
    import nanoinfer  # noqa: F401
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = nicpp.Model(args.weights_dir, nicpp.quant_mode(args.quant))
    eos_id = model.config.eos_token_id

    sched = Scheduler(model, max_batch=args.max_batch)
    prompts = {}
    for i, text in enumerate(args.prompts):
        rid = f"req{i}"
        ids = tokenizer(text, return_tensors=None)["input_ids"]
        prompts[rid] = text
        sched.add(Request(
            rid, ids, max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, seed=args.seed, eos_id=eos_id,
        ))

    t0 = time.perf_counter()
    out = sched.run()
    dt = time.perf_counter() - t0

    n_tok = sum(len(v) for v in out.values())
    print(f"served {len(prompts)} prompts, max_batch={args.max_batch}: "
          f"{sched.steps} steps, peak_batch={sched.peak_batch}, "
          f"{n_tok} tokens in {dt:.2f}s ({n_tok / dt:.1f} tok/s)\n")
    for rid, text in prompts.items():
        print(f"Prompt: {text}")
        print(f"Completion: {tokenizer.decode(out[rid], skip_special_tokens=True)}\n")


if __name__ == "__main__":
    main()
