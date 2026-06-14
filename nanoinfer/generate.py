"""Autoregressive generation loop + CLI.

This is the part most inference tutorials hand to `model.generate()`; we do it
ourselves so the loop is visible: forward -> take last-position logits -> pick
next token -> feed it back -> repeat until EOS or max length.

Next-token selection lives in `sampling.py`; this loop just feeds it the
last-position logits. The default `SamplingParams()` is greedy (temperature 0),
so the default stays fully deterministic — pass a populated `SamplingParams` (or
the CLI flags) to sample.

Two paths, selected by `use_cache`:
  - Cached (stage 1, default): prefill the prompt once to fill the KV cache, then
    feed only the new token each step. O(n) per step.
  - Uncached (stage 0): re-feed the whole sequence every step. O(n^2) overall,
    kept for A/B benchmarking and as the parity reference for the cache.
Both share next-token selection and EOS handling, so for the same params and
seed they emit identical tokens.
"""

from __future__ import annotations

import argparse

import torch

from .sampling import SamplingParams, sample_next_token
from .weights import load_model

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 32,
    device: str = "cpu",
    use_cache: bool = True,
    sampling: SamplingParams | None = None,
) -> str:
    """Continue `prompt` for up to `max_tokens` new tokens.

    `sampling` selects the decoding strategy; the default is greedy.
    """
    sampling = sampling or SamplingParams()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Cached path: one cache, sized to hold the prompt plus everything we'll emit.
    cache = model.init_cache(batch=1, max_seq=prompt_len + max_tokens) if use_cache else None

    # Seed the sampler's RNG for reproducible draws; None leaves it default.
    generator = (
        torch.Generator(device=device).manual_seed(sampling.seed)
        if sampling.seed is not None
        else None
    )

    eos_id = tokenizer.eos_token_id
    generated: list[int] = []
    # Running context for the repetition penalty (prompt + generated). Only
    # materialized into a tensor when the penalty is actually on.
    penalize = sampling.repetition_penalty != 1.0
    context_ids: list[int] = input_ids[0].tolist()

    # `cur` is what we feed this step: the full prompt on step 0 (prefill), then a
    # single new token (decode) once the cache holds the past. Without a cache,
    # `cur` stays the whole running sequence and we recompute it every step.
    cur = input_ids
    for _ in range(max_tokens):
        logits = model(cur, cache=cache)[:, -1, :].squeeze(0)  # [vocab]
        ctx = torch.tensor(context_ids, device=device) if penalize else None
        next_id = sample_next_token(logits, sampling, ctx, generator)

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        context_ids.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        cur = next_token if use_cache else torch.cat([cur, next_token], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="nanoinfer text generation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="use the stage-0 full-recompute path instead of the KV cache",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="softmax temperature; 0 (default) is greedy/deterministic",
    )
    parser.add_argument("--top-k", type=int, default=None, help="keep the k highest-logit tokens")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus: keep cumprob >= p")
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0,
        help="penalty (>1) on already-seen tokens; 1.0 is off",
    )
    parser.add_argument("--seed", type=int, default=None, help="seed the sampler for reproducible draws")
    args = parser.parse_args()

    sampling = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)

    print(f"\nPrompt: {args.prompt}")
    completion = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens, device=args.device,
        use_cache=not args.no_cache, sampling=sampling,
    )
    print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
