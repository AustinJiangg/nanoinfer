"""Autoregressive generation loop + CLI.

Greedy decoding only — this is the part most inference tutorials hand to
`model.generate()`; we do it ourselves so the loop is visible: forward -> take
last-position logits -> pick next token -> feed it back -> repeat until EOS or
max length.

Two paths, selected by `use_cache`:
  - Cached (stage 1, default): prefill the prompt once to fill the KV cache, then
    feed only the new token each step. O(n) per step.
  - Uncached (stage 0): re-feed the whole sequence every step. O(n^2) overall,
    kept for A/B benchmarking and as the parity reference for the cache.
Both share next-token selection and EOS handling, so they emit identical tokens.

Sampling (temperature/top-k/top-p) arrives in stage 2; for now next-token
selection is a plain argmax so output is fully deterministic and easy to verify.
"""

from __future__ import annotations

import argparse

import torch

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
) -> str:
    """Greedily continue `prompt` for up to `max_tokens` new tokens."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Cached path: one cache, sized to hold the prompt plus everything we'll emit.
    cache = model.init_cache(batch=1, max_seq=prompt_len + max_tokens) if use_cache else None

    eos_id = tokenizer.eos_token_id
    generated: list[int] = []

    # `cur` is what we feed this step: the full prompt on step 0 (prefill), then a
    # single new token (decode) once the cache holds the past. Without a cache,
    # `cur` stays the whole running sequence and we recompute it every step.
    cur = input_ids
    for _ in range(max_tokens):
        logits = model(cur, cache=cache)         # [1, seq, vocab]
        next_id = int(logits[:, -1, :].argmax(dim=-1))  # greedy: last position

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        next_token = torch.tensor([[next_id]], device=device)
        cur = next_token if use_cache else torch.cat([cur, next_token], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="nanoinfer greedy generation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="use the stage-0 full-recompute path instead of the KV cache",
    )
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)

    print(f"\nPrompt: {args.prompt}")
    completion = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens, device=args.device,
        use_cache=not args.no_cache,
    )
    print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
