"""Autoregressive generation loop + CLI.

Stage 0: greedy decoding only, full recompute each step. This is the part most
inference tutorials hand to `model.generate()` — we do it ourselves so the loop
is visible: forward -> take last-position logits -> pick next token -> append ->
repeat until EOS or max length.

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
) -> str:
    """Greedily continue `prompt` for up to `max_tokens` new tokens."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    eos_id = tokenizer.eos_token_id
    generated: list[int] = []

    for _ in range(max_tokens):
        logits = model(input_ids)               # [1, seq, vocab]
        next_logits = logits[:, -1, :]           # [1, vocab] — only last position
        next_id = int(next_logits.argmax(dim=-1))  # greedy

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(next_id)
        # Append and feed the whole thing back (full recompute — stage 1 fixes this).
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="nanoinfer greedy generation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, device=args.device)

    print(f"\nPrompt: {args.prompt}")
    completion = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens, device=args.device,
    )
    print(f"Completion: {completion}")


if __name__ == "__main__":
    main()
