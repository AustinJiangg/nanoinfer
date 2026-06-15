"""Text generation on the C++ engine via the F6 pybind11 binding — the fusion in
one script: HF tokenizes, our own C++ kernels run the forward + sampling loop, HF
decodes. The vLLM shape in miniature (Python orchestration, C++ compute).

Mirrors nanoinfer/generate.py's CLI, but model() and the sampling loop live in C++
(nicpp) instead of torch. Only the tokenizer is HF — the golden rule still holds.

    python tools/generate.py --prompt "The capital of France is" --max-tokens 20
    python tools/generate.py --prompt "Once upon a time" --temperature 0.8 --top-p 0.95 --seed 1234
    python tools/generate.py --prompt "..." --quant q8        # run quantized weights

Greedy (temperature 0, the default) is deterministic and matches nanoinfer
token-for-token. Sampled output does NOT — the C++ RNG differs from torch (see
cpp/src/sampling.hpp) — so a --seed is reproducible only within this engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CPP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CPP / "build"))  # nicpp.*.so

import nicpp  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    p = argparse.ArgumentParser(description="nanoinfer-cpp text generation (F6 binding)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--weights-dir", default=str(CPP / "weights/qwen2.5-0.5b"),
                   help="exported NIT0 weights dir (tools/export_weights.py)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id for the tokenizer only")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--quant", default="fp32", choices=["fp32", "q8", "q4", "q4g"],
                   help="weight-only quantization mode")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 (default) is greedy/deterministic")
    p.add_argument("--top-k", type=int, default=0, help="keep the k highest-logit tokens (0=off)")
    p.add_argument("--top-p", type=float, default=1.0, help="nucleus: keep cumprob >= p (1=off)")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0, help="seed the C++ sampler")
    args = p.parse_args()

    # HF only for tokenization (allowed by the golden rule). Importing nanoinfer
    # first applies its HF Hub proxy shim (IPv6 no_proxy entries crash httpx
    # otherwise — see nanoinfer/_env.py), the same shim export_weights.py relies on.
    import nanoinfer  # noqa: F401
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors=None)["input_ids"]

    model = nicpp.Model(args.weights_dir, nicpp.quant_mode(args.quant))
    params = nicpp.SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    eos_id = model.config.eos_token_id
    gen_ids = model.generate(input_ids, max_tokens=args.max_tokens, params=params,
                             seed=args.seed, eos_id=eos_id)

    print(f"Prompt: {args.prompt}")
    print(f"Completion: {tokenizer.decode(gen_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
