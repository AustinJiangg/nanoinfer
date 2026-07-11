"""Run nanoinfer on a prompt and dump the token ids + reference logits.

The C++ engine reads `ref_ids.txt`, runs its own forward over those ids, and
compares its logits against `ref_logits.bin` (cpp/tests/run_parity.cpp). This is
the C++ stage-C1 analogue of nanoinfer's parity-with-HuggingFace test — and since
nanoinfer is itself HF-verified, matching it transitively matches HF.

Usage:  python dump_reference.py <weights_dir> [prompt] [model_name]
Writes ref_ids.txt and ref_logits.bin into <weights_dir>.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ni.nit0 import save_bin, write_ids  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: dump_reference.py <weights_dir> [prompt] [model_name]")
        raise SystemExit(2)
    out = Path(sys.argv[1])
    prompt = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-0.5B"

    import torch

    from nanoinfer.weights import load_model

    model, tokenizer = load_model(model_name, dtype=torch.float32, device="cpu")
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids)[0]  # [seq, vocab]

    id_list = ids[0].tolist()
    write_ids(out / "ref_ids.txt", id_list)
    save_bin(out / "ref_logits.bin", logits.numpy())

    # Greedy continuation (full-recompute, no EOS stop) for the C++ generate parity
    # test — fixed length so both engines emit the same number of tokens.
    n_gen = 12
    cur = ids.clone()
    gen: list[int] = []
    with torch.no_grad():
        for _ in range(n_gen):
            nxt = int(model(cur)[0, -1].argmax())
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    write_ids(out / "ref_gen_ids.txt", gen)

    print(f"prompt: {prompt!r}")
    print(f"seq={len(id_list)}  next-token argmax={int(logits[-1].argmax())}")
    print(f"greedy {n_gen} tokens: {gen}")
    print(f"wrote ref_ids.txt + ref_logits.bin + ref_gen_ids.txt to {out}")


if __name__ == "__main__":
    main()
