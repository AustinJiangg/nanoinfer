"""Export a loaded nanoinfer model into NIT0 weights + config.txt for the C++ engine.

Reuses nanoinfer to load the HF checkpoint into our own naming, then dumps every
state-dict tensor as `<name>.bin` plus a `config.txt`. The C++ Model reads this
directory directly, so both engines run the *same* Qwen2.5 tensors and their
logits can be compared (see cpp/tools/dump_reference.py + cpp/tests/run_parity).

Usage:  python export_weights.py <out_dir> [model_name]
The fp32 export is ~2 GB; the directory is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from ni.nit0 import save_bin  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("weights/qwen2.5-0.5b")
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-0.5B"
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from nanoinfer.weights import load_model

    print(f"loading {model_name} ...")
    model, tokenizer = load_model(model_name, dtype=torch.float32, device="cpu")
    cfg = model.cfg
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    with open(out / "config.txt", "w") as f:
        f.write(f"vocab_size {cfg.vocab_size}\n")
        f.write(f"hidden_size {cfg.hidden_size}\n")
        f.write(f"intermediate_size {cfg.intermediate_size}\n")
        f.write(f"num_layers {cfg.num_layers}\n")
        f.write(f"num_attention_heads {cfg.num_attention_heads}\n")
        f.write(f"num_kv_heads {cfg.num_kv_heads}\n")
        f.write(f"head_dim {cfg.head_dim}\n")
        f.write(f"max_position_embeddings {cfg.max_position_embeddings}\n")
        f.write(f"rms_norm_eps {cfg.rms_norm_eps}\n")
        f.write(f"rope_theta {cfg.rope_theta}\n")
        f.write(f"tie_word_embeddings {int(cfg.tie_word_embeddings)}\n")
        f.write(f"eos_token_id {eos_id}\n")

    sd = model.state_dict()
    count = 0
    for name, t in sd.items():
        if name.startswith("rope_"):  # non-persistent buffers; skip if ever present
            continue
        # Tied models alias lm_head to the embedding — don't write it twice (~0.5 GB).
        # The C++ Model reads the tie flag and reuses embed_tokens for the lm_head.
        if name == "lm_head.weight" and cfg.tie_word_embeddings:
            continue
        assert t.dtype == torch.float32, f"{name} is {t.dtype}, expected float32"
        save_bin(out / f"{name}.bin", t.detach().cpu().numpy())
        count += 1
    print(f"wrote {count} weight tensors + config.txt to {out}")


if __name__ == "__main__":
    main()
