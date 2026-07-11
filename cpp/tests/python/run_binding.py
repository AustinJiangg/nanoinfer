"""F6 parity check — drive the C++ core through the pybind11 binding and confirm it
reproduces nanoinfer's reference dumps. Same MATCH bar as run_parity.cpp /
run_generate.cpp, but from Python: the fusion (Python orchestration over our own
C++ kernels), with the binding itself as the thing under test.

It loads `nicpp` from the build dir, runs forward over ref_ids.txt and compares the
logits to ref_logits.bin, then greedy-generates and compares to ref_gen_ids.txt, and
checks the cached forward path agrees with the uncached one.

Run after building the module and dumping a reference (weights are already exported):
    cmake --build build -j
    python tools/dump_reference.py weights/qwen2.5-0.5b "The capital of France is"
    python tests/python/run_binding.py weights/qwen2.5-0.5b
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import load_bin, read_ids  # noqa: E402


def main() -> int:
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights_dir()

    model = nicpp.Model(str(wd))
    cfg = model.config
    ids = read_ids(wd / "ref_ids.txt")
    print(f"model: {cfg.num_layers} layers, vocab {cfg.vocab_size}, "
          f"hidden {cfg.hidden_size}; seq {len(ids)}")

    # --- forward parity vs nanoinfer's logit dump (the C1/run_parity bar) ---
    logits = model.forward(ids)  # [seq, vocab] numpy float32
    ref = load_bin(wd / "ref_logits.bin")
    if logits.shape != ref.shape:
        print(f"FAIL: logit shape {logits.shape} vs {ref.shape}")
        return 1
    maxd = float(np.max(np.abs(logits - ref)))
    argmax_mism = int(np.sum(logits.argmax(-1) != ref.argmax(-1)))
    next_cpp, next_ref = int(logits[-1].argmax()), int(ref[-1].argmax())
    print(f"forward: max|diff|={maxd:.3g}  per-position argmax mismatches="
          f"{argmax_mism}/{len(ids)}  next-token cpp={next_cpp} ref={next_ref}")
    # Correctness is argmax agreement; the abs-diff bound guards float drift (same
    # threshold as run_parity.cpp).
    forward_ok = argmax_mism == 0 and next_cpp == next_ref and maxd < 0.1

    # --- greedy generation parity vs nanoinfer's continuation (the C2 bar) ---
    ref_gen = read_ids(wd / "ref_gen_ids.txt")
    got = model.generate(ids, max_tokens=len(ref_gen))  # default greedy, no EOS stop
    print(f"greedy ref: {ref_gen}")
    print(f"greedy cpp: {got}")
    gen_ok = got == ref_gen

    # --- cached forward path agrees with the uncached one (the C3 bar) ---
    cache = model.make_cache(len(ids) + 4)
    cached = model.forward(ids, cache)
    cache_ok = (cache.length == len(ids)
                and int(cached[-1].argmax()) == next_cpp
                and float(np.max(np.abs(cached - logits))) < 1e-4)

    ok = forward_ok and gen_ok and cache_ok
    print(f"forward parity: {'MATCH' if forward_ok else 'MISMATCH'}")
    print(f"greedy parity : {'MATCH' if gen_ok else 'MISMATCH'}")
    print(f"cache parity  : {'MATCH' if cache_ok else 'MISMATCH'}")
    print("run_binding:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
