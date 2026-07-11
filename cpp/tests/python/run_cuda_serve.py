"""G4c — the Python continuous-batching scheduler running on the GPU (the vLLM shape:
Python orchestration over our own CUDA kernels). Two checks:

  1. Parity: the scheduler's interleaved output must equal each request run standalone
     (model.generate), at several max_batch sizes — greedy is deterministic and each
     sequence has its own device KV cache, so batching/queueing must not change a token.
  2. Throughput: fixed total work (R requests x T tokens) run at increasing max_batch;
     tok/s should climb as continuous batching packs more sequences into each
     forward_batch (the batched projection GEMM is one big matmul, the GPU's favorite).

Contiguous + batched path only (block_size=0). The paged scheduler on GPU needs the
device BlockPool / PagedKVCache bound into nicpp — a separate step.

    cmake -S . -B build-cuda -DNI_CUDA=ON && cmake --build build-cuda -j   # nicpp on the CUDA core
    python tests/python/run_cuda_serve.py weights/qwen2.5-0.5b
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.nit0 import read_ids  # noqa: E402
from ni.scheduler import Request, Scheduler  # noqa: E402

from gateutil import build_requests, standalone  # noqa: E402  (sibling: the shared reference path)


def main() -> int:
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else default_weights_dir()
    try:
        model = nicpp.Model(str(wd), device="cuda")
    except Exception as e:  # no CUDA build / no GPU
        print(f"run_cuda_serve: CUDA model unavailable — skipping ({e})")
        return 0
    base = read_ids(wd / "ref_ids.txt")
    print(f"model: {model.config.num_layers} layers, vocab {model.config.vocab_size} (CUDA)")

    reqs = build_requests(base)
    ref = {r.request_id: standalone(model, r) for r in reqs}

    # --- 1. Parity: scheduler interleaving == standalone, across batch sizes ---
    ok = True
    for mb in (1, 2, 4, 8):
        sched = Scheduler(model, max_batch=mb, batched=True, block_size=0)
        for r in reqs:
            sched.add(r)
        out = sched.run()
        match = all(out[r.request_id] == ref[r.request_id] for r in reqs)
        ok = ok and match
        print(f"batched mb={mb:<2}: steps={sched.steps} peak_batch={sched.peak_batch} "
              f"-> {'MATCH' if match else 'MISMATCH'}")

    # --- 1b. Paged + prefix-sharing parity (GPU). block_size=4 so the prompt spans a
    # full block: r0/r2/r4 all start with `base`, so a prefix block is actually shared. ---
    for label, kw in [("paged", dict(block_size=4, num_blocks=256)),
                      ("paged+prefix", dict(block_size=4, num_blocks=256, prefix_sharing=True))]:
        sched = Scheduler(model, max_batch=4, batched=True, **kw)
        for r in reqs:
            sched.add(r)
        out = sched.run()
        match = all(out[r.request_id] == ref[r.request_id] for r in reqs)
        ok = ok and match
        extra = f" shared_prefill={sched.shared_prefill_tokens}" if "prefix" in label else ""
        sched.clear_prefix_cache()
        print(f"{label:<13}: steps={sched.steps} -> {'MATCH' if match else 'MISMATCH'}{extra}")

    # --- 2. Throughput: fixed work (R x T) at increasing max_batch ---
    R, T = 32, 32
    work = [Request(f"w{i}", base, max_tokens=T) for i in range(R)]
    print(f"\ncontinuous-batching throughput ({R} requests x {T} tokens):")
    print("  max_batch   tok/s")
    for mb in (1, 4, 8, 16, 32):
        sched = Scheduler(model, max_batch=mb, batched=True, block_size=0)
        for r in work:
            sched.add(r)
        t0 = time.perf_counter()
        sched.run()
        dt = time.perf_counter() - t0
        print(f"  {mb:<9} {R * T / dt:6.1f}")

    print("\nrun_cuda_serve:", "ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
