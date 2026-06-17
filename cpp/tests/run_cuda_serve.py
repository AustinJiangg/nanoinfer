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

    cmake --build build -j -DNI_CUDA=ON     # builds nicpp against the CUDA core
    python tests/run_cuda_serve.py weights/qwen2.5-0.5b
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPP = HERE.parent
sys.path.insert(0, str(CPP / "build"))   # nicpp.*.so
sys.path.insert(0, str(CPP / "python"))  # scheduler

import nicpp  # noqa: E402
from scheduler import Request, Scheduler  # noqa: E402


def read_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().split()]


def standalone(model, r: Request) -> list[int]:
    params = nicpp.SamplingParams(temperature=r.temperature, top_k=r.top_k, top_p=r.top_p,
                                  repetition_penalty=r.repetition_penalty)
    return model.generate(r.prompt_ids, max_tokens=r.max_tokens, params=params, seed=r.seed,
                          eos_id=r.eos_id)


def main() -> int:
    wd = Path(sys.argv[1] if len(sys.argv) > 1 else "weights/qwen2.5-0.5b")
    try:
        model = nicpp.Model(str(wd), device="cuda")
    except Exception as e:  # no CUDA build / no GPU
        print(f"run_cuda_serve: CUDA model unavailable — skipping ({e})")
        return 0
    base = read_ids(wd / "ref_ids.txt")
    print(f"model: {model.config.num_layers} layers, vocab {model.config.vocab_size} (CUDA)")

    # Distinct prompts + lengths so sequences finish at different steps (that staggering
    # is what exercises continuous batching). r4 carries a repetition penalty.
    reqs = [
        Request("r0", base, max_tokens=12),
        Request("r1", base[:3], max_tokens=6),
        Request("r2", base + [12095], max_tokens=8),
        Request("r3", base[:2], max_tokens=4),
        Request("r4", base, max_tokens=10, repetition_penalty=1.3),
        Request("r5", base[1:4], max_tokens=7),
    ]
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
