"""Serve the C++ engine over HTTP — the serving last mile (V-track).

The full stack, top to bottom: this CLI wires an HF tokenizer (text in/out,
golden rule: tokenization only) to `ni.server.HttpServer` (hand-rolled HTTP/1.1
+ SSE) over `ni.async_engine.AsyncEngine` (the asyncio bridge) over the
continuous-batching `Scheduler` — or the speculative `SpecScheduler` — over the
`nicpp` kernels, CPU or CUDA.

    # plain continuous batching, paged KV, CPU
    python tools/serve_http.py --block-size 16 --num-blocks 256

    # CUDA + prefix sharing
    NI_BUILD_DIR=cpp/build-cuda python tools/serve_http.py --device cuda \
        --block-size 16 --num-blocks 512 --prefix-sharing

    # speculative decoding: 0.5B draft proposing for a 1.5B target
    python tools/serve_http.py --weights-dir weights/qwen2.5-1.5b \
        --spec draft --draft-weights-dir weights/qwen2.5-0.5b

Then, from anywhere:

    curl -s localhost:8000/v1/completions \
        -d '{"prompt": "The capital of France is", "max_tokens": 24}'
    curl -sN localhost:8000/v1/completions \
        -d '{"prompt": "Once upon a time", "max_tokens": 48, "stream": true}'
    curl -s localhost:8000/metrics
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from ni.async_engine import AsyncEngine  # noqa: E402
from ni.engine import default_weights_dir, nicpp  # noqa: E402
from ni.scheduler import Scheduler  # noqa: E402
from ni.server import HttpServer  # noqa: E402
from ni.spec_scheduler import SpecScheduler  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


async def amain(args) -> None:
    if args.no_tokenizer:
        tokenizer = None
    else:
        # HF only for tokenization (golden rule). Import nanoinfer first for its
        # HF Hub proxy shim — see tools/generate.py / nanoinfer/_env.py.
        import nanoinfer  # noqa: F401
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = nicpp.Model(args.weights_dir, nicpp.quant_mode(args.quant),
                        device=args.device)
    kw = dict(max_batch=args.max_batch, block_size=args.block_size,
              num_blocks=args.num_blocks, prefix_sharing=args.prefix_sharing)
    if args.spec == "off":
        sched = Scheduler(model, batched=True, **kw)
    else:
        draft = (nicpp.Model(args.draft_weights_dir, device=args.device)
                 if args.spec == "draft" else None)
        sched = SpecScheduler(model, draft=draft, batched=True, **kw)

    engine = AsyncEngine(sched)
    await engine.start()
    server = HttpServer(engine, tokenizer=tokenizer)
    port = await server.start(args.host, args.port)

    cache = (f"paged(bs={args.block_size}, {args.num_blocks} blocks)"
             if args.block_size else "contiguous")
    spec = f", spec={args.spec}" if args.spec != "off" else ""
    print(f"nanoinfer-cpp serving on http://{args.host}:{port}  "
          f"[{args.device}, {args.quant}, max_batch={args.max_batch}, {cache}{spec}, "
          f"tokenizer={'yes' if tokenizer else 'ids-only'}]")
    print("  POST /v1/completions   GET /metrics   GET /healthz   (Ctrl-C to stop)")
    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()
        await engine.stop()


def main() -> None:
    p = argparse.ArgumentParser(description="nanoinfer-cpp HTTP serving (V-track)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--weights-dir", default=str(default_weights_dir()))
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF id for the tokenizer only")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="ids-in/ids-out API; no HF download needed")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--quant", default="fp32", choices=["fp32", "q8", "q4", "q4g", "w8a8"])
    p.add_argument("--max-batch", type=int, default=8, help="concurrent sequences")
    p.add_argument("--block-size", type=int, default=0,
                   help="paged KV cache (F8b) with blocks of this size; 0 = contiguous")
    p.add_argument("--num-blocks", type=int, default=256,
                   help="paged pool size in blocks (used when --block-size > 0)")
    p.add_argument("--prefix-sharing", action="store_true",
                   help="reuse shared prompt-prefix KV across requests (needs --block-size)")
    p.add_argument("--spec", default="off", choices=["off", "lookup", "draft"],
                   help="speculative decoding: prompt-lookup or a draft model")
    p.add_argument("--draft-weights-dir",
                   default=str(default_weights_dir("qwen2.5-0.5b")),
                   help="NIT0 export for the draft model (--spec draft)")
    args = p.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
