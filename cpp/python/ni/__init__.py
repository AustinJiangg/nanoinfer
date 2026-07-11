"""nanoinfer's Python orchestration layer over the C++ engine (the vLLM shape).

The package owns everything Python-side of the pybind11 seam:

  engine          locate + import the nicpp extension from a build tree
  scheduler       continuous batching (F7/F8): Scheduler, Request, PrefixCache
  speculative     speculative decoding (S-track): proposers, accept rules, spec loops
  spec_scheduler  speculative decoding folded into continuous batching (S3)
  nit0            the NIT0 tensor format + token-id text dumps (mirrors src/serialize)

Nothing here imports torch — the oracle package (nanoinfer/) stays a separate,
frozen reference. Entry scripts (cpp/tools, cpp/tests) put cpp/python on sys.path
and import from `ni`; inside the package every import is relative. This file
deliberately imports nothing: nit0-only users (the fixture generator, the weight
exporter) must work without a built nicpp module.
"""
