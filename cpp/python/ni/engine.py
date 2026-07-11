"""Locate and import the C++ engine's pybind11 module (`nicpp`).

The extension is built by CMake and lands in the build tree, not the source tree
(deliberate: during development the build dir IS the artifact; there's no wheel).
Every Python entry point used to hand-roll `sys.path.insert(0, .../"build")` with
the directory name hardcoded — this module is now the one place that knows where
the engine lives.

Search order:
  1. $NI_BUILD_DIR      — explicit override (point it at any build tree)
  2. cpp/build          — the historical default (kept first so behavior is unchanged)
  3. cpp/build-*        — any other configured tree (build-cuda, ...)

Usage:  from ni.engine import nicpp
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent.parent.parent  # .../cpp


def default_weights_dir(model: str = "qwen2.5-0.5b") -> Path:
    """The conventional NIT0 export location for `model`: cpp/weights/<model>.
    (tools/export_weights.py writes here; CMake's `weights` ctest suite and the
    Python gates read it.) Absolute, so callers are CWD-independent."""
    return CPP_DIR / "weights" / model


def _candidates() -> list[Path]:
    env = os.environ.get("NI_BUILD_DIR")
    if env:
        return [Path(env)]
    dirs = [CPP_DIR / "build"]
    dirs += sorted(d for d in CPP_DIR.glob("build-*") if d.is_dir())
    return dirs


def _import_nicpp():
    tried = []
    for d in _candidates():
        tried.append(str(d))
        # The module file is nicpp.<abi-tag>.so (or .pyd); a bare glob is enough to
        # know this tree built the binding.
        if d.is_dir() and any(d.glob("nicpp.*")):
            sys.path.insert(0, str(d))
            return importlib.import_module("nicpp")
    raise ImportError(
        "nicpp extension not found (looked in: " + ", ".join(tried) + "). "
        "Build it first:  cmake -S cpp -B cpp/build && cmake --build cpp/build -j"
        "  — or point NI_BUILD_DIR at a configured build tree.")


nicpp = _import_nicpp()
