"""Generate numpy reference fixtures for the C++ ops parity test.

Writes inputs and numpy-computed expected outputs in the little-endian "NIT0"
format (see cpp/src/serialize.hpp) that the C++ test reads back. The reference
math here is the source of truth the C++ ops must reproduce — the C++ stage C0
analogue of nanoinfer parity-testing against HuggingFace.

Usage:  python gen_fixtures.py <out_dir>
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

RMS_EPS = 1e-6  # must match kRmsEps in cpp/tests/ops_parity.cpp


def save_bin(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"NIT0")
        f.write(struct.pack("<i", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", int(d)))
        f.write(arr.tobytes())


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * w


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: gen_fixtures.py <out_dir>")
        raise SystemExit(2)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)  # fixed seed -> reproducible fixtures

    # matmul: [m, k] x [k, n]
    a = rng.standard_normal((7, 5)).astype(np.float32)
    b = rng.standard_normal((5, 3)).astype(np.float32)
    save_bin(out / "matmul_a.bin", a)
    save_bin(out / "matmul_b.bin", b)
    save_bin(out / "matmul_expected.bin", a.astype(np.float64) @ b.astype(np.float64))

    # rmsnorm over the last dim of a [rows, d] tensor
    x = rng.standard_normal((4, 16)).astype(np.float32)
    w = rng.standard_normal((16,)).astype(np.float32)
    save_bin(out / "rmsnorm_x.bin", x)
    save_bin(out / "rmsnorm_weight.bin", w)
    save_bin(out / "rmsnorm_expected.bin", rmsnorm(x, w, RMS_EPS))

    # softmax over the last dim, including a large-magnitude row (stability)
    sx = rng.standard_normal((4, 10)).astype(np.float32)
    sx[0] *= 50.0
    save_bin(out / "softmax_x.bin", sx)
    save_bin(out / "softmax_expected.bin", softmax(sx))

    # add
    pa = rng.standard_normal((3, 4)).astype(np.float32)
    pb = rng.standard_normal((3, 4)).astype(np.float32)
    save_bin(out / "add_a.bin", pa)
    save_bin(out / "add_b.bin", pb)
    save_bin(out / "add_expected.bin", pa + pb)

    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
