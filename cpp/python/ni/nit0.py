"""The reference-dump file formats shared by the fixtures, the weight exporter,
the reference dumper, and the parity gates.

  * NIT0 tensors — magic "NIT0" | int32 ndim | int32*ndim shape | float32*N data
    (row-major, little-endian). Mirrors cpp/src/serialize.{hpp,cpp}.
  * token-id text — space-separated ints (ref_ids.txt / ref_gen_ids.txt).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def save_bin(path: str | Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        f.write(b"NIT0")
        f.write(struct.pack("<i", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", int(d)))
        f.write(arr.tobytes())


def load_bin(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        if f.read(4) != b"NIT0":
            raise ValueError(f"bad magic in {path}")
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<i", f.read(4))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype="<f4")
    return data.reshape(shape)


def read_ids(path: str | Path) -> list[int]:
    return [int(x) for x in Path(path).read_text().split()]


def write_ids(path: str | Path, ids: list[int]) -> None:
    Path(path).write_text(" ".join(str(i) for i in ids))
