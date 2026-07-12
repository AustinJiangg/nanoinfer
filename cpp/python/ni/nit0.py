"""The reference-dump file formats shared by the fixtures, the weight exporter,
the reference dumper, and the parity gates.

  * NIT0 tensors — magic "NIT0" | int32 ndim | int32*ndim shape | float32*N data
    (row-major, little-endian). Mirrors cpp/src/serialize.{hpp,cpp}.
  * NIT1 tensors (B1) — magic "NIT1" | int32 dtype (0=f32, 1=bf16) | int32 ndim |
    int32*ndim shape | payload (f32*N or bf16-as-u16*N). A bf16 payload halves the
    file; the loaders (here and in C++) inflate it back to fp32 (bits << 16 —
    exact, bf16 is fp32's top half), so everything downstream stays fp32.
  * token-id text — space-separated ints (ref_ids.txt / ref_gen_ids.txt).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Round an fp32 array to bf16, returned as raw uint16 bits (round-to-nearest-even).

    The RNE trick: add 0x7FFF plus the LSB of the surviving mantissa half, then
    truncate — ties round to the even candidate. Exact (a no-op round) whenever the
    input is already bf16-representable, which every weight of a bf16-shipped
    checkpoint is after our lossless fp32 upcast at export time.
    """
    bits = np.ascontiguousarray(arr, dtype="<f4").view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Inflate raw bf16 bits (uint16) to fp32 — exact (bf16 is fp32's top 16 bits)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def save_bin(path: str | Path, arr: np.ndarray, dtype: str = "f32") -> None:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    with open(path, "wb") as f:
        if dtype == "bf16":  # NIT1: dtype-tagged header, u16 payload (half the file)
            f.write(b"NIT1")
            f.write(struct.pack("<i", 1))
        elif dtype == "f32":  # NIT0, byte-identical to every pre-B1 export
            f.write(b"NIT0")
        else:
            raise ValueError(f"save_bin: unknown dtype {dtype!r}")
        f.write(struct.pack("<i", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", int(d)))
        f.write(f32_to_bf16_bits(arr).tobytes() if dtype == "bf16" else arr.tobytes())


def load_bin(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.read(4)
        dtype = 0  # NIT0 is implicitly fp32
        if magic == b"NIT1":
            dtype = struct.unpack("<i", f.read(4))[0]
            if dtype not in (0, 1):
                raise ValueError(f"unknown dtype {dtype} in {path}")
        elif magic != b"NIT0":
            raise ValueError(f"bad magic in {path}")
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<i", f.read(4))[0] for _ in range(ndim)]
        if dtype == 1:
            data = bf16_bits_to_f32(np.frombuffer(f.read(), dtype="<u2"))
        else:
            data = np.frombuffer(f.read(), dtype="<f4")
    return data.reshape(shape)


def read_ids(path: str | Path) -> list[int]:
    return [int(x) for x in Path(path).read_text().split()]


def write_ids(path: str | Path, ids: list[int]) -> None:
    Path(path).write_text(" ".join(str(i) for i in ids))
