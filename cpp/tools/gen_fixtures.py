"""Generate numpy reference fixtures for the C++ ops parity test.

Writes inputs and numpy-computed expected outputs in the little-endian "NIT0"
format (see cpp/src/serialize.hpp) that the C++ test reads back. The reference
math here is the source of truth the C++ ops must reproduce — the C++ stage C0
analogue of nanoinfer parity-testing against HuggingFace.

Usage:  python gen_fixtures.py <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from nit0 import save_bin

RMS_EPS = 1e-6  # must match kRmsEps in cpp/tests/ops_parity.cpp


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)) * w


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def build_rope_cache(seq_len: int, head_dim: int, theta: float):
    # neox / half-split convention (matches nanoinfer.layers.build_rope_cache).
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))  # [head_dim/2]
    freqs = np.outer(np.arange(seq_len), inv_freq)                      # [seq, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)                       # [seq, head_dim]
    return np.cos(emb), np.sin(emb)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    # x: [heads, seq, head_dim]; cos/sin: [seq, head_dim] broadcast over heads.
    half = x.shape[-1] // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)  # rotate_half
    return x * cos + rot * sin


def split_heads(x: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    s = x.shape[0]
    return x.reshape(s, n_heads, head_dim).transpose(1, 0, 2)  # [H, S, D]


def merge_heads(x: np.ndarray) -> np.ndarray:
    h, s, d = x.shape
    return x.transpose(1, 0, 2).reshape(s, h * d)  # [S, H*D]


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    return np.repeat(x, n_rep, axis=0)  # [H*n_rep, S, D]


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool) -> np.ndarray:
    h, sq, d = q.shape
    sk = k.shape[1]
    scale = 1.0 / np.sqrt(d)
    qd, kd, vd = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
    out = np.zeros((h, sq, d), dtype=np.float64)
    for hi in range(h):
        scores = (qd[hi] @ kd[hi].T) * scale  # [sq, sk]
        if causal:
            mask = np.triu(np.ones((sq, sk), dtype=bool), k=1)  # j > i
            scores = np.where(mask, -np.inf, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(axis=-1, keepdims=True)
        out[hi] = attn @ vd[hi]
    return out


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

    # linear: y = x @ w.T + bias  (nn.Linear weight layout [out, in])
    lx = rng.standard_normal((4, 8)).astype(np.float32)
    lw = rng.standard_normal((6, 8)).astype(np.float32)
    lb = rng.standard_normal((6,)).astype(np.float32)
    save_bin(out / "linear_x.bin", lx)
    save_bin(out / "linear_w.bin", lw)
    save_bin(out / "linear_bias.bin", lb)
    save_bin(out / "linear_expected.bin", lx.astype(np.float64) @ lw.astype(np.float64).T + lb)

    # silu: x * sigmoid(x)
    ux = rng.standard_normal((4, 8)).astype(np.float32)
    save_bin(out / "silu_x.bin", ux)
    save_bin(out / "silu_expected.bin", ux / (1.0 + np.exp(-ux.astype(np.float64))))

    # mul: elementwise
    ma = rng.standard_normal((3, 4)).astype(np.float32)
    mb = rng.standard_normal((3, 4)).astype(np.float32)
    save_bin(out / "mul_a.bin", ma)
    save_bin(out / "mul_b.bin", mb)
    save_bin(out / "mul_expected.bin", ma * mb)

    # embedding: table[ids]; ids fixed and mirrored in ops_parity.cpp
    etable = rng.standard_normal((20, 5)).astype(np.float32)
    eids = [3, 0, 19, 7]
    save_bin(out / "embedding_table.bin", etable)
    save_bin(out / "embedding_expected.bin", etable[eids])

    # rope: cache (seq=4, head_dim=8, theta=10000) + apply to q [heads=2, seq=4, dim=8].
    # ROPE_SEQ/HEAD_DIM/THETA mirror the constants in ops_parity.cpp.
    rcos, rsin = build_rope_cache(4, 8, 10000.0)
    save_bin(out / "rope_cos.bin", rcos)
    save_bin(out / "rope_sin.bin", rsin)
    rq = rng.standard_normal((2, 4, 8)).astype(np.float32)
    save_bin(out / "rope_q.bin", rq)
    save_bin(out / "rope_applied_expected.bin", apply_rope(rq, rcos, rsin))

    # attention blocks. Shapes mirror the constants in ops_parity.cpp.
    # split/merge heads: [S=3, H=2, D=4]
    sh_x = rng.standard_normal((3, 8)).astype(np.float32)
    save_bin(out / "split_x.bin", sh_x)
    save_bin(out / "split_expected.bin", split_heads(sh_x, 2, 4))
    mh_x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    save_bin(out / "merge_x.bin", mh_x)
    save_bin(out / "merge_expected.bin", merge_heads(mh_x))

    # repeat_kv: [H=2, S=3, D=4], n_rep=3
    rk_x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    save_bin(out / "repeatkv_x.bin", rk_x)
    save_bin(out / "repeatkv_expected.bin", repeat_kv(rk_x, 3))

    # causal attention: q/k/v [H=2, S=4, D=4]
    aq = rng.standard_normal((2, 4, 4)).astype(np.float32)
    ak = rng.standard_normal((2, 4, 4)).astype(np.float32)
    av = rng.standard_normal((2, 4, 4)).astype(np.float32)
    save_bin(out / "attn_q.bin", aq)
    save_bin(out / "attn_k.bin", ak)
    save_bin(out / "attn_v.bin", av)
    save_bin(out / "attn_expected.bin", attention(aq, ak, av, causal=True))

    print(f"wrote fixtures to {out}")


if __name__ == "__main__":
    main()
