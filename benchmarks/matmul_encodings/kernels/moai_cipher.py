"""MOAI CKKS matmul kernels.

Ports of:
- Algorithm 3 (Col x Col -> Diag, BSGS) from matmult/moai_cipher.go
- Algorithm 4 (Diag x Col -> Col, BSGS) [TODO; this commit covers only Alg 3]

The Col x Col variant is the natural fit for Q . K^T in transformer
self-attention. Q and K are both column-packed; the output is
diag-packed and feeds Algorithm 4 (the next matmul, by V) without
re-encoding.

BSGS structure (matmult/moai_plain.go moai_col_col_bsgs):
    b = ceil(sqrt(m)), g = ceil(m/b)
    baby steps: rotate K[i] by r*stride for r in [1, b)
    giant steps: for alpha in [0, g):
        rotate each Q[i] by (m - alpha*b) * stride
        for r in [0, b):
            j = alpha*b + r; if j >= m: break
            partial = sum_i Q_rot[i] * K_baby[i][r]
            rolled  = Rot(partial, alpha*b * stride) if alpha != 0
            out[j] += rolled

This first port uses eager relinearization (mul_rl) like the BMM-I /
THOR kernels in this branch; lazy-relin is a follow-up optimization.

Hoisting is applied to the baby-step rotations (one ctx.rot_batch per
i across (b-1) shifts). The giant-step rotation is a single rotation
per (alpha, i) pair so there is no hoisting opportunity there.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..context import Context
from ..plaintext.moai_plain import (
    interleaved_column_pack,
    interleaved_diag_unpack,
)


def _tile_to_slots(vec: np.ndarray, slots: int) -> np.ndarray:
    """Tile cyclically to fill slot count -- see kernels/thor_cipher.py."""
    L = vec.size
    assert slots % L == 0, f"slots {slots} must be a multiple of encoded length {L}"
    return np.tile(vec, slots // L)


def _int_ceil_sqrt(n: int) -> int:
    return int(math.isqrt(n - 1) + 1) if n > 1 else 1


def encrypt_col_packed(
    ctx: Context,
    Xs: np.ndarray,
    n_he: int,
) -> list[int]:
    """Pack n_batch column-packed (m x d) matrices into d ciphertexts."""
    packed = interleaved_column_pack(Xs, n_he=n_he)  # (d, n_he)
    return [ctx.encrypt(_tile_to_slots(packed[i], ctx.slots)) for i in range(packed.shape[0])]


def moai_col_col_bsgs_he(
    ctx: Context,
    Q_cts: list[int],
    K_cts: list[int],
    m: int,
    d_prime: int,
    rot_stride: int,
    n_he: int,
) -> list[int]:
    """BSGS Algorithm 3 on encrypted Q, K. Returns m output ciphertexts.

    `n_he` is the algorithm's logical wrap length (= n_batch * m typically),
    NOT the slot count. The encoded vector is tiled across the slot count
    so cyclic rotations behave like length-n_he rotations.
    """
    assert len(Q_cts) == d_prime
    assert len(K_cts) == d_prime
    b = _int_ceil_sqrt(m)
    g = (m + b - 1) // b

    # Baby steps: for each i, hoisted batch rotate K[i] by r*stride, r in [1, b).
    baby_shifts = [r * rot_stride for r in range(1, b)]
    beta: list[list[int]] = []
    for i in range(d_prime):
        row: list[int] = [K_cts[i]]  # r=0 identity
        if baby_shifts:
            rotated = ctx.rot_batch(K_cts[i], baby_shifts)
            row.extend(rotated)
        beta.append(row)

    out_cts: list[int | None] = [None] * m

    for alpha in range(g):
        shift = (alpha * b) % m
        q_rot_shift = ((m - shift) * rot_stride) % n_he

        # Giant-step rotation on each Q[i]. One rotation per i (no hoisting:
        # each Q[i] is rotated by the same single delta — different inputs).
        q_rot: list[int] = []
        for i in range(d_prime):
            if q_rot_shift == 0:
                q_rot.append(Q_cts[i])
            else:
                q_rot.append(ctx.rot(Q_cts[i], q_rot_shift))

        for r in range(b):
            j = alpha * b + r
            if j >= m:
                break

            # Inner accumulator: partial = sum_i Q_rot[i] * K_baby[i][r]
            partial = ctx.mul_rl(q_rot[0], beta[0][r])
            partial = ctx.rescale(partial)
            for i in range(1, d_prime):
                prod = ctx.mul_rl(q_rot[i], beta[i][r])
                prod = ctx.rescale(prod)
                partial = ctx.add(partial, prod)

            final_shift = (shift * rot_stride) % n_he
            if final_shift != 0:
                partial = ctx.rot(partial, final_shift)

            if out_cts[j] is None:
                out_cts[j] = partial
            else:
                out_cts[j] = ctx.add(out_cts[j], partial)

    return out_cts  # type: ignore[return-value]


def moai_col_col_bsgs_end_to_end(
    ctx: Context,
    Qs: np.ndarray,
    Ks: np.ndarray,
) -> np.ndarray:
    """Encrypt n_batch (m x d) Q and K matrices, run Alg 3, decrypt + unpack.

    Returns shape (n_batch, m, m) with result[s] = Qs[s] @ Ks[s].T.
    """
    n_batch, m, d_prime = Qs.shape
    n_batch2, m2, d_prime2 = Ks.shape
    assert (n_batch, m, d_prime) == (n_batch2, m2, d_prime2)
    n_he = n_batch * m
    rot_stride = n_batch

    Q_cts = encrypt_col_packed(ctx, Qs, n_he=n_he)
    K_cts = encrypt_col_packed(ctx, Ks, n_he=n_he)
    out_cts = moai_col_col_bsgs_he(
        ctx, Q_cts, K_cts, m=m, d_prime=d_prime, rot_stride=rot_stride, n_he=n_he,
    )

    decoded = np.stack([ctx.decrypt(ct)[:n_he] for ct in out_cts], axis=0)
    return interleaved_diag_unpack(decoded, m=m, n_batch=n_batch)
