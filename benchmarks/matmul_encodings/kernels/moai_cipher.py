"""MOAI CKKS matmul kernels.

Ports of:
- Algorithm 3 (Col x Col -> Diag, BSGS) from matmult/moai_cipher.go
- Algorithm 4 (Diag x Col -> Col, BSGS) from matmult/moai_cipher.go

The Col x Col variant is the natural fit for Q . K^T in transformer
self-attention. Q and K are both column-packed; the output is
diag-packed and feeds Algorithm 4 (the next matmul, by V) without
re-encoding -- which is the back-to-back-matmul property that motivates
MOAI as an encoding choice in the first place. With both algorithms
landed, a Q . K^T . V chain runs end-to-end on encrypted inputs without
any intermediate decrypt-and-repack.

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
    interleaved_column_unpack,
    interleaved_diag_pack,
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


# ---------------------------------------------------------------------------
# Algorithm 4 -- Diag x Col -> Col (BSGS)
#
# Direct port of matmult/moai_cipher.go::MoaiDiagColBSGSHE. Mirrors the
# plaintext oracle moai_plain.moai_diag_col_bsgs structure 1:1.
# ---------------------------------------------------------------------------


def encrypt_diag_packed(
    ctx: Context,
    Cs: np.ndarray,
    n_he: int,
) -> list[int]:
    """Pack n_batch diag-packed (m x m) matrices into m ciphertexts."""
    packed = interleaved_diag_pack(Cs, n_he=n_he)  # (m, n_he)
    return [
        ctx.encrypt(_tile_to_slots(packed[i], ctx.slots))
        for i in range(packed.shape[0])
    ]


def moai_diag_col_bsgs_he(
    ctx: Context,
    C_cts: list[int],
    V_cts: list[int],
    m: int,
    d_prime: int,
    rot_stride: int,
    n_he: int,
) -> list[int]:
    """BSGS Algorithm 4 on encrypted (diag-packed C, col-packed V).

    Returns d' column-packed output ciphertexts. Outer loop is over j in
    [0, d'); the inner BSGS decomposition is identical in structure to
    Algorithm 3 but with the roles of "diag operand" and "col operand"
    swapped: baby steps rotate V[j] by r*stride, giant step rotates the
    sum (not the col operand) by alpha*b * stride at the end of each
    block.
    """
    assert len(C_cts) == m
    assert len(V_cts) == d_prime
    b = _int_ceil_sqrt(m)
    g = (m + b - 1) // b

    out_cts: list[int | None] = [None] * d_prime

    for j in range(d_prime):
        # Baby steps on V[j]: hoisted batch rotate by r*stride for r in [1, b).
        baby_shifts = [r * rot_stride for r in range(1, b)]
        beta: list[int] = [V_cts[j]]  # r=0 identity
        if baby_shifts:
            rotated = ctx.rot_batch(V_cts[j], baby_shifts)
            beta.extend(rotated)

        for alpha in range(g):
            shift = (alpha * b) % m
            c_rot_shift = ((m - shift) * rot_stride) % n_he

            # Inner accumulator: inner = sum_r Rot(C[idx], c_rot_shift) * beta[r]
            inner: int | None = None
            for r in range(b):
                idx = alpha * b + r
                if idx >= m:
                    break

                if c_rot_shift == 0:
                    rot_c = C_cts[idx]
                else:
                    rot_c = ctx.rot(C_cts[idx], c_rot_shift)

                prod = ctx.mul_rl(rot_c, beta[r])
                prod = ctx.rescale(prod)
                inner = prod if inner is None else ctx.add(inner, prod)

            assert inner is not None  # at least r=0 ran

            final_shift = (shift * rot_stride) % n_he
            if final_shift != 0:
                inner = ctx.rot(inner, final_shift)

            if out_cts[j] is None:
                out_cts[j] = inner
            else:
                out_cts[j] = ctx.add(out_cts[j], inner)

    return out_cts  # type: ignore[return-value]


def moai_diag_col_bsgs_end_to_end(
    ctx: Context,
    Cs: np.ndarray,
    Vs: np.ndarray,
) -> np.ndarray:
    """Encrypt n_batch (m x m) diag-packed C and (m x d') col-packed V,
    run Alg 4, decrypt + col-unpack.

    Returns shape (n_batch, m, d') with result[s] = Cs[s] @ Vs[s].
    """
    n_batch, m, m2 = Cs.shape
    assert m == m2, f"C must be square per batch, got {Cs.shape}"
    n_batch2, m3, d_prime = Vs.shape
    assert (n_batch, m) == (n_batch2, m3), (
        f"batch / m mismatch: C is {Cs.shape}, V is {Vs.shape}"
    )
    n_he = n_batch * m
    rot_stride = n_batch

    C_cts = encrypt_diag_packed(ctx, Cs, n_he=n_he)
    V_cts = encrypt_col_packed(ctx, Vs, n_he=n_he)
    out_cts = moai_diag_col_bsgs_he(
        ctx, C_cts, V_cts, m=m, d_prime=d_prime, rot_stride=rot_stride, n_he=n_he,
    )

    decoded = np.stack([ctx.decrypt(ct)[:n_he] for ct in out_cts], axis=0)
    return interleaved_column_unpack(decoded, m=m, n_batch=n_batch)


# ---------------------------------------------------------------------------
# Convenience: end-to-end Q . K^T . V chain (back-to-back matmul)
# ---------------------------------------------------------------------------


def moai_qkt_v_he(
    ctx: Context,
    Qs: np.ndarray,
    Ks: np.ndarray,
    Vs: np.ndarray,
) -> np.ndarray:
    """Run Q . K^T (Alg 3) then (Q . K^T) . V (Alg 4) end-to-end on encrypted
    inputs. Demonstrates that MOAI's encoding-output of Alg 3 (diag-pack)
    feeds directly into Alg 4 with no decrypt-and-repack -- the property
    that motivates MOAI as an encoding choice.

    Returns shape (n_batch, m, d_v) with result[s] = Qs[s] @ Ks[s].T @ Vs[s].
    """
    n_batch, m, d_prime = Qs.shape
    qkt = moai_col_col_bsgs_end_to_end(ctx, Qs, Ks)  # (n_batch, m, m), diag
    return moai_diag_col_bsgs_end_to_end(ctx, qkt, Vs)
