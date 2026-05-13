"""Plaintext packing, unpacking, and kernels for MOAI.

Port of matmult/moai_plain.go.

Algorithm 3 -- Col x Col -> Diag (naive and BSGS)
Algorithm 4 -- Diag x Col -> Col (BSGS)

Slot layout (interleaved over n_batch matrices packed per ciphertext):

    slot[r * n_batch + s]  =  sequence s, position r

A batch of n_batch = n_he / m matrices can be processed in lock-step inside
a single ciphertext.

The plaintext kernels mirror their HE counterparts in moai_cipher.go
line-for-line, so the OpCounts returned here are exactly the ops an HE run
would perform at the same config.
"""

from __future__ import annotations

import math

import numpy as np

from .bmm1_plain import rotate_vec
from .op_counts import OpCounts


# ---------------------------------------------------------------------------
# Interleaved packing helpers
# ---------------------------------------------------------------------------


def interleaved_column_pack(Xs: np.ndarray, n_he: int) -> np.ndarray:
    """Pack n_batch (m x d) matrices into d slot-vectors of length n_he.

    `Xs` has shape (n_batch, m, d). Returns an array of shape (d, n_he).
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    n_batch, m, d = Xs.shape
    out = np.zeros((d, n_he), dtype=np.float64)
    for j in range(d):
        for r in range(m):
            for s in range(n_batch):
                out[j, r * n_batch + s] = Xs[s, r, j]
    return out


def interleaved_column_unpack(cts: np.ndarray, m: int, n_batch: int) -> np.ndarray:
    """Invert interleaved_column_pack.

    `cts` has shape (d, n_he). Returns shape (n_batch, m, d).
    """
    cts = np.asarray(cts, dtype=np.float64)
    d = cts.shape[0]
    out = np.zeros((n_batch, m, d), dtype=np.float64)
    for j in range(d):
        for r in range(m):
            for s in range(n_batch):
                out[s, r, j] = cts[j, r * n_batch + s]
    return out


def interleaved_diag_pack(Cs: np.ndarray, n_he: int) -> np.ndarray:
    """Pack n_batch (m x m) matrices into m diagonal slot-vectors.

    The i-th output vector holds the i-th lower-diagonal of every matrix,
    interleaved in the n_batch dimension.

    `Cs` has shape (n_batch, m, m). Returns shape (m, n_he).
    """
    Cs = np.asarray(Cs, dtype=np.float64)
    n_batch, m, _ = Cs.shape
    out = np.zeros((m, n_he), dtype=np.float64)
    for i in range(m):
        for r in range(m):
            for s in range(n_batch):
                out[i, r * n_batch + s] = Cs[s, r, (r + i) % m]
    return out


def interleaved_diag_unpack(cts: np.ndarray, m: int, n_batch: int) -> np.ndarray:
    """Invert interleaved_diag_pack.

    `cts` has shape (m, n_he). Returns shape (n_batch, m, m).
    """
    cts = np.asarray(cts, dtype=np.float64)
    out = np.zeros((n_batch, m, m), dtype=np.float64)
    for i in range(m):
        for r in range(m):
            for s in range(n_batch):
                out[s, r, (r + i) % m] = cts[i, r * n_batch + s]
    return out


# ---------------------------------------------------------------------------
# Helpers (rotation aliased to match Go's rotateSlots semantics)
# ---------------------------------------------------------------------------


def _rotate_slots(v: np.ndarray, k: int) -> np.ndarray:
    """Alias of rotate_vec; left rotation by k positions modulo len(v)."""
    return rotate_vec(v, k)


def _int_ceil_sqrt(n: int) -> int:
    return int(math.ceil(math.sqrt(n)))


def _pos_mod_int(a: int, m: int) -> int:
    r = a % m
    if r < 0:
        r += m
    return r


# ---------------------------------------------------------------------------
# Algorithm 3 -- Col x Col -> Diag
# ---------------------------------------------------------------------------


def moai_col_col_naive(
    Q: np.ndarray,
    K: np.ndarray,
    m: int,
    d_prime: int,
    rot_stride: int,
) -> tuple[np.ndarray, OpCounts]:
    """Naive Algorithm 3.

    diag_j = sum_i Q[i] (*) Rot_{j*stride}( K[i] )

    Q, K are arrays of shape (d_prime, n_he). Returns (out, counts) where
    `out` has shape (m, n_he).

    Costs O((m - 1) * d_prime) rotations and m * d_prime multiplications.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n_he = Q.shape[1]
    counts = OpCounts()

    out = np.zeros((m, n_he), dtype=np.float64)
    for j in range(m):
        acc = np.zeros(n_he, dtype=np.float64)
        for i in range(d_prime):
            if j == 0:
                rot_k = K[i]
            else:
                rot_k = _rotate_slots(K[i], j * rot_stride)
                counts.rotations += 1
            prod = Q[i] * rot_k
            counts.ct_ct_muls += 1
            acc = acc + prod
        out[j] = acc
    return out, counts


def moai_col_col_bsgs(
    Q: np.ndarray,
    K: np.ndarray,
    m: int,
    d_prime: int,
    rot_stride: int,
) -> tuple[np.ndarray, OpCounts]:
    """BSGS Algorithm 3.

    j = alpha * b + r, baby-step rotations on K precomputed once per i,
    giant-step rotations on Q done per alpha. Rotation count drops from
    (m - 1)*d_prime to (b + g - 2)*d_prime + (g - 1).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    b = _int_ceil_sqrt(m)
    g = (m + b - 1) // b
    n_he = Q.shape[1]
    counts = OpCounts()

    # Baby steps: beta[i][r] = Rot_{r*stride}(K[i]).  r=0 is identity.
    beta: list[list[np.ndarray]] = []
    for i in range(d_prime):
        row: list[np.ndarray] = [K[i]]
        for r in range(1, b):
            row.append(_rotate_slots(K[i], r * rot_stride))
            counts.rotations += 1
        beta.append(row)

    out = np.zeros((m, n_he), dtype=np.float64)

    for alpha in range(g):
        shift = (alpha * b) % m
        # Giant-step rotation on Q by (m - shift) * stride. When alpha=0 this
        # reduces to 0 mod n_he (since m*stride = n_he in typical configs)
        # and collapses to the identity, so we skip it.
        q_rot_shift = _pos_mod_int((m - shift) * rot_stride, n_he)
        q_rot: list[np.ndarray] = []
        for i in range(d_prime):
            if q_rot_shift == 0:
                q_rot.append(Q[i])
            else:
                q_rot.append(_rotate_slots(Q[i], q_rot_shift))
                counts.rotations += 1

        for r in range(b):
            j = alpha * b + r
            if j >= m:
                break
            partial = q_rot[0] * beta[0][r]
            counts.ct_ct_muls += 1
            for i in range(1, d_prime):
                prod = q_rot[i] * beta[i][r]
                counts.ct_ct_muls += 1
                partial = partial + prod

            final_shift = _pos_mod_int(shift * rot_stride, n_he)
            if final_shift == 0:
                rolled = partial
            else:
                rolled = _rotate_slots(partial, final_shift)
                counts.rotations += 1
            out[j] = out[j] + rolled
    return out, counts


# ---------------------------------------------------------------------------
# Algorithm 4 -- Diag x Col -> Col (BSGS)
# ---------------------------------------------------------------------------


def moai_diag_col_bsgs(
    C: np.ndarray,
    V: np.ndarray,
    m: int,
    d_prime: int,
    rot_stride: int,
) -> tuple[np.ndarray, OpCounts]:
    """BSGS Algorithm 4.

    (C . V)_j = sum_i diag_i(C) (*) Rot_{i*stride}( V[j] )

    decomposed as i = alpha*b + r. Outer loop over j in [0, d_prime); the
    two inner loops are the baby-step / giant-step decomposition.

    `C` has shape (m, n_he); `V` has shape (d_prime, n_he). Returns out of
    shape (d_prime, n_he).
    """
    C = np.asarray(C, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    b = _int_ceil_sqrt(m)
    g = (m + b - 1) // b
    n_he = V.shape[1]
    counts = OpCounts()

    out = np.zeros((d_prime, n_he), dtype=np.float64)
    for j in range(d_prime):
        # Baby steps on V[j]: beta[r] = Rot_{r*stride}(V[j]).
        beta: list[np.ndarray] = [V[j]]
        for r in range(1, b):
            beta.append(_rotate_slots(V[j], r * rot_stride))
            counts.rotations += 1

        acc = np.zeros(n_he, dtype=np.float64)
        for alpha in range(g):
            shift = (alpha * b) % m
            c_rot_shift = _pos_mod_int((m - shift) * rot_stride, n_he)

            inner = np.zeros(n_he, dtype=np.float64)
            for r in range(b):
                idx = alpha * b + r
                if idx >= m:
                    break
                if c_rot_shift == 0:
                    rot_c = C[idx]
                else:
                    rot_c = _rotate_slots(C[idx], c_rot_shift)
                    counts.rotations += 1
                prod = rot_c * beta[r]
                counts.ct_ct_muls += 1
                inner = inner + prod

            final_shift = _pos_mod_int(shift * rot_stride, n_he)
            if final_shift == 0:
                rolled = inner
            else:
                rolled = _rotate_slots(inner, final_shift)
                counts.rotations += 1
            acc = acc + rolled
        out[j] = acc
    return out, counts


# ---------------------------------------------------------------------------
# Theoretical cost formulas (paper-side, ignoring identity rotations)
# ---------------------------------------------------------------------------


def theoretical_naive(m: int, d: int) -> tuple[int, int, int]:
    """Return (multiplications, rotations, total) for naive Col x Col."""
    mul = m * d
    rot = (m - 1) * d
    return mul, rot, mul + rot


def theoretical_bsgs(m: int, d: int) -> tuple[int, int, int]:
    """Return (multiplications, rotations, total) for BSGS Col x Col."""
    b = _int_ceil_sqrt(m)
    g = (m + b - 1) // b
    mul = m * d
    rot = (b - 1) * d + (g - 1) * d + (g - 1)
    return mul, rot, mul + rot
