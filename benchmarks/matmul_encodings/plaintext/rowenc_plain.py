"""Plaintext simulation of row-packing matrix multiplication.

Port of matmult/rowenc_plain.go.

Encoding
--------
An n x n matrix M is row-major flattened into a length-n^2 slot vector:

    slot[i*n + j] = M[i][j]

Algorithm (per outer iteration i in [0, n)):

    1. extract_col(A, i)  -- isolate column i of A: every n-th slot
    2. extract_row(B, i)  -- isolate row i of B: contiguous n-slot block
    3. replicate_col      -- spread each column value rightward
                             via log2(n) rotate-and-add steps
    4. replicate_row(i)   -- bring row i to slot 0, then spread downward
    5. diagonal-align     -- rotate A_rep left by i
    6. accumulate         -- C += A_rep * B_rep

Constraint: n must be a power of 2 so the log2(n) replication steps land
on integer rotation amounts.

Status in the paper. Row-packing is the simplest of the four encodings
(no BSGS, no chunk stitching, no diagonal interleaving), but its cost
profile -- O(n log n) rotations on a single ciphertext per matmul -- is
a useful baseline against the more clever encodings (THOR, MOAI,
Bicycle/BMM-{I,III}). Negar's Go runs it only at small n (n in
{4, 8, 16, 32}) since s = n^2 has to fit inside the slot count.
"""

from __future__ import annotations

import math

import numpy as np

from .op_counts import OpCounts


__all__ = [
    "row_pack",
    "row_unpack",
    "rowenc_plain",
    "rowenc_matmul_plain",
    "theoretical_rowenc_costs",
]


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------


def row_pack(M: np.ndarray) -> np.ndarray:
    """Row-major flatten of an (n, n) matrix into a length-n^2 vector."""
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    assert M.shape == (n, n), f"row_pack expects a square matrix, got {M.shape}"
    return M.ravel().copy()


def row_unpack(vec: np.ndarray, n: int) -> np.ndarray:
    """Inverse of row_pack: read the first n*n slots into an (n, n) matrix."""
    vec = np.asarray(vec, dtype=np.float64)
    return vec[: n * n].reshape(n, n).copy()


# ---------------------------------------------------------------------------
# Mask generation -- length-s, paired with the row-packed vector
# ---------------------------------------------------------------------------


def _make_col_mask(s: int, i: int, n: int) -> np.ndarray:
    """Length-s mask with 1.0 at slots i, i+n, i+2n, ... -- isolates column i."""
    mask = np.zeros(s, dtype=np.float64)
    mask[i:s:n] = 1.0
    return mask


def _make_row_mask(s: int, i: int, n: int) -> np.ndarray:
    """Length-s mask with 1.0 in slots [i*n, i*n + n) -- isolates row i."""
    mask = np.zeros(s, dtype=np.float64)
    start = i * n
    mask[start : start + n] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Slot helpers (numpy plaintext analogs of CKKS rotate / add / mul)
#
# rotate_vec is a left rotation: out[i] = v[(i + k) mod n]. This matches
# Lattigo's RotateNew(ct, k) convention -- the element at position k moves
# to position 0.
# ---------------------------------------------------------------------------


def _rotate_vec(v: np.ndarray, k: int) -> np.ndarray:
    s = v.size
    if s == 0:
        return v.copy()
    k %= s
    if k == 0:
        return v.copy()
    return np.concatenate([v[k:], v[:k]])


# ---------------------------------------------------------------------------
# Extraction + replication helpers (plaintext)
# ---------------------------------------------------------------------------


def _row_extract_col(vec: np.ndarray, i: int, n: int, counts: OpCounts) -> np.ndarray:
    counts.ct_pt_muls += 1
    return vec * _make_col_mask(vec.size, i, n)


def _row_extract_row(vec: np.ndarray, i: int, n: int, counts: OpCounts) -> np.ndarray:
    counts.ct_pt_muls += 1
    return vec * _make_row_mask(vec.size, i, n)


def _row_replicate_col(vec: np.ndarray, n: int, counts: OpCounts) -> np.ndarray:
    """Spread each isolated column value rightward via log2(n) doublings.

    Right shift by 2^k is left rotation by (s - 2^k) in the rotate_vec
    convention. The accumulator pattern is x' = x + Rot(x, s - 2^k), which
    over log2(n) steps fills every "row" with the column's i-th value.
    """
    log2n = int(math.log2(n))
    s = vec.size
    out = vec.copy()
    for k in range(log2n):
        shift = s - (1 << k)
        rot = _rotate_vec(out, shift)
        counts.rotations += 1
        out = out + rot
    return out


def _row_replicate_row(vec: np.ndarray, i: int, n: int, counts: OpCounts) -> np.ndarray:
    """Bring row i to slot 0, then spread downward via log2(n) doublings."""
    log2n = int(math.log2(n))
    s = vec.size

    out = _rotate_vec(vec, n * i)  # left-rotate by n*i
    if (n * i) % s != 0:
        counts.rotations += 1

    pow2log = 1 << log2n  # == n
    for k in range(log2n):
        shift = (s - (pow2log * (1 << k)) % s) % s
        if shift == 0:
            continue
        rot = _rotate_vec(out, shift)
        counts.rotations += 1
        out = out + rot
    return out


# ---------------------------------------------------------------------------
# Plaintext kernel
# ---------------------------------------------------------------------------


def rowenc_plain(
    A_flat: np.ndarray, B_flat: np.ndarray, n: int, counts: OpCounts
) -> np.ndarray:
    """Run row-encoding matmul on flattened inputs. Returns C_flat (length n^2).

    Mirrors RowPackMatMulHE in matmult/rowenc_cipher.go line-for-line, so
    `counts` predicts exactly the ops the HE kernel will perform (rotations,
    ct_pt_muls, ct_ct_muls).
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"rowenc_plain: n={n} must be a positive power of 2")

    A_flat = np.asarray(A_flat, dtype=np.float64)
    B_flat = np.asarray(B_flat, dtype=np.float64)
    assert A_flat.size == n * n and B_flat.size == n * n

    C = np.zeros(n * n, dtype=np.float64)
    for i in range(n):
        a_col = _row_extract_col(A_flat, i, n, counts)
        b_row = _row_extract_row(B_flat, i, n, counts)

        a_rep = _row_replicate_col(a_col, n, counts)
        b_rep = _row_replicate_row(b_row, i, n, counts)

        # Diagonal alignment: rotate A_rep left by i.
        if i != 0:
            a_rep = _rotate_vec(a_rep, i)
            counts.rotations += 1

        prod = a_rep * b_rep
        counts.ct_ct_muls += 1
        C = C + prod

    return C


def rowenc_matmul_plain(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, OpCounts]:
    """End-to-end: pack, run, unpack."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    assert A.shape == (n, n) and B.shape == (n, n), (
        f"rowenc requires square matrices, got A={A.shape}, B={B.shape}"
    )

    counts = OpCounts()
    C_flat = rowenc_plain(row_pack(A), row_pack(B), n, counts)
    return row_unpack(C_flat, n), counts


# ---------------------------------------------------------------------------
# Theoretical cost formulas
# ---------------------------------------------------------------------------


def theoretical_rowenc_costs(n: int) -> tuple[int, int, int, int, int]:
    """Return (n_rot, n_pmult, n_mult, n_add, n_ks) for one row-packing
    n x n matmul. Matches matmult/rowenc_plain.go::TheoreticalRowCosts:

        Rot   = n * (2*log2(n) + 2)
        PMult = 2*n
        Mult  = n
        Add   = n * (2*log2(n) + 1)
        Ks    = Rot + Mult
    """
    log2n = int(math.log2(n))
    n_rot = n * (2 * log2n + 2)
    n_pmult = 2 * n
    n_mult = n
    n_add = n * (2 * log2n + 1)
    n_ks = n_rot + n_mult
    return n_rot, n_pmult, n_mult, n_add, n_ks
