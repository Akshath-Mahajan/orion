"""Plaintext simulation of BMM-I bicyclic matrix multiplication.

Port of matmult/bmm1_plain.go (Zheng et al., IEEE TIFS 2024).

Bicyclic encoding
-----------------
An (n x m) matrix A is encoded into a length-(n*m) slot vector by
    vec[k] = A[k mod n, k mod m]   for k in [0, n*m).

When s_n, s_m, s_p are pairwise coprime, this lets matrix multiplication
be done with only rotations and pointwise products -- no masks.

This module is the plaintext oracle used by the GPU ciphertext kernels'
correctness tests.
"""

from __future__ import annotations

import math

import numpy as np

from .op_counts import OpCounts


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------


def bicyclic_encode(M: np.ndarray) -> np.ndarray:
    """Return the length-(n*m) slot vector with vec[k] = M[k mod n, k mod m]."""
    n, m = M.shape
    k = np.arange(n * m)
    return M[k % n, k % m].astype(np.float64, copy=False)


def bicyclic_decode(vec: np.ndarray, n: int, p: int) -> np.ndarray:
    """Recover an (n x p) matrix C from a bicyclic-encoded vector of length n*p.

    C[k mod n, k mod p] = vec[k].
    """
    C = np.zeros((n, p), dtype=np.float64)
    k = np.arange(n * p)
    C[k % n, k % p] = vec
    return C


def repeat_vector(vec: np.ndarray, times: int) -> np.ndarray:
    """Tile `vec` `times` times. Plaintext analogue of repeat_vector.

    Produces a vector long enough that any rotation used in bmm1 leaves the
    first n*p slots populated with valid (non-zero-padding) data.
    """
    return np.tile(vec, times)


def encode_blocks(
    A: np.ndarray,
    B: np.ndarray,
    s_n: int,
    s_m: int,
    s_p: int,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Split A (NxM) and B (MxP) into blocks, bicyclic-encode each.

    a_enc[i][k] = bicyclic_encode(A[i*s_n:(i+1)*s_n, k*s_m:(k+1)*s_m])
    b_enc[k][j] = bicyclic_encode(B[k*s_m:(k+1)*s_m, j*s_p:(j+1)*s_p])
    """
    N, M = A.shape
    _, P = B.shape

    a_rows = N // s_n
    k_blocks = M // s_m
    b_cols = P // s_p

    a_enc: list[list[np.ndarray]] = []
    for i in range(a_rows):
        row = []
        for k in range(k_blocks):
            row.append(
                bicyclic_encode(A[i * s_n : (i + 1) * s_n, k * s_m : (k + 1) * s_m])
            )
        a_enc.append(row)

    b_enc: list[list[np.ndarray]] = []
    for k in range(k_blocks):
        row = []
        for j in range(b_cols):
            row.append(
                bicyclic_encode(B[k * s_m : (k + 1) * s_m, j * s_p : (j + 1) * s_p])
            )
        b_enc.append(row)

    return a_enc, b_enc


# ---------------------------------------------------------------------------
# rotateVec (left rotation, matches Go rotateVec / rotateSlots semantics)
# ---------------------------------------------------------------------------


def rotate_vec(v: np.ndarray, k: int) -> np.ndarray:
    """Left-rotate v by k positions modulo len(v).

    Matches Go's rotateVec: out[i] = v[(i + k) mod n]. The element at
    position k moves to position 0.
    """
    n = len(v)
    if n == 0:
        return v.copy()
    k %= n
    if k == 0:
        return v.copy()
    return np.concatenate([v[k:], v[:k]])


def _int_ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Single-block BMM-I (plaintext, with op counter)
# ---------------------------------------------------------------------------


def bmm1_plain(
    a_enc: np.ndarray,
    b_enc: np.ndarray,
    n: int,
    m: int,
    p: int,
    counts: OpCounts,
) -> np.ndarray:
    """Run BMM-I on a single bicyclic-encoded block.

    Returns the length-(n*p) output slot vector. Rotations and pointwise
    products are accumulated into `counts` in place (matches Go pass-by-
    pointer semantics).
    """
    step = n * p

    # Tile so that rotations do not wrap into zero padding.
    a_tiles = _int_ceil_div(n * m + n * p, n * m)
    b_tiles = _int_ceil_div(m * p + n * p, m * p)
    a_rep = repeat_vector(a_enc, a_tiles)
    b_rep = repeat_vector(b_enc, b_tiles)

    C = np.zeros(step, dtype=np.float64)
    for i in range(m):
        rot_a = (i * step) % (n * m)
        rot_b = (i * step) % (m * p)

        a_rot = rotate_vec(a_rep, rot_a)
        b_rot = rotate_vec(b_rep, rot_b)
        if rot_a != 0:
            counts.rotations += 1
        if rot_b != 0:
            counts.rotations += 1

        C += a_rot[:step] * b_rot[:step]
        counts.ct_ct_muls += 1

    return C


# ---------------------------------------------------------------------------
# Block BMM-I -- falls back to single-block when dims == s_dims
# ---------------------------------------------------------------------------


def bmm1_matmul_plain(
    a_enc: list[list[np.ndarray]],
    b_enc: list[list[np.ndarray]],
    N: int,
    M: int,
    P: int,
    s_n: int,
    s_m: int,
    s_p: int,
) -> tuple[np.ndarray, OpCounts]:
    """Run block BMM-I over pre-encoded block arrays.

    Returns the (N x P) product matrix alongside the cumulative op counts.
    """
    a_rows = N // s_n
    k_blocks = M // s_m
    b_cols = P // s_p

    counts = OpCounts()
    C = np.zeros((N, P), dtype=np.float64)

    for i in range(a_rows):
        for j in range(b_cols):
            acc = np.zeros(s_n * s_p, dtype=np.float64)
            for k in range(k_blocks):
                block = bmm1_plain(
                    a_enc[i][k], b_enc[k][j], s_n, s_m, s_p, counts
                )
                acc += block
            dec = bicyclic_decode(acc, s_n, s_p)
            C[i * s_n : (i + 1) * s_n, j * s_p : (j + 1) * s_p] = dec

    return C, counts


# ---------------------------------------------------------------------------
# Theoretical cost formulas
# ---------------------------------------------------------------------------


def theoretical_bmm1_costs(
    N: int, M: int, P: int, s_n: int, s_m: int, s_p: int
) -> tuple[int, int, int, int]:
    """Return (n_rot, n_mult, n_add, n_ks) for block BMM-I.

    Per single-block call: 2*s_m rotations, s_m mults, s_m adds.
    Total blocks: (N/s_n) * (M/s_m) * (P/s_p).
    """
    total_blocks = (N // s_n) * (M // s_m) * (P // s_p)
    n_rot = 2 * s_m * total_blocks
    n_mult = s_m * total_blocks
    n_add = s_m * total_blocks
    n_ks = n_rot + n_mult
    return n_rot, n_mult, n_add, n_ks


def bmm1_required_slots(s_n: int, s_m: int, s_p: int) -> int:
    """Minimum n_he that encode_and_encrypt_blocks expects."""
    a = s_n * s_m * _int_ceil_div(s_m + s_p, s_m)
    b = s_m * s_p * _int_ceil_div(s_m + s_n, s_m)
    return int(max(a, b))
