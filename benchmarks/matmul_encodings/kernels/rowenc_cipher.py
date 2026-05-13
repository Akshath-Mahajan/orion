"""Row-encoding (row-pack) CKKS matmul kernel.

Port of matmult/rowenc_cipher.go::RowPackMatMulHE. Computes C = A @ B for
square (n, n) matrices encoded row-major into a single ciphertext.

This is the simplest of the four encodings -- no BSGS, no chunk
stitching, no diagonal interleaving. The cost profile is O(n log n)
rotations per matmul on a single ciphertext, which makes it the natural
"baseline" against the more clever encodings (THOR, MOAI, Bicycle).
Negar's Go runs row-pack only at small n (n in {4, 8, 16, 32}) because
s = n^2 has to fit inside one ciphertext.

Algorithm (per outer iteration i in [0, n)):

    1. extract column i of A: ct * col_mask[i] + Rescale
    2. extract row i of B:    ct * row_mask[i] + Rescale
    3. replicate column rightward over log2(n) rotate-and-add steps
    4. replicate row downward (after initial alignment by n*i)
    5. diagonal-align A_rep by rotating left by i
    6. ct * ct, accumulate

Multiplicative depth: 2 (one ct*pt mask + one ct*ct multiply). One
terminal Relinearize after the accumulator is built; eager rescale per
multiply (matches the Go's RowPackMatMulHE control flow).

Slot-count vs s = n^2
---------------------
CKKS rotations are cyclic over the full slot count, NOT over s = n^2.
Inputs are zero-padded to ctx.slots (NOT tiled -- the row-pack algorithm
relies on the [s, slots) tail being zero so the replication shifts can
spread into it cleanly). Replication shifts are therefore expressed as
``slots - 2^k``, ``slots - n*2^k mod slots`` etc. -- the "right shift by
delta in the active window" recipe rephrased in left-rotation terms.
"""

from __future__ import annotations

import math

import numpy as np

from ..context import Context
from ..plaintext.rowenc_plain import (
    _make_col_mask,
    _make_row_mask,
    row_pack,
    row_unpack,
)


# ---------------------------------------------------------------------------
# Mask cache
#
# Encodes the 2n extraction masks (n column + n row) once per problem
# size. Each mask is zero outside its target slots within the active
# n^2-slot window, then zero-padded out to ctx.slots.
# ---------------------------------------------------------------------------


class _RowMasks:
    """(n column-extraction, n row-extraction) plaintext id bundle."""

    def __init__(self, ctx: Context, n: int, level: int) -> None:
        s = n * n
        scale = ctx.scheme.params.get_default_scale()
        self.cols: list[int] = []
        self.rows: list[int] = []
        for i in range(n):
            self.cols.append(self._encode(ctx, _make_col_mask(s, i, n), level, scale))
            self.rows.append(self._encode(ctx, _make_row_mask(s, i, n), level, scale))

    @staticmethod
    def _encode(ctx: Context, mask: np.ndarray, level: int, scale) -> int:
        padded = np.zeros(ctx.slots, dtype=np.float64)
        padded[: mask.size] = mask
        return ctx.backend.Encode(padded.tolist(), level, scale)


# ---------------------------------------------------------------------------
# Per-step HE helpers
# ---------------------------------------------------------------------------


def _extract_col_he(ctx: Context, ct: int, mask_pt: int) -> int:
    """ct * column-mask + Rescale. Bumps ct_pt_muls."""
    return ctx.rescale(ctx.mul_pt(ct, mask_pt))


def _extract_row_he(ctx: Context, ct: int, mask_pt: int) -> int:
    """ct * row-mask + Rescale. Bumps ct_pt_muls."""
    return ctx.rescale(ctx.mul_pt(ct, mask_pt))


def _replicate_col_he(ctx: Context, ct: int, n: int) -> int:
    """Spread each isolated column value rightward via log2(n) doublings.

    Implementation note: the running accumulator changes every iteration,
    so the rotations don't share an input -- hoisting via rot_batch does
    NOT apply here. Each step is one independent ctx.rot.
    """
    log2n = int(math.log2(n))
    slots = ctx.slots
    out = ct
    for k in range(log2n):
        shift = slots - (1 << k)
        rot = ctx.rot(out, shift)
        out = ctx.add(out, rot)
    return out


def _replicate_row_he(ctx: Context, ct: int, i: int, n: int) -> int:
    """Bring row i to slot 0, then spread downward via log2(n) doublings."""
    log2n = int(math.log2(n))
    slots = ctx.slots

    if (n * i) % slots == 0:
        out = ct
    else:
        out = ctx.rot(ct, n * i)

    pow2log = 1 << log2n  # == n
    for k in range(log2n):
        shift = (slots - (pow2log * (1 << k)) % slots) % slots
        if shift == 0:
            continue
        rot = ctx.rot(out, shift)
        out = ctx.add(out, rot)
    return out


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


def rowenc_he_kernel(
    ctx: Context,
    ct_a: int,
    ct_b: int,
    n: int,
    input_level: int,
) -> int:
    """Run row-packing matmul on encrypted inputs.

    `ct_a` and `ct_b` are row-packed ciphertexts at `input_level`. Returns
    a single ciphertext at `input_level - 2` whose first n*n slots decode
    (via row_unpack) to the (n x n) product matrix.

    Lazy relinearization: every inner ct*ct uses ctx.mul_nr; the
    accumulator runs at degree 2; one terminal ctx.relin produces the
    degree-1 result. On lattigo this degrades to eager (binding fallback).
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"rowenc_he_kernel: n={n} must be a positive power of 2")

    masks = _RowMasks(ctx, n, level=input_level)

    acc: int | None = None
    for i in range(n):
        a_col = _extract_col_he(ctx, ct_a, masks.cols[i])
        b_row = _extract_row_he(ctx, ct_b, masks.rows[i])

        a_rep = _replicate_col_he(ctx, a_col, n)
        b_rep = _replicate_row_he(ctx, b_row, i, n)

        # Diagonal alignment of A_rep (left rotate by i).
        if i != 0:
            a_rep = ctx.rot(a_rep, i)

        prod = ctx.mul_nr(a_rep, b_rep)
        prod = ctx.rescale(prod)
        acc = prod if acc is None else ctx.add(acc, prod)

    assert acc is not None
    return ctx.relin(acc)


def rowenc_he(
    ctx: Context,
    A: np.ndarray,
    B: np.ndarray,
    input_level: int | None = None,
) -> np.ndarray:
    """End-to-end: row-pack, encrypt, run, decrypt, row-unpack.

    Convenience for tests; real benchmarks split encrypt + kernel + decrypt
    into separately timed sections.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    assert A.shape == (n, n) and B.shape == (n, n), (
        f"rowenc requires square matrices, got A={A.shape}, B={B.shape}"
    )
    assert n * n <= ctx.slots, (
        f"n*n={n*n} exceeds ctx.slots={ctx.slots} -- use a bigger ring or smaller n"
    )

    if input_level is None:
        input_level = ctx.max_level

    ct_a = ctx.encrypt(row_pack(A), level=input_level)
    ct_b = ctx.encrypt(row_pack(B), level=input_level)
    ct_c = rowenc_he_kernel(ctx, ct_a, ct_b, n, input_level)
    vec = ctx.decrypt(ct_c)
    return row_unpack(vec, n)
