"""BMM-I (bicycle, single-ciphertext) CKKS kernel for the matmul-encoding paper.

Port of matmult/bmm1_cipher.go (Zheng et al., IEEE TIFS 2024). Computes
C = A @ B on bicyclic-encoded inputs that each fit in one CKKS ciphertext.

Algorithm at a glance
---------------------
Given the bicyclic encoding ``vec_X[k] = X[k mod n, k mod m]`` (X is n x m),
the matmul C = A @ B on (n x m) A and (m x p) B reduces to:

    for i in [0, m):
        rot_a = (i * n*p) mod (n*m)
        rot_b = (i * n*p) mod (m*p)
        C += Rot(a_enc, rot_a)[:n*p] * Rot(b_enc, rot_b)[:n*p]

When n, m, p are pairwise coprime, the bicyclic encoding's orbit
property makes this exact (no masking needed).

Optimizations exercised here
----------------------------
1. **Hoisted batch rotation** -- both A and B are rotated by ``m-1`` distinct
   shifts each; we issue them in one batched call so the ModUp on the input
   ciphertext is amortized across all key-switches.
2. **Lazy relinearization** -- products land as degree-2 ciphertexts, the
   accumulator runs at degree 2, and we pay one Relinearize at the end
   instead of m of them.

Result
------
Single ciphertext whose first ``n*p`` slots decode (via bicyclic_decode)
to the (n x p) product matrix C, modulo CKKS noise.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..context import Context
from ..plaintext.bmm1_plain import bicyclic_decode, bicyclic_encode, repeat_vector


def block_rot_lists(n: int, m: int, p: int) -> tuple[list[int], list[int]]:
    """Return the two rotation schedules: rotsA[i] = (i*n*p) % (n*m),
    rotsB[i] = (i*n*p) % (m*p), for i in [0, m).

    Matches matmult/bmm1_cipher.go blockRotLists.
    """
    step = n * p
    rots_a = [(i * step) % (n * m) for i in range(m)]
    rots_b = [(i * step) % (m * p) for i in range(m)]
    return rots_a, rots_b


def required_bmm1_rotations(n: int, m: int, p: int) -> list[int]:
    """Distinct non-zero shifts the kernel performs (union over A and B).

    Used both to mint Galois keys ahead of time and to size the batched
    rotation call.
    """
    rots_a, rots_b = block_rot_lists(n, m, p)
    return sorted({k for k in rots_a + rots_b if k != 0})


def encrypt_a(ctx: Context, A: np.ndarray, p: int) -> int:
    """Bicyclic-encode A (n x m), tile so rotations stay valid, encrypt.

    We tile A's encoding ``ceil((m + p) / m)`` times. After CKKS rotation
    by up to (n*m - 1), we read ``n*p`` consecutive slots; that read
    window only sits inside populated data if the tiled vector has at
    least ``n*m + n*p`` non-padding slots, i.e. >= ``ceil((m+p)/m)``
    copies of the n*m encoding. Matches matmult/bmm1_cipher.go ``aTiles``.
    """
    n, m = A.shape
    a_tiles = max(1, math.ceil((m + p) / m))
    vec = repeat_vector(bicyclic_encode(A), a_tiles)
    return ctx.encrypt(vec)


def encrypt_b(ctx: Context, B: np.ndarray, n: int) -> int:
    """Bicyclic-encode B (m x p), tile ``ceil((m+n)/m)`` times, encrypt."""
    m, p = B.shape
    b_tiles = max(1, math.ceil((m + n) / m))
    vec = repeat_vector(bicyclic_encode(B), b_tiles)
    return ctx.encrypt(vec)


def bmm1_he_hoisted(
    ctx: Context,
    ct_a: int,
    ct_b: int,
    n: int,
    m: int,
    p: int,
) -> int:
    """Compute C = A @ B on bicyclic-encoded ciphertexts, hoisted + lazy relin.

    Returns a single ciphertext id; the first n*p slots decode to C.

    The control flow exactly matches Negar's Bmm1HEHoisted (Bmm1Accumulate
    fed with pre-hoisted shift maps for both operands), except:
    - we use ctx.mul_nr (no immediate relin) per term,
    - we accumulate at degree 2 (the desilo engine.add tolerates this),
    - we apply one ctx.relin at the end before returning.

    The decrypt path uses degree-1 ciphertexts in desilo, so the final
    relin is required (not optional like in the Lattigo reference, which
    leaves it commented out).
    """
    rots_a, rots_b = block_rot_lists(n, m, p)

    # Distinct non-zero shifts for hoisted batched rotation.
    distinct_a = sorted({k for k in rots_a if k != 0})
    distinct_b = sorted({k for k in rots_b if k != 0})

    a_hoist: dict[int, int] = {}
    b_hoist: dict[int, int] = {}

    if distinct_a:
        rotated = ctx.rot_batch(ct_a, distinct_a)
        a_hoist.update(dict(zip(distinct_a, rotated)))
    if distinct_b:
        rotated = ctx.rot_batch(ct_b, distinct_b)
        b_hoist.update(dict(zip(distinct_b, rotated)))

    def pick(src: int, shift: int, hoist: dict[int, int]) -> int:
        if shift == 0:
            return src
        return hoist[shift]

    acc: int | None = None
    for i in range(m):
        a = pick(ct_a, rots_a[i], a_hoist)
        b = pick(ct_b, rots_b[i], b_hoist)
        prod = ctx.mul_nr(a, b)
        prod = ctx.rescale(prod)
        if acc is None:
            acc = prod
        else:
            acc = ctx.add(acc, prod)
    assert acc is not None
    return ctx.relin(acc)


def decode_output(ctx: Context, ct_id: int, n: int, p: int) -> np.ndarray:
    """Decrypt + bicyclic_decode to recover the (n x p) product."""
    vec = ctx.decrypt(ct_id)
    return bicyclic_decode(vec[: n * p], n, p)


def bmm1_he(
    ctx: Context,
    A: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """End-to-end: encrypt A and B, run the kernel, decrypt to C.

    Convenience for tests / single-block benchmarks. ``A.shape == (n, m)``,
    ``B.shape == (m, p)``; requires n, m, p pairwise coprime and n*m, m*p
    each <= ctx.slots after tiling.
    """
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, f"matmul shape mismatch: A is {A.shape}, B is {B.shape}"
    ct_a = encrypt_a(ctx, A, p)
    ct_b = encrypt_b(ctx, B, n)
    ct_c = bmm1_he_hoisted(ctx, ct_a, ct_b, n, m, p)
    return decode_output(ctx, ct_c, n, p)
