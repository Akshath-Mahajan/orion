"""Oracle tests for the BMM-III (multi-chunk bicycle) CKKS kernel.

Runs against both backends via the conftest.py ctx fixture. The tests use
n_he < ctx.slots to exercise the multi-chunk LongRot path with shapes
small enough to keep test runtime in the seconds, not minutes -- the
algorithm parameter ``n_he`` is decoupled from ``ctx.slots`` precisely so
this is possible (the chunk is tiled across slots before encryption).

Three nested checks per backend:

1. Plaintext oracle vs numpy.matmul -- already covered by
   benchmarks/matmul_encodings/plaintext/test_oracles.py; not duplicated.
2. Ciphertext kernel decrypt vs numpy.matmul (catches arithmetic bugs).
3. Ciphertext kernel decrypt vs the encoding-aware bmm3_matmul_plain
   oracle (catches encoding/chunking bugs the numpy reference would
   silently absorb).

Tolerance: atol=1e-1 (CKKS noise floor at this depth is ~1e-3 in practice;
generous bound to keep flaky-CI risk low. PLAN.md §7 already flags this
to be tightened across the suite once we have a stable baseline).
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.bmm3_cipher import bmm3_he
from benchmarks.matmul_encodings.plaintext.bmm3_plain import bmm3_matmul_plain


# (n, m, p, n_he). Each picks pairwise-coprime (n, m, p) and an n_he small
# enough that n*m and m*p exceed it, so multi-chunk LongRot is exercised.
SHAPES = [
    (5, 7, 11, 16),    # small smoke test, multi-chunk on both sides
    (4, 5, 7, 16),     # ct_a needs more tiling (4*5=20 < 4*7=28, n_he=16)
    (8, 9, 11, 32),    # bigger inputs, more output chunks
]


@pytest.mark.parametrize("mode", ["cached", "hoisted"])
@pytest.mark.parametrize("nmpn", SHAPES)
def test_bmm3_cipher_matches_numpy(ctx, nmpn, mode):
    """Decrypt of CKKS BMM-III matches numpy.matmul within CKKS tolerance.

    Parametrised over both ``cached`` and ``hoisted`` modes so the
    hoist-planning + precompute path is exercised identically to the
    one-rotation-at-a-time path. Negar's paper preset is hoisted.
    """
    n, m, p, n_he = nmpn
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    got = bmm3_he(ctx, A, B, n_he=n_he, mode=mode)
    want = A @ B

    assert got.shape == want.shape
    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"BMM-III cipher (mode={mode}) deviates from numpy.matmul by "
        f"{max_err:.3e} (n={n}, m={m}, p={p}, n_he={n_he}, "
        f"backend={type(ctx.backend).__name__})"
    )


@pytest.mark.parametrize("mode", ["cached", "hoisted"])
def test_bmm3_cipher_matches_plaintext_oracle(ctx, mode):
    """Decrypt of the CKKS kernel matches the encoding-aware plaintext oracle.

    Catches encoding/chunking bugs that the numpy reference would absorb:
    if the chunked LongRot is wrong but happens to land slots in positions
    that bicyclic_decode also gets wrong in the same way, the numpy check
    could pass and the oracle check would still catch it.
    """
    n, m, p, n_he = 5, 7, 11, 16
    rng = np.random.default_rng(seed=123)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    got = bmm3_he(ctx, A, B, n_he=n_he, mode=mode)
    want, _counts = bmm3_matmul_plain(A, B, n_he=n_he)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"BMM-III cipher (mode={mode}) vs plaintext oracle: "
        f"max_err={max_err:.3e} "
        f"(backend={type(ctx.backend).__name__})"
    )


@pytest.mark.parametrize("hoist_block_size", [1, 4, 8, 16])
def test_bmm3_hoisted_block_sizes(ctx, hoist_block_size):
    """Hoisted mode is correct across the block-size sweep we care about
    for the paper preset (Negar's notes: 8/16/32, 16 best for larger).

    Uses ``m=7`` so block_size=1 / 4 / 8 / 16 all exercise distinct
    boundary cases (m % block_size != 0 for some, == 0 for others).
    """
    n, m, p, n_he = 5, 7, 11, 16
    rng = np.random.default_rng(seed=99)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    got = bmm3_he(
        ctx, A, B, n_he=n_he,
        mode="hoisted", hoist_block_size=hoist_block_size,
    )
    want = A @ B
    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"BMM-III hoisted (block={hoist_block_size}) deviates from "
        f"numpy.matmul by {max_err:.3e} "
        f"(backend={type(ctx.backend).__name__})"
    )
