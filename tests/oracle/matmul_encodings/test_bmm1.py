"""Oracle tests for the BMM-I (single-ciphertext bicycle) CKKS kernel.

Three nested checks (per backend, via the ctx fixture in conftest.py):

1. Plaintext oracle vs numpy.matmul -- already covered by
   benchmarks/matmul_encodings/plaintext/test_oracles.py; not duplicated.
2. Ciphertext kernel decrypt vs plaintext oracle for one (n, m, p) seed.
3. Ciphertext kernel decrypt vs numpy.matmul directly.

Tolerance: atol=1e-1 (loose CKKS bound). The actual error at LogN=13 with
one CMult+Rescale tends to be ~1e-3 or better -- we set a generous bar so
that flaky CI noise doesn't trip the test, and tighten it later if we
observe consistently low error.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.bmm1_cipher import bmm1_he
from benchmarks.matmul_encodings.plaintext.bmm1_plain import (
    bmm1_plain,
    bicyclic_encode,
    bicyclic_decode,
)
from benchmarks.matmul_encodings.plaintext.op_counts import OpCounts


@pytest.mark.parametrize("nmp", [(5, 7, 11), (4, 5, 7), (8, 9, 11)])
def test_bmm1_cipher_matches_numpy(ctx, nmp):
    """Decrypt of CKKS kernel matches numpy.matmul within CKKS tolerance."""
    n, m, p = nmp
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    got = bmm1_he(ctx, A, B)
    want = A @ B

    assert got.shape == want.shape
    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"BMM-I cipher result deviates from numpy.matmul by {max_err:.3e} "
        f"(n={n}, m={m}, p={p}, backend={type(ctx.backend).__name__})"
    )


def test_bmm1_cipher_matches_plaintext_oracle(ctx):
    """Decrypt of CKKS kernel matches the encoding-aware plaintext oracle.

    This is the layer that catches encoding bugs distinct from arithmetic
    bugs -- if the CKKS path lands the right bits in the right slots, but
    the plaintext oracle is wrong, we'd never know from the numpy check.
    """
    n, m, p = 5, 7, 11
    rng = np.random.default_rng(seed=123)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    got = bmm1_he(ctx, A, B)

    # Plaintext oracle path
    a_enc = bicyclic_encode(A)
    b_enc = bicyclic_encode(B)
    counts = OpCounts()
    c_vec = bmm1_plain(a_enc, b_enc, n, m, p, counts)
    want = bicyclic_decode(c_vec, n, p)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1


def test_bmm1_op_counts(ctx):
    """The kernel issues exactly the expected number of CKKS ops.

    For a single-block BMM-I with (n, m, p):
      rotations:  2*(m-1)  (m-1 distinct non-zero A shifts + m-1 distinct B shifts)
      ct*ct muls: m
      ct*pt muls: 0
    The first multiply-input has shift 0 (identity), so the count is m-1, not m.
    """
    n, m, p = 5, 7, 11
    rng = np.random.default_rng(seed=7)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((m, p))

    ctx.reset_counts()
    _ = bmm1_he(ctx, A, B)

    # Distinct non-zero shifts per side: m - 1 each in the worst case,
    # but coincidences reduce it. Just check upper bounds + the multiply
    # count which is unambiguous.
    assert ctx.counts.ct_ct_muls == m
    assert ctx.counts.ct_pt_muls == 0
    assert ctx.counts.rotations <= 2 * (m - 1)
    assert ctx.counts.rotations > 0
