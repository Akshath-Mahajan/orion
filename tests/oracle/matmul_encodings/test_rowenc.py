"""Oracle tests for the row-encoding (row-pack) CKKS kernel.

Three nested checks per backend:

1. Plaintext oracle vs numpy.matmul -- already covered by
   benchmarks/matmul_encodings/plaintext/test_oracles.py; not duplicated.
2. Ciphertext kernel decrypt vs numpy.matmul.
3. Ciphertext kernel decrypt vs the encoding-aware rowenc_matmul_plain
   oracle (catches encoding bugs the numpy reference would absorb).

Constraint: n must be a power of 2 (the log2(n) replication steps need
integer shift amounts) and n^2 must fit in ctx.slots. Tested at n in
{2, 4, 8} -- with ctx.slots = 4096 the kernel comfortably runs up to
n = 64, but small n keeps test runtime tight.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.rowenc_cipher import rowenc_he
from benchmarks.matmul_encodings.plaintext.rowenc_plain import rowenc_matmul_plain


@pytest.mark.parametrize("n", [2, 4, 8])
def test_rowenc_cipher_matches_numpy(ctx, n):
    """Decrypt of CKKS row-encoding kernel matches numpy.matmul."""
    if n * n > ctx.slots:
        pytest.skip(f"n*n={n*n} > ctx.slots={ctx.slots}")

    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))

    got = rowenc_he(ctx, A, B)
    want = A @ B

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"rowenc cipher result deviates from numpy.matmul by {max_err:.3e} "
        f"(n={n}, backend={type(ctx.backend).__name__})"
    )


def test_rowenc_cipher_matches_plaintext_oracle(ctx):
    """Decrypt of CKKS kernel matches the encoding-aware plaintext oracle."""
    n = 4
    if n * n > ctx.slots:
        pytest.skip(f"n*n={n*n} > ctx.slots={ctx.slots}")

    rng = np.random.default_rng(seed=123)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))

    got = rowenc_he(ctx, A, B)
    want, _counts = rowenc_matmul_plain(A, B)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 1e-1, (
        f"rowenc cipher vs plaintext oracle: max_err={max_err:.3e} "
        f"(n={n}, backend={type(ctx.backend).__name__})"
    )
