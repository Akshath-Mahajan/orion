"""Oracle tests for MOAI Algorithms 3 and 4.

Algorithm 3: Col x Col -> Diag (BSGS) -- Q . K^T in transformer attention.
Algorithm 4: Diag x Col -> Col (BSGS) -- (Q . K^T) . V in transformer attention.

The two compose without re-encoding (Alg 3's diag output = Alg 4's diag input),
which is the property that motivates MOAI as an encoding choice. The chained
test covers that composition end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.moai_cipher import (
    moai_col_col_bsgs_end_to_end,
    moai_diag_col_bsgs_end_to_end,
    moai_qkt_v_he,
)


def _qkt_per_batch(Qs: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    """Reference: for each batch s, Qs[s] @ Ks[s].T."""
    n_batch, m, _ = Qs.shape
    out = np.zeros((n_batch, m, m), dtype=np.float64)
    for s in range(n_batch):
        out[s] = Qs[s] @ Ks[s].T
    return out


def _cv_per_batch(Cs: np.ndarray, Vs: np.ndarray) -> np.ndarray:
    """Reference: for each batch s, Cs[s] @ Vs[s]."""
    n_batch, m, _ = Cs.shape
    _, _, d_prime = Vs.shape
    out = np.zeros((n_batch, m, d_prime), dtype=np.float64)
    for s in range(n_batch):
        out[s] = Cs[s] @ Vs[s]
    return out


@pytest.mark.parametrize("n_batch_m_d", [(2, 4, 4), (2, 8, 4), (4, 4, 2)])
def test_moai_col_col_bsgs_matches_qkt(ctx, n_batch_m_d):
    """MOAI Col x Col BSGS HE kernel matches per-batch Q @ K.T."""
    n_batch, m, d_prime = n_batch_m_d
    n_he = n_batch * m
    if n_he > ctx.slots or ctx.slots % n_he != 0:
        pytest.skip(f"slots {ctx.slots} not a positive multiple of n_he={n_he}")

    rng = np.random.default_rng(seed=42)
    Qs = rng.standard_normal((n_batch, m, d_prime))
    Ks = rng.standard_normal((n_batch, m, d_prime))

    got = moai_col_col_bsgs_end_to_end(ctx, Qs, Ks)
    want = _qkt_per_batch(Qs, Ks)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 5e-1, (
        f"MOAI Col x Col BSGS deviates from per-batch Q@K.T by {max_err:.3e} "
        f"(n_batch={n_batch}, m={m}, d_prime={d_prime}, "
        f"backend={type(ctx.backend).__name__})"
    )


@pytest.mark.parametrize("n_batch_m_d", [(2, 4, 4), (2, 8, 4), (4, 4, 2)])
def test_moai_diag_col_bsgs_matches_cv(ctx, n_batch_m_d):
    """MOAI Diag x Col BSGS HE kernel matches per-batch C @ V.

    C is (m, m) diag-packed, V is (m, d') col-packed; output is (m, d')
    col-packed. Test fills C with random values per diagonal -- the kernel
    only ever reads diag-packed slots, so the rest of the matrix is implicit
    in the diagonal layout.
    """
    n_batch, m, d_prime = n_batch_m_d
    n_he = n_batch * m
    if n_he > ctx.slots or ctx.slots % n_he != 0:
        pytest.skip(f"slots {ctx.slots} not a positive multiple of n_he={n_he}")

    rng = np.random.default_rng(seed=42)
    Cs = rng.standard_normal((n_batch, m, m))
    Vs = rng.standard_normal((n_batch, m, d_prime))

    got = moai_diag_col_bsgs_end_to_end(ctx, Cs, Vs)
    want = _cv_per_batch(Cs, Vs)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 5e-1, (
        f"MOAI Diag x Col BSGS deviates from per-batch C@V by {max_err:.3e} "
        f"(n_batch={n_batch}, m={m}, d_prime={d_prime}, "
        f"backend={type(ctx.backend).__name__})"
    )


def test_moai_qkt_v_chain_matches_reference(ctx):
    """End-to-end Q . K^T . V on encrypted inputs matches the numpy chain.

    This is the property that motivates MOAI: Algorithm 3's diag-packed
    output is Algorithm 4's diag-packed input; the chain runs without any
    decrypt-and-repack between matmuls.
    """
    n_batch, m, d_prime, d_v = 2, 4, 4, 4
    n_he = n_batch * m
    if n_he > ctx.slots or ctx.slots % n_he != 0:
        pytest.skip(f"slots {ctx.slots} not a positive multiple of n_he={n_he}")

    rng = np.random.default_rng(seed=42)
    Qs = rng.standard_normal((n_batch, m, d_prime))
    Ks = rng.standard_normal((n_batch, m, d_prime))
    Vs = rng.standard_normal((n_batch, m, d_v))

    got = moai_qkt_v_he(ctx, Qs, Ks, Vs)

    want = np.zeros_like(got)
    for s in range(n_batch):
        want[s] = Qs[s] @ Ks[s].T @ Vs[s]

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 5e-1, (
        f"MOAI Q.K^T.V chain deviates from numpy by {max_err:.3e} "
        f"(backend={type(ctx.backend).__name__})"
    )
