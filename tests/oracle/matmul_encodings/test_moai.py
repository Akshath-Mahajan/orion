"""Oracle tests for MOAI Algorithm 3 (Col x Col -> Diag, BSGS)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.moai_cipher import (
    moai_col_col_bsgs_end_to_end,
)


def _qkt_per_batch(Qs: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    """Reference: for each batch s, Qs[s] @ Ks[s].T."""
    n_batch, m, _ = Qs.shape
    out = np.zeros((n_batch, m, m), dtype=np.float64)
    for s in range(n_batch):
        out[s] = Qs[s] @ Ks[s].T
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
