"""Oracle tests for THOR (Algorithm 2) on the matmul-encoding harness."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.matmul_encodings.kernels.thor_cipher import thor_he_end_to_end
from benchmarks.matmul_encodings.plaintext.thor_plain import (
    build_masks_for_kernel,
    encode_batched,
    replication,
    thor_cc_matmul_plain,
    decode_batched,
)


def _matmul_per_head(As: np.ndarray, Bs: np.ndarray) -> np.ndarray:
    """Reference: for each head h in [0, H), result[h] = As[h] @ Bs[h]."""
    H, m, n = As.shape
    _, _, n_b = Bs.shape
    out = np.zeros((H, m, n_b), dtype=np.float64)
    for h in range(H):
        out[h] = As[h] @ Bs[h]
    return out


@pytest.mark.parametrize("dnH_c", [(4, 4, 2, 2), (8, 8, 2, 4)])
def test_thor_cipher_matches_per_head_matmul(ctx, dnH_c):
    """End-to-end: THOR HE kernel matches numpy per-head matmul.

    Parameters are (d, n, H, c). Required: d == n (square per-head matmul),
    d % c == 0, n % c == 0, c * n * H <= ctx.slots.
    """
    d, n, H, c = dnH_c
    s = c * n * H
    assert s <= ctx.slots, f"need slots >= {s}, have {ctx.slots}"

    rng = np.random.default_rng(seed=42)
    As = rng.standard_normal((H, d, n))
    Bs = rng.standard_normal((H, d, n))

    got = thor_he_end_to_end(ctx, As, Bs, c=c)
    want = _matmul_per_head(As, Bs)

    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 5e-1, (
        f"THOR cipher result deviates from per-head matmul by {max_err:.3e} "
        f"(d={d}, n={n}, H={H}, c={c}, backend={type(ctx.backend).__name__})"
    )


def test_thor_cipher_matches_plaintext_oracle(ctx):
    """Decrypt of CKKS THOR kernel matches the plaintext THOR oracle."""
    d, n, H, c = 4, 4, 2, 2
    s = c * n * H
    if s > ctx.slots:
        pytest.skip(f"slots {ctx.slots} too small for s={s}")

    rng = np.random.default_rng(seed=2026)
    As = rng.standard_normal((H, d, n))
    Bs = rng.standard_normal((H, d, n))

    # Plaintext oracle path: same encoding, same kernel structure
    p_as = encode_batched(As, c=c, H=H)
    p_b = encode_batched(Bs, c=c, H=H)
    p_b_rep = replication(p_b, n=n, c=c, H=H)
    masks = build_masks_for_kernel(c=c, n=n, H=H, s=s)
    plain_out, _ = thor_cc_matmul_plain(p_as, p_b_rep, masks, d=d, n=n, H=H, s=s)
    want = decode_batched(plain_out, m=d, n=n, c=c, H=H)

    got = thor_he_end_to_end(ctx, As, Bs, c=c)
    max_err = float(np.max(np.abs(got - want)))
    assert max_err < 5e-1
