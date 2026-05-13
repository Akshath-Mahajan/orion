"""Plaintext packing, replication, mask construction, and matmul kernel for
THOR (Moon et al., CCS 2025, Algorithm 2).

Port of matmult/thor_plain.go (packing + masks) and matmult/thor_runner.go
(the `thorCCMatMulPlain` kernel).

Layout of every packed vector of length s = c*n*H:
        idx  =  r * (n*H)  +  h  +  H * t
with
    r in [0, c)  -- block index (which diagonal group)
    t in [0, n)  -- position within an interleaved diagonal
    h in [0, H)  -- head index (heads are interleaved, not stacked)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bmm1_plain import rotate_vec
from .op_counts import OpCounts


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _rotate_slots(v: np.ndarray, k: int) -> np.ndarray:
    """Alias of rotate_vec; matches Go rotateSlots semantics."""
    return rotate_vec(v, k)


def _pos_mod(a: int, m: int) -> int:
    r = a % m
    if r < 0:
        r += m
    return r


def _any_non_zero(v: np.ndarray) -> bool:
    return bool(np.any(v != 0.0))


# ---------------------------------------------------------------------------
# Diagonal extraction
# ---------------------------------------------------------------------------


def lower_diagonal(A: np.ndarray, ell: int) -> np.ndarray:
    """Extract the lower diagonal at offset ell from an (m, n) matrix.

    lower_diagonal(A, ell)[t] = A[(t + ell) mod m, t], t in [0, n).
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        out[t] = A[(t + ell) % m, t]
    return out


# ---------------------------------------------------------------------------
# Encode / decode (batched, interleaved diagonals)
# ---------------------------------------------------------------------------


def encode_batched(Ms: np.ndarray, c: int, H: int) -> np.ndarray:
    """Pack H matrices of shape (m, n) into m_c = m/c flat vectors of length
    s = c * n * H.

    Each output vector contains c interleaved diagonals, concatenated. The
    j-th output vector encodes diagonals {c*j, c*j+1, ..., c*j+c-1}.

    `Ms` has shape (H, m, n). Returns shape (m_c, s).
    """
    Ms = np.asarray(Ms, dtype=np.float64)
    H_in, m, n = Ms.shape
    assert H_in == H, f"Ms first dim {H_in} != H {H}"
    nH = n * H
    s = c * nH
    mc = m // c

    out = np.zeros((mc, s), dtype=np.float64)
    for j in range(mc):
        for r in range(c):
            ell = c * j + r
            base = r * nH
            for z in range(H):
                for t in range(n):
                    out[j, base + z + H * t] = Ms[z, (t + ell) % m, t]
    return out


def decode_batched(
    vecs: np.ndarray, m: int, n: int, c: int, H: int
) -> np.ndarray:
    """Invert encode_batched; recover H matrices of shape (m, n) from m_c
    decrypted vectors.

    `vecs` has shape (m_c, s). Returns shape (H, m, n).
    """
    vecs = np.asarray(vecs, dtype=np.float64)
    nH = n * H

    out = np.zeros((H, m, n), dtype=np.float64)
    for j, vec in enumerate(vecs):
        for r in range(c):
            ell = c * j + r
            base = r * nH
            for z in range(H):
                for t in range(n):
                    out[z, (t + ell) % m, t] = vec[base + z + H * t]
    return out


def replication(b_packed: np.ndarray, n: int, c: int, H: int) -> np.ndarray:
    """Expand the n_c = n/c packed B vectors into the n ciphertext-ready
    vectors consumed by the kernel.

    Each output vector holds c copies of one interleaved diagonal (length
    nH, tiled c times to length s).

    `b_packed` has shape (n_c, s). Returns shape (n, s).
    """
    b_packed = np.asarray(b_packed, dtype=np.float64)
    nH = n * H
    s = c * nH
    nc = n // c

    out_list: list[np.ndarray] = []
    for j in range(nc):
        ct = b_packed[j]
        for k in range(c):
            diag = ct[k * nH : (k + 1) * nH]
            rep = np.tile(diag, c)
            assert rep.shape[0] == s
            out_list.append(rep)
    return np.stack(out_list, axis=0)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------


def build_ell_masks(
    ell: int, c: int, n: int, H: int, s: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the three Algorithm-2 routing masks for a given ell.

    Returns (mu0, mu1, mu2). mu3 is computed by the kernel as
        v - v*mu0 - v*mu1 - v*mu2
    to save one plaintext multiplication.
    """
    nH = n * H
    ell_c = _pos_mod(ell, c)

    mu0 = np.zeros(s, dtype=np.float64)
    mu1 = np.zeros(s, dtype=np.float64)
    mu2 = np.zeros(s, dtype=np.float64)

    for r in range(c):
        for t in range(n):
            r_prime = _pos_mod(r - ell_c + (t + ell) // n, c)
            same_block = r_prime < (c - ell_c)
            r_out = _pos_mod(r_prime + ell, c)
            needs_rot = r_out != r

            for h in range(H):
                idx = r * nH + h + H * t
                if (not same_block) and (not needs_rot):
                    mu0[idx] = 1.0
                elif same_block and (not needs_rot):
                    mu1[idx] = 1.0
                elif (not same_block) and needs_rot:
                    mu2[idx] = 1.0
                # else: same_block & needs_rot -> mu3, obtained for free.
    return mu0, mu1, mu2


# ---------------------------------------------------------------------------
# Plain-ell-mask container used by the kernel
# ---------------------------------------------------------------------------


@dataclass
class PlainEllMask:
    mu0: np.ndarray
    mu1: np.ndarray
    mu2: np.ndarray
    has_mu0: bool
    has_mu2: bool


def build_masks_for_kernel(c: int, n: int, H: int, s: int) -> list[PlainEllMask]:
    """Build masks for ell in [1, n) and wrap them in PlainEllMask records."""
    masks: list[PlainEllMask] = []
    for ell in range(1, n):
        mu0, mu1, mu2 = build_ell_masks(ell, c, n, H, s)
        masks.append(
            PlainEllMask(
                mu0=mu0,
                mu1=mu1,
                mu2=mu2,
                has_mu0=_any_non_zero(mu0),
                has_mu2=_any_non_zero(mu2),
            )
        )
    return masks


# ---------------------------------------------------------------------------
# Plaintext kernel -- a literal port of ThorCCMatMulHE / thorCCMatMulPlain.
# ---------------------------------------------------------------------------


def thor_cc_matmul_plain(
    p_as: np.ndarray,
    p_b_rep: np.ndarray,
    masks: list[PlainEllMask],
    d: int,
    n: int,
    H: int,
    s: int,
) -> tuple[np.ndarray, OpCounts]:
    """Algorithm 2 on plaintext slot vectors.

    Mirrors thorCCMatMulPlain in matmult/thor_runner.go line-for-line.

    Returns the output slot-vectors of shape (m_c, s) and an OpCounts tally.
    """
    p_as = np.asarray(p_as, dtype=np.float64)
    p_b_rep = np.asarray(p_b_rep, dtype=np.float64)

    c = s // (n * H)
    nH = n * H
    mc = d // c

    counts = OpCounts()

    # Lines 4-8: intermediate products.
    p_cjl: list[list[np.ndarray | None]] = [
        [None for _ in range(n)] for _ in range(mc)
    ]
    for j in range(mc):
        p_cjl[j][0] = p_as[j] * p_b_rep[0]
        counts.ct_ct_muls += 1

        for ell in range(1, n):
            shift = (-n * (ell % c) + ell) * H

            rot = _rotate_slots(p_as[j], shift)
            counts.rotations += 1

            p_cjl[j][ell] = rot * p_b_rep[ell]
            counts.ct_ct_muls += 1

    # Lines 9-17: masking and accumulation.
    acc_prime: list[np.ndarray | None] = [None for _ in range(mc)]
    acc_d_prime: list[np.ndarray | None] = [None for _ in range(mc)]

    for ell in range(1, n):
        m = masks[ell - 1]
        ell_q = ell // c
        for j in range(mc):
            v = p_cjl[j][ell]
            j_same = (j + ell_q) % mc
            j_next = (j + ell_q + 1) % mc

            v1 = v * m.mu1
            counts.ct_pt_muls += 1
            if acc_prime[j_same] is None:
                acc_prime[j_same] = v1.copy()
            else:
                acc_prime[j_same] = acc_prime[j_same] + v1

            v0 = None
            v2 = None
            if m.has_mu0:
                v0 = v * m.mu0
                counts.ct_pt_muls += 1
                if acc_prime[j_next] is None:
                    acc_prime[j_next] = v0.copy()
                else:
                    acc_prime[j_next] = acc_prime[j_next] + v0
            if m.has_mu2:
                v2 = v * m.mu2
                counts.ct_pt_muls += 1
                if acc_d_prime[j_next] is None:
                    acc_d_prime[j_next] = v2.copy()
                else:
                    acc_d_prime[j_next] = acc_d_prime[j_next] + v2

            # v3 = v - v0 - v1 - v2 (free mu3, NOT counted)
            v3 = v.copy()
            if v0 is not None:
                v3 = v3 - v0
            v3 = v3 - v1
            if v2 is not None:
                v3 = v3 - v2
            if acc_d_prime[j_same] is None:
                acc_d_prime[j_same] = v3
            else:
                acc_d_prime[j_same] = acc_d_prime[j_same] + v3

    # Line 18: final assembly.
    out = np.zeros((mc, s), dtype=np.float64)
    for j in range(mc):
        res = p_cjl[j][0].copy()
        if acc_prime[j] is not None:
            res = res + acc_prime[j]
        if acc_d_prime[j] is not None:
            rolled = _rotate_slots(acc_d_prime[j], -nH)
            counts.rotations += 1
            res = res + rolled
        out[j] = res
    return out, counts
