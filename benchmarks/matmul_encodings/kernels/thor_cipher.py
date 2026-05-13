"""THOR (Moon et al., CCS 2025) CKKS matmul kernel.

Port of Algorithm 2 (CC-MatMul, interleaved diagonal packing) following
matmult/thor_runner.go's thorCCMatMulPlain reference and
matmult/thor_cipher.go's ThorCCMatMul* HE entry points.

Encoding (see plaintext.thor_plain for the math):
    s = c * n * H        -- slots per ciphertext, c = s/(n*H)
    p_as[j]   in C^s     -- A side, c interleaved diagonals per ciphertext (m_c total)
    p_b_rep[ell] in C^s  -- B side, replicated diagonal, n total

Kernel sketch:
    for j in [0, m_c):
        p_cjl[j][0]   = p_as[j] * p_b_rep[0]
        for ell in [1, n):
            shift          = (-n*(ell%c) + ell) * H
            p_cjl[j][ell]  = Rot(p_as[j], shift) * p_b_rep[ell]
    accumulate via four routing masks mu0/mu1/mu2/mu3, then
    res[j] = p_cjl[j][0] + acc_prime[j] + Rot(acc_d_prime[j], -nH)

This first port uses eager relinearization. A later commit can swap
the ct.ct multiplies for ctx.mul_nr + a single relin at the end to
recover the lazy-relin speedup (Negar's ThorCCMatMulHELazyRelin path);
the kernel logic is unchanged, only the binding verbs differ.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..context import Context
from ..plaintext.thor_plain import (
    PlainEllMask,
    build_masks_for_kernel,
    decode_batched,
    encode_batched,
    replication,
)


def required_thor_rotations(c: int, n: int, H: int) -> list[int]:
    """Distinct non-zero shifts the kernel issues per j-block.

    For ell in [1, n): shift = (-n*(ell%c) + ell) * H.
    Plus one final shift of -nH for the d_prime accumulator.
    """
    nH = n * H
    shifts = {((-n) * (ell % c) + ell) * H for ell in range(1, n)}
    shifts.add(-nH)
    shifts.discard(0)
    return sorted(shifts)


def _tile_to_slots(vec: np.ndarray, slots: int) -> np.ndarray:
    """Tile `vec` cyclically to fill `slots` so that CKKS slot-rotations
    wrap inside valid data (instead of into zero padding).

    Required because CKKS rotates over all slot_count slots, while the
    encoding only naturally fills c*n*H slots. Without tiling, rotated
    elements from outside the encoded length would be zeros, not the
    cyclic wraparound the algorithm assumes.
    """
    L = vec.size
    assert slots % L == 0, f"slots {slots} must be a multiple of encoded length {L}"
    return np.tile(vec, slots // L)


def encrypt_a(ctx: Context, A: np.ndarray, c: int, H: int) -> list[int]:
    """Pack A (shape (H, m, n)) into the m_c interleaved-diagonal vectors,
    tile each to fill the slot count, and encrypt.
    """
    p_as = encode_batched(A, c=c, H=H)   # shape (m_c, s)
    return [ctx.encrypt(_tile_to_slots(p_as[j], ctx.slots)) for j in range(p_as.shape[0])]


def encrypt_b_rep(ctx: Context, B: np.ndarray, c: int, H: int) -> list[int]:
    """Pack B (shape (H, m, n)) into the n replicated-diagonal vectors,
    tile, encrypt.

    Note: B uses the same encode_batched, but with a downstream `replication`
    step to produce the n vectors the kernel consumes (vs A's m_c vectors).
    """
    H_, m, n = B.shape
    assert H_ == H
    b_packed = encode_batched(B, c=c, H=H)        # (n_c, s)
    b_rep = replication(b_packed, n=n, c=c, H=H)  # (n, s)
    return [ctx.encrypt(_tile_to_slots(b_rep[ell], ctx.slots)) for ell in range(n)]


def _encode_mask(ctx: Context, mask_vec: np.ndarray, level: int) -> int | None:
    """Encode a mask at the given level. Returns None if mask is all-zero.

    Tiles the length-s mask to fill the slot count so the v*mu product
    stays non-zero across all tiles -- the final Rot(acc_d_prime, -nH)
    relies on cyclic wraparound through the full slot vector, not just
    the first s slots.
    """
    if not np.any(mask_vec != 0.0):
        return None
    tiled = _tile_to_slots(mask_vec, ctx.slots)
    return ctx.backend.Encode(
        tiled.tolist(),
        level,
        ctx.scheme.params.get_default_scale(),
    )


def _mu3_vec(mu0: np.ndarray | None, mu1: np.ndarray, mu2: np.ndarray | None, s: int) -> np.ndarray:
    """mu3 = 1 - mu0 - mu1 - mu2 (filling zero where a mask is absent)."""
    mu3 = np.ones(s, dtype=np.float64)
    if mu0 is not None:
        mu3 -= mu0
    mu3 -= mu1
    if mu2 is not None:
        mu3 -= mu2
    return mu3


def thor_cc_matmul_he(
    ctx: Context,
    p_as_cts: list[int],
    p_b_rep_cts: list[int],
    masks: list[PlainEllMask],
    n: int,
    H: int,
    c: int,
) -> list[int]:
    """Run Algorithm 2 on encrypted inputs. Returns m_c result ciphertexts.

    Output ciphertexts decrypt + decode_batched back to the (H, m, n)-shaped
    product matrix.

    ``c`` is the algorithm parameter (number of diagonals interleaved per
    ciphertext), NOT slot_count / (n*H). The encoded vector inside each
    ciphertext has length s = c*n*H and is tiled across the slot vector;
    confusing these two lengths produces a wrong shift schedule.
    """
    mc = len(p_as_cts)
    if mc == 0:
        return []
    nH = n * H
    s_enc = c * nH  # encoded length, NOT ctx.slots
    assert ctx.slots % s_enc == 0, f"slot count {ctx.slots} must be a multiple of c*n*H={s_enc}"

    # Mask plaintext encoding: each mask is consumed by v*mu where v is at
    # level L-1 (post ct*ct + rescale). After the v*mu + rescale, the
    # result lands at L-2. Encode at L-1.
    mul_level = ctx.scheme.params.get_max_level()
    encoded_masks: list[tuple[int | None, int | None, int | None, int]] = []
    for m_ in masks:
        mu3_vec = _mu3_vec(
            m_.mu0 if m_.has_mu0 else None, m_.mu1,
            m_.mu2 if m_.has_mu2 else None, s_enc,
        )
        encoded_masks.append((
            _encode_mask(ctx, m_.mu0, mul_level - 1) if m_.has_mu0 else None,
            _encode_mask(ctx, m_.mu1, mul_level - 1),
            _encode_mask(ctx, m_.mu2, mul_level - 1) if m_.has_mu2 else None,
            _encode_mask(ctx, mu3_vec, mul_level - 1),
        ))

    # All-ones mask used to drop p_cjl[j][0] one level (so it aligns
    # with the L-2 accumulators in the final assembly). Encoded across
    # the full slot vector so the multiply is non-zero everywhere.
    ones_pt = ctx.backend.Encode(
        [1.0] * ctx.slots, mul_level - 1, ctx.scheme.params.get_default_scale()
    )

    # Lines 4-8 of Algorithm 2: build all p_cjl[j][ell] products.
    p_cjl: list[list[int | None]] = [[None] * n for _ in range(mc)]
    for j in range(mc):
        # ell=0 product (no rotation)
        prod = ctx.mul_rl(p_as_cts[j], p_b_rep_cts[0])
        prod = ctx.rescale(prod)
        p_cjl[j][0] = prod

        # ell in [1, n): hoisted batched rotation of p_as[j], then mul.
        shifts = [((-n) * (ell % c) + ell) * H for ell in range(1, n)]
        rotated = ctx.rot_batch(p_as_cts[j], shifts)
        for ell, rot_ct in zip(range(1, n), rotated):
            prod = ctx.mul_rl(rot_ct, p_b_rep_cts[ell])
            prod = ctx.rescale(prod)
            p_cjl[j][ell] = prod

    # Lines 9-17: masking + accumulation into acc_prime / acc_d_prime.
    acc_prime: list[int | None] = [None] * mc
    acc_d_prime: list[int | None] = [None] * mc

    for ell in range(1, n):
        mu0_pt, mu1_pt, mu2_pt, mu3_pt = encoded_masks[ell - 1]
        ell_q = ell // c
        for j in range(mc):
            v = p_cjl[j][ell]
            j_same = (j + ell_q) % mc
            j_next = (j + ell_q + 1) % mc

            v1 = ctx.rescale(ctx.mul_pt(v, mu1_pt))
            acc_prime[j_same] = v1 if acc_prime[j_same] is None else ctx.add(acc_prime[j_same], v1)

            if mu0_pt is not None:
                v0 = ctx.rescale(ctx.mul_pt(v, mu0_pt))
                acc_prime[j_next] = v0 if acc_prime[j_next] is None else ctx.add(acc_prime[j_next], v0)
            if mu2_pt is not None:
                v2 = ctx.rescale(ctx.mul_pt(v, mu2_pt))
                acc_d_prime[j_next] = v2 if acc_d_prime[j_next] is None else ctx.add(acc_d_prime[j_next], v2)

            # v3 = v * mu3 (computed directly rather than v - v0 - v1 - v2,
            # which would require a DropLevel that neither binding exposes).
            # Pays one extra ct*pt per ell-iter; semantically identical.
            v3 = ctx.rescale(ctx.mul_pt(v, mu3_pt))
            acc_d_prime[j_same] = v3 if acc_d_prime[j_same] is None else ctx.add(acc_d_prime[j_same], v3)

    # Line 18: final assembly. p_cjl[j][0] is at level L-1, accumulators at
    # L-2. Drop p_cjl[j][0] one level by multiplying by an all-ones plaintext
    # and rescaling. (No DropLevel verb in either binding.)
    out: list[int] = []
    for j in range(mc):
        base = ctx.rescale(ctx.mul_pt(p_cjl[j][0], ones_pt))
        res = base
        if acc_prime[j] is not None:
            res = ctx.add(res, acc_prime[j])
        if acc_d_prime[j] is not None:
            rolled = ctx.rot(acc_d_prime[j], -nH)
            res = ctx.add(res, rolled)
        out.append(res)
    return out


def thor_he_end_to_end(
    ctx: Context,
    As: np.ndarray,
    Bs: np.ndarray,
    c: int,
) -> np.ndarray:
    """Encrypt H matrices of A and B, run the kernel, decrypt + decode.

    `As.shape == Bs.shape == (H, m, n)`. Returns the (H, m, n) product
    where for each head h, result[h] = As[h] @ Bs[h].
    """
    H, m, n = As.shape
    H2, m2, n2 = Bs.shape
    assert (H, m, n) == (H2, m2, n2)
    s = ctx.slots

    s_enc = c * n * H
    masks = build_masks_for_kernel(c=c, n=n, H=H, s=s_enc)

    p_as_cts = encrypt_a(ctx, As, c=c, H=H)
    p_b_rep_cts = encrypt_b_rep(ctx, Bs, c=c, H=H)
    out_cts = thor_cc_matmul_he(ctx, p_as_cts, p_b_rep_cts, masks, n=n, H=H, c=c)

    decoded_vecs = np.stack([ctx.decrypt(ct)[:s_enc] for ct in out_cts], axis=0)
    return decode_batched(decoded_vecs, m=m, n=n, c=c, H=H)
