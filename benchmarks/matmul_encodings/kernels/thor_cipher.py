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

Lazy relinearization (matches Negar's ThorCCMatMulHELazyRelin):
  * Each ct*ct multiply uses ctx.mul_nr, leaving a degree-2 product.
  * acc_prime / acc_d_prime accumulate at degree-2 (mul_pt and add
    preserve degree). The base term p_cjl[j][0] is also degree-2.
  * Final assembly:
      - merge base + acc_prime at degree-2, relinearize ONCE -> degree-1.
      - relinearize acc_d_prime ONCE -> degree-1.
      - rotate the relinearized acc_d_prime (rotation requires degree-1).
  * Net cost: m_c * n ct*ct multiplies pay only 2 * m_c relins instead
    of m_c * n. On desilo this is real (binding has MulNoRelinCiphertextNew
    + RelinearizeNew). On lattigo it degrades to eager via the binding
    fallback (mul_nr -> mul_rl, relin -> no-op); same numerical result,
    no speedup.
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

    # Lines 4-17 of Algorithm 2, fused: for each input row j, build
    # p_j[0..n-1], fold p_j[1..n-1] into the global accumulators, then
    # free p_j[1..n-1]. Only p_j[0] survives across outer-j (it's reused
    # in the final assembly). The original Negar formulation builds the
    # full mc*n p_cjl array up front; on GPU that OOMs around d=256 even
    # with eager free of rescale temporaries. Fusing drops peak working
    # set from O(mc*n) to O(mc + n) -- the mc per-row col-0 ciphertexts
    # plus one current row's n ciphertexts -- which lets the kernel
    # scale to d=2048 within the 24 GiB on an RTX 3090.
    #
    # GPU memory note: ctx.rescale on desilo is a clone (RescaleNew
    # calls engine.clone), so we also free pre-rescale temporaries.
    acc_prime: list[int | None] = [None] * mc
    acc_d_prime: list[int | None] = [None] * mc
    p_col0: list[int | None] = [None] * mc  # p_cjl[j][0] for each input j

    def _accumulate(acc_list, idx, v):
        """acc_list[idx] = (acc_list[idx] + v) or v; free the old acc and v
        if both were present. Returns nothing."""
        if acc_list[idx] is None:
            acc_list[idx] = v
        else:
            new_acc = ctx.add(acc_list[idx], v)
            ctx.free(acc_list[idx], v)
            acc_list[idx] = new_acc

    shifts = [((-n) * (ell % c) + ell) * H for ell in range(1, n)]

    for j in range(mc):
        # Build p_j[0]
        prod = ctx.mul_nr(p_as_cts[j], p_b_rep_cts[0])
        rescaled = ctx.rescale(prod)
        ctx.free(prod)
        p_col0[j] = rescaled

        # Build p_j[1..n-1] via hoisted batched rotation, then immediately
        # fold each into the accumulators and free.
        rotated = ctx.rot_batch(p_as_cts[j], shifts)
        for ell, rot_ct in zip(range(1, n), rotated):
            prod = ctx.mul_nr(rot_ct, p_b_rep_cts[ell])
            ctx.free(rot_ct)
            v = ctx.rescale(prod)
            ctx.free(prod)

            # Fold v into accumulators (same logic as the original Phase 2
            # inner body, just running per-(j, ell) inline).
            mu0_pt, mu1_pt, mu2_pt, mu3_pt = encoded_masks[ell - 1]
            ell_q = ell // c
            j_same = (j + ell_q) % mc
            j_next = (j + ell_q + 1) % mc

            v1_pre = ctx.mul_pt(v, mu1_pt)
            v1 = ctx.rescale(v1_pre)
            ctx.free(v1_pre)
            _accumulate(acc_prime, j_same, v1)

            if mu0_pt is not None:
                v0_pre = ctx.mul_pt(v, mu0_pt)
                v0 = ctx.rescale(v0_pre)
                ctx.free(v0_pre)
                _accumulate(acc_prime, j_next, v0)
            if mu2_pt is not None:
                v2_pre = ctx.mul_pt(v, mu2_pt)
                v2 = ctx.rescale(v2_pre)
                ctx.free(v2_pre)
                _accumulate(acc_d_prime, j_next, v2)

            # v3 = v * mu3 (computed directly rather than v - v0 - v1 - v2,
            # which would require a DropLevel that neither binding exposes).
            # Pays one extra ct*pt per ell-iter; semantically identical.
            v3_pre = ctx.mul_pt(v, mu3_pt)
            v3 = ctx.rescale(v3_pre)
            ctx.free(v3_pre)
            _accumulate(acc_d_prime, j_same, v3)

            ctx.free(v)

    # Line 18: final assembly. All accumulators are degree-2 at L-2; the
    # base term is degree-2 at L-1. Drop the base to L-2 by multiplying by
    # an all-ones plaintext and rescaling (no DropLevel in either binding).
    # Then merge base + acc_prime, relinearize once. Relinearize acc_d_prime
    # separately (the rotation in the next step requires degree-1).
    out: list[int] = []
    for j in range(mc):
        base_pre = ctx.mul_pt(p_col0[j], ones_pt)
        base = ctx.rescale(base_pre)
        ctx.free(base_pre, p_col0[j])
        p_col0[j] = None

        if acc_prime[j] is not None:
            new_base = ctx.add(base, acc_prime[j])
            ctx.free(base, acc_prime[j])
            acc_prime[j] = None
            base = new_base
        base_relin = ctx.relin(base)
        ctx.free(base)
        base = base_relin

        if acc_d_prime[j] is not None:
            acc_d = ctx.relin(acc_d_prime[j])
            ctx.free(acc_d_prime[j])
            acc_d_prime[j] = None
            rolled = ctx.rot(acc_d, -nH)
            ctx.free(acc_d)
            new_base = ctx.add(base, rolled)
            ctx.free(base, rolled)
            base = new_base
        out.append(base)
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
