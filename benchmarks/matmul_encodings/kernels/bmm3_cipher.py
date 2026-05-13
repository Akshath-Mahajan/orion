"""BMM-III (multi-chunk bicycle) CKKS kernel for the matmul-encoding paper.

Port of matmult/bmm3_cipher.go (Zheng et al., IEEE TIFS 2024). Computes
C = A @ B on bicyclic-encoded inputs that exceed a single CKKS ciphertext.

When n*m or m*p is larger than ``n_he`` (the BMM-III chunk size, typically
the slot count), the encoding is split into ``w = ceil(enc_len / n_he)``
chunks. The kernel rotates these chunks in lockstep via the LongRot
primitive, multiplies A-chunks by B-chunks per output position, and
accumulates into ``stop = ceil(n*p / n_he)`` output chunks.

This first port implements the **cached** mode (matmult/bmm3_cipher.go's
``Bmm3ModeCached``): plaintext masks are encoded once per (start, end)
key and reused across all m outer iterations and both A/B sides. The
naive mode (re-encode on every call) bloats Lattigo's plaintext table to
the point where Go panics on shapes larger than (5, 7, 11) at small
n_he. The hoisted mode (block-hoisted rotations on top of the cache)
is a follow-up commit.

Lazy rescale + lazy relinearization
-----------------------------------
The inner accumulation deliberately:
  * uses ``ctx.mul_nr`` (no immediate relinearization), leaving each
    product a degree-2 ciphertext;
  * skips the per-product Rescale, so the accumulator stays at scale
    S^2 and level L-1 throughout.

Across all m outer iterations this pays the inner cost of m * stop
multiplications with ZERO Relinearizes and ZERO Rescales. The deferred
work is performed once per output chunk in ``_finalize``:
  * Rescale     : (deg 2, S^2, L-1) -> (deg 2, S, L-2)
  * Relinearize : (deg 2, S,   L-2) -> (deg 1, S, L-2)

Total CKKS multiplicative depth: 2 (one ct*pt mask + one ct*ct multiply).

Tile-to-slots invariant
-----------------------
Each n_he-slot chunk is tiled cyclically to fill ``ctx.slots`` before
encryption. CKKS rotates over all slot positions, not over n_he, so the
tile keeps the rotation wraparound inside valid data instead of into
zero padding. ``ctx.slots % n_he == 0`` is the only constraint.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..context import Context
from ..plaintext.bmm1_plain import bicyclic_decode, bicyclic_encode
from ..plaintext.bmm3_plain import break_into_chunks, smallest_r


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _pos_mod(a: int, b: int) -> int:
    r = a % b
    if r < 0:
        r += b
    return r


def _tile_to_slots(vec: np.ndarray, slots: int) -> np.ndarray:
    """Tile `vec` cyclically to fill `slots` so CKKS slot rotations wrap
    inside valid data.

    Same helper as in the THOR / MOAI / BMM-I kernels; duplicated here to
    keep each kernel module self-contained.
    """
    L = vec.size
    assert slots % L == 0, (
        f"slots {slots} must be a multiple of encoded length {L}"
    )
    return np.tile(vec, slots // L)


def encrypt_chunks(
    ctx: Context,
    chunks: list[np.ndarray],
    n_he: int,
    input_level: int,
) -> list[int]:
    """Tile each n_he-slot chunk to ctx.slots, encode at input_level, encrypt.

    Returns a list of ciphertext IDs aligned 1:1 with `chunks`.
    """
    assert ctx.slots % n_he == 0, (
        f"ctx.slots ({ctx.slots}) must be a multiple of n_he ({n_he})"
    )
    return [
        ctx.encrypt(_tile_to_slots(c, ctx.slots), level=input_level)
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Mask cache
#
# Masks are length-ctx.slots plaintexts with 1.0 in [start, end) within each
# n_he-tile and 0 elsewhere. Tiling makes ct * mask correct across the
# full slot vector, the same way input chunks are tiled.
#
# A single (start, end) key is hit O(m) times across BMM-III's outer loop;
# without caching the lattigo binding's plaintext table grows large enough
# to crash the Go runtime on bigger shapes. We encode each key at most once
# per BMM-III call and reuse the plaintext id for every subsequent hit.
#
# Encoding level matches the Go reference (`inputLevel - 1`) so a single
# ct * mask + Rescale lands the post-mask ciphertext at level inputLevel-1
# at the default scale, ready for the inner ct * ct multiply.
# ---------------------------------------------------------------------------


class _MaskCache:
    """(start, end) -> plaintext id cache, scoped to one BMM-III call."""

    def __init__(self, ctx: Context, n_he: int, level: int) -> None:
        self.ctx = ctx
        self.n_he = n_he
        self.level = level
        self._cache: dict[tuple[int, int], int] = {}

    def get(self, start: int, end: int) -> int:
        key = (start, end)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        base = np.zeros(self.n_he, dtype=np.float64)
        base[start:end] = 1.0
        tiled = _tile_to_slots(base, self.ctx.slots)
        scale = self.ctx.scheme.params.get_default_scale()
        pt = self.ctx.backend.Encode(tiled.tolist(), self.level, scale)
        self._cache[key] = pt
        return pt


# ---------------------------------------------------------------------------
# LongRot (cipher)
#
# Rotate a logical length-enc_len vector (stored as `enc_cts`, w chunks of
# n_he slots) by `rot` positions, producing `stop = ceil(output_len/n_he)`
# output ciphertexts. Direct port of matmult/bmm3_cipher.go::longRotHE.
# ---------------------------------------------------------------------------


def _ct_mul_mask_rescale(ctx: Context, ct: int, mask_pt: int) -> int:
    """ct * mask plaintext, then Rescale. Bumps ct_pt_muls."""
    return ctx.rescale(ctx.mul_pt(ct, mask_pt))


def _long_rot_select(
    ctx: Context,
    v_t: int,
    n_he: int,
    ct_a: int,
    ct_b: int,
    mc: _MaskCache,
) -> int:
    """Plaintext analog of longRotSelect at the cipher level:

        v_t == 0    -> identity (return ct_a)
        v_t == n_he -> next chunk (return ct_b)
        else        -> ct_a * mask[0, n_he-v_t] + ct_b * mask[n_he-v_t, n_he]

    The two-mask path lands a ciphertext at scale S, level mask_level - 1,
    degree 1, ready for the inner ct * ct multiply.
    """
    if v_t == 0:
        return ct_a
    if v_t == n_he:
        return ct_b
    m_a = mc.get(0,         n_he - v_t)
    m_b = mc.get(n_he - v_t, n_he)
    left  = _ct_mul_mask_rescale(ctx, ct_a, m_a)
    right = _ct_mul_mask_rescale(ctx, ct_b, m_b)
    return ctx.add(left, right)


def _long_rot_he(
    ctx: Context,
    enc_cts: list[int],
    rot: int,
    output_len: int,
    enc_len: int,
    n_he: int,
    mc: _MaskCache,
) -> list[int]:
    """Rotate the chunked logical vector by `rot`, return stop output cts.

    Mirrors matmult/bmm3_cipher.go::longRotHE step by step. Three steps:
      1. rotate each source chunk by v_tmp = rot mod n_he;
      2. stitch adjacent rotated chunks into output chunks;
      3. produce the (possibly partial) final output chunk.
    """
    rot = _pos_mod(rot, enc_len)
    w = len(enc_cts)
    v = _pos_mod(rot, n_he)
    u = rot // n_he
    stop_length = (output_len + n_he - 1) // n_he
    last_r = output_len % n_he
    r_wrap = enc_len % n_he

    # --- Step 1: per-chunk rotations. -------------------------------------
    rotate_cts: list[int] = []
    v_tmp = v
    k = 1
    i = 0
    while i < stop_length + k:
        idx = _pos_mod(u + i, w)
        if v_tmp == 0:
            rotate_cts.append(enc_cts[idx])
        else:
            rotate_cts.append(ctx.rot(enc_cts[idx], v_tmp))

        if idx == w - 1:
            if v_tmp <= r_wrap:
                v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
                if v_tmp == 0:
                    k += 1
            else:
                v_tmp = _pos_mod(v_tmp - r_wrap, n_he)
                k += 1
        i += 1

    rc_len = len(rotate_cts)

    # --- Step 2: stitch rotated chunks into output chunks. ----------------
    destination: list[int] = []
    v_tmp = v
    i = 0

    while len(destination) < stop_length - 1:
        if _pos_mod(u + i, w) == w - 1 and v_tmp <= r_wrap:
            v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
            if v_tmp == 0:
                i += 1
                continue

        if _pos_mod(u + i, w) == w - 2 and v_tmp > r_wrap:
            m1 = mc.get(0,                       n_he - v_tmp)
            m2 = mc.get(n_he - v_tmp,            n_he - v_tmp + r_wrap)
            m3 = mc.get(n_he - v_tmp + r_wrap,   n_he)
            p1 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i,     rc_len)], m1)
            p2 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 1, rc_len)], m2)
            p3 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 2, rc_len)], m3)
            destination.append(ctx.add(ctx.add(p1, p2), p3))
            v_tmp = _pos_mod(v_tmp - r_wrap, n_he)
            i += 2
            continue

        destination.append(_long_rot_select(
            ctx, v_tmp, n_he,
            rotate_cts[_pos_mod(i,     rc_len)],
            rotate_cts[_pos_mod(i + 1, rc_len)],
            mc,
        ))
        i += 1

    # --- Step 3: final (possibly partial) output chunk. -------------------
    if len(destination) == stop_length - 1:
        lr = last_r if last_r != 0 else n_he

        if _pos_mod(u + i, w) == w - 2 and v_tmp > r_wrap:
            if lr <= n_he - v_tmp:
                m1 = mc.get(0, lr)
                destination.append(_ct_mul_mask_rescale(
                    ctx, rotate_cts[_pos_mod(i, rc_len)], m1
                ))
            elif lr <= n_he - v_tmp + r_wrap:
                m1 = mc.get(0,           n_he - v_tmp)
                m2 = mc.get(n_he - v_tmp, lr)
                p1 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i,     rc_len)], m1)
                p2 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 1, rc_len)], m2)
                destination.append(ctx.add(p1, p2))
            else:
                m1 = mc.get(0,                       n_he - v_tmp)
                m2 = mc.get(n_he - v_tmp,            n_he - v_tmp + r_wrap)
                m3 = mc.get(n_he - v_tmp + r_wrap,   lr)
                p1 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i,     rc_len)], m1)
                p2 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 1, rc_len)], m2)
                p3 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 2, rc_len)], m3)
                destination.append(ctx.add(ctx.add(p1, p2), p3))
        else:
            if _pos_mod(u + i, w) == w - 1 and v_tmp <= r_wrap:
                v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
                if v_tmp == 0:
                    i += 1

            if lr <= n_he - v_tmp:
                m1 = mc.get(0, lr)
                destination.append(_ct_mul_mask_rescale(
                    ctx, rotate_cts[_pos_mod(i, rc_len)], m1
                ))
            else:
                m1 = mc.get(0,           n_he - v_tmp)
                m2 = mc.get(n_he - v_tmp, lr)
                p1 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i,     rc_len)], m1)
                p2 = _ct_mul_mask_rescale(ctx, rotate_cts[_pos_mod(i + 1, rc_len)], m2)
                destination.append(ctx.add(p1, p2))

    return destination


# ---------------------------------------------------------------------------
# BMM-III dispatcher (naive mode)
# ---------------------------------------------------------------------------


def bmm3_he_cached(
    ctx: Context,
    a_cts: list[int],
    b_cts: list[int],
    n: int,
    m: int,
    p: int,
    n_he: int,
    input_level: int,
) -> list[int]:
    """Run BMM-III in cached mode on encrypted chunks.

    Returns ``stop = ceil(n*p / n_he)`` output ciphertexts, each at
    (deg 1, scale S, level input_level - 2) after the lazy finalize step.

    Inner loop: ``m`` LongRot pairs + ``m * stop`` ct*ct multiplies (lazy
    relin). Finalize: ``stop`` Rescales + ``stop`` Relinearizes. Plaintext
    masks are encoded once per (start, end) key and reused.
    """
    r = smallest_r(n, m, p)
    nm, mp, np_ = n * m, m * p, n * p
    stop = (np_ + n_he - 1) // n_he
    mc = _MaskCache(ctx, n_he, input_level - 1)

    dest: list[int | None] = [None] * stop

    for i in range(m):
        rot_a = _pos_mod(-i * n, nm)
        rot_b = _pos_mod((r * m - n) * i, mp)

        a_rot = _long_rot_he(ctx, a_cts, rot_a, np_, nm, n_he, mc)
        b_rot = _long_rot_he(ctx, b_cts, rot_b, np_, mp, n_he, mc)

        for s in range(stop):
            # Lazy: degree-2 product, no rescale, no relin yet.
            prod = ctx.mul_nr(a_rot[s], b_rot[s])
            if dest[s] is None:
                dest[s] = prod
            else:
                dest[s] = ctx.add(dest[s], prod)

    # Finalize: one Rescale + one Relinearize per output chunk.
    out: list[int] = []
    for s in range(stop):
        ct = dest[s]
        assert ct is not None
        ct = ctx.rescale(ct)
        ct = ctx.relin(ct)
        out.append(ct)
    return out


def decode_output(
    ctx: Context,
    ct_chunks: list[int],
    n: int,
    p: int,
    n_he: int,
) -> np.ndarray:
    """Decrypt every output chunk, stitch the first n*p slots, bicyclic-decode."""
    raw = []
    for ct in ct_chunks:
        vec = ctx.decrypt(ct)
        raw.append(vec[:n_he])
    flat = np.concatenate(raw)[: n * p]
    return bicyclic_decode(flat, n, p)


def bmm3_he(
    ctx: Context,
    A: np.ndarray,
    B: np.ndarray,
    n_he: int | None = None,
    input_level: int | None = None,
) -> np.ndarray:
    """End-to-end: encrypt A and B, run the naive BMM-III kernel, decrypt to C.

    `A.shape == (n, m)`, `B.shape == (m, p)`. Requires (n, m, p) pairwise
    coprime (BMM-III precondition). When n_he is None, defaults to ctx.slots
    (the runner case); tests typically pass a smaller n_he to exercise the
    multi-chunk path with smaller matrices.
    """
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, f"matmul shape mismatch: A is {A.shape}, B is {B.shape}"

    if n_he is None:
        n_he = ctx.slots
    if input_level is None:
        input_level = ctx.max_level

    a_chunks = break_into_chunks(bicyclic_encode(A), n * m, n * p, n_he)
    b_chunks = break_into_chunks(bicyclic_encode(B), m * p, n * p, n_he)

    a_cts = encrypt_chunks(ctx, a_chunks, n_he, input_level)
    b_cts = encrypt_chunks(ctx, b_chunks, n_he, input_level)

    out_cts = bmm3_he_cached(ctx, a_cts, b_cts, n, m, p, n_he, input_level)
    return decode_output(ctx, out_cts, n, p, n_he)
