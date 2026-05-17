"""BMM-III (multi-chunk bicycle) CKKS kernel for the matmul-encoding paper.

Port of matmult/bmm3_cipher.go (Zheng et al., IEEE TIFS 2024). Computes
C = A @ B on bicyclic-encoded inputs that exceed a single CKKS ciphertext.

When n*m or m*p is larger than ``n_he`` (the BMM-III chunk size, typically
the slot count), the encoding is split into ``w = ceil(enc_len / n_he)``
chunks. The kernel rotates these chunks in lockstep via the LongRot
primitive, multiplies A-chunks by B-chunks per output position, and
accumulates into ``stop = ceil(n*p / n_he)`` output chunks.

Two of Negar's three Go modes are ported:

  * ``bmm3_he_cached``  — Bmm3ModeCached. Plaintext masks encoded once
    per (start, end) and reused across all m iterations + both A/B
    sides. Rotations issued one at a time inside ``_long_rot_he``.

  * ``bmm3_he_hoisted`` — Bmm3ModeHoisted. Cached masks PLUS
    block-hoisted Step-1 rotations on top. For every block of
    ``hoist_block_size`` outer iterations we (1) walk the LongRot
    Step-1 plan in plan-only mode via ``_build_plan`` to learn which
    chunks need which rotation amounts, (2) take the union across the
    block, (3) call ``ctx.rot_batch`` once per chunk to compute all
    needed rotates in one ModUp + N keyswitches, (4) run the
    block's ``hoist_block_size`` iterations of ``_bmm3_loop``, each
    reading rotated chunks from the precomputed dict instead of
    issuing individual rotates. (5) free the hoisted ciphertexts
    before advancing to the next block.

    Net rotation cost on a single LongRot drops from "one full
    keyswitch per (chunk, v_tmp) pair" to "one ModUp per chunk +
    one keyswitch per (chunk, v_tmp)". For the paper's BMM-III
    sweep at n*m*p ~ 128^3 with hoist_block_size=16, the LongRot
    Step-1 rotations dominate cached-mode wall-clock, so the
    hoisted-mode speedup is substantial.

The naive mode (re-encode masks on every call) is intentionally not
ported: it bloated Lattigo's plaintext table enough to crash the Go
runtime on shapes larger than (5, 7, 11) at small n_he during the
session-1 port. Cached mode is the correctness baseline; hoisted
mode is the paper preset.

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
    rotate_dict: dict[int, dict[int, int]] | None = None,
) -> list[int]:
    """Rotate the chunked logical vector by `rot`, return stop output cts.

    Mirrors matmult/bmm3_cipher.go::longRotHE step by step. Three steps:
      1. rotate each source chunk by v_tmp = rot mod n_he;
      2. stitch adjacent rotated chunks into output chunks;
      3. produce the (possibly partial) final output chunk.

    If ``rotate_dict`` is provided, Step 1 reads pre-rotated chunks from
    it (chunk_idx -> v_tmp -> ct_id) instead of calling ``ctx.rot``. The
    dict must contain every (chunk_idx, v_tmp != 0) pair that Step 1
    walks; ``_build_plan`` enumerates exactly that set and
    ``_precompute_hoisted`` populates the dict via ``ctx.rot_batch``.
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
        elif rotate_dict is not None:
            try:
                rotate_cts.append(rotate_dict[idx][v_tmp])
            except KeyError as e:
                raise RuntimeError(
                    f"_long_rot_he: hoist dict missing entry "
                    f"(idx={idx}, v_tmp={v_tmp}); "
                    f"_build_plan/_precompute_hoisted bug"
                ) from e
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
# Hoisted-mode planning + precompute
#
# These two helpers implement the "hoisted" half of Bmm3ModeHoisted in
# Negar's Go (matmult/bmm3_cipher.go::buildPlan + precomputeHoisted).
# _build_plan walks LongRot Step 1 in plan-only mode and records which
# chunks need which v_tmp values. _precompute_hoisted unions plans over a
# block of outer iterations and issues one ctx.rot_batch per chunk so the
# ModUp half of each key-switch is amortised across all v_tmp values for
# that chunk.
# ---------------------------------------------------------------------------


def _build_plan(
    rot: int,
    output_len: int,
    enc_len: int,
    n_he: int,
    w: int,
) -> dict[int, set[int]]:
    """Plan-only walk of LongRot Step 1: which chunk needs which v_tmp.

    Returns chunk_idx -> set of distinct v_tmp values. Skips v_tmp == 0
    (identity, no rotation needed). No ciphertexts touched.
    """
    rot = _pos_mod(rot, enc_len)
    v = _pos_mod(rot, n_he)
    u = rot // n_he
    stop_length = (output_len + n_he - 1) // n_he
    r_wrap = enc_len % n_he

    needed: dict[int, set[int]] = {}
    v_tmp = v
    k = 1
    i = 0
    while i < stop_length + k:
        idx = _pos_mod(u + i, w)
        if v_tmp != 0:
            needed.setdefault(idx, set()).add(v_tmp)
        if idx == w - 1:
            if v_tmp <= r_wrap:
                v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
                if v_tmp == 0:
                    k += 1
            else:
                v_tmp = _pos_mod(v_tmp - r_wrap, n_he)
                k += 1
        i += 1
    return needed


def _precompute_hoisted(
    ctx: Context,
    rots_block: Sequence[int],
    output_len: int,
    enc_len: int,
    n_he: int,
    enc_cts: list[int],
) -> dict[int, dict[int, int]]:
    """Hoist the Step-1 rotations needed by a block of LongRot calls.

    For every (rot in rots_block) we run ``_build_plan`` to learn the
    per-chunk shift sets, union them across the block, then call
    ``ctx.rot_batch(enc_cts[idx], sorted_shifts)`` once per chunk so the
    ModUp half of the key-switch is paid once per chunk regardless of
    how many shifts that chunk needs.

    Returns chunk_idx -> {v_tmp -> rotated_ct_id}. The caller is
    responsible for calling ``ctx.free()`` on the returned ciphertexts
    when the block is done (see ``bmm3_he_hoisted``).
    """
    w = len(enc_cts)

    union: dict[int, set[int]] = {}
    for rot in rots_block:
        per = _build_plan(rot, output_len, enc_len, n_he, w)
        for idx, vset in per.items():
            union.setdefault(idx, set()).update(vset)

    hoisted: dict[int, dict[int, int]] = {}
    for idx, vset in union.items():
        if not vset:
            continue
        shifts = sorted(vset)
        rotated = ctx.rot_batch(enc_cts[idx], shifts)
        hoisted[idx] = dict(zip(shifts, rotated))
    return hoisted


# ---------------------------------------------------------------------------
# Per-iteration body (shared by cached + hoisted)
# ---------------------------------------------------------------------------


def _bmm3_loop(
    ctx: Context,
    a_cts: list[int],
    b_cts: list[int],
    rot_a: int,
    rot_b: int,
    nm: int,
    mp: int,
    np_: int,
    n_he: int,
    mc: _MaskCache,
    rotate_dict_a: dict[int, dict[int, int]] | None,
    rotate_dict_b: dict[int, dict[int, int]] | None,
    dest: list[int | None],
) -> list[int | None]:
    """One outer-iteration body: LongRot A, LongRot B, accumulate
    ``a_rot[s] * b_rot[s]`` into ``dest[s]`` for every output chunk s.

    Lazy: ``ctx.mul_nr`` (degree-2 product, no relin) + ``ctx.add``
    (preserves degree). Rescale and Relinearize are deferred to the
    caller's finalize step.

    ``rotate_dict_a`` / ``rotate_dict_b`` are forwarded to ``_long_rot_he``
    -- ``None`` means cached mode (issue rotations one at a time), a
    populated dict means hoisted mode (read from precomputed map).
    """
    stop = (np_ + n_he - 1) // n_he

    a_rot = _long_rot_he(
        ctx, a_cts, rot_a, np_, nm, n_he, mc, rotate_dict=rotate_dict_a,
    )
    b_rot = _long_rot_he(
        ctx, b_cts, rot_b, np_, mp, n_he, mc, rotate_dict=rotate_dict_b,
    )

    for s in range(stop):
        prod = ctx.mul_nr(a_rot[s], b_rot[s])
        if dest[s] is None:
            dest[s] = prod
        else:
            dest[s] = ctx.add(dest[s], prod)
    return dest


def _bmm3_finalize(ctx: Context, dest: list[int | None]) -> list[int]:
    """Rescale + Relinearize each accumulated dest chunk once.

    Drops (deg 2, S^2, L-1) -> (deg 1, S, L-2). Independent of m.
    """
    out: list[int] = []
    for s, ct in enumerate(dest):
        assert ct is not None, f"dest[{s}] was never written"
        ct = ctx.rescale(ct)
        ct = ctx.relin(ct)
        out.append(ct)
    return out


# ---------------------------------------------------------------------------
# BMM-III dispatchers (cached + hoisted)
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
    masks are encoded once per (start, end) key and reused. Rotations are
    issued one at a time (no hoisting); see ``bmm3_he_hoisted`` for the
    block-hoisted variant that amortises ModUp across rotations.
    """
    r = smallest_r(n, m, p)
    nm, mp, np_ = n * m, m * p, n * p
    stop = (np_ + n_he - 1) // n_he
    mc = _MaskCache(ctx, n_he, input_level - 1)

    dest: list[int | None] = [None] * stop

    for i in range(m):
        rot_a = _pos_mod(-i * n, nm)
        rot_b = _pos_mod((r * m - n) * i, mp)
        dest = _bmm3_loop(
            ctx, a_cts, b_cts, rot_a, rot_b,
            nm, mp, np_, n_he, mc,
            rotate_dict_a=None, rotate_dict_b=None, dest=dest,
        )

    return _bmm3_finalize(ctx, dest)


def bmm3_he_hoisted(
    ctx: Context,
    a_cts: list[int],
    b_cts: list[int],
    n: int,
    m: int,
    p: int,
    n_he: int,
    input_level: int,
    hoist_block_size: int = 16,
) -> list[int]:
    """Run BMM-III in hoisted mode on encrypted chunks.

    Identical numerics + op count to ``bmm3_he_cached``; the difference
    is the per-block precompute that calls ``ctx.rot_batch`` once per
    chunk so the ModUp half of each Step-1 keyswitch is paid once per
    (block, chunk) instead of once per (iteration, chunk, v_tmp).

    Negar's Go default is ``hoistBlockSize=8`` but her notes flag
    ``16`` as best for larger dimensions on the paper sweep, so we
    default to 16 here. Sweep at 8/16/32 if you want to characterise.

    The per-block hoist dicts are freed before advancing to the next
    block via ``ctx.free()`` so peak GPU memory grows like
    ``O(hoist_block_size)`` rather than ``O(m)``.
    """
    if hoist_block_size <= 0:
        hoist_block_size = 8
    r = smallest_r(n, m, p)
    nm, mp, np_ = n * m, m * p, n * p
    stop = (np_ + n_he - 1) // n_he
    mc = _MaskCache(ctx, n_he, input_level - 1)

    rots_a = [_pos_mod(-i * n, nm) for i in range(m)]
    rots_b = [_pos_mod((r * m - n) * i, mp) for i in range(m)]

    dest: list[int | None] = [None] * stop

    for base in range(0, m, hoist_block_size):
        end = min(base + hoist_block_size, m)

        hoisted_a = _precompute_hoisted(
            ctx, rots_a[base:end], np_, nm, n_he, a_cts,
        )
        hoisted_b = _precompute_hoisted(
            ctx, rots_b[base:end], np_, mp, n_he, b_cts,
        )

        for i in range(base, end):
            dest = _bmm3_loop(
                ctx, a_cts, b_cts, rots_a[i], rots_b[i],
                nm, mp, np_, n_he, mc,
                rotate_dict_a=hoisted_a, rotate_dict_b=hoisted_b,
                dest=dest,
            )

        # Release the per-block hoist dicts before the next block
        # allocates its own. ctx.free() is a no-op on lattigo (per the
        # backend design) and a real DeleteCiphertext on desilo so GPU
        # memory peak grows like O(hoist_block_size), not O(m).
        for chunk_dict in (hoisted_a, hoisted_b):
            for v_map in chunk_dict.values():
                for ct in v_map.values():
                    ctx.free(ct)

    return _bmm3_finalize(ctx, dest)


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
    *,
    mode: str = "cached",
    hoist_block_size: int = 16,
) -> np.ndarray:
    """End-to-end: encrypt A and B, run the BMM-III kernel, decrypt to C.

    `A.shape == (n, m)`, `B.shape == (m, p)`. Requires (n, m, p) pairwise
    coprime (BMM-III precondition). When n_he is None, defaults to ctx.slots
    (the runner case); tests typically pass a smaller n_he to exercise the
    multi-chunk path with smaller matrices.

    ``mode``  -- "cached" or "hoisted". Hoisted matches Negar's paper
    preset (Bmm3ModeHoisted in matmult/bmm3_cipher.go).
    ``hoist_block_size`` -- only used when mode="hoisted". Negar's
    default is 8; her notes flag 16 as best for larger dimensions.
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

    if mode == "cached":
        out_cts = bmm3_he_cached(ctx, a_cts, b_cts, n, m, p, n_he, input_level)
    elif mode == "hoisted":
        out_cts = bmm3_he_hoisted(
            ctx, a_cts, b_cts, n, m, p, n_he, input_level,
            hoist_block_size=hoist_block_size,
        )
    else:
        raise ValueError(f"unknown bmm3 mode {mode!r}; use 'cached' or 'hoisted'")
    return decode_output(ctx, out_cts, n, p, n_he)
