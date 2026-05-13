"""BMM-III bicycle matmul helpers + plaintext oracle.

Helpers (smallest_r, break_into_chunks) are a port of matmult/bmm3_plain.go.
The plaintext LongRot simulator and the BMM-III plaintext kernel below have
no Go counterpart -- Negar's bmm3_plain.go is HE-only. They are added here
so that the BMM-III ciphertext kernel can be checked against an encoding-
aware plaintext oracle in addition to the raw numpy.matmul reference.

LongRot at a glance
-------------------
Bicyclic encoding of an (n x m) matrix has length n*m. When n*m exceeds the
slot count, BMM-III splits the encoding into w = ceil(n*m / n_he) chunks of
n_he slots each and rotates them in lockstep with cross-chunk stitching.

LongRot rotates the LOGICAL vector by ``rot`` positions and returns
``stop_length = ceil(output_len / n_he)`` output chunks. Three steps mirror
matmult/bmm3_cipher.go::longRotHE exactly:

    1. Step 1: rotate each source chunk by v_tmp = rot mod n_he. The
       chunk index walks (u + i) mod w; v_tmp is adjusted whenever the
       last source chunk is hit (its real data fills only r_wrap slots).
    2. Step 2: stitch adjacent rotated chunks via two-way masks (default)
       or three-way masks (when the next-to-last chunk is in play and
       v_tmp > r_wrap) to form whole output chunks.
    3. Step 3: produce the possibly-partial final chunk, choosing the
       mask layout from the same case analysis.

The plaintext implementation uses numpy multiplies in place of CKKS
ct * mask * Rescale, and a chunk-array slice in place of ct rotation. The
arithmetic structure is identical, so a bug in the chunk-stitching logic
will surface here before it bites the cipher kernel.
"""

from __future__ import annotations

import math

import numpy as np

from .bmm1_plain import bicyclic_decode, bicyclic_encode  # re-exported
from .op_counts import OpCounts


__all__ = [
    "bicyclic_encode",
    "bicyclic_decode",
    "smallest_r",
    "break_into_chunks",
    "long_rot_plain",
    "bmm3_plain",
    "bmm3_matmul_plain",
]


# ---------------------------------------------------------------------------
# Small helpers (Go port)
# ---------------------------------------------------------------------------


def smallest_r(n: int, m: int, p: int) -> int:
    """Smallest r >= 1 such that (r*m - n) is divisible by p AND >= 0.

    Used by BMM-III to compute the B-side rotation amount:
        rot_b = ((r*m - n) * i) mod (m*p).
    """
    r = 1
    s = m - n
    while (s % p != 0) or s < 0:
        r += 1
        s += m
    return r


def break_into_chunks(
    enc: np.ndarray,
    enc_len: int,
    output_len: int,
    n_he: int,
) -> list[np.ndarray]:
    """Cyclically tile `enc` to support the worst-case rotation window, then
    split into ceil(len/n_he) chunks of n_he slots each.

    The last chunk is zero-padded on the right if it falls short.
    """
    needed = output_len + n_he
    work = np.asarray(enc[:enc_len], dtype=np.float64)
    if enc_len < needed:
        reps = math.ceil(needed / enc_len) + 1
        work = np.tile(work, reps)

    w = (len(work) + n_he - 1) // n_he
    chunks: list[np.ndarray] = []
    for i in range(w):
        chunk = np.zeros(n_he, dtype=np.float64)
        start = i * n_he
        end = min(start + n_he, len(work))
        chunk[: end - start] = work[start:end]
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# LongRot (plaintext simulator)
# ---------------------------------------------------------------------------


def _pos_mod(a: int, b: int) -> int:
    """Python `a % b` is already non-negative for positive b, but state the
    invariant explicitly so the port reads 1:1 with the Go posMod3."""
    r = a % b
    if r < 0:
        r += b
    return r


def _mask(start: int, end: int, n_he: int) -> np.ndarray:
    """Length-n_he mask vector with 1.0 in [start, end), 0 elsewhere.

    Mirrors maskFn in the cipher kernel: any range outside [0, n_he) is
    clipped, so callers can pass intervals derived from arithmetic without
    pre-clamping.
    """
    out = np.zeros(n_he, dtype=np.float64)
    lo = max(0, start)
    hi = min(n_he, end)
    if hi > lo:
        out[lo:hi] = 1.0
    return out


def _rotate_chunk(chunk: np.ndarray, v_tmp: int) -> np.ndarray:
    """Left-rotate a single n_he-slot chunk by v_tmp (Go's eval.RotateNew)."""
    if v_tmp == 0:
        return chunk
    return np.concatenate([chunk[v_tmp:], chunk[:v_tmp]])


def _select(
    v_t: int,
    n_he: int,
    a: np.ndarray,
    b: np.ndarray,
    counts: OpCounts | None,
) -> np.ndarray:
    """Plaintext analog of longRotSelect: stitch [v_t, n_he) of `a` with
    [0, v_t) of `b` via masks (or just return one operand for the trivial
    boundaries).
    """
    if v_t == 0:
        return a
    if v_t == n_he:
        return b
    m_a = _mask(0, n_he - v_t, n_he)
    m_b = _mask(n_he - v_t, n_he, n_he)
    if counts is not None:
        counts.ct_pt_muls += 2
    return a * m_a + b * m_b


def long_rot_plain(
    chunks: list[np.ndarray],
    rot: int,
    output_len: int,
    enc_len: int,
    n_he: int,
    counts: OpCounts | None = None,
) -> list[np.ndarray]:
    """Rotate a logical length-enc_len vector (stored as `chunks` of n_he
    slots) by `rot` positions; return ceil(output_len / n_he) output chunks.

    Counts (when ``counts`` is provided):
      - rotations: number of non-trivial Step-1 chunk rotations.
      - ct_pt_muls: number of mask multiplies across Steps 2 and 3.
      - ct_ct_muls: not touched here.

    See module docstring for the three-step structure.
    """
    rot = _pos_mod(rot, enc_len)
    w = len(chunks)
    v = _pos_mod(rot, n_he)
    u = rot // n_he
    stop_length = (output_len + n_he - 1) // n_he
    last_r = output_len % n_he
    r_wrap = enc_len % n_he

    # --- Step 1: rotate each source chunk by v_tmp. -----------------------
    rotate_chunks: list[np.ndarray] = []
    v_tmp = v
    k = 1
    i = 0
    while i < stop_length + k:
        idx = _pos_mod(u + i, w)
        if v_tmp == 0:
            ch = chunks[idx]
        else:
            ch = _rotate_chunk(chunks[idx], v_tmp)
            if counts is not None:
                counts.rotations += 1
        rotate_chunks.append(ch)

        if idx == w - 1:
            if v_tmp <= r_wrap:
                v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
                if v_tmp == 0:
                    k += 1
            else:
                v_tmp = _pos_mod(v_tmp - r_wrap, n_he)
                k += 1
        i += 1

    rc_len = len(rotate_chunks)

    # --- Step 2: stitch rotated chunks into output chunks. ----------------
    destination: list[np.ndarray] = []
    v_tmp = v
    i = 0

    while len(destination) < stop_length - 1:
        # Case A: last input chunk, v_tmp fits within r_wrap -> skip ahead.
        if _pos_mod(u + i, w) == w - 1 and v_tmp <= r_wrap:
            v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
            if v_tmp == 0:
                i += 1
                continue

        # Case B: second-to-last chunk, v_tmp > r_wrap -> 3-way stitch.
        if _pos_mod(u + i, w) == w - 2 and v_tmp > r_wrap:
            m1 = _mask(0, n_he - v_tmp, n_he)
            m2 = _mask(n_he - v_tmp, n_he - v_tmp + r_wrap, n_he)
            m3 = _mask(n_he - v_tmp + r_wrap, n_he, n_he)
            p1 = rotate_chunks[_pos_mod(i,     rc_len)] * m1
            p2 = rotate_chunks[_pos_mod(i + 1, rc_len)] * m2
            p3 = rotate_chunks[_pos_mod(i + 2, rc_len)] * m3
            if counts is not None:
                counts.ct_pt_muls += 3
            destination.append(p1 + p2 + p3)
            v_tmp = _pos_mod(v_tmp - r_wrap, n_he)
            i += 2
            continue

        # Default case: classic two-way stitch.
        destination.append(_select(
            v_tmp, n_he,
            rotate_chunks[_pos_mod(i,     rc_len)],
            rotate_chunks[_pos_mod(i + 1, rc_len)],
            counts,
        ))
        i += 1

    # --- Step 3: final (possibly partial) output chunk. -------------------
    if len(destination) == stop_length - 1:
        lr = last_r if last_r != 0 else n_he

        if _pos_mod(u + i, w) == w - 2 and v_tmp > r_wrap:
            if lr <= n_he - v_tmp:
                m1 = _mask(0, lr, n_he)
                if counts is not None:
                    counts.ct_pt_muls += 1
                destination.append(
                    rotate_chunks[_pos_mod(i, rc_len)] * m1
                )
            elif lr <= n_he - v_tmp + r_wrap:
                m1 = _mask(0,         n_he - v_tmp, n_he)
                m2 = _mask(n_he - v_tmp, lr,         n_he)
                p1 = rotate_chunks[_pos_mod(i,     rc_len)] * m1
                p2 = rotate_chunks[_pos_mod(i + 1, rc_len)] * m2
                if counts is not None:
                    counts.ct_pt_muls += 2
                destination.append(p1 + p2)
            else:
                m1 = _mask(0,                       n_he - v_tmp,           n_he)
                m2 = _mask(n_he - v_tmp,            n_he - v_tmp + r_wrap, n_he)
                m3 = _mask(n_he - v_tmp + r_wrap,   lr,                     n_he)
                p1 = rotate_chunks[_pos_mod(i,     rc_len)] * m1
                p2 = rotate_chunks[_pos_mod(i + 1, rc_len)] * m2
                p3 = rotate_chunks[_pos_mod(i + 2, rc_len)] * m3
                if counts is not None:
                    counts.ct_pt_muls += 3
                destination.append(p1 + p2 + p3)
        else:
            if _pos_mod(u + i, w) == w - 1 and v_tmp <= r_wrap:
                v_tmp = _pos_mod(n_he - r_wrap + v_tmp, n_he)
                if v_tmp == 0:
                    i += 1

            if lr <= n_he - v_tmp:
                m1 = _mask(0, lr, n_he)
                if counts is not None:
                    counts.ct_pt_muls += 1
                destination.append(
                    rotate_chunks[_pos_mod(i, rc_len)] * m1
                )
            else:
                m1 = _mask(0,           n_he - v_tmp, n_he)
                m2 = _mask(n_he - v_tmp, lr,           n_he)
                p1 = rotate_chunks[_pos_mod(i,     rc_len)] * m1
                p2 = rotate_chunks[_pos_mod(i + 1, rc_len)] * m2
                if counts is not None:
                    counts.ct_pt_muls += 2
                destination.append(p1 + p2)

    return destination


# ---------------------------------------------------------------------------
# BMM-III plaintext kernel
# ---------------------------------------------------------------------------


def bmm3_plain(
    a_chunks: list[np.ndarray],
    b_chunks: list[np.ndarray],
    n: int,
    m: int,
    p: int,
    n_he: int,
    counts: OpCounts | None = None,
) -> list[np.ndarray]:
    """Run BMM-III on bicyclic-encoded chunked inputs (numpy plaintext).

    `a_chunks` are the break_into_chunks output of bicyclic_encode(A) at
    enc_len = n*m, output_len = n*p; `b_chunks` likewise for B at
    enc_len = m*p, output_len = n*p.

    Returns ``stop = ceil(n*p / n_he)`` output chunks. The first n*p slots
    of their concatenation, fed to bicyclic_decode(_, n, p), recover C = A @ B.

    Mirrors matmult/bmm3_cipher.go's bmm3HENaive structure exactly. Counts
    (when provided): one ct_ct_mul per (m * stop) inner iteration; rotations
    and ct_pt_muls accumulate across the LongRot calls.
    """
    r = smallest_r(n, m, p)
    nm, mp, np_ = n * m, m * p, n * p
    stop = (np_ + n_he - 1) // n_he

    dest: list[np.ndarray | None] = [None] * stop
    for i in range(m):
        rot_a = _pos_mod(-i * n, nm)
        rot_b = _pos_mod((r * m - n) * i, mp)
        a_rot = long_rot_plain(a_chunks, rot_a, np_, nm, n_he, counts)
        b_rot = long_rot_plain(b_chunks, rot_b, np_, mp, n_he, counts)

        for s in range(stop):
            prod = a_rot[s] * b_rot[s]
            if counts is not None:
                counts.ct_ct_muls += 1
            dest[s] = prod if dest[s] is None else dest[s] + prod

    # No None entries possible after the loop (m >= 1).
    return [d for d in dest]  # type: ignore[misc]


def bmm3_matmul_plain(
    A: np.ndarray,
    B: np.ndarray,
    n_he: int,
) -> tuple[np.ndarray, OpCounts]:
    """End-to-end BMM-III plaintext: encode + chunk + run + decode.

    Convenience wrapper for tests; real benchmarks use the cipher path.
    """
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, f"matmul shape mismatch: A is {A.shape}, B is {B.shape}"

    a_enc = bicyclic_encode(A)
    b_enc = bicyclic_encode(B)
    a_chunks = break_into_chunks(a_enc, n * m, n * p, n_he)
    b_chunks = break_into_chunks(b_enc, m * p, n * p, n_he)

    counts = OpCounts()
    dest = bmm3_plain(a_chunks, b_chunks, n, m, p, n_he, counts)
    flat = np.concatenate(dest)[: n * p]
    C = bicyclic_decode(flat, n, p)
    return C, counts
