"""BMM-III helpers used by the HE kernel.

Port of matmult/bmm3_plain.go. Adds the small math helpers
(`smallest_r`, `break_into_chunks`) needed on top of the bicyclic encoding
already provided by `bmm1_plain.py`.

There is no plaintext runner for BMM-III in this port (HE-only). The
plaintext bicyclic encode/decode are imported from bmm1_plain to avoid
duplication, mirroring the Go file's reuse.
"""

from __future__ import annotations

import math

import numpy as np

from .bmm1_plain import bicyclic_encode, bicyclic_decode  # re-exported


__all__ = [
    "bicyclic_encode",
    "bicyclic_decode",
    "smallest_r",
    "break_into_chunks",
]


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
