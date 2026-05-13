"""CKKS context wrapper for the matmul-encoding kernels.

Wraps an Orion Scheme + its (desilo or lattigo) backend in a small
counted-op API. Every multiplication / rotation / addition is funneled
through this object so the OpCounts attached to each kernel run
reflects exactly what the kernel did, with no separate instrumentation.

The Go reference (Negar's matmult repo) uses `*OpCounts` pointers
passed explicitly into every helper; in Python we hang the counter
off the Context so kernel call sites read like the Go ones without
threading a counter through every helper signature.

This module is deliberately minimal -- it is not an FHE library, just
a friendly facade so kernel code matches the Go reference line-for-line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from orion.core.orion import Scheme

from .plaintext.op_counts import OpCounts


@dataclass
class Context:
    """Scheme + counters bundle handed to every cipher kernel.

    Attributes:
        scheme:    Orion Scheme (carries the backend, params, encoder).
        counts:    Live OpCounts tally (mutated by every counted op).
        slots:     Cached slot count for convenience.
        max_level: Cached top mult level.
    """

    scheme: Scheme
    counts: OpCounts = field(default_factory=OpCounts)
    slots: int = 0
    max_level: int = 0

    # -- factory -----------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict | str) -> "Context":
        """Initialise a Scheme from a config dict or YAML path, return a Context."""
        scheme = Scheme()
        scheme.init_scheme(config)
        return cls(
            scheme=scheme,
            slots=scheme.params.get_slots(),
            max_level=scheme.params.get_max_level(),
        )

    @property
    def backend(self):
        return self.scheme.backend

    # -- encode / encrypt / decrypt ----------------------------------------

    def encode(self, values: np.ndarray, level: int | None = None) -> int:
        """Encode a length-`slots` float64 vector at the given level.

        If ``values`` is shorter than ``slots``, it is zero-padded.
        Returns a plaintext id.

        Note: backends expect a Python list (Lattigo's ctypes wrapper
        only auto-expands list[float] -> (ptr, len); a np.ndarray of
        float64 falls through and crashes with "takes 4 args"). Pass a
        list to keep both backends happy.
        """
        if level is None:
            level = self.max_level
        scale = self.scheme.params.get_default_scale()
        padded = self._pad(values)
        return self.backend.Encode(padded.tolist(), level, scale)

    def encrypt(self, values: np.ndarray, level: int | None = None) -> int:
        """Encode + encrypt; returns a ciphertext id."""
        pt_id = self.encode(values, level=level)
        return self.backend.Encrypt(pt_id)

    def decrypt(self, ct_id: int) -> np.ndarray:
        """Decrypt + decode a ciphertext id back to a real np.ndarray."""
        pt_id = self.backend.Decrypt(ct_id)
        result = self.backend.Decode(pt_id)
        return np.asarray(result, dtype=np.float64)

    # -- counted ops -------------------------------------------------------

    def rot(self, ct_id: int, k: int) -> int:
        """Cyclic rotation by k. Increments counts.rotations (k != 0)."""
        if k % self.slots == 0:
            return self.backend.RotateBatchNew(ct_id, [0])[0]  # cheap clone
        self.counts.rotations += 1
        return self.backend.RotateNew(ct_id, int(k))

    def rot_batch(self, ct_id: int, ks: Sequence[int]) -> list[int]:
        """Hoisted N-way rotation. Increments counts.rotations by the
        number of non-zero deltas.

        Backend dispatch:
        - desilo exposes ``RotateBatchNew`` (with a segfault-safe fallback
          to individual rotates today).
        - lattigo does not expose batched rotation; we issue N individual
          ``RotateNew`` calls in a loop, matching the same semantics.
        """
        ks = list(ks)
        non_zero = [k for k in ks if k % self.slots != 0]
        self.counts.rotations += len(non_zero)
        if hasattr(self.backend, "RotateBatchNew"):
            return self.backend.RotateBatchNew(ct_id, ks)
        return [self.backend.RotateNew(ct_id, int(k)) for k in ks]

    def add(self, ct1: int, ct2: int) -> int:
        return self.backend.AddCiphertextNew(ct1, ct2)

    def sub(self, ct1: int, ct2: int) -> int:
        return self.backend.SubCiphertextNew(ct1, ct2)

    def mul_rl(self, ct1: int, ct2: int) -> int:
        """ct1 * ct2 with eager relinearization. Bumps ct_ct_muls."""
        self.counts.ct_ct_muls += 1
        return self.backend.MulRelinCiphertextNew(ct1, ct2)

    def mul_nr(self, ct1: int, ct2: int) -> int:
        """ct1 * ct2 leaving a degree-2 ciphertext (lazy relin). Bumps ct_ct_muls.

        Falls back to eager-relin (MulRelinCiphertextNew) when the backend
        lacks ``MulNoRelinCiphertextNew``. Result is numerically identical
        -- the only observable difference is that eager-relin pays an
        extra key-switch per multiply.
        """
        self.counts.ct_ct_muls += 1
        if hasattr(self.backend, "MulNoRelinCiphertextNew"):
            return self.backend.MulNoRelinCiphertextNew(ct1, ct2)
        return self.backend.MulRelinCiphertextNew(ct1, ct2)

    def relin(self, ct_id: int) -> int:
        """Standalone relinearization (used after a chain of mul_nr + adds).

        Does NOT bump ct_ct_muls -- the multiply that produced the
        degree-2 ciphertext already did.

        No-op fallback on backends without ``RelinearizeNew``: if mul_nr
        fell back to eager relin, the ciphertext is already degree-1, so
        relin is a meaningless extra step we skip.
        """
        if hasattr(self.backend, "RelinearizeNew"):
            return self.backend.RelinearizeNew(ct_id)
        return ct_id

    def mul_pt(self, ct_id: int, pt_id: int) -> int:
        """ct * pt. Bumps ct_pt_muls."""
        self.counts.ct_pt_muls += 1
        return self.backend.MulPlaintextNew(ct_id, pt_id)

    def rescale(self, ct_id: int) -> int:
        """Consume one level. Counts as a rescale not a separate op."""
        return self.backend.RescaleNew(ct_id)

    # -- utility -----------------------------------------------------------

    def _pad(self, values: np.ndarray) -> np.ndarray:
        """Zero-pad a 1-D array up to ``self.slots``."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == self.slots:
            return values
        if values.size > self.slots:
            raise ValueError(
                f"vector length {values.size} exceeds slot count {self.slots}"
            )
        out = np.zeros(self.slots, dtype=np.float64)
        out[: values.size] = values
        return out

    def reset_counts(self) -> None:
        """Zero the op counter (for timing one kernel call cleanly)."""
        self.counts = OpCounts()

    def free(self, *ct_ids: int) -> None:
        """Hint the backend that these ciphertexts are no longer needed.

        Currently a no-op on desilo since the binding holds them in a
        Python dict that GC eventually reclaims; left as a hook for
        when we add an explicit DeleteCiphertext path.
        """
        return
