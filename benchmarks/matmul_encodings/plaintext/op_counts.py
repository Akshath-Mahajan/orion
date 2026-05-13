"""Shared operation counter dataclass used by every plaintext kernel port.

Mirrors the Go `OpCounts` struct in matmult/util.go. Counts are mutated in
place by the kernels so that callers can merge across many block calls,
matching the Go pass-by-pointer semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpCounts:
    """Tally of HE-equivalent operations a plaintext kernel would perform.

    Fields:
        rotations:   slot rotations (one Galois key-switch each in HE).
        ct_ct_muls:  ciphertext x ciphertext multiplications.
        ct_pt_muls:  ciphertext x plaintext multiplications.
    """

    rotations: int = 0
    ct_ct_muls: int = 0
    ct_pt_muls: int = 0

    def __add__(self, other: "OpCounts") -> "OpCounts":
        """Merge two op counts into a fresh OpCounts (non-mutating)."""
        if not isinstance(other, OpCounts):
            return NotImplemented
        return OpCounts(
            rotations=self.rotations + other.rotations,
            ct_ct_muls=self.ct_ct_muls + other.ct_ct_muls,
            ct_pt_muls=self.ct_pt_muls + other.ct_pt_muls,
        )

    def __iadd__(self, other: "OpCounts") -> "OpCounts":
        if not isinstance(other, OpCounts):
            return NotImplemented
        self.rotations += other.rotations
        self.ct_ct_muls += other.ct_ct_muls
        self.ct_pt_muls += other.ct_pt_muls
        return self

    def __str__(self) -> str:
        total = self.rotations + self.ct_ct_muls + self.ct_pt_muls
        return (
            f"rot={self.rotations}  ct.ct={self.ct_ct_muls}  "
            f"ct.pt={self.ct_pt_muls}  (total={total})"
        )
