"""Paper shape tables for the D7 benchmark sweep.

These mirror Negar's Go runners (matmult/{bmm1,bmm3,thor,moai,rowenc}_runner.go)
so GPU numbers from this harness line up shape-for-shape with the CPU
baseline she gathers from the same tables.

Two presets per kernel:

  * ``smoke``   -- one or two tiny shapes; the harness self-test runs at
                   this preset to confirm CSV emission and kernel wiring
                   without burning minutes per shape.
  * ``paper``   -- the (possibly truncated) configuration list pulled
                   directly from the corresponding Go runner. These are
                   the shapes the figure should report.

Several entries in Negar's Go are commented out at the largest sizes
(e.g. BMM-III at 2048; BMM-I at 2064). They're listed below as comments
in case you want to extend the sweep, but the default ``paper`` preset
sticks with the configurations Negar's runner actively executes today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Shape dataclasses (one per kernel API surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bmm1Shape:
    """BMM-I: matmul of (N, M) @ (M, P) via blocks (s_n, s_m, s_p)."""
    N: int; M: int; P: int
    s_n: int; s_m: int; s_p: int

    @property
    def label(self) -> str:
        return f"({self.N},{self.M},{self.P})/blk({self.s_n},{self.s_m},{self.s_p})"


@dataclass(frozen=True)
class Bmm3Shape:
    """BMM-III: pairwise-coprime (n, m, p) matmul."""
    n: int; m: int; p: int

    @property
    def label(self) -> str:
        return f"({self.n},{self.m},{self.p})"


@dataclass(frozen=True)
class ThorShape:
    """THOR: per-head (m, n) @ (n, n) over H heads with c diagonals/ct."""
    H: int; m: int; n: int; c: int

    @property
    def label(self) -> str:
        return f"H={self.H},m={self.m},n={self.n},c={self.c}"


@dataclass(frozen=True)
class MoaiShape:
    """MOAI Alg 3 / Alg 4: n_batch matmuls of (m, d') in lock-step."""
    n_batch: int; m: int; d_prime: int

    @property
    def label(self) -> str:
        return f"batch={self.n_batch},m={self.m},d'={self.d_prime}"


@dataclass(frozen=True)
class RowEncShape:
    """RowEnc: (n, n) @ (n, n); n must be a power of 2."""
    n: int

    @property
    def label(self) -> str:
        return f"n={self.n}"


# ---------------------------------------------------------------------------
# Per-kernel shape sets
#
# The "paper" preset matches what Negar's *_runner.go actually executes
# today (uncommented entries). The "smoke" preset is one cheap shape
# usable for self-test / CI.
# ---------------------------------------------------------------------------


BMM1_SMOKE: Sequence[Bmm1Shape] = (
    Bmm1Shape(N=43, M=45, P=44, s_n=43, s_m=45, s_p=44),
)
BMM1_PAPER: Sequence[Bmm1Shape] = (
    # Mirrors matmult/bmm1_runner.go::BMM1CiphertextSuite (uncommented entries).
    Bmm1Shape(N=516, M=540, P=528, s_n=43, s_m=45, s_p=44),
    # Negar's larger shapes commented out in the Go; uncomment to extend:
    # Bmm1Shape(N=129, M=135, P=132, s_n=43, s_m=45, s_p=44),
    # Bmm1Shape(N=258, M=270, P=264, s_n=43, s_m=45, s_p=44),
    # Bmm1Shape(N=1032, M=1080, P=1056, s_n=43, s_m=45, s_p=44),
    # Bmm1Shape(N=2064, M=2070, P=2068, s_n=43, s_m=45, s_p=44),
)


BMM3_SMOKE: Sequence[Bmm3Shape] = (
    Bmm3Shape(n=5, m=7, p=11),
)
BMM3_PAPER: Sequence[Bmm3Shape] = (
    # Mirrors matmult/bmm3_runner.go::BMM3CiphertextSuite.
    Bmm3Shape(n=128, m=131, p=129),
    Bmm3Shape(n=256, m=259, p=257),
    Bmm3Shape(n=512, m=515, p=513),
    Bmm3Shape(n=1024, m=1027, p=1025),
    # Bmm3Shape(n=2048, m=2051, p=2049),  # commented out in Go
)


THOR_SMOKE: Sequence[ThorShape] = (
    ThorShape(H=2, m=2, n=2, c=2),
)
THOR_PAPER: Sequence[ThorShape] = (
    # Mirrors matmult/thor_runner.go::ThorCiphertextSuite (size=2048, H=1).
    # c is chosen so c*n*H <= ctx.slots (4096): for n=2048 H=1, c must = 2.
    ThorShape(H=1, m=2048, n=2048, c=2),
)


MOAI_SMOKE: Sequence[MoaiShape] = (
    MoaiShape(n_batch=2, m=4, d_prime=4),
)
MOAI_PAPER: Sequence[MoaiShape] = (
    # Mirrors matmult/moai_runner.go::MoaiCiphertextSuite (size=2048).
    # n_batch must keep n_batch * m within ctx.slots; with m=2048 we need
    # n_batch=1 (fits exactly in 4096-slot config? no, 2048 < 4096 so fits).
    # Use n_batch=1 to match Negar's "single matmul" benchmark.
    MoaiShape(n_batch=1, m=2048, d_prime=2048),
)


ROWENC_SMOKE: Sequence[RowEncShape] = (
    RowEncShape(n=4),
)
ROWENC_PAPER: Sequence[RowEncShape] = (
    # Mirrors matmult/rowenc_runner.go::RowCiphertextSuite (n in {4,8,16,32}).
    RowEncShape(n=4),
    RowEncShape(n=8),
    RowEncShape(n=16),
    RowEncShape(n=32),
    # n=64 fits (64^2=4096 slots, exactly), uncomment to extend:
    # RowEncShape(n=64),
)


SHAPE_SETS: dict[str, dict[str, Sequence]] = {
    "smoke": {
        "bmm1":   BMM1_SMOKE,
        "bmm3":   BMM3_SMOKE,
        "thor":   THOR_SMOKE,
        "moai":   MOAI_SMOKE,
        "rowenc": ROWENC_SMOKE,
    },
    "paper": {
        "bmm1":   BMM1_PAPER,
        "bmm3":   BMM3_PAPER,
        "thor":   THOR_PAPER,
        "moai":   MOAI_PAPER,
        "rowenc": ROWENC_PAPER,
    },
}
