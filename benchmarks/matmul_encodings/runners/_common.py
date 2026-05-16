"""Shared bench harness: timing, HBM measurement, CSV emission.

The per-kernel runners under this directory invoke ``bench_kernel`` with
a closure that takes a fresh ``Context`` and runs ONE kernel invocation.
The harness wraps the closure to:

  1. Run a configurable number of warmup iterations (untimed).
  2. Run ``n_trials`` timed iterations, recording wall-clock per trial.
  3. Sample GPU memory before / after the timed block (only when the
     desilo backend runs in gpu mode -- nvidia-smi is a no-op for the
     lattigo CPU oracle).
  4. Snapshot the OpCounts captured during the kernel via ``ctx.counts``.

Output shape is one ``BenchResult`` per (backend, kernel, shape_label).
The CLI driver (``runners/__main__.py``) collects results across kernels
and shapes and writes a single CSV.
"""

from __future__ import annotations

import dataclasses
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from benchmarks.matmul_encodings.context import Context
from benchmarks.matmul_encodings.plaintext.op_counts import OpCounts


@dataclasses.dataclass
class BenchResult:
    """One row of the harness output CSV."""

    backend: str           # "lattigo" or "desilo"
    device: str            # "cpu" or "gpu"
    kernel: str            # "bmm1", "bmm3", "thor", "moai_alg3", "moai_alg4", "rowenc"
    shape: str             # human-readable shape label
    n_he: int              # algorithm n_he (== ctx.slots when not chunked)
    n_trials: int
    mean_seconds: float
    std_seconds: float     # 0.0 when n_trials == 1
    rotations: int
    ct_ct_muls: int
    ct_pt_muls: int
    peak_hbm_mb: float | None  # None for cpu / lattigo runs
    max_abs_err: float | None  # None when verify is disabled

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def csv_row(self) -> list[str]:
        return [str(getattr(self, f.name)) for f in dataclasses.fields(self)]


def nvidia_smi_used_mb() -> float | None:
    """Return GPU0 used memory in MB; None if nvidia-smi isn't available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        return float(out.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def bench_kernel(
    *,
    backend: str,
    device: str,
    kernel: str,
    shape: str,
    n_he: int,
    ctx: Context,
    run_fn: Callable[[], object],
    n_trials: int = 3,
    warmup: int = 1,
    verify_fn: Callable[[object], float] | None = None,
) -> BenchResult:
    """Time `run_fn`, record op counts and (when on GPU) peak HBM.

    Args:
        run_fn: Closure that invokes one kernel call. May return a value
            (typically the decrypted output) for `verify_fn` to consume.
        verify_fn: Optional. Receives the result of the LAST timed call
            and returns max absolute error vs a numpy reference.

    Returns:
        BenchResult with mean / std wall-clock, op counts, and HBM delta.
    """
    # Warmup (untimed). Counts and HBM samples taken AFTER warmup so they
    # reflect only the timed region.
    for _ in range(warmup):
        run_fn()

    # GPU memory baseline (after warmup so any one-shot allocations are out)
    measure_hbm = (device == "gpu")
    hbm_before = nvidia_smi_used_mb() if measure_hbm else None

    ctx.reset_counts()

    times: list[float] = []
    last_result: object | None = None
    for _ in range(n_trials):
        t0 = time.perf_counter()
        last_result = run_fn()
        times.append(time.perf_counter() - t0)

    hbm_after = nvidia_smi_used_mb() if measure_hbm else None
    peak_hbm_mb = (hbm_after - hbm_before) if (
        measure_hbm and hbm_before is not None and hbm_after is not None
    ) else None

    counts: OpCounts = ctx.counts

    max_err: float | None = None
    if verify_fn is not None and last_result is not None:
        try:
            max_err = float(verify_fn(last_result))
        except Exception as e:  # verify failure shouldn't kill the row
            print(f"  [warn] verify_fn raised on {kernel}/{shape}: {e}")

    return BenchResult(
        backend=backend,
        device=device,
        kernel=kernel,
        shape=shape,
        n_he=n_he,
        n_trials=n_trials,
        # Mean/std over n_trials. n_trials==1 -> std defaults to 0.
        mean_seconds=statistics.fmean(times),
        std_seconds=(statistics.stdev(times) if len(times) > 1 else 0.0),
        # OpCounts cumulated across n_trials -- divide for per-trial counts.
        rotations=counts.rotations // n_trials,
        ct_ct_muls=counts.ct_ct_muls // n_trials,
        ct_pt_muls=counts.ct_pt_muls // n_trials,
        peak_hbm_mb=peak_hbm_mb,
        max_abs_err=max_err,
    )


def write_csv(results: Iterable[BenchResult], path: Path | str) -> None:
    """Emit one CSV with `BenchResult.csv_header()` as the first row."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BenchResult.csv_header())
        for r in results:
            w.writerow(r.csv_row())


# CKKS parameter presets. Each preset is a fully-specified ``ckks_params``
# dict. The orion/backend/device slice is filled in by ``make_context``.
#
# - ``default``: the test-suite config used by everything prior to this
#   commit -- LogN=13 ConjugateInvariant, 5 levels at 29/26-bit primes.
#   Lands at 8192 slots on both backends.
# - ``negar``: a 1:1 port of Negar's Go DefaultParams in
#   matmul-encoding-material/MatMult/matmult/init_lattigo.go --
#   LogN=13 Standard, LogQ=55+4*45, LogP=61. Lands at true 4096 slots on
#   both backends thanks to the desilo binding fix that wires ``slot_count``
#   through to the ``Engine(slot_count=..., max_level=...)`` overload.
CKKS_PRESETS: dict[str, dict[str, object]] = {
    "default": {
        "LogN": 13,
        "LogQ": [29, 26, 26, 26, 26],
        "LogP": [29],
        "LogScale": 26,
        "H": 8192,
        "RingType": "ConjugateInvariant",
    },
    "negar": {
        "LogN": 13,
        "LogQ": [55, 45, 45, 45, 45],
        "LogP": [61],
        "LogScale": 45,
        "H": 8192,
        "RingType": "Standard",
    },
}


def make_context(
    backend: str,
    device: str = "cpu",
    *,
    preset: str = "default",
) -> Context:
    """Build a fresh Context at the requested CKKS preset.

    ``preset="default"`` keeps the historical test-suite config (LogN=13
    ConjugateInvariant, 8192 slots). ``preset="negar"`` matches the Go
    DefaultParams used by the D8 CPU baseline (LogN=13 Standard,
    LogQ=55+4*45, LogP=61, 4096 slots on both backends).

    ``device`` only matters for the desilo backend; lattigo ignores it.
    """
    if preset not in CKKS_PRESETS:
        raise ValueError(
            f"Unknown CKKS preset {preset!r}. "
            f"Available: {sorted(CKKS_PRESETS)}"
        )
    config = {
        "ckks_params": dict(CKKS_PRESETS[preset]),
        "orion": {
            "backend": backend,
            "io_mode": "none",
            "debug": False,
            "device": device,
        },
    }
    return Context.from_config(config)
