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
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from benchmarks.matmul_encodings.context import Context
from benchmarks.matmul_encodings.plaintext.op_counts import OpCounts
from benchmarks.matmul_encodings.runners.gpu_sampler import GpuMonitor


@dataclasses.dataclass
class BenchResult:
    """One row of the harness output CSV.

    GPU columns (``peak_hbm_mb`` onwards) are only populated when the
    desilo backend runs on ``device=gpu`` and a ``GpuMonitor`` is
    attached. They are ``None`` otherwise, which CSV writes as the
    empty string.
    """

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
    # GPU memory columns (true peak via NVML sampling thread):
    #  - peak_hbm_mb       cumulative process high-water mark during
    #                      the timed window. Includes residual state
    #                      from prior kernels in the same run since
    #                      the harness reuses one Context across all
    #                      kernels.
    #  - peak_hbm_delta_mb peak - baseline_at_window_enter. The memory
    #                      THIS kernel added on top of its starting
    #                      state. Comparable kernel-to-kernel.
    peak_hbm_mb: float | None
    peak_hbm_delta_mb: float | None
    # Per-trial GPU energy (joules). gross is the raw counter delta /
    # n_trials; kernel_energy_j subtracts (resident_idle_w * window)
    # so it approximates the kernel's marginal energy above the
    # engine+keys-resident idle floor.
    gross_energy_j: float | None
    kernel_energy_j: float | None
    # mean_power_w = gross_energy_total / window_seconds. Useful sanity
    # check (should land between idle and TDP).
    mean_power_w: float | None
    # True idle power, sampled once before any FHE state is allocated.
    # Same value on every row from a single run; included so the CSV
    # is self-describing (paper figure can compare true vs resident
    # idle without an external reference).
    true_idle_w: float | None
    # False if any compute PID other than ours was on the GPU during
    # the timed window. When False, the energy columns above are
    # overcounted (whole-GPU counter, not per-process attributable).
    single_tenant: bool | None
    max_abs_err: float | None  # None when verify is disabled

    @classmethod
    def csv_header(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    def csv_row(self) -> list[str]:
        return [
            "" if v is None else str(v)
            for v in (getattr(self, f.name) for f in dataclasses.fields(self))
        ]


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
    gpu_monitor: GpuMonitor | None = None,
) -> BenchResult:
    """Time `run_fn`, record op counts, and (when on GPU) peak memory +
    integrated energy.

    Args:
        run_fn: Closure that invokes one kernel call. May return a value
            (typically the decrypted output) for `verify_fn` to consume.
        verify_fn: Optional. Receives the result of the LAST timed call
            and returns max absolute error vs a numpy reference.
        gpu_monitor: Optional NVML wrapper. When present *and* device is
            "gpu", the timed window is bracketed by ``gpu_monitor.measure()``
            which fills in peak_hbm_mb and the energy columns. The
            monitor's idle baseline (if calibrated) is used to compute
            kernel_energy_j; otherwise that column is left None.

    Returns:
        BenchResult with mean/std wall-clock, op counts, and (when
        applicable) GPU memory + energy.
    """
    # Warmup (untimed). Counts and GPU samples are taken AFTER warmup
    # so they reflect only the timed region.
    for _ in range(warmup):
        run_fn()

    ctx.reset_counts()

    times: list[float] = []
    last_result: object | None = None

    measure_gpu = (device == "gpu" and gpu_monitor is not None)

    if measure_gpu:
        sampler_cm = gpu_monitor.measure(n_trials=n_trials)
    else:
        # Null context manager so we can share the body.
        from contextlib import nullcontext
        sampler_cm = nullcontext(None)

    with sampler_cm as gpu_sample:
        for _ in range(n_trials):
            t0 = time.perf_counter()
            last_result = run_fn()
            times.append(time.perf_counter() - t0)

    counts: OpCounts = ctx.counts

    max_err: float | None = None
    if verify_fn is not None and last_result is not None:
        try:
            max_err = float(verify_fn(last_result))
        except Exception as e:  # verify failure shouldn't kill the row
            print(f"  [warn] verify_fn raised on {kernel}/{shape}: {e}")

    if gpu_sample is None:
        peak_hbm = peak_hbm_delta = gross_e = kern_e = mean_p = None
        single_tenant = None
    else:
        peak_hbm = gpu_sample.peak_hbm_mb
        peak_hbm_delta = gpu_sample.peak_hbm_delta_mb
        gross_e = gpu_sample.gross_energy_j
        kern_e = gpu_sample.kernel_energy_j
        mean_p = gpu_sample.mean_power_w
        single_tenant = gpu_sample.single_tenant

    # true_idle_w is identical across every row from a single run; it
    # lives on the GpuMonitor (sampled once before any ctx existed).
    true_idle = (
        gpu_monitor.true_idle_w if gpu_monitor is not None else None
    )

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
        peak_hbm_mb=peak_hbm,
        peak_hbm_delta_mb=peak_hbm_delta,
        gross_energy_j=gross_e,
        kernel_energy_j=kern_e,
        mean_power_w=mean_p,
        true_idle_w=true_idle,
        single_tenant=single_tenant,
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
