"""GPU energy + memory measurement for the D7 harness.

Built on NVML (via ``pynvml``). The 3090's driver does not expose
``total_energy_consumption`` through ``nvidia-smi --query-gpu=...`` (the
CLI rejects the field) but **does** expose it through NVML's
``nvmlDeviceGetTotalEnergyConsumption``. That counter is a cumulative
millijoule reading since driver load, so we get exact kernel energy
from one read before the timed block and one read after -- no sampling
integration error.

GPU memory peak still needs sampling: NVML has no per-process peak
counter. ``GpuMonitor`` spawns a daemon thread that polls
``nvmlDeviceGetComputeRunningProcesses`` at ``poll_ms`` intervals and
keeps the running max of *our* process's used GPU memory. Default
5 ms gives ~200 samples/sec with negligible overhead (each poll is
~20us in-process).

Per-process power attribution is NOT supported on this driver
(``nvmlDeviceGetProcessUtilization`` returns ``Not Found`` because
accounting mode is disabled and would require root to enable). To
keep the energy column honest we instead assert single-tenancy at run
start: if any compute process other than our own PID is found on the
GPU, the monitor records a contention flag and the row's energy
columns become "best effort -- GPU not single-tenant".

Usage:

    mon = GpuMonitor(device_index=0)
    mon.calibrate_idle(duration_s=0.5)
    with mon.measure() as m:
        run_kernel()
    print(m.peak_hbm_mb, m.kernel_energy_j, m.gross_energy_j, m.mean_power_w)

The measure() context manager starts/stops the memory sampler and
takes the energy counter reads at the boundaries. The returned object
has all four metrics filled in. If the monitor was not constructed
(no GPU, ``pynvml`` missing) ``GpuMonitor.create()`` returns ``None``
and the harness falls back to leaving energy/memory columns empty.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator


@dataclasses.dataclass
class GpuSample:
    """One row of GPU measurement, produced by ``GpuMonitor.measure()``.

    Two memory columns:
      * peak_hbm_mb       cumulative process high-water mark during the
                          timed window. Includes any state left behind
                          by prior kernels sharing the same ctx.
      * peak_hbm_delta_mb peak - baseline_at_window_enter. Approximates
                          the memory this kernel *added* on top of its
                          starting state. Comparable kernel-to-kernel.

    Energy columns are filled when the NVML energy counter is available
    AND single-tenancy held throughout the timed block. ``mean_power_w``
    is gross_energy / window_seconds; useful as a sanity check.
    """

    peak_hbm_mb: float | None
    peak_hbm_delta_mb: float | None
    gross_energy_j: float | None       # raw counter delta / n_trials
    kernel_energy_j: float | None      # gross minus idle*duration / n_trials
    mean_power_w: float | None         # gross_energy / total_window_s
    window_seconds: float
    single_tenant: bool                # False if other PIDs were seen


class GpuMonitor:
    """NVML-backed peak memory + energy counter wrapper.

    Construct with ``GpuMonitor.create(device_index=0)`` which returns
    ``None`` cleanly if pynvml isn't installed or no GPU is visible.
    """

    def __init__(self, handle, my_pid: int, energy_supported: bool):
        self._h = handle
        self._my_pid = my_pid
        self._energy_supported = energy_supported
        # Resident idle (engine+keys loaded): set by calibrate_idle()
        # AFTER ctx creation. Used to compute kernel_energy_j.
        self._idle_power_w: float | None = None
        # True idle (no FHE state allocated yet): sampled once inside
        # create() before the caller allocates a Context. Informational
        # column in every BenchResult; never used for energy subtraction.
        self._true_idle_w: float | None = None

    @classmethod
    def create(cls, device_index: int = 0,
               *, true_idle_duration_s: float = 0.5) -> "GpuMonitor | None":
        """Build a monitor and capture the true-idle baseline.

        Call this BEFORE ``make_context`` / any other GPU work in the
        process so the true idle reading reflects "no FHE state
        allocated". Returns None cleanly if pynvml is missing or no
        GPU is visible.
        """
        try:
            import pynvml
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            return None
        # Probe whether the energy counter is callable. Some drivers /
        # GPUs (older Pascal, etc) don't expose it.
        energy_ok = True
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except Exception:
            energy_ok = False

        mon = cls(handle=handle, my_pid=os.getpid(),
                  energy_supported=energy_ok)

        # True-idle baseline: sample BEFORE the caller creates a Context.
        # Uses the same counter-delta method as calibrate_idle() but
        # under the (caller-controlled) precondition of no resident
        # FHE state.
        if energy_ok:
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
            t0 = time.perf_counter()
            time.sleep(true_idle_duration_s)
            e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
            t1 = time.perf_counter()
            mon._true_idle_w = (e1 - e0) / (t1 - t0)

        return mon

    # ------------------------------------------------------------------
    # Low-level NVML reads
    # ------------------------------------------------------------------

    def _energy_j(self) -> float | None:
        """Cumulative GPU energy in joules since driver load (None if
        the counter is unavailable on this device)."""
        if not self._energy_supported:
            return None
        import pynvml
        # nvmlDeviceGetTotalEnergyConsumption returns millijoules.
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h) / 1000.0

    def _power_w(self) -> float:
        """Instantaneous power draw in watts."""
        import pynvml
        return pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0

    def _used_mem_mb_for_pid(self) -> float:
        """Used GPU memory in MB attributed to our PID, or 0 if not
        currently present in the compute-process list."""
        import pynvml
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._h)
        except Exception:
            return 0.0
        for p in procs:
            if p.pid == self._my_pid and p.usedGpuMemory:
                return p.usedGpuMemory / (1024 * 1024)
        return 0.0

    def _other_compute_pids(self) -> list[int]:
        """List of compute PIDs on this GPU other than our own."""
        import pynvml
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._h)
        except Exception:
            return []
        return [p.pid for p in procs if p.pid != self._my_pid]

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_idle(self, duration_s: float = 0.5) -> float | None:
        """Sample idle power for ``duration_s`` and remember it as the
        baseline. Use *before* allocating any large GPU state.

        Returns the measured idle watts, or None if the energy counter
        is unavailable (in which case kernel-attributable energy is
        also unavailable).
        """
        if not self._energy_supported:
            self._idle_power_w = None
            return None
        e_before = self._energy_j()
        t_before = time.perf_counter()
        time.sleep(duration_s)
        e_after = self._energy_j()
        t_after = time.perf_counter()
        self._idle_power_w = (e_after - e_before) / (t_after - t_before)
        return self._idle_power_w

    @property
    def idle_power_w(self) -> float | None:
        """Resident idle (post-ctx) used for kernel_energy_j subtraction."""
        return self._idle_power_w

    @property
    def true_idle_w(self) -> float | None:
        """True idle (pre-ctx) -- informational only."""
        return self._true_idle_w

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @contextmanager
    def measure(self, *, n_trials: int = 1,
                poll_ms: float = 5.0) -> Iterator[GpuSample]:
        """Context manager bracketing the timed region.

        Memory: a daemon thread polls per-PID used memory every
        ``poll_ms`` and keeps the running max for our PID. Energy: read
        the counter at enter/exit and subtract; divide by n_trials.

        On exit, the yielded ``GpuSample`` is mutated in place with the
        final metrics. Other compute processes appearing on the GPU at
        sample time flip ``single_tenant=False``.
        """
        # Snapshot baseline memory at window enter so the delta column
        # reflects "memory this kernel added on top of its starting
        # state", not cumulative process memory.
        baseline_mb = self._used_mem_mb_for_pid()
        peak_holder = {
            "peak": baseline_mb,
            "single_tenant": True,
        }
        stop = threading.Event()

        def _sampler() -> None:
            while not stop.is_set():
                used = self._used_mem_mb_for_pid()
                if used > peak_holder["peak"]:
                    peak_holder["peak"] = used
                if self._other_compute_pids():
                    peak_holder["single_tenant"] = False
                time.sleep(poll_ms / 1000.0)

        # Initial single-tenancy check before starting the thread.
        if self._other_compute_pids():
            peak_holder["single_tenant"] = False

        e_before = self._energy_j()
        t_before = time.perf_counter()

        thread = threading.Thread(target=_sampler, daemon=True)
        thread.start()

        sample = GpuSample(
            peak_hbm_mb=None,
            peak_hbm_delta_mb=None,
            gross_energy_j=None,
            kernel_energy_j=None,
            mean_power_w=None,
            window_seconds=0.0,
            single_tenant=True,
        )

        try:
            yield sample
        finally:
            stop.set()
            thread.join(timeout=1.0)
            t_after = time.perf_counter()
            e_after = self._energy_j()

            window = t_after - t_before
            sample.window_seconds = window
            sample.peak_hbm_mb = peak_holder["peak"]
            sample.peak_hbm_delta_mb = max(
                0.0, peak_holder["peak"] - baseline_mb
            )
            sample.single_tenant = peak_holder["single_tenant"]

            if e_before is not None and e_after is not None and window > 0:
                gross = e_after - e_before
                sample.gross_energy_j = gross / n_trials
                sample.mean_power_w = gross / window
                if self._idle_power_w is not None:
                    idle_energy = self._idle_power_w * window
                    sample.kernel_energy_j = max(
                        0.0, (gross - idle_energy) / n_trials
                    )
