"""D7 benchmark harness: per-kernel shape sweep -> single CSV.

Usage:

    python -m benchmarks.matmul_encodings.runners \
        --backend desilo --device gpu --preset paper \
        --output benchmarks/matmul_encodings/results/desilo_gpu_paper.csv

    # quick smoke run on the cpu oracle:
    python -m benchmarks.matmul_encodings.runners \
        --backend lattigo --preset smoke --n-trials 1 --warmup 0 \
        --output /tmp/smoke.csv

    # restrict to a subset of kernels:
    python -m benchmarks.matmul_encodings.runners \
        --backend desilo --kernels bmm3,thor --preset paper \
        --output benchmarks/matmul_encodings/results/bmm3_thor.csv

The harness builds ONE Context per (backend, device) pair and reuses it
across kernels and shapes, matching Negar's runner pattern (one
HEContext, many kernel calls). HBM is sampled before / after the timed
block when device is gpu; otherwise that column is empty.

**GPU note.** `--device gpu` requires a desilofhe build with CUDA support
linked in. The default pip wheel installed in `myenv2` is CPU-only and
will raise ``RuntimeError: Not supported mode`` at Context init. Build
or install the GPU variant of desilofhe before running the GPU sweep.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import CKKS_PRESETS, BenchResult, make_context, write_csv
from .gpu_sampler import GpuMonitor
from .kernels import KERNEL_TABLE
from .shapes import SHAPE_SETS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="benchmarks.matmul_encodings.runners")
    p.add_argument("--backend", choices=["lattigo", "desilo"], required=True,
                   help="CKKS backend.")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="desilo device. 'gpu' triggers nvidia-smi HBM sampling.")
    p.add_argument("--ckks-preset", choices=list(CKKS_PRESETS.keys()),
                   default="default",
                   help="CKKS parameter preset. 'default' = LogN=13 "
                        "ConjugateInvariant (8192 slots, historical). "
                        "'negar' = LogN=13 Standard + LogQ=55+4*45 + LogP=61 "
                        "(4096 slots, matches D8 CPU baseline).")
    p.add_argument("--preset", choices=list(SHAPE_SETS.keys()), default="smoke",
                   help="Shape table preset. 'smoke' = one tiny shape per kernel.")
    p.add_argument("--kernels", default=",".join(KERNEL_TABLE.keys()),
                   help="Comma-separated kernel ids. Default = all.")
    p.add_argument("--n-trials", type=int, default=3,
                   help="Timed iterations per shape.")
    p.add_argument("--warmup", type=int, default=1,
                   help="Untimed iterations before the timed block.")
    p.add_argument("--no-verify", dest="verify", action="store_false",
                   help="Skip max_abs_err computation (saves a decrypt per shape).")
    p.add_argument("--output", required=True, type=Path,
                   help="Output CSV path. Parent dirs are created if needed.")
    p.add_argument("--lock-gpu-clocks", type=int, default=None, metavar="MHZ",
                   help="If set, run `nvidia-smi --lock-gpu-clocks=MHZ,MHZ` "
                        "at start and `--reset-gpu-clocks` on exit. Needs "
                        "sudo or appropriate permissions; failure is "
                        "logged but does not abort the run. Recommended "
                        "for paper sweeps to remove boost-clock variance.")
    p.add_argument("--cold-start", action="store_true",
                   help="Pause between shapes until GPU temperature is "
                        "within --cold-start-tolerance-c of the baseline "
                        "captured at start. Removes thermal drift across "
                        "the sweep at the cost of wall-clock time.")
    p.add_argument("--cold-start-tolerance-c", type=float, default=2.0,
                   help="Tolerance in Celsius for --cold-start pacing.")
    p.add_argument("--cold-start-timeout-s", type=float, default=60.0,
                   help="Max seconds --cold-start will wait per shape "
                        "before giving up and proceeding (the row's "
                        "start_temp_c column flags the actual temp).")
    p.set_defaults(verify=True)
    return p.parse_args(argv)


def _set_gpu_clocks(mhz: int | None) -> bool:
    """Lock the GPU graphics clock via nvidia-smi. Returns True on
    success, False on failure (logged). Caller is responsible for
    calling _reset_gpu_clocks() on exit even if this returned False
    (no-op in that case)."""
    if mhz is None:
        return True
    import subprocess
    cmd = ["nvidia-smi", "--lock-gpu-clocks=" + f"{mhz},{mhz}", "-i", "0"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=10)
        print(f"[bench] locked GPU clock to {mhz} MHz")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[bench] WARNING: clock lock failed "
              f"(stderr: {e.stderr.strip()}). Continuing unpinned.",
              flush=True)
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[bench] WARNING: clock lock unavailable: {e}. "
              f"Continuing unpinned.", flush=True)
        return False


def _reset_gpu_clocks() -> None:
    import subprocess
    try:
        subprocess.run(["nvidia-smi", "--reset-gpu-clocks", "-i", "0"],
                       check=False, capture_output=True, text=True,
                       timeout=10)
    except Exception:
        pass


def _selected_kernels(spec: str) -> list[str]:
    requested = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in requested if k not in KERNEL_TABLE]
    if unknown:
        raise SystemExit(
            f"Unknown kernel ids: {unknown}. "
            f"Valid options: {sorted(KERNEL_TABLE.keys())}"
        )
    return requested


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    kernels = _selected_kernels(args.kernels)
    shape_set = SHAPE_SETS[args.preset]

    print(f"[bench] backend={args.backend} device={args.device} "
          f"ckks_preset={args.ckks_preset} preset={args.preset} "
          f"kernels={kernels} "
          f"n_trials={args.n_trials} warmup={args.warmup} "
          f"verify={args.verify} "
          f"lock_clock={args.lock_gpu_clocks} "
          f"cold_start={args.cold_start}")

    # Optional clock pinning. Done BEFORE GpuMonitor.create() so the
    # true-idle baseline reflects the pinned clock too.
    clock_pin_attempted = (args.lock_gpu_clocks is not None
                           and args.device == "gpu")
    if clock_pin_attempted:
        _set_gpu_clocks(args.lock_gpu_clocks)

    # GPU monitor setup. MUST run before make_context() so the
    # true-idle baseline reflects "no FHE state allocated yet". The
    # resident-idle baseline (idle with engine+keys loaded, used for
    # kernel_energy_j subtraction) is sampled after make_context below.
    gpu_monitor: GpuMonitor | None = None
    if args.device == "gpu":
        gpu_monitor = GpuMonitor.create(device_index=0)
        if gpu_monitor is None:
            print("[bench] WARNING: pynvml unavailable; GPU energy/memory "
                  "columns will be empty.", flush=True)
        else:
            if gpu_monitor.true_idle_w is not None:
                print(f"[bench] true idle (pre-ctx): "
                      f"{gpu_monitor.true_idle_w:.2f} W")
            if gpu_monitor.baseline_temp_c is not None:
                print(f"[bench] baseline GPU temp: "
                      f"{gpu_monitor.baseline_temp_c:.1f} C "
                      f"(cold-start target = baseline + "
                      f"{args.cold_start_tolerance_c:.1f} C)")

    ctx = make_context(args.backend, device=args.device, preset=args.ckks_preset)
    print(f"[bench] context: slots={ctx.slots} max_level={ctx.max_level}")

    # Resident-idle baseline (engine + keys loaded). Used to compute
    # kernel_energy_j -- the kernel's marginal energy above its
    # already-loaded state.
    if gpu_monitor is not None:
        idle = gpu_monitor.calibrate_idle(duration_s=0.5)
        if idle is not None:
            print(f"[bench] resident idle (post-ctx): {idle:.2f} W "
                  f"(used for kernel_energy_j subtraction)")
        # Hand the cold-start config off to the monitor; bench_kernel
        # applies it post-warmup so the timed window starts from a
        # consistent thermal state on every shape.
        if args.cold_start:
            gpu_monitor.cold_start_tolerance_c = args.cold_start_tolerance_c
            gpu_monitor.cold_start_timeout_s = args.cold_start_timeout_s

    rows: list[BenchResult] = []
    t0 = time.perf_counter()
    try:
        for kernel_id in kernels:
            bench_fn, shape_key = KERNEL_TABLE[kernel_id]
            for shape in shape_set[shape_key]:
                print(f"[bench] {kernel_id:10s}  {shape.label}", flush=True)
                try:
                    r = bench_fn(
                        ctx, shape,
                        n_trials=args.n_trials, warmup=args.warmup,
                        verify=args.verify, device=args.device,
                        gpu_monitor=gpu_monitor,
                    )
                except Exception as e:
                    print(f"[bench]   FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                err_str = (
                    f"err={r.max_abs_err:.2e}  " if r.max_abs_err is not None else ""
                )
                hbm_str = (
                    f"hbm_d={r.peak_hbm_delta_mb:.1f}MB "
                    f"(tot={r.peak_hbm_mb:.1f}MB)  "
                    if r.peak_hbm_delta_mb is not None else ""
                )
                energy_str = (
                    f"E_kern={r.kernel_energy_j:.3f}J  "
                    f"E_gross={r.gross_energy_j:.3f}J  "
                    f"P={r.mean_power_w:.1f}W  "
                    if r.kernel_energy_j is not None else ""
                )
                env_str = (
                    f"T={r.start_temp_c:.1f}C  "
                    f"clk={r.mean_clock_mhz:.0f}MHz  "
                    if r.start_temp_c is not None
                    and r.mean_clock_mhz is not None else ""
                )
                tenancy_str = (
                    "[CONTENDED] " if r.single_tenant is False else ""
                )
                print(
                    f"[bench]   {tenancy_str}"
                    f"{r.mean_seconds*1000:.1f}ms ± "
                    f"{r.std_seconds*1000:.1f}ms  "
                    f"{err_str}{hbm_str}{energy_str}{env_str}"
                    f"rot={r.rotations} ct.ct={r.ct_ct_muls} ct.pt={r.ct_pt_muls}",
                    flush=True,
                )
                rows.append(r)
    finally:
        ctx.scheme.delete_scheme()
        if clock_pin_attempted:
            _reset_gpu_clocks()
            print("[bench] reset GPU clocks")

    write_csv(rows, args.output)
    elapsed = time.perf_counter() - t0
    print(f"[bench] wrote {len(rows)} rows -> {args.output}  ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
