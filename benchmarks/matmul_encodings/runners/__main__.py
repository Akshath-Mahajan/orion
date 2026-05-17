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
    p.set_defaults(verify=True)
    return p.parse_args(argv)


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
          f"verify={args.verify}")

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
        elif gpu_monitor.true_idle_w is not None:
            print(f"[bench] true idle (pre-ctx): "
                  f"{gpu_monitor.true_idle_w:.2f} W")
        else:
            print(f"[bench] true idle unavailable "
                  f"(energy counter: disabled)")

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
                tenancy_str = (
                    "[CONTENDED] " if r.single_tenant is False else ""
                )
                print(
                    f"[bench]   {tenancy_str}"
                    f"{r.mean_seconds*1000:.1f}ms ± "
                    f"{r.std_seconds*1000:.1f}ms  "
                    f"{err_str}{hbm_str}{energy_str}"
                    f"rot={r.rotations} ct.ct={r.ct_ct_muls} ct.pt={r.ct_pt_muls}",
                    flush=True,
                )
                rows.append(r)
    finally:
        ctx.scheme.delete_scheme()

    write_csv(rows, args.output)
    elapsed = time.perf_counter() - t0
    print(f"[bench] wrote {len(rows)} rows -> {args.output}  ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
