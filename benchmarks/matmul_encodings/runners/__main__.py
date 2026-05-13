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

from ._common import BenchResult, make_context, write_csv
from .kernels import KERNEL_TABLE
from .shapes import SHAPE_SETS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="benchmarks.matmul_encodings.runners")
    p.add_argument("--backend", choices=["lattigo", "desilo"], required=True,
                   help="CKKS backend.")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="desilo device. 'gpu' triggers nvidia-smi HBM sampling.")
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
          f"preset={args.preset} kernels={kernels} "
          f"n_trials={args.n_trials} warmup={args.warmup} "
          f"verify={args.verify}")

    ctx = make_context(args.backend, device=args.device)
    print(f"[bench] context: slots={ctx.slots} max_level={ctx.max_level}")

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
                    )
                except Exception as e:
                    print(f"[bench]   FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                err_str = (
                    f"err={r.max_abs_err:.2e}  " if r.max_abs_err is not None else ""
                )
                hbm_str = (
                    f"hbm={r.peak_hbm_mb:.1f}MB  "
                    if r.peak_hbm_mb is not None else ""
                )
                print(
                    f"[bench]   {r.mean_seconds*1000:.1f}ms ± "
                    f"{r.std_seconds*1000:.1f}ms  "
                    f"{err_str}{hbm_str}"
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
