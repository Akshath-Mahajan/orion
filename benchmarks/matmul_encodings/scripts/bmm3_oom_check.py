#!/usr/bin/env python
"""BMM-3 hoisted-mode OOM probe across the shapes that previously
cached-mode-OOM'd in session 2 (per HANDOFF2.md).

Session-2 paper-preset cached-mode results on the RTX 3090:

  shape              cpu       gpu (cached)
  (128, 131, 129)    8.35s     12.43s        (last one that fit)
  (256, 259, 257)    47.24s    OOM           <- first OOM
  (512, 515, 513)    303.65s   OOM
  (1024, 1027, 1025) 1896.83s  OOM

This script re-tests those three OOM'd shapes with the new hoisted
mode + ctx.free()-per-block memory discipline, sweeping
``hoist_block_size`` from smallest (least memory) upward at each
shape. We stop the block-size sweep at the first failure inside one
shape (smaller blocks already tested + larger will only use more);
we move to the next shape regardless, so we get a clear OOM-cliff
picture in one run.

For each (shape, block_size) we record:
  * runtime in seconds (None if it crashed)
  * peak per-process GPU memory in MB (best snapshot before fault)
  * decrypt max_abs_err vs numpy.matmul
  * success flag + exception class/message on failure

Progress is flushed to the JSON log after every iteration so the
user can ``cat`` it during execution without waiting for the sweep
to end. Each (shape, block_size) starts from a fresh Context so
prior runs' state can't push later ones into OOM.

Run via tmux so disconnects don't kill the job; see
``benchmarks/matmul_encodings/BMM3_OOM_CHECK.md`` for instructions.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from benchmarks.matmul_encodings.kernels import bmm3_cipher
from benchmarks.matmul_encodings.runners._common import make_context


# Each (shape, block_sizes): block sizes ordered smallest-mem first so a
# failure tells us the OOM cliff. Later block sizes are aggressive
# enough that they'd OOM if the smallest already failed.
TEST_PLAN = [
    {"shape": (256, 259, 257),    "block_sizes": [4, 8, 16, 32]},
    {"shape": (512, 515, 513),    "block_sizes": [4, 8, 16]},
    {"shape": (1024, 1027, 1025), "block_sizes": [4, 8]},
]
LOG_PATH = Path(
    "/home/avm6288/orion/benchmarks/matmul_encodings/results/"
    "bmm3_oom_check.json"
)


def _gpu_used_mb_for_pid() -> float | None:
    """Per-process GPU memory in MB via NVML; None if unavailable."""
    try:
        import os
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        my_pid = os.getpid()
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        for p in procs:
            if p.pid == my_pid and p.usedGpuMemory:
                return p.usedGpuMemory / (1024 * 1024)
    except Exception:
        pass
    return None


def _whole_gpu_used_mb() -> float | None:
    """Whole-GPU used memory in MB (fallback if per-process fails)."""
    try:
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
    except Exception:
        return None


def _save(results: list[dict], *, completed: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps({
        "test_plan": [
            {"shape": list(p["shape"]), "block_sizes": p["block_sizes"]}
            for p in TEST_PLAN
        ],
        "completed": completed,
        "written_at_unix": time.time(),
        "results": results,
    }, indent=2))


def _run_one(shape: tuple[int, int, int], block_size: int) -> dict:
    """Build a fresh ctx, run one BMM-3 hoisted, return a result row."""
    n, m, p = shape
    print(
        f"\n----- shape={shape}  hoist_block_size={block_size} -----",
        flush=True,
    )
    ctx = None
    runtime_s: float | None = None
    peak_mem_mb: float | None = None
    max_err: float | None = None
    err_msg: str | None = None
    success = False
    try:
        t_ctx = time.perf_counter()
        ctx = make_context("desilo", device="gpu", preset="negar")
        print(
            f"  [setup] ctx build: {time.perf_counter() - t_ctx:.1f}s"
            f"  (GPU now: {_whole_gpu_used_mb():.0f} MB whole-gpu)",
            flush=True,
        )

        rng = np.random.default_rng(seed=42)
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))

        t_run = time.perf_counter()
        result = bmm3_cipher.bmm3_he(
            ctx, A, B,
            mode="hoisted", hoist_block_size=block_size,
        )
        runtime_s = time.perf_counter() - t_run

        peak_mem_mb = _gpu_used_mb_for_pid() or _whole_gpu_used_mb()
        max_err = float(np.max(np.abs(result - A @ B)))
        success = True
        print(
            f"  [done] {runtime_s:.1f}s  "
            f"peak_mem={peak_mem_mb:.0f}MB  "
            f"max_err={max_err:.2e}",
            flush=True,
        )
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        peak_mem_mb = _gpu_used_mb_for_pid() or _whole_gpu_used_mb()
        print(f"  [FAIL] {err_msg}", flush=True)
        print(f"  [FAIL] peak_mem at fault: {peak_mem_mb} MB", flush=True)
        traceback.print_exc()
    finally:
        if ctx is not None:
            try:
                ctx.scheme.delete_scheme()
            except Exception as e:
                print(f"  [warn] delete_scheme: {e}", flush=True)

    return {
        "shape": list(shape),
        "block_size": block_size,
        "success": success,
        "runtime_s": runtime_s,
        "peak_gpu_mem_mb": peak_mem_mb,
        "max_abs_err": max_err,
        "error": err_msg,
    }


def main() -> int:
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as e:
        print(f"[setup] pynvml init failed: {e} "
              f"(continuing without per-process mem)", flush=True)

    results: list[dict] = []

    print(f"\n==== BMM-3 hoisted OOM check ====", flush=True)
    print(f"==== log: {LOG_PATH} ====", flush=True)
    for entry in TEST_PLAN:
        print(f"==== plan: shape={entry['shape']}  "
              f"block_sizes={entry['block_sizes']}", flush=True)
    print("", flush=True)

    for entry in TEST_PLAN:
        shape = entry["shape"]
        block_sizes = entry["block_sizes"]
        shape_had_failure = False
        for block_size in block_sizes:
            row = _run_one(shape, block_size)
            results.append(row)
            _save(results, completed=False)
            if not row["success"]:
                shape_had_failure = True
                print(
                    f"  [stop-shape] {shape} block_size={block_size} failed; "
                    f"larger block sizes will use more memory -- "
                    f"skipping rest of this shape's sweep.",
                    flush=True,
                )
                break

        if shape_had_failure and block_sizes[0] == min(block_sizes):
            # Smallest block size for this shape failed -- larger shapes
            # will definitely OOM too. Bail out of the outer sweep.
            print(
                f"  [stop-plan] smallest block size for shape {shape} "
                f"failed; abandoning bigger shapes.",
                flush=True,
            )
            break

    _save(results, completed=True)
    print("\n==== DONE ====", flush=True)
    for r in results:
        flag = "OK" if r["success"] else "FAIL"
        rt = f"{r['runtime_s']:.1f}s" if r["runtime_s"] else "—"
        pm = f"{r['peak_gpu_mem_mb']:.0f}MB" if r["peak_gpu_mem_mb"] else "—"
        sh = "x".join(str(s) for s in r["shape"])
        print(f"  shape={sh:>16s}  block={r['block_size']:>3d}  "
              f"{flag:>4s}  runtime={rt:>10s}  peak={pm:>10s}", flush=True)

    return 0 if all(r["success"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
