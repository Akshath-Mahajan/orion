# D8 — CPU baseline (Negar's Go binary, instrumented for CSV)

This directory holds the patches + driver script that produce the CPU
baseline numbers consumed by D9 (the headline figure).

## What's here

- `csv_emit.go` — package-local CSV writer hung off `BENCH_CSV_OUT`.
  Goes into `matmul-encoding-material/MatMult/matmult/`.
- `memsnap.go` — minimal `TakeMemSnap`/`PrintMemDelta` stubs that the
  upstream repo references but never defines (without these, `go build`
  fails out of the box).
- `matmult_csv_emit.patch` — diff against
  `matmul-encoding-material/MatMult/matmult/{rowenc,bmm1,bmm3,thor,moai}_runner.go`
  that adds one `csvEmit(...)` call at the end of each per-shape report
  block, mirroring the existing pretty-print but in our harness CSV
  schema.
- `run.sh` — driver: copies the shims into the gitignored matmult clone,
  applies the patch, builds the binary, drives the menu via stdin to
  run all five HE suites (Row → BMM-III → THOR → MOAI → BMM-I), and
  writes one CSV under `benchmarks/matmul_encodings/results/`.

## Why patches and not a fork

`matmul-encoding-material/` is a sibling clone gitignored from orion
(see `.gitignore`). Direct edits there can't be tracked in orion's
history, so we keep the diffs as committable artifacts in this dir.
Apply with `git apply matmult_csv_emit.patch` from the
`matmul-encoding-material/MatMult/` directory; `run.sh` does this
automatically.

## CSV schema

Matches `benchmarks/matmul_encodings/runners/_common.py` `BenchResult`
exactly — same columns, same units, so a D9 plotting script can `pd.read_csv`
both this CSV and the GPU sweep CSV and concatenate them.

```
backend, device, kernel, shape, n_he, n_trials, mean_seconds,
std_seconds, rotations, ct_ct_muls, ct_pt_muls, peak_hbm_mb, max_abs_err
```

`peak_hbm_mb` is empty for CPU rows. `rotations` / `ct_ct_muls` /
`ct_pt_muls` come from the theoretical formulas Negar already prints
in stdout (the cipher kernels don't track measured op tallies); see
the corresponding `Theoretical*Costs` functions in her repo.

## How to re-run

```bash
ORION_ROOT=$HOME/orion ./benchmarks/matmul_encodings/cpu_baseline/run.sh
# bump trial count for tighter stdev:
MATMULT_TRIALS=3 ./benchmarks/matmul_encodings/cpu_baseline/run.sh
```

Output: `benchmarks/matmul_encodings/results/cpu_baseline_paper.csv`.

## Notes / caveats

- **Wall-clock budget.** The default paper preset includes 2048-class
  shapes for THOR + MOAI and (1024, 1027, 1025) for BMM-III; one run
  of the full sweep at `-trials 1` takes on the order of an hour on
  CPU. For smaller smoke runs, edit the shape lists in each `_runner.go`
  before running.
- **Slot-count caveat (PLAN.md §6 D7).** Negar's Go uses LogN=13 with
  the Standard ring → 4096 slots. Our Python harness lands at
  ConjugateInvariant → 8192 slots. Per-rotation cost differs by ~2x.
  Either rerun the Python harness at matching slot count or document
  the difference in the methodology.
