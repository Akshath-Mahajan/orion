# Session handoff — matmul-encoding paper

> Written 2026-05-13 by Akshath + the prior Claude instance. Read this
> first if you're a fresh Claude picking up after he reconnects.
> [`PLAN.md`](PLAN.md) is the canonical long-term plan; this doc only
> captures what's specific to handing off across sessions.
>
> **Last updated 2026-05-13 ~07:30**: CPU baseline sweep finished
> cleanly (12 rows, 6h30m). See "Background sweep" section for the
> numbers and the next-step menu Akshath was asked.

## TL;DR for the next instance

- **Active branch**: `feat/matmul-encoding` off `main`. All work in this
  session merged via `--no-ff` from short-lived sub-branches; topology
  is preserved.
- **What's in flight right now**: a CPU-baseline benchmark sweep is
  running inside a detached tmux session named `cpu-baseline-d8`. It
  was launched by the prior Claude before Akshath disconnected. **Do
  not kill it** — see "Background sweep" section below for status
  checks.
- **What's next**: D9 (the headline figure). D7 (Python harness) +
  D8 (Go CPU baseline tooling) both landed; the figure consumes the
  CSVs both produce.
- **Read for context, in this order**: (1) `PLAN.md`, especially §6
  ("Plan — what's left") and §7 ("Suggested next move"); (2) the
  `cpu_baseline/README.md`; (3) the per-kernel docstrings in
  `kernels/*.py` — they explain the lazy-relin / hoisting design.
- **Auto-memory entries** to consult: `user_role.md`,
  `project_matmul_encoding_paper.md`,
  `project_desilo_optimization_gaps.md`,
  `project_matmul_open_questions.md`,
  `project_matmul_testing_strategy.md`,
  `reference_matmul_encoding_context_repo.md`.

## What landed this session (2026-05-12 to 2026-05-13)

Five sub-branches, all merged back into `feat/matmul-encoding`. In
chronological order on the branch graph:

| Merge   | Subbranch                                   | Adds                                                                                                                                                       |
|---------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| f02526e | `feat/matmul-encoding-kernel-coverage`      | MOAI Algorithm 4 (Diag×Col→Col) + the Q·Kᵀ·V chain runner; row-encoding plaintext + cipher kernel; conftest `gc.collect()` fix to keep the suite stable.   |
| c583c2d | `feat/matmul-encoding-lazy-relin-hoisting`  | Lazy-relin port for THOR + MOAI (all 5 kernels lazy in source); desilo `RotateBatchNew` rewritten to use the safe `FixedRotationKey` overload (hoisting!). |
| e14d0cb | `feat/matmul-encoding-benchmark-harness`    | D7: `benchmarks/matmul_encodings/runners/` — CLI sweep that times each kernel over a chosen shape preset, captures op counts + nvidia-smi HBM, emits CSV.  |
| (current)| `feat/matmul-encoding-cpu-baseline`        | D8 tooling: `cpu_baseline/{csv_emit.go,memsnap.go,matmult_csv_emit.patch,run.sh,README.md}` — patches Negar's Go to emit our schema, plus driver script.   |

`git log --oneline --graph -20` shows the full picture.

### Three decisions worth remembering, with the why

1. **Hoisting fix used overload (2), not "fix overload (1)".**
   `desilofhe.Engine.rotate_batch` has two overloads:
   `(ct, RotationKey, list[int])` (deltas-based) and
   `(ct, list[FixedRotationKey])` (per-delta keys). Overload (1)
   segfaults on negative deltas, mid-stride positive batches, and
   most BSGS patterns. Overload (2) was robust on every pattern we
   tested. The fix is in `orion/backend/desilo/bindings.py::RotateBatchNew`
   with a per-delta `FixedRotationKey` cache cleared on `DeleteScheme`.
   Suite went from 197s → 165s (~17%) just from this one binding flip.

2. **MOAI lazy-relin needs per-block relin, not just a terminal one.**
   The inner `final_shift` rotation requires degree-1 input. Each
   (alpha, r) block accumulates degree-2 products, then relinearizes
   once before the rotation. THOR is different — it can defer all relins
   until the outer assembly because there's no rotation between the
   accumulator builds and the final write. `kernels/moai_cipher.py`
   and `kernels/thor_cipher.py` docstrings spell this out.

3. **D8 ships as patches, not a fork.**
   `matmul-encoding-material/` is a sibling clone gitignored from
   orion (it's Negar's separate repo). Direct edits there can't be
   tracked, so we keep the diffs (`csv_emit.go`, `memsnap.go`, the
   `matmult_csv_emit.patch`) inside our repo under `cpu_baseline/`,
   and `run.sh` re-applies them to a fresh matmul-encoding-material/
   clone on demand.

## Background sweep — DONE (2026-05-13 00:34 → 07:03, 6h30m)

The CPU-baseline sweep that the prior Claude launched under
`tmux new-session -d -s cpu-baseline-d8` ran to completion. Final
log line:

    === cpu-baseline-d8 sweep ended Wed May 13 07:03:54 AM EDT 2026 (elapsed 23387s) ===
    === final CSV: 12 rows ===

The tmux session is empty / sweep process has exited. To clean up:

```bash
tmux kill-session -t cpu-baseline-d8 2>/dev/null
```

(Optional — leaving it idle costs nothing.)

### Results (single trial, no warmup, all max_err < 1e-5 except thor + bmm1)

CSV at `benchmarks/matmul_encodings/results/cpu_baseline_paper.csv`
(gitignored — regenerate via `cpu_baseline/run.sh`):

| kernel    | shape                                | mean (s)    | max_err  | notes |
|-----------|--------------------------------------|------------:|----------|-------|
| rowenc    | n=4                                  | 0.16        | 1.5e-08  |       |
| rowenc    | n=8                                  | 0.39        | 3.2e-08  |       |
| rowenc    | n=16                                 | 1.00        | 7.2e-08  |       |
| rowenc    | n=32                                 | 2.29        | 1.2e-07  |       |
| bmm3      | (128,131,129)                        | 8.35        | 4.5e-08  |       |
| bmm3      | (256,259,257)                        | 47.24       | 6.2e-08  |       |
| bmm3      | (512,515,513)                        | 303.65      | 1.1e-07  |       |
| bmm3      | (1024,1027,1025)                     | 1896.83     | 9.9e-08  |       |
| **thor**  | d=2048,n=2048,H=1,s=4096,c=2         | **17610.44**| 6.6e-06  | 4h53m — dominates everything |
| moai_alg3 | batch=2,m=2048,d'=2048,bsgs=true     | 2604.04     | 3.1e-08  |       |
| bmm1      | (516,540,528)/blk(43,45,44)          | 605.34      | 2.2e-06  |       |

**Headline observations the paper can lean on:**

- THOR is **6-7x slower than MOAI** at the same size (17610s vs 2604s
  at d=n=2048, H=1). On CPU that's the lots-of-mask-PMults profile
  biting hard. The "GPU ranking inverts" thesis predicts THOR closes
  the gap (or wins) once rotations become bandwidth-bound on GPU.
- BMM-III scales ~5-6x per shape doubling (8 → 47 → 304 → 1897s).
  Clean exponential — fits a regression for extrapolation if the
  paper wants to project to shapes too big to actually run.
- Errors are tight everywhere except THOR (6.6e-06) and BMM-I
  (2.2e-06). THOR's depth-2 BSGS structure accumulates more noise
  than BMM-III's (which sits at ~1e-07).

### What's NOT in the CSV (read before D9)

- **No `moai_alg4` row.** Negar's `MoaiCiphertextSuite` only runs
  `colCol=true` (Alg 3). Adding `RunMoaiHE(..., colCol: false, ...)`
  in `moai_runner.go` and re-running option 4 would close that gap;
  budget ~30 min runtime at size=2048.
- **No multi-trial stdev.** Sweep was `-trials 1` to keep the
  wall-clock under ~hour-scale per shape. For paper-grade error bars,
  re-run with `MATMULT_TRIALS=3` (multiplies wall-clock by ~3 — at
  6h30m baseline, that's ~20h on this box).
- **No `n_he=8192` runs.** Negar's Go uses LogN=13 Standard → 4096
  slots; our Python harness uses LogN=13 ConjugateInvariant → 8192
  slots. The slot-count alignment decision (PLAN.md §6) is still open.

If the sweep had failed mid-way, the recipe is the same per-suite:
each menu option is independent, so re-launching just the failed
suite via `echo -e "<opt>\n0" | /tmp/matmult_runner -trials 1 -verify`
with `BENCH_CSV_OUT` set in append mode gets you back on track.

## What's left — sorted

**Required for the paper deadline (2026-05-21):**

- **D9 — Headline figure** (~0.5 day). matplotlib bar chart, 2×N
  (CPU/GPU per encoding), log-scale Y, annotated with rotation counts.
  Single Python script under `benchmarks/matmul_encodings/`. Consume
  both the GPU sweep CSV (from `runners/__main__.py`) and the CPU
  sweep CSV (from `cpu_baseline/run.sh`); they share the same schema.
  CPU CSV is now ready — see "Background sweep" above.
- **GPU-enabled desilofhe build** — blocking for actual GPU numbers.
  The `myenv2` wheel is CPU-only; `--device gpu` aborts with
  `RuntimeError: Not supported mode`. Everything else GPU-side is
  ready: every kernel is implemented against the desilo backend,
  every kernel runs hoisting + lazy-relin on desilo, the Python
  harness has `--device gpu` wired through, and the harness samples
  HBM via nvidia-smi when device=gpu. The ONLY missing piece is the
  CUDA-built desilofhe wheel/binary. Akshath needs to install or
  build it (or swap to a different env that already has it).
- **Slot-count alignment decision** (~0.1 day). Negar's Go uses LogN=13
  Standard → 4096 slots; our Python harness uses LogN=13
  ConjugateInvariant → 8192 slots. Either rerun Negar's Go at
  matching slot count (edit `init_lattigo.go::DefaultParams`) or
  document the ~2× per-rotation cost difference in the methodology.
  Decide once GPU numbers are in (the figure makes the choice
  obvious one way or the other).
- **Final regression check** (~0.1 day).
  `/example-test-mm-encodings` on LoLA / MLP / ResNet vs
  `examples/results/` transcripts. Catches slow regressions in the
  inference paths from binding/kernel work.

### Decision the prior Claude left for the next instance

After the sweep finished, Akshath was given three options for what to
do next (no answer recorded yet — ask him on reconnect):

  - **(a)** Move on to D9 (figure script) using just CPU numbers + a
    "GPU TBD" placeholder. Unblocks the figure layout work; can drop
    GPU bars in once the desilofhe-CUDA install lands.
  - **(b)** First investigate the desilofhe-CUDA install path
    (`pip search`, the desilofhe upstream repo's install docs, see
    if there's a separate `desilofhe-gpu` package, etc.). Highest
    paper value but uncertain how long it takes — could be 5 minutes
    or several hours depending on what's available.
  - **(c)** Add MOAI Algorithm 4 to Negar's Go runner so we get the
    full Q·Kᵀ·V chain numbers on CPU (~30 min runtime at size=2048)
    before D9. Closes the only gap in the CPU CSV, useful only if
    the figure / paper plans to show MOAI's chained-matmul story
    explicitly.

Recommendation: probably **(b) → (a)** in that order. Without GPU
numbers the figure has half its data; the install is on the critical
path and shouldn't be deferred. (c) is a small follow-up that can
slot in after.

**Open follow-ups (paper-irrelevant unless flagged):**

- Tighten oracle test `atol` 1e-1 → 5e-2 (observed errors ~1e-3).
- File desilofhe `rotate_batch` overload-(1) segfault upstream (we
  work around via overload (2) but a fix would let us drop the per-delta
  key cache).
- Add `RotateBatchNew` + `MulNoRelinCiphertextNew` + `RelinearizeNew`
  to the lattigo Go binding so `Context.rot_batch` doesn't degrade to a
  Python loop and `Context.mul_nr` doesn't fall back to eager. Only
  matters if we ever benchmark lattigo through Python — D8 uses
  Negar's Go binary directly.

## Key facts that aren't obvious from reading the code

- **Aggregate test suite is 86 tests in ~165s** on this box (post-hoisting).
  The conftest `gc.collect()` between modules is load-bearing — without
  it, the lattigo binding's plaintext heap balloons across modules and
  Go aborts in test_moai's first item.
- **All 5 kernels (BMM-I, BMM-III, RowEnc, THOR, MOAI Alg 3 + Alg 4)
  use lazy-relin and hoisting on desilo.** On lattigo they degrade to
  eager / un-hoisted via the binding fallback, which is documented and
  paper-irrelevant.
- **`ctx.slots` is 8192** in our test config (LogN=13 ConjugateInvariant
  in this Orion build) — not 4096 as you might expect from the LogN
  alone. Both backends report 8192 with our config, so the per-shape
  arithmetic is consistent across backends.
- **The CSV that D8 produces lives under
  `benchmarks/matmul_encodings/results/`** which is gitignored. The
  CSV itself isn't tracked; the script that produces it is.

## Quick reference

- **Python env**: `/home/avm6288/miniconda3/envs/myenv2/bin/python`.
- **Run all matmul-encoding tests** (~165s):
  ```bash
  /home/avm6288/miniconda3/envs/myenv2/bin/python -m pytest \
      tests/oracle/matmul_encodings/ benchmarks/matmul_encodings/plaintext/ -v
  ```
- **Run the Python harness** (smoke):
  ```bash
  python -m benchmarks.matmul_encodings.runners \
      --backend lattigo --preset smoke --n-trials 1 --warmup 0 --output /tmp/smoke.csv
  ```
- **Re-run the CPU baseline** (full sweep):
  ```bash
  ORION_ROOT=$HOME/orion ./benchmarks/matmul_encodings/cpu_baseline/run.sh
  ```
- **Git identity for new commits**: Akshath Mahajan
  `<akshathmahajan13@gmail.com>` — set via env vars rather than
  `git config`:
  ```bash
  GIT_AUTHOR_NAME="Akshath Mahajan" GIT_AUTHOR_EMAIL="akshathmahajan13@gmail.com" \
  GIT_COMMITTER_NAME="Akshath Mahajan" GIT_COMMITTER_EMAIL="akshathmahajan13@gmail.com" \
  git commit -m "..."
  ```
- **Branching pattern in use this session**: short-lived
  `feat/matmul-encoding-<topic>` sub-branches off `feat/matmul-encoding`,
  merged back via `--no-ff` so the merge commits preserve scope. Sub-
  branches are kept locally until clearly safe to delete.
