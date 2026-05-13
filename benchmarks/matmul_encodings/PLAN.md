# Matmul-encoding paper — handoff plan

> Living plan for the IISWC 2026 characterization paper. Read this if you are
> a fresh instance picking up the `feat/matmul-encoding` branch. Last updated:
> 2026-05-12 (D6 BMM-III landed; D7/D8/D9 remain).

## 1. Context

**Paper.** A characterization paper for **IISWC 2026** comparing three FHE (CKKS)
matrix-multiplication encodings on GPU:

- **THOR** (Moon et al., CCS 2025) — interleaved diagonal packing + BSGS.
- **MOAI** — column × diagonal hybrid, BSGS, natural fit for back-to-back matmuls
  (Q·Kᵀ then ·V in transformer attention).
- **Bicycle / BMM-I / BMM-III** (Zheng et al., IEEE TIFS 2024) — single-ciphertext
  CRT packing with the "transpose is free" property. BMM-I handles matrices that
  fit in one ciphertext; BMM-III extends to large matrices via a multi-chunk
  `LongRot` primitive.

**Deadlines.**
- Abstract: **2026-05-14** (2 days from this writing). Need framing claim + Negar's
  existing CPU numbers — no new measurement required.
- Paper: **2026-05-21** (9 days). Need GPU numbers, CPU baseline rerun on the
  same machine, headline figure.

**People.**
- **Negar** — lead author. Owns the CPU Lattigo implementations (the
  `matmul-encoding-material/MatMult/` reference) and the algorithmic
  derivations (notably THOR mat-mat which is not in the published paper).
- **Akshath (avm6288@nyu.edu)** — me / the user. GPU port, correctness
  verification, benchmarks. Just shipped the DeSiLo backend on main (commit
  `1253a58`).
- **Brandon** — stakeholder; wants everything eventually living in Orion.
- **Austin** — adjacent context.

**Context repo.** Read-only sibling clone at
`/home/avm6288/orion/matmul-encoding-material/` (gitignored from orion).
Contains:
- `CLAUDE.md` and `notes/project_overview.md` — primer + extended writeup.
- `material/` — PDFs of all three encoding papers + CiFlow + Negar's
  encoding doc.
- `MatMult/matmult/` — Negar's Lattigo Go reference implementation (~5.3k LoC
  across 18 files), the source of every algorithmic port in this branch.

## 2. Goal

GPU implementations of all three encodings on **Orion's DeSiLo backend**, with:
1. **Correctness** verified against Negar's plaintext reference and against
   `numpy.matmul`, plus a **cross-backend differential** (lattigo vs desilo
   give bit-equivalent decrypts within CKKS noise).
2. **Runtime / memory / rotation-count / mult-count** numbers across a shape
   sweep, on the same machine that runs Negar's Go for a fair CPU baseline.
3. A **headline figure**: 2×N CPU/GPU bars per encoding, annotated with
   rotation counts. Expected story: GPU ranking **inverts** vs CPU because
   rotation cost shifts from NTT-compute-bound (CPU) to HBM-bandwidth-bound
   (GPU eval key reads).

## 3. Key decisions made

These are settled. Don't rehash without a strong reason.

| # | Decision | Why |
|---|---|---|
| **1** | **Standalone harness under `benchmarks/matmul_encodings/`**, NOT integrated into Orion's `nn.operations`. | Orion's `Linear` op assumes a fixed encoding (row-pack-ish); the paper is precisely about comparing different encodings. Fighting `nn.operations` for three encoding variants is expensive. Negar's `*_runner.go` is already the right shape — mirror it. |
| **2** | **Both BMM-I and BMM-III** in scope. | BMM-I for small-shape benchmarks; BMM-III for large shapes where bicycle's bandwidth-bound profile actually shows. The "GPU ranking inverts" story needs the large shapes. |
| **3** | **Abstract MVP = framing claim + Negar's existing CPU numbers**, no new measurement needed by 2026-05-14. | Lowest deadline risk. Real GPU vs CPU bar lands in the paper, not the abstract. |
| **4** | **Two-layer testing:** (a) oracle tests under `tests/oracle/matmul_encodings/` with backend-parameterized `Context` fixture, and (b) example regression: `examples/run_{lola,mlp,resnet}.py` diffed against `examples/results/` after every binding change. | Mirrors the pattern that validated the DeSiLo backend. Lattigo serves as the CKKS oracle; desilo is the GPU target. Two skills land this: `/oracle-test-mm-encodings` and `/example-test-mm-encodings`. |
| **5** | **Eager relinearization (`mul_rl`) in all kernels for the first pass.** Lazy relin is a separate optimization commit. | Correctness comes first. Negar's lazy variants exist (`ThorCCMatMulHELazyRelin`, `bmm1Accumulate` deferring relin) — we can swap `mul_rl` → `mul_nr` + terminal `relin` per kernel once correctness is locked. The binding verbs already exist. |
| **6** | **`engine.rotate_batch` segfault workaround = `all_safe = False` gate; fall back to individual rotates.** | Negative deltas crash desilofhe v1.11.2; some large positive deltas crash too. Investigation captured in commit `f3bd7bf`. Hoisting amortization is gated off until upstream stabilizes — flipping one flag re-enables it instantly. |
| **7** | **Tile every encoded vector to fill `slot_count`** before encrypting. Also tile masks. | CKKS rotates over all `slot_count` slots, not the algorithm's logical `n_he` or `s = c·n·H`. Without tiling, the wraparound is into zero padding and rotated values are wrong. `slot_count % n_he == 0` is the only constraint. |

## 4. What is done (the 8 + 1 commits on this branch)

Branch: `feat/matmul-encoding`, off `main`.

| Commit | What | Tests |
|---|---|---|
| `2ccda18` | gitignore `matmul-encoding-material/`; add two testing skills (`oracle-test-mm-encodings`, `example-test-mm-encodings`). | — |
| `2810dee` | desilo: lazy-relin bindings (`MulNoRelinCiphertext{,New}` + `Relinearize{,New}`). | Verified lazy ≡ eager within CKKS noise (~1e-8). Existing oracle tests still pass. |
| `f3bd7bf` | desilo: `RotateBatchNew` API with **safe fallback** to individual rotates. Hoisted path gated off until upstream segfault fixed. | Existing oracle tests pass; rotate_batch failure modes documented in commit body. |
| `30257d6` | Plaintext numpy oracle ports: `bmm1_plain`, `bmm3_plain`, `moai_plain`, `thor_plain`, `op_counts`. Scaffolding for `kernels/` and `runners/`. | 17 pytest tests pass at `atol=1e-10` against `numpy.matmul`. |
| `bae1808` | `benchmarks/matmul_encodings/context.py`: `Context` wrapper around an Orion Scheme with counted CKKS ops (`rot`, `rot_batch`, `add`, `mul_rl`, `mul_nr`, `relin`, `mul_pt`, `rescale`) and an `OpCounts` tally. Includes fallbacks for backends without lazy-relin / batched rotate (so Lattigo works as oracle). | Smoke test: roundtrip 8.9e-16, all ops match expected. |
| `ffaf48e` | **BMM-I kernel** (`bmm1_cipher.py`) — hoisted bicycle. Plus `tests/oracle/matmul_encodings/conftest.py` (backend-parameterized) and `test_bmm1.py`. | **10/10 passing** (5 lattigo + 5 desilo). |
| `2501d25` | **THOR kernel** (`thor_cipher.py`) — Algorithm 2 with mu3 explicit + ones-mask for level alignment (avoids DropLevel which neither binding exposes). `test_thor.py`. | **6/6 passing** (3 shapes × 2 backends). |
| `f87b298` | **MOAI BSGS Col×Col kernel** (`moai_cipher.py`) — Algorithm 3. `test_moai.py`. | **6/6 passing** (3 shapes × 2 backends). |
| `8f12317` | This PLAN.md + commit-author rewrite. | — |
| (D6) | **BMM-III LongRot kernel** (`bmm3_cipher.py`) — cached-mode dispatcher with lazy-relin/lazy-rescale finalize. Plaintext oracle (`bmm3_plain.py` adds `long_rot_plain` + `bmm3_plain` + `bmm3_matmul_plain`). `test_bmm3.py` + 13 plaintext tests. | **8/8 cipher passing** (3 shapes × 2 backends + 2 oracle checks); **17/17 plaintext passing**. |

**Aggregate oracle suite:** 30 tests under `tests/oracle/matmul_encodings/`,
plus 35 plaintext tests under `benchmarks/matmul_encodings/plaintext/`.
**65 total, all green on both backends, ~144s.**

**Non-regression check after binding additions:** LoLA on desilo runs clean —
MAE 0.0000, Precision 22.8241 (reference 22.7654), Runtime 15.3s (reference
51.4s). See commit `bohlcyh67` output (the smoke-test transcript).

## 5. Known gotchas the next instance will hit

These are non-obvious. Internalize before writing more kernels.

1. **`engine.rotate_batch` segfault.** Negative deltas crash. Some non-consecutive
   large positive deltas crash (`[8191, 8187, 8175]` reproducer in
   `f3bd7bf` body). Workaround in `RotateBatchNew` is `all_safe = False` →
   fall back to individual rotates. To repro and report upstream:
   ```python
   import desilofhe, numpy as np
   eng = desilofhe.Engine(max_level=4, mode='cpu')
   sk = eng.create_secret_key(); pk = eng.create_public_key(sk)
   rk = eng.create_rotation_key(sk)
   ct = eng.encrypt(eng.encode(np.zeros(eng.slot_count), level=4), pk)
   eng.rotate_batch(ct, rk, [8191, 8187, 8175])   # segfaults
   eng.rotate_batch(ct, rk, [-1, -5, -17])        # segfaults
   eng.rotate_batch(ct, rk, [1, 2, 3])            # OK
   ```

2. **`s = c·n·H` vs `s = ctx.slots` are NOT the same.** Conflating them was
   the bug that sent THOR to a 3+ absolute error (`2501d25`). The kernel's
   algorithm parameter `c` MUST be passed explicitly; never derive
   `c = ctx.slots // (n*H)`.

3. **Tile encoded vectors to slot_count BEFORE encrypting.** CKKS rotations
   wrap modulo `slot_count`, not modulo the encoded length. `_tile_to_slots`
   helper lives in `kernels/{thor,moai,bmm1}_cipher.py`. BMM-III will need it
   too. Constraint: `ctx.slots % n_he == 0`.

4. **Level-management without `DropLevel`.** Neither binding exposes the
   Lattigo `DropLevel` verb. Workarounds in THOR: compute `v3 = v · mu3`
   explicitly instead of `v - v0 - v1 - v2`; drop `p_cjl[j][0]` via
   `mul_pt(_, ones_at_LM1) + rescale`. Each pays one extra ct·pt per
   alignment site; semantically identical.

5. **Lattigo binding lacks `MulNoRelinCiphertextNew`, `RelinearizeNew`,
   `RotateBatchNew`.** `Context` falls back gracefully: `mul_nr` → `mul_rl`,
   `relin` → no-op, `rot_batch` → list of individual `RotateNew`. Same
   numerical result, just less efficient. Good — Lattigo is the oracle.

6. **`backend.Encode` wants a Python `list`, not a `np.ndarray`.**
   Lattigo's ctypes wrapper auto-expands `list[float]` to `(ptr, len)`;
   `np.ndarray` of float64 falls through and ctypes errors with "takes
   4 args". `Context.encode` does `.tolist()` before passing.

## 6. Plan — what's left

Ordered, with rough estimates. Deadlines: abstract 2026-05-14, paper 2026-05-21.

### D6 — BMM-III LongRot kernel + tests (DONE)
Picked Option A — ported the plaintext oracle (`long_rot_plain` +
`bmm3_plain` + `bmm3_matmul_plain` in `plaintext/bmm3_plain.py`) before
the cipher kernel, validated chunk stitching against
`concat(chunks)[:enc_len] -> rotate -> chunk` reference at 12 LongRot
configs, then ran the cipher kernel against both numpy.matmul and the
plaintext oracle on both backends.

**Implementation note on cached vs naive mode.** The first port used
"naive" (re-encode masks every call) which crashed Lattigo with a Go
panic on shape (8, 9, 11) at n_he=32 — the binding's plaintext table
balloons under m × stop × ~7 mask sites per LongRot. Switched to "cached"
mode (matmult/bmm3_cipher.go::Bmm3ModeCached): one encode per (start,
end) per BMM-III call, reused across iterations and both A/B sides.
Suite stable in ~144s.

**Hoisted mode** (matmult/bmm3_cipher.go::Bmm3ModeHoisted, block-hoisted
rotations on top of the cache) is not ported. Would slot in after D7 if
benchmark numbers show LongRot Step-1 dominates wall-clock — but the
rotate_batch segfault (Decision #6) currently gates hoisting off
anyway, so the cached vs hoisted comparison can't go in the paper until
upstream desilo fixes that.

### D7 — Benchmark harness (~1 day)
Mirror `matmult/main.go`'s runner pattern: per-encoding shape sweep,
collect OpCounts + wall-clock + peak HBM (via `nvidia-smi`), emit CSV.
Live under `benchmarks/matmul_encodings/runners/`. Reuse Negar's
shape tables from `*_runner.go` so the GPU numbers line up with her
CPU numbers shape-for-shape.

### D8 — CPU baseline rerun (~0.5 day)
Build and run `matmul-encoding-material/MatMult/matmult/` on this
machine to produce a fair CPU baseline at the exact same shapes the
GPU runs at. Requires Go installed; should already be — try
`go version`. If not, install via conda.

### D9 — Headline figure (~0.5 day)
matplotlib bar chart, 2×N (CPU/GPU per encoding), annotated with
rotation count, log-scale Y axis. Drop into the paper's figure
section. Single Python script under `benchmarks/matmul_encodings/`.

### Optimization pass (lazy-relin everywhere; ~0.5 day)
Swap `ctx.mul_rl` for `ctx.mul_nr` in each kernel, then `ctx.relin` at
the end. Re-run the oracle tests to confirm bit-equivalence. Re-run
benchmarks to quantify the speedup (this is one of the paper's
data points). Defer if time-constrained.

### Final regression check
`/example-test-mm-encodings` on LoLA/MLP/ResNet against the
`examples/results/` transcripts. Already passed once at `2810dee` for
LoLA; rerun after all kernel work to confirm no slow regressions.

## 7. Suggested next move for the new instance

**With D6 landed**, the path to the May-21 paper is D7 → D8 → D9.

Order recommendation: **D7 (harness) first**. Reasons:
- Harness output is the data feed for both the CPU-baseline rerun (D8)
  and the headline figure (D9). Building it first lets D8/D9 just
  consume CSVs.
- Mirrors `matmult/main.go`'s shape sweep so GPU vs CPU lines up
  shape-for-shape. Negar's tables in `*_runner.go` are the source of
  truth for the sweep grid.
- Roughly 1 day of mostly-mechanical wiring; no algorithmic risk.

Once D7 emits CSVs for all four encodings (BMM-I, BMM-III, THOR, MOAI),
D8 (CPU baseline) and D9 (headline figure) follow naturally and can
overlap.

**Independent of D6:**
1. Tighten `atol` in the oracle tests from `5e-1` down to `5e-2` —
   observed errors are ~1e-3, the loose bound was just being safe
   during development. Documents the actual noise floor.
2. Add a `runners/_common.py` with timing helpers, then port one of
   Negar's runners as a sanity benchmark (BMM-I is simplest).
3. File the `rotate_batch` segfault upstream (desilo issue tracker)
   with the minimal repro from §5 — flips on hoisting amortization
   "for free" if they fix it before the paper deadline.

## 8. Quick reference

**Active branch:** `feat/matmul-encoding` off `main`.
**Python env:** `/home/avm6288/miniconda3/envs/myenv2/bin/python` (has
`desilofhe`, `orion`, `lattigo`).
**Run all matmul-encoding oracle tests:**
```bash
/home/avm6288/miniconda3/envs/myenv2/bin/python -m pytest \
    tests/oracle/matmul_encodings/ benchmarks/matmul_encodings/plaintext/ -v
```
**Run a single encoding's tests on one backend:**
```bash
pytest tests/oracle/matmul_encodings/test_thor.py -v -k "lattigo"
```
**Smoke-test LoLA on desilo (~15s):**
```bash
cd examples && python run_lola.py ../configs/lola_desilo.yml
```

**Memory entries that should be consulted:**
- `user_role.md` (Akshath, NYU)
- `project_matmul_encoding_paper.md`
- `project_desilo_optimization_gaps.md`
- `project_matmul_open_questions.md`
- `project_matmul_testing_strategy.md`
- `reference_matmul_encoding_context_repo.md`

**Negar's Go reference files mapped to our Python ports:**

| Go (matmul-encoding-material/MatMult/matmult/) | Python |
|---|---|
| `bmm1_plain.go` | `benchmarks/matmul_encodings/plaintext/bmm1_plain.py` |
| `bmm1_cipher.go` | `benchmarks/matmul_encodings/kernels/bmm1_cipher.py` |
| `thor_plain.go` + `thor_runner.go::thorCCMatMulPlain` | `benchmarks/matmul_encodings/plaintext/thor_plain.py` |
| `thor_cipher.go` (`ThorCCMatMulHE`) | `benchmarks/matmul_encodings/kernels/thor_cipher.py` |
| `moai_plain.go` | `benchmarks/matmul_encodings/plaintext/moai_plain.py` |
| `moai_cipher.go` | `benchmarks/matmul_encodings/kernels/moai_cipher.py` (Alg 3 only) |
| `bmm3_plain.go` (helpers only) | `benchmarks/matmul_encodings/plaintext/bmm3_plain.py` |
| `bmm3_cipher.go` (cached mode) | `benchmarks/matmul_encodings/kernels/bmm3_cipher.py` |
| `init_lattigo.go` | `benchmarks/matmul_encodings/context.py` |
| `util.go::OpCounts` | `benchmarks/matmul_encodings/plaintext/op_counts.py` |
