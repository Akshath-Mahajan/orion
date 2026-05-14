# Session 2 handoff — matmul-encoding paper

> Written 2026-05-13 (evening) by Akshath + Claude. Follow-up to
> [`HANDOFF.md`](HANDOFF.md). The earlier doc covers the CPU baseline
> sweep (D8 results) and the GPU plumbing bring-up. This one covers
> what happened *after* the GPU wheel was confirmed working: the THOR
> memory rewrite, the apples-to-apples d=256 measurement vs Negar's
> Go, the desilo-GPU "paper preset" sweep, and what still blocks
> d=2048 numbers.

## TL;DR

- **Three sub-branches merged into `feat/matmul-encoding`** this session: GPU mode opt-in for tests ([54edb1e](.)), GPU validation writeup ([731b56e](.)), and `ctx.free()` primitive activation ([f552251](.)).
- **One sub-branch sitting unmerged**: `feat/matmul-encoding-thor-memory` at [8288960](.) — the THOR kernel rewrite (eager-free + fused Phase 1+2). Tests pass on both backends; not merged because we haven't decided whether to apply the same treatment to MOAI/BMM before landing.
- **THOR scales on GPU to d=256** (~3.7 min kernel). Direct CPU↔GPU comparison at matching shape `d=256, c=16`: Negar Go CPU 43.0s vs our Python GPU 35.0s → **1.23× GPU speedup** — much less than the 15× we saw on individual rotations in the micro-bench, because THOR is `ct×pt + rescale` dominated, not rotation-dominated.
- **Paper-preset GPU CSV at [results/desilo_gpu_paper.csv](results/desilo_gpu_paper.csv)** mirrors Negar's [results/cpu_baseline_paper.csv](results/cpu_baseline_paper.csv) schema exactly, but is **6 of 12 rows** because the other 5 kernels haven't received the THOR-style memory cleanup yet, and THOR/MOAI at d=2048 still hits a separate FixedRotationKey-cache wall.
- **Two distinct memory bottlenecks** identified, listed in §3.

## 1. What landed (commits, in branch-graph order)

| Merge | Sub-branch | Adds |
|---|---|---|
| `54edb1e` | `feat/matmul-encoding-gpu-plumbing` | one-line `ORION_DESILO_DEVICE` env-var toggle in oracle conftest. Engine plumbing was already there. |
| `731b56e` | `docs/matmul-encoding-gpu-validation-report` | [`GPU_VALIDATION.md`](GPU_VALIDATION.md): wheel install steps, binary-level evidence for the optimization codepaths, oracle results, micro-bench tables, **honest caveats** on bias of measurements. |
| `f552251` | `feat/matmul-encoding-ctx-free` | activates the `ctx.free()` hook in [`context.py`](context.py) so kernels can evict ciphertexts from the desilo binding's `_objects` dict. Lattigo path intentionally no-op (known GC-termination segfault, see [`orion/backend/python/tensors.py:103`](../../orion/backend/python/tensors.py#L103)). |
| **`8288960`** (unmerged) | `feat/matmul-encoding-thor-memory` | THOR kernel rewrite, two layered changes: eager free of pre-rescale / consumed-rotation intermediates, and **fused Phase 1 + Phase 2** per input row. Same algorithm, same op count, same numerics. Matches the "MEMORY OPTIMIZED VERSION" Negar herself wrote and commented out at the bottom of `matmul-encoding-material/MatMult/matmult/thor_cipher.go`. Tests pass on lattigo CPU, desilo CPU, desilo GPU. |

`git log --oneline --graph` from HEAD shows all four cleanly.

## 2. What we measured

### 2.1 Apples-to-apples THOR at d=256 (one trial, no warmup)

CPU number is from a re-run of Negar's Go runner at `size=256` (edited `thor_runner.go:45` then reverted). GPU is `/tmp/thor_gpu_timing.py 256 256 1 16 1`.

| Side | Shape | Kernel time | max_err | hardware |
|---|---|---:|---:|---|
| Negar Go, CPU (Lattigo) | `d=n=256, H=1, c=16` | **43.02s** | 1.92e-06 | this box's CPUs |
| Our Python GPU (desilofhe-cu129) | `d=n=256, H=1, c=16` | **35.00s** | 6.07e-05 | RTX 3090 (1× of 2) |
| **Speedup** | — | **1.23×** | — | — |

The earlier 219.5s GPU number I quoted at d=256 used `c=2` — 8× more outer iterations (`m_c = d/c`). Not directly comparable to Negar's c=16; my mistake to use it as the headline number. The c=16 measurement is the right one.

### 2.2 desilo-GPU paper preset sweep ([results/desilo_gpu_paper.csv](results/desilo_gpu_paper.csv))

Same schema as Negar's `cpu_baseline_paper.csv`. Ran each kernel in a fresh Python process so per-kernel GPU state starts clean (the harness reuses one Context, but only THOR has `ctx.free()` calls, so non-THOR kernels poison the GPU for everything after them — see §3).

| kernel | shape | CPU (Negar) | GPU (ours) | GPU vs CPU |
|---|---|---:|---:|---|
| rowenc | n=4 | 0.16s | 0.17s | 0.94× |
| rowenc | n=8 | 0.39s | 0.37s | 1.05× |
| rowenc | n=16 | 1.00s | 0.77s | 1.30× |
| rowenc | n=32 | 2.29s | 1.59s | **1.44×** |
| bmm3 | (128,131,129) | 8.35s | **12.43s** | 0.67× *(CPU faster)* |
| bmm3 | (256,259,257) | 47.24s | **OOM** | — |
| bmm3 | (512,515,513) | 303.65s | **OOM** | — |
| bmm3 | (1024,1027,1025) | 1896.83s | **OOM** | — |
| bmm1 (per-block) | block(43,45,44) | ~350ms | 742ms | 0.47× *(CPU faster)* |
| thor | d=2048 | 17610s | **OOM** | — |
| moai_alg3 | d=2048 | 2604s | **OOM** | — |

**Important caveat on bmm1**: the bench function times a **single block call** (`s_n × s_m × s_p`), not the whole `(N, M, P)` matmul. Negar's 605s for `(516,540,528)/blk(43,45,44)` is 12³ = 1728 blocks. Per-block CPU ≈ 350ms; per-block GPU 742ms → GPU is **~2.1× slower per block**. The CSV's raw 0.742s row is per-block; downstream comparison code must multiply by block count.

### 2.3 Micro-bench (already in [`GPU_VALIDATION.md`](GPU_VALIDATION.md))

Headline at N=16 batch size:
- Hoisted rotation: GPU 47.7 ms vs CPU 744 ms → **15.6× on GPU** for individual rotations
- Lazy ct×ct multiply: GPU 70 ms vs CPU 607 ms → **8.6× on GPU**
- Eager ct×ct + relin: GPU 103 ms vs CPU 1696 ms → **16.5× on GPU**

The **gap between micro-bench (15×) and end-to-end (1.23× best, often <1×)** is the *implementation tax*: ct×pt isn't measured but evidently doesn't enjoy the same GPU acceleration as rotation/mul; Python ↔ C call overhead per op; our per-op `ctx.free()` bookkeeping in THOR; small-shape launch overhead. **ct×pt is unmeasured** — running that micro-bench would quantify the gap.

## 3. Two distinct memory bottlenecks on GPU

These are independent, fixed by different changes, and both currently in the way of d=2048 paper numbers.

### 3.1 In-flight ciphertext leak (per-kernel)

**Symptom**: kernel OOMs at modest d (~64 to ~256) even though the GPU is otherwise idle.
**Cause**: every operation creates a new ciphertext ID in the desilo binding's `_objects` dict; without explicit `ctx.free()` the dict pins underlying GPU memory until `DeleteScheme`. On desilo, `RescaleNew` is a `clone`, so the leak doubles with each rescale.
**Fix** (already applied to THOR via `feat/matmul-encoding-thor-memory`):
1. `ctx.free()` calls after every consumed temp (Change A).
2. Fuse build/consume per input row so peak working set drops from `O(m_c · n)` to `O(m_c + n)` (Change B).

**Status**: THOR done. MOAI Alg 3 / MOAI Alg 4 / BMM-1 / BMM-3 / RowEnc all still untouched. Pattern from THOR transfers directly; ~half a day each, mostly mechanical.

### 3.2 Persistent FixedRotationKey cache (per-shape, not per-kernel)

**Symptom**: THOR (and likely MOAI) OOMs at large `n` (e.g. n=2048) during the very first `rot_batch` call, before any kernel work happens.
**Cause**: the desilo binding caches one `FixedRotationKey` per distinct rotation delta (in `_fixed_rot_keys`) so keygen is paid once per scheme rather than per call. THOR at n=2048 has 2047 distinct deltas. Each key is ~5 MB on GPU (LogN=13 ConjugateInvariant, L+K=6 primes, hybrid decomp factor 2). 2047 × ~5 MB ≈ **10 GB of persistent keys**, plus engine workspace + the actual ciphertext working set + transient workspace inside `create_fixed_rotation_key` itself ≈ 5-10 GB more. Tips past 24 GB during the keygen loop of the *first* `rot_batch`. Change A+B can't help — the leak is rotation keys, not intermediate ciphertexts.

**Why chunked rotation-key creation alone wouldn't help**: chunking just defers the OOM. THOR's access pattern hits every one of the 2047 deltas per input row, so an LRU cache with any limit < 2047 has effectively 0% hit rate after the first j iteration. Without cache, every j re-generates all 2047 keys (~10ms each × 2047 keys × 1024 j-iterations ≈ 5+ hours of pure keygen). Caching is required for performance; the 10 GB cost is structural at this parameter set.

**Real options** (from cheapest to most invasive):

1. **Smaller LogN** (likely sufficient). Drop the test config to `LogN=12` (ConjugateInvariant → 4096 slots). Polynomial halves, FixedRotationKey halves to ~2.5 MB. 2047 × 2.5 MB ≈ **5 GB** of persistent keys — fits in 24 GB with headroom. Caveats: still need `s_enc = c·n·H ≤ slot_count`; for THOR at d=n=2048, H=1, c=2 we need s=4096 slots which **exactly** equals LogN=12 ConjugateInvariant. Need to also confirm the mult depth still fits (LogQ=[29,26,26,26,26] gives 4 levels — should still work but verify oracle tests pass at LogN=12).

2. **Drop hoisting at large n**. Call `ctx.rot()` in a loop with the single shared `RotationKey` instead of `rot_batch` with FixedRotationKeys. Eliminates the 10 GB cache entirely. Loses the hoisting speedup (~2.9× on GPU at N=16 from the micro-bench). Paper-honest only if reported with a note. Easiest to implement (5-10 lines in the kernel).

3. **Hybrid working-set cache + lazy regen**. Cache only the K most-recent deltas (say K=512, ~2.5 GB). For deltas not in cache, fall back to single `rot()` with the shared `RotationKey`. THOR's pattern means most deltas miss; effectively this is option (2) with extra complexity. Not recommended unless we can find a kernel pattern that has high cache reuse.

4. **Bigger GPU** (out-of-band). An H100 80 GB or A100 80 GB fits the full 10 GB cache trivially. If lab has one available, that's the simplest unblocker.

5. **Different ring**. LogN=13 *Standard* gives 4096 slots like Negar's Go config (not ConjugateInvariant's 8192). Keys would be effectively half-size again (different ring structure but similar bytes). Could be tried as a third axis if (1) isn't enough.

**Recommended path**: try (1) first — it's a one-line config change, oracle tests catch any depth regression immediately, and it should unblock d=2048 without changing the kernel or losing hoisting. If oracle tests fail at LogN=12, fall back to (2) — accept the no-hoisting result for the d=2048 row and document it. Only consider (4) if neither works.

## 4. What's left for paper deadline (2026-05-21, 8 days out)

In priority order:

1. **Fix the FixedRotationKey cache for THOR d=2048** (§3.2 above). Try LogN=12 first (~30 min including oracle re-run); if that works, the rest of THOR's d=2048 GPU number is one more long background run.
2. **Apply THOR-style memory rewrite to MOAI Alg 3** (§3.1). MOAI is the second headline kernel. Same pattern, similar shape constraints.
3. **Fill remaining paper preset rows**: MOAI Alg 4, BMM-3 at 256/512/1024, BMM-1 (proper whole-shape timing, not per-block). Each needs the per-kernel memory cleanup.
4. **ct×pt micro-bench** to quantify the implementation-tax gap. 15 minutes of work; tells the paper *why* end-to-end speedup is so much smaller than primitive speedup.
5. **D9 headline figure** (matplotlib) once we have enough rows in both CSVs.
6. **Cross-check at least one GPU result against Negar's Go reference** before freezing the paper figure.

## 5. Quick-reference paths

- Python env: `/home/avm6288/miniconda3/envs/myenv2/bin/python` (GPU wheel `desilofhe-cu129==1.11.2` installed).
- THOR-only timing script: `/tmp/thor_gpu_timing.py d n H c trials` (kernel-only + total split).
- Micro-bench: `/tmp/desilo_microbench.py {cpu|gpu}` (hoisted rotation + lazy-relin tables).
- Oracle tests (CPU): `python -m pytest tests/oracle/matmul_encodings/ -v`.
- Oracle tests (GPU, desilo only): `ORION_DESILO_DEVICE=gpu python -m pytest tests/oracle/matmul_encodings/ -v`.
- Full paper-preset sweep: `python -m benchmarks.matmul_encodings.runners --backend desilo --device gpu --preset paper --output ...`.
- Per-kernel sweep (avoids cross-kernel poisoning while §3.1 is unfixed): see the shell loop in this conversation's `desilo_gpu_paper_per_kernel.log` for the recipe.
- Negar Go CPU baseline regen: `ORION_ROOT=$HOME/orion ./benchmarks/matmul_encodings/cpu_baseline/run.sh`.
- Git identity for commits: `Akshath Mahajan <akshathmahajan13@gmail.com>` via `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL` env vars.

## 6. Decisions to revisit

- The unmerged `feat/matmul-encoding-thor-memory` branch (Change B's fused-phase rewrite). It's correct, tests pass on both backends, mirrors Negar's commented-out reference structure. Hold-up was deciding whether to land it alone or as part of a broader "memory-cleanup pass across all kernels" series. With the deadline 8 days out and §3.1 needing to be done anyway for MOAI/BMM/RowEnc, the cleanest move is probably: land THOR memory now, then a follow-up branch per remaining kernel.
- Whether to switch the test config to LogN=12 globally, or keep LogN=13 ConjugateInvariant and only override for the d=2048 GPU run. Affects oracle test reproducibility and CPU comparison fairness.
- For the paper, whether the headline number is **THOR at d=2048** (apples-to-apples with Negar's CPU baseline, requires solving §3.2) or **THOR at d=256 / d=512** (already runs after Change A+B, smaller but honest). Both have merit; the second sets a lower bar but is bulletproof.
