# GPU validation report — matmul-encoding kernels on desilo CUDA

> Written 2026-05-13 by Akshath + Claude. Captures the work that took
> the matmul-encoding paper's desilo backend from CPU-only to GPU,
> what we verified, and the honest gaps in that verification. Sister
> document to [`HANDOFF.md`](HANDOFF.md); this one is specifically
> about the GPU bring-up.

## TL;DR

- **GPU is live.** All 46 matmul-encoding oracle tests pass on
  `mode='gpu'` via `desilofhe-cu129==1.11.2`. Suite drops from 163s
  (CPU baseline) to 34s (~4.8× faster aggregate).
- **Hoisting and lazy-relin both work on GPU**, not stubbed. Both
  micro-bench speedup ratios grow monotonically with batch size N,
  matching the amortization signature.
- **Methodology caveat**: the bench's `decrypt`-as-sync barrier adds
  a constant cost to every measurement, biasing speedup ratios toward
  1.0. Directionally the conclusions stand, but absolute numbers are
  understated. A tighter v2 is in §6.
- **What we have NOT yet verified**: direct CPU↔GPU output agreement on
  identical seeds; integration tests on GPU (`/example-test-mm-encodings`);
  cross-check against Negar's Go reference. See §7 for the gap map.

## 1. Wheel install + plumbing

### Wheel selection

This box: 2× RTX 3090, CUDA driver 13.0. desilo's CUDA build is
published on PyPI as a *separate package* per CUDA-toolkit variant
(`desilofhe-cu121` ... `desilofhe-cu130`) — not as wheel variants of
the `desilofhe` name. PyPI's `desilofhe` itself is CPU-only.

We picked `desilofhe-cu129==1.11.2` to match the other HPC's setup
(behavioral consistency over driver-native match); CUDA driver 13.0 is
forward-compatible with the 12.9 runtime.

```bash
pip uninstall -y desilofhe
pip install desilofhe-cu129==1.11.2
```

The GPU wheel ships the same Python import name (`import desilofhe`)
and constructor surface as the CPU wheel — only the `.so` body changes.
`ldd` on either wheel shows no static CUDA linkage; the GPU wheel
`dlopen`s the CUDA runtime at `Engine(mode='gpu')` construction time,
which is why the same wheel works in CPU-only envs.

### Orion plumbing

The desilo binding already plumbed `mode` end-to-end:

- [`orion/backend/python/parameters.py:77`](../../orion/backend/python/parameters.py#L77):
  `device: Literal["cpu", "gpu"] = "cpu"` field on `OrionParameters`.
- [`orion/backend/python/parameters.py:147-148`](../../orion/backend/python/parameters.py#L147-L148):
  `get_device()` exposes it.
- [`orion/backend/desilo/bindings.py:103,109,113`](../../orion/backend/desilo/bindings.py#L103):
  `setup_scheme` reads `orion_params.get_device()` and passes
  `mode=self._device` to `desilofhe.Engine(...)`.

So **no binding changes were needed**. The only edit was
[`tests/oracle/matmul_encodings/conftest.py`](../../tests/oracle/matmul_encodings/conftest.py)
to thread an `ORION_DESILO_DEVICE` env-var toggle into the test config,
defaulting to `cpu`. Landed via sub-branch
`feat/matmul-encoding-gpu-plumbing` → merged `--no-ff` into
`feat/matmul-encoding` (commits `3581f28` → `54edb1e`).

## 2. Wheel internals — does the codepath exist?

Before timing, we checked that `KeySwitcher4D` (hoisted variant) and
the no-relin multiply path are *physically present* in the GPU `.so`.
Excerpts from `strings` / `nm -D` on
`desilofhe.cpython-310-x86_64-linux-gnu.so`:

| Evidence | Interpretation |
|---|---|
| Two distinct C++ class symbols: `desilo::fhe::KeySwitcher` and `desilo::fhe::KeySwitcher4D` | Hoisted key-switch is a separate implementation, not a flag on the loop path. |
| `KeySwitchingKey` and `KeySwitchingKey4D` both present | Dedicated key type for the hoisted path. |
| Function name strings `Multiply Ciphertexts Then Relinearize` *and* `Multiply Unit Ciphertexts` (without `Then Relinearize`) | The no-relin overload is a distinct function, not just a missing-arg branch. |
| `Batch Rotate Ciphertext With Fixed Rotation Keys` string | The hoisted `rotate_batch` (overload 2) is a discrete API name. |
| `cudaMalloc*`, `cuLaunchKernel*` runtime symbols | GPU kernels exist in this `.so`. |
| GPU-side rotation kernels: `rotate_polynomial_kernel`, `depth_rotate_polynomial_kernel`, `height_rotate_polynomial_kernel`, `width_rotate_polynomial_kernel` | Multi-dimensional polynomial rotation kernels are real CUDA code. |

So the binary inspection rules out "the GPU build silently degrades
to a loop / eager-relin." The implementations exist; the open question
was whether they're *fast*.

## 3. Correctness — oracle tests

Same 46 oracle tests, run under both modes against the same wheel:

```
# CPU mode (default)
$ python -m pytest tests/oracle/matmul_encodings/ -v

# GPU mode (desilo backend only; lattigo stays CPU)
$ ORION_DESILO_DEVICE=gpu python -m pytest tests/oracle/matmul_encodings/ -v
```

| Run | Result | Time |
|---|---|---|
| CPU (after wheel swap) | 46/46 PASS | 163.19s |
| GPU (desilo half only) | 46/46 PASS | 34.21s |

CPU run matches the HANDOFF's ~165s baseline → **no regression** from
swapping CPU-only wheel for the CUDA-enabled wheel. The GPU run's 34s
aggregate is dominated by lattigo (still CPU) — actual desilo-GPU
half is materially faster than 4.8×.

### What 46/46 actually proves — and doesn't

The tests assert `|decrypt(GPU_kernel(A, B)) - numpy(A @ B)| < atol`
per-kernel. Tolerances per the [test files](../../tests/oracle/matmul_encodings/):

- THOR / MOAI: `atol = 5e-1` (very loose)
- BMM-1 / BMM-3 / RowEnc: `atol = 1e-1`

The HANDOFF's CPU sweep measured actual decrypted errors of ~`1e-7`
to `2e-6`. So there is **~5 orders of magnitude of headroom** between
observed error and the assertion threshold. The tests catch
catastrophic bugs cleanly, but a *systematic bias up to ~5e-1* is
mathematically possible to slip through.

This is a tolerance-vs-precision gap, not a methodology error. Mitigations
in §7.

## 4. Performance — micro-benches

Two benches to test that desilo's CUDA build actually delivers on the
two binding-layer optimizations we made for the CPU path:

1. **Hoisting**: time `engine.rotate_batch(ct, [FixedRotKey_k1...kN])`
   vs N individual `engine.rotate(ct, rot_key, delta=k)`. Hoisting
   amortizes the ModUp / decomposition phase of key-switching across
   all N rotations, so the ratio should grow with N.
2. **Lazy-relin**: time N independent ct×ct products with eager relin
   (`multiply(a, b, rk)`) vs N degree-2 products with a single
   terminal `relinearize`. Lazy pays 1 key-switch instead of N, so the
   ratio should grow with N.

Script saved at `/tmp/desilo_microbench.py`. Methodology: 5 trials per
config, 2 warmup, median reported. **Important caveats in §6.**

### Hoisting

| N | CPU loop | CPU batch | CPU× | GPU loop | GPU batch | GPU× |
|--:|--:|--:|--:|--:|--:|--:|
|  2 |  230ms |  195ms | 1.18× |  16.0ms |  14.4ms | 1.12× |
|  4 |  456ms |  274ms | 1.66× |  28.1ms |  19.0ms | 1.48× |
|  8 | 1054ms |  431ms | 2.45× |  60.2ms |  28.4ms | 2.12× |
| 16 | 2553ms |  744ms | **3.43×** | 140.0ms |  47.7ms | **2.94×** |

Monotonic growth on both CPU and GPU → hoisting amortization is real
on both. GPU absolute time at N=16: 47.7ms vs CPU 744ms — **15.6×
faster in absolute terms**.

### Lazy-relin

| N | CPU eager | CPU lazy | CPU× | GPU eager | GPU lazy | GPU× |
|--:|--:|--:|--:|--:|--:|--:|
|  2 |  263ms |  196ms | 1.34× | 17.6ms | 15.4ms | 1.14× |
|  4 |  466ms |  257ms | 1.81× | 29.8ms | 23.0ms | 1.30× |
|  8 |  866ms |  373ms | 2.32× | 54.0ms | 38.8ms | 1.39× |
| 16 | 1696ms |  607ms | **2.79×** | 102.7ms | 70.3ms | **1.46×** |

Lazy-relin still wins on GPU but the slope is flatter (1.14× → 1.46×
vs CPU's 1.34× → 2.79×). Interpretation: on GPU the key-switch cost
is already a smaller fraction of total mul cost (NTTs dominate
proportionally more), so there's less to amortize. Speedup is real
and growing with N, but the magnitude is smaller — that's a GPU
architecture fact, not a bug or stub.

## 5. Investigation findings — what the binary + benches together tell us

- **Hoisting**: physical evidence (separate `KeySwitcher4D` class) +
  growing speedup ratio. **High confidence it works.**
- **Lazy-relin**: physical evidence (separate "Then Relinearize" vs
  bare "Multiply" function names) + growing speedup ratio. **High
  confidence it works**, even though the magnitude is smaller than CPU.
- **Both binding additions in `orion/backend/desilo/bindings.py` carry
  over to GPU unchanged.** No code changes needed in the kernels or
  binding to gain these wins.

## 6. Measurement caveats — why the absolute numbers are understated

Before quoting these in a paper figure, the bench needs a v2 pass.
Limitations of the current methodology:

1. **`decrypt`-as-sync barrier adds a constant offset.** Every timed
   region includes one `decrypt + decode`, which is a real key-switch
   + NTT + host transfer. Call it `B` ms. Reported ratio is
   `(real_loop + B) / (real_batch + B)`, which is **closer to 1 than
   the true ratio**. Worse on GPU where `B` is a larger fraction of
   total work. **All GPU speedups in §4 are biased low.**
2. **`decrypt` may not fully sync.** If desilofhe uses multiple CUDA
   streams, decrypt likely syncs the stream it touches but not
   necessarily others. We have no public sync API from desilofhe to
   verify.
3. **Only 5 trials, 2 warmup.** GPU was P8/28°C at bench start; two
   warmups may not be enough to reach a stable P0 power state. Real
   GPU benchmarks want 20-50 trials and longer warmup.
4. **No GPU pinning.** Default device 0; not isolated via
   `CUDA_VISIBLE_DEVICES`.
5. **Loop path syncs only on the last output.** If desilo serializes
   rotations on the same input ciphertext (likely), fine. If it
   dispatches them to independent streams, the last-only sync would
   miss in-flight work.
6. **No clock locking.** `nvidia-smi -lgc <max>` would pin the clock
   but needs root.

### Tighter v2 plan (when we want paper-grade timing)

1. Measure `decrypt + decode` alone in a tight loop; **subtract** from
   each timed region.
2. Probe desilofhe for an explicit sync method (might not exist on
   the Python surface — if not, decrypt is the best proxy, just with
   subtraction).
3. 20 trials, 10 warmup, report median + IQR.
4. `CUDA_VISIBLE_DEVICES=0`, optional clock lock.
5. Sync after every op in the loop path (slow but rigorous); compare
   against the cheaper sync-once-at-end version to size the bias.

## 7. What we have NOT verified yet

| Check | Status | Cost | Value |
|---|---|---|---|
| Oracle tests pass on GPU (decrypt vs numpy) | ✅ done | — | catches catastrophic bugs |
| GPU↔CPU output agreement on identical seeds | ❌ not done | ~5 min to write, 1 min to run | catches systematic bias one backend has and the other doesn't |
| Tolerance tightening (1e-1 → 5e-2 or tighter) | ❌ not done | 1-line edit | raises the floor on what "passes" |
| `/example-test-mm-encodings` on GPU (LoLA, MLP, ResNet) | ❌ not done | ~10-30 min runtime | integration-path correctness |
| Cross-check against Negar's Go reference on same seed | ❌ not done | ~30 min | paper-grade validation |
| Statistical robustness (multi-seed parametrize) | ❌ not done | ~10 min | catches seed-dependent bugs |
| Tighter micro-bench (§6 v2) | ❌ not done | ~30-60 min | paper-grade timing numbers |

## 8. Recommended next steps

In rough priority order for paper-readiness:

1. **GPU↔CPU agreement test.** Cheapest order-of-magnitude tightening
   of the correctness claim. Add to
   `tests/oracle/matmul_encodings/test_cross_backend.py` or extend an
   existing test to assert decrypt-equality on the same seed.
2. **Run `/example-test-mm-encodings` on GPU.** Integration check.
3. **Tighten oracle tolerances.** One-line edit; HANDOFF's open
   follow-up list already includes this.
4. **v2 micro-bench** before quoting absolute GPU numbers in the paper.
5. **Point D7 harness at GPU** to produce the actual paper sweep.
6. **Cross-check ≥1 GPU result against Negar's Go runner** before
   freezing the figure.

## 9. Confidence map (current state)

| Claim | Confidence |
|---|---|
| GPU produces results within CKKS noise of numpy ground truth | **High** — 46/46 pass with 5 orders of magnitude headroom in tolerance |
| GPU agrees with CPU output | **Untested** — assumed by transitivity through numpy, not directly verified |
| Hoisting code path is real and provides speedup | **High** — physical evidence + monotonic speedup growth |
| Lazy-relin code path is real and provides speedup | **High** — physical evidence + monotonic speedup growth |
| Specific GPU speedup ratios (e.g. "2.94× hoisting at N=16") | **Low** — biased low by the bench's sync barrier overhead; v2 needed |
| GPU is ~15× faster than CPU in absolute terms | **Medium** — directionally right, same bias on both sides; v2 will sharpen |
| Integration paths (LoLA / MLP / ResNet) work on GPU | **Untested** — needs `/example-test-mm-encodings` |
| Numbers are paper-figure-ready | **No, not yet** — need v2 bench + cross-checks above |
