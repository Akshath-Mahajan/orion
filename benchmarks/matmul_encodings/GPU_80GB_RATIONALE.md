# 80 GB-class GPU rationale — matmul-encoding paper

> Written 2026-05-13 by Akshath + Claude. Companion to
> [`HANDOFF2.md`](HANDOFF2.md) §3.2. Captures the case for running
> the d=2048 paper sweep on an A100 80 GB or H100 80 GB instead of
> the current RTX 3090 24 GB.

## TL;DR

- THOR at d=n=2048 needs ~10 GB of persistent `FixedRotationKey`
  cache + 5-10 GB of in-flight working set + transient keygen
  workspace. Peaks past **24 GB** on the 3090.
- LogN-shrinking (LogN=12) avoids the cache problem but **changes
  the ring degree relative to Negar's CPU baseline**, which
  invalidates apples-to-apples speedup numbers.
- An 80 GB GPU (A100 80GB or H100 80GB) fits the cache trivially
  *and* keeps hoisting on *and* keeps the ring degree aligned with
  Negar's CPU run. This is the cleanest path to a defensible d=2048
  GPU number for the paper.

## 1. Why 24 GB is the wall

THOR at the paper shape `d=n=2048, H=1, c=2` issues 2047 distinct
rotation deltas (one per `ell ∈ [1, n)`). The desilo binding's
`RotateBatchNew` (`bindings.py:305`) uses
`engine.rotate_batch(ct, list[FixedRotationKey])` — overload (2) of
desilo's `rotate_batch` — because overload (1) segfaults on this
exact shift pattern (HANDOFF.md §"Three decisions worth remembering").

Each `FixedRotationKey` at LogN=13 (N=8192) with `L+K=6` RNS primes
and hybrid decomposition factor 2 is approximately:

    2 polys × N × (L+K) × 8 bytes × decomp_factor
    = 2 × 8192 × 6 × 8 × 2  ≈ 1.5 MB per polynomial component
    × 2 key components       ≈ 3-5 MB per FixedRotationKey

Caching all 2047 distinct keys persistently in `_fixed_rot_keys`:

    2047 × ~5 MB  =  ~10 GB of persistent GPU memory

Then add: engine workspace and tables (~2-3 GB), the actual ciphertext
working set during THOR (~5-7 GB after the §3.1 fixes from HANDOFF2),
and the transient workspace inside `create_fixed_rotation_key` itself.
Total peak: **15-25 GB**. Tips past 24 GB during the keygen loop of
the very first `rot_batch` call (we observed this empirically — OOM
after 32 s, all in keygen).

The cache **cannot be made smaller without losing performance**.
THOR's access pattern reads every one of the 2047 deltas on every
input row (m_c = d/c = 1024 rows for d=2048, c=2). Any cache with
capacity < 2047 has effectively 0% hit rate after the first row.
Without caching, full regeneration costs ~10 ms × 2047 keys × 1024
rows ≈ 5+ hours of pure keygen — worse than the CPU baseline.

## 2. Why this can't be parameter-tuned around (cleanly)

The HANDOFF2 §3.2 lists three "cheaper" options. Each has a fatal
flaw for the paper story:

| Option | Fixes memory? | Apples-to-apples vs Negar? |
|---|---|---|
| LogN=12 (smaller ring) | ✅ keys halve to ~2.5 MB → cache ≈ 5 GB | ❌ **Different ring degree.** Per-rotation work scales with N; running at N=4096 vs Negar's N=8192 means our GPU number is ~2× faster from the ring alone, not from GPU advantage. To make it fair we'd have to re-run Negar's full CPU sweep at LogN=12 — ~6 h of CPU time, and the paper would be using a smaller ring than the THOR/MOAI source papers. |
| Drop hoisting (single `rot()` in a loop) | ✅ no key cache | ✅ Same ring, but **loses the ~3× hoisting speedup**. Paper has to footnote: "GPU d=2048 row excludes hoisting due to 24 GB memory ceiling." Defensible, but a weaker headline. |
| LogN=13 Standard ring | ❌ Same N=8192 → same key size → same 10 GB cache. Doesn't help. | ✅ Matches Negar's ring exactly. But doesn't fix the memory problem. |

The current operating-point split between us (LogN=13 ConjugateInvariant
→ 8192 slots) and Negar (LogN=13 Standard → 4096 slots) is itself an
open item in the HANDOFF (slot-count alignment). An 80 GB GPU lets us
fix that *and* the cache problem simultaneously by running at LogN=13
Standard with hoisting fully enabled.

## 3. What an 80 GB GPU gives us

| Resource | RTX 3090 (current) | A100 80GB | H100 80GB |
|---|---|---|---|
| HBM | 24 GB GDDR6X | 80 GB HBM2e | 80 GB HBM3 |
| HBM bandwidth | 936 GB/s | 1.9-2.0 TB/s | 3.0-3.4 TB/s |
| FP64 / INT64 throughput | low (consumer) | high | high |
| TF32 / FP32 | high | high | very high |
| Headroom for the 10 GB FixedRotationKey cache | **no** (tips over) | **yes** (~7×) | **yes** (~7×) |
| Headroom for 16k×64-bit polynomial workspaces | tight | comfortable | comfortable |

For *this* workload, the relevant axes are:

1. **HBM capacity.** 80 GB removes the cache wall completely; even
   d=4096 (4095 keys × 5 MB = 20 GB) would fit. This is the *headline*
   reason to switch.

2. **HBM bandwidth.** CKKS rotations are bandwidth-bound on GPU per
   the project's mental model
   (see `matmul-encoding-material/CLAUDE.md`: "each rotation reads
   a large evaluation key from HBM, so the workload becomes
   bandwidth-bound"). The H100's 3 TB/s vs A100's 1.9 TB/s is **a
   ~60% bandwidth advantage** that goes directly to per-rotation
   throughput. The 3090 is 0.94 TB/s — half of A100, a third of H100.

3. **FP64 / INT64 throughput.** Desilo's CKKS implementation likely
   uses 64-bit integer arithmetic for the polynomial coefficients
   (each prime modulus < 2^60). The 3090 is a consumer card with
   anemic 64-bit throughput; A100/H100 are full-speed. This may be a
   silent contributor to why end-to-end GPU speedup is so modest at
   d=256 (1.23× vs Negar's CPU, far from the 15× we measured on
   pure hoisted rotation).

## 4. A100 vs H100 for our specific workload

**Recommendation: A100 80GB if available.** Reasoning:

- For CKKS at LogN=13 (N=8192), the polynomial sizes don't push the
  H100's compute units to peak — the workload is dominated by
  memory traffic, not arithmetic intensity.
- A100's 1.9 TB/s HBM bandwidth is **already 2× the 3090**. The H100
  adds another ~60%, which is real but smaller in relative terms.
- A100 is broadly available (NYU Greene, AWS p4d, GCP a2 series).
  H100 is harder to schedule and roughly 3-5× more expensive per hour.
- If both are accessible at similar cost/availability, H100 is
  strictly better — no architectural reason to prefer A100.

For a *single d=2048 trial* run, either would suffice. The choice is
practical (availability / cost), not technical.

## 5. Where to get one

Options Akshath has access to or should evaluate, in rough order of
ease:

1. **NYU Greene HPC cluster.** Has A100 80GB nodes. Standard slurm
   submission. Free for research compute. Probably the path of least
   resistance — request a single GPU for ~6 hours (one trial of THOR
   d=2048 estimated at 1-2 h on A100 + buffer).

2. **Anthropic / collaborator cloud credits.** If a cloud account is
   already provisioned, AWS `p4d.24xlarge` (8× A100 40GB) or
   `p4de.24xlarge` (8× A100 80GB) on-demand is ~$30-40/h. Single trial
   = single-digit dollars. H100 on `p5.48xlarge` is ~$100/h.

3. **Lambda Labs / Vast.ai / RunPod.** Spot or community-pool A100
   80GB at $1-2/h. Convenient for one-shot benchmarks but worth
   vetting whether desilofhe CUDA wheels work cleanly on whatever
   image they boot.

4. **Negar's local lab.** Ask whether her institution has A100 nodes
   we could borrow on for a half-day. Avoids the cloud setup tax.

## 6. What to actually do on the bigger box

Minimum-viable d=2048 run, in priority order:

1. **THOR d=2048** at the paper shape (`H=1, m=2048, n=2048, c=2`,
   LogN=13 *Standard* for slot-count alignment with Negar). Single
   trial, verify ON, capture kernel time + peak HBM. Compare against
   Negar's 17610 s. Expected runtime: 1-3 h on A100 80GB based on
   our d=256 measurement and 4× per-doubling scaling.

2. **MOAI Alg 3 d=2048** after applying the same Change A+B-style
   memory rewrite (HANDOFF2 §3.1). Expected runtime: probably less
   than THOR (Negar's CPU number is 6.7× faster than THOR's).

3. **BMM-3 sweep at 256/512/1024/2048**. Memory rewrite needed; same
   pattern as THOR. The 4-point scaling curve is the strongest data
   point in the paper.

4. **All five kernels at the smoke shapes** as a regression check
   against today's CSV from the 3090.

## 7. Open questions

- Does desilo's CUDA build expose `cuda_grid_size_multiplier`
  tuning that would help A100/H100 specifically? (We never touched
  the default of 4 — it might be more or less optimal on different
  SM counts.)
- Will the GPU wheel work cleanly on whatever container/AMI we end
  up using? Confirmed on Ubuntu + CUDA 13 driver here; A100 nodes
  typically run CUDA 12.x. `desilofhe-cu129` (12.9 runtime) should
  be forward-compatible with both.
- Is there any reason to *not* use Standard ring on the bigger GPU?
  The slot-count alignment with Negar is a free win at no extra
  cost.
