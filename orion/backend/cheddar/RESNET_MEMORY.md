# Cheddar backend: ResNet memory experiment (`cheddar-resnet-memory`)

Status of the `cheddar-resnet-memory` branch. This branch investigates
whether ResNet20 can run on the Cheddar backend given a single 24 GB card,
by measuring the true memory footprint and testing host-RAM spill via
CUDA managed memory. Bootstrap itself is already implemented and merged on
`cheddar-backend`; this is a follow-on memory study, not new crypto.

## Headline findings

1. **ResNet20 needs ~90 GB of GPU-pool memory** (peak 89.5 GB with
   `min_ks=1`, the memory-optimized rotation-key setting, measured via a
   full `compile()` run — see per-layer breakdown below; an earlier,
   coarser measurement said 79.65 GB, same ballpark). That is **~3.7x
   over** the 24 GB RTX 3090. Even an 80 GB A100/H100 is too tight (no
   headroom for bootstrap working set); realistically you want ~96 GB+ or
   multiple cards. The network performs 38 bootstrap operations.

2. **Managed memory works end-to-end for `compile()`.** Swapping Cheddar's
   RMM upstream to `managed_memory_resource` lets the full 89.5 GB
   footprint allocate by keeping ~24 GB resident in VRAM (confirmed via
   `nvidia-smi`, which plateaus at the card's physical capacity for the
   back half of compile) and paging the rest to host RAM (244 GB free).
   `compile()` now completes fully without OOM (previously it aborted
   during conv1's key generation on plain `cuda_async`). Inference itself
   (actually running the bootstraps) has not yet been measured — only
   `compile()`, which is where essentially all the key/diagonal memory
   described below gets allocated.

3. **Per-layer breakdown — only ~15 of ~140 compiled nodes cost any GPU
   memory at all**, and repeated blocks within the same ResNet stage fully
   *reuse* their stage's first block's rotation keys at zero extra cost
   (same shape ⇒ same rotation distances ⇒ same keys). Measured with
   `ORION_CHEDDAR_MANAGED_MEMORY=1 ORION_CHEDDAR_BOOT_MIN_KS=1`, logical
   peak-pool bytes (see `GetPeakDeviceMemoryMB`, not physical VRAM):

   | Phase | Cumulative | Increment |
   |---|---|---|
   | Baseline (scheme setup, before any model work) | 8.8 GB | — |
   | + boot circuits (3 slot counts: 16384/8192/4096) | 14.6 GB | +5.8 GB |
   | + conv1 (stem) | 27.9 GB | **+13.3 GB** |
   | + stage 0 (3 blocks, 16ch) | 27.9 GB | +0 (full reuse) |
   | + stage 1 transition (conv1/shortcut/conv2, 32ch) | 56.2 GB | +27.7 GB |
   | + stage 1 remaining blocks | 56.2 GB | +0 (full reuse) |
   | + stage 2 transition (64ch) | 79.5 GB | +23.3 GB |
   | + stage 2 remaining blocks | 79.5 GB | +0 (full reuse) |
   | + avgpool (grouped conv) | 87.8 GB | +8.3 GB |
   | + final linear | 89.5 GB | +1.7 GB |

   **Decisive consequence:** the preamble (baseline + boot circuits, 14.6
   GB) plus conv1 alone (13.3 GB) already totals 27.9 GB — over the 24 GB
   card *before a single non-stem conv layer is compiled*. This means
   **no amount of per-layer rotation-key streaming (`io_mode`-style
   eviction) can make ResNet20 fit on a 24 GB card**, even with perfect,
   immediate eviction of every layer's keys the instant they're no longer
   needed. `io_mode` is therefore not viable as a *sufficient* fix on its
   own here (see Open decisions); it could still reduce how much managed
   memory needs to page (from ~65 GB down to as little as ~4 GB in the
   best case), but only in combination with managed memory, and building
   it correctly is harder than it first looks: stage-repeated blocks
   *reuse* identical keys, so a naive "evict right after each layer"
   policy would force regenerating those shared keys 2-3x over instead of
   saving anything.

   Reproduce with `examples/measure_resnet_memory.py configs/resnet_cheddar.yml`
   (see bottom of this doc).

## The boot-slot-count crash (fixed)

Inference used to abort at conv1 with:

```
Cheddar: no bootstrapper prepared for slots=32768. Call NewBootstrapper first.
```

Root cause is an **Orion bootstrap-placement gap that desilo masks and
Cheddar cannot**:

- Orion's placement calls `generate_bootstrapper(slots)` for the slot
  counts it decides on (observed: logslots 14/13/12). At runtime conv1
  requests **logslots 15 (32768)** — a count that was never generated.
- **Desilo** (`orion/backend/desilo/bindings.py`): `NewBootstrapper`
  builds one *slot-agnostic* key via `engine.create_bootstrap_key(sk)` —
  the `slots` argument is ignored on both `NewBootstrapper` and
  `Bootstrap`. One key bootstraps any slot count, so the unrequested 15
  just works. The placement gap is invisible.
- **Cheddar** (`orion/backend/cheddar/ext/cheddar_pybind.cu`): bootstrap is
  *slot-specific* — each slot count needs its own
  `PrepareEvalSpecialFFT(slots)` + `AddRequiredRotations(slots)`, and
  `Bootstrap` threw for any count not in `boot_prepared_slots`. It could
  not paper over the missing 15.

### Fix applied

`Bootstrap` (`cheddar_pybind.cu:882`) now lazily prepares an unprepared
slot count instead of throwing, mirroring desilo's "one key covers all"
behavior:

```cpp
int Bootstrap(int ct_id, int slots) {
    ensure_keys();
    if (!g_state.boot_prepared_slots.count(slots))
        NewBootstrapper(0, 0, 0, slots);   // prepare on demand
    ...
}
```

`NewBootstrapper` was already idempotent per slot count and already
ignores `num_cts_levels`/`num_stc_levels`/`log_message_ratio` (the
`BootContext` those configure is built once at `setup_scheme` time), so
`0, 0, 0` for those args is safe.

**Verified:** a standalone repro (generate a bootstrapper for 16384 only,
then bootstrap a full 32768-slot ciphertext without ever calling
`generate_bootstrapper(32768)`) previously threw the "no bootstrapper
prepared" error; with the fix it lazily prepares 32768 and gets past that
check entirely, failing only on a CUDA OOM from holding two full boot
circuits' worth of keys at once on one 24 GB card. That OOM is the
already-documented aggregate-key-memory cost (see headline finding 1),
not a new bug. `pytest -m slow -k cheddar tests/oracle/test_bootstrap.py`
(2 passed) and the full `pytest tests/oracle/ --backend=cheddar` (71
passed) show no regression on the existing single-slot-count path.

The cleaner alternative — fix Orion's placement to generate exactly the
slot counts inference requests — touches backend-agnostic core logic that
desilo currently hides the bug in, so it remains out of scope here.

Since ResNet20's full working set is ~90 GB regardless (headline finding
1), this fix alone does not make ResNet20 complete on a single 24 GB
card without also solving the memory problem — it just removes a crash
that would otherwise hit regardless of how much memory is available. The
managed-memory experiment (finding 2) is what actually lets `compile()`
get all the way through.

## State of the managed-memory instrumentation

This was built, reverted, and rebuilt once during this investigation (an
earlier pass left it as an uncommitted, easy-to-lose diff — that mistake
is not repeated here). It is now committed in both repos:

- **Shared clone `matmul-encoding-material/cheddar`** (its own git repo,
  remote `orion-private`, branch `orion-patches`):
  `include/core/MemoryPool.h` / `src/core/MemoryPool.cpp` — RMM upstream
  is chosen at runtime via `ORION_CHEDDAR_MANAGED_MEMORY` (unset/`0`
  keeps the original `cuda_async_memory_resource`; any other value swaps
  in `managed_memory_resource`), wrapped in a `statistics_resource_adaptor`
  so `GetPeakDeviceBytes()` reports the true logical peak regardless of
  physical residency. Default behavior (env unset) is unchanged, so other
  branches rebuilding `libcheddar` are unaffected.
- **`orion/backend/cheddar/ext/cheddar_pybind.cu`** /
  **`orion/backend/cheddar/bindings.py`**: `GetPeakDeviceMemoryMB()`
  binding exposing the above to Python.
- The previously-reported "libfmt relink needed for spdlog 1.10.0 /
  `fmt::v12::vformat`" issue **did not recur** on this rebuild (`ldd -r`
  clean, managed-memory path runs fine). Not persisted in CMake since it
  wasn't needed; if it resurfaces on a different machine/toolchain, add
  an explicit `-Wl,--no-as-needed <path>/libfmt.so` to the `cheddar`
  target's link flags in `CMakeLists.txt`.

## Why desilo needs so much less memory

Same network, same CKKS params (`configs/resnet_desilo.yml`, LogN=16,
same LogQ/LogP/scale/H), same measurement script
(`examples/measure_resnet_memory.py`), on the same GPU. Note:
`resnet_desilo.yml` had no `device: gpu` set — Orion defaults `device` to
`"cpu"` (`OrionParameters.device`), so running it as-shipped silently
measures a CPU run (near-zero "GPU memory" for the wrong reason). Fixed
in this branch; re-run with `device: gpu` added:

| Phase | Cumulative | Increment |
|---|---|---|
| Baseline (engine + secret/public/relin/rotation/conj keys) | 4.1 GB | — |
| + diagonal generation (all layers) | 4.1 GB | +0 |
| + boot circuits (3 slot counts) | 16.5 GB | +12.4 GB |
| + conv1 (stem) | 16.6 GB | +0.12 GB |
| + everything else (19 more conv/linear/shortcut layers + avgpool) | 16.6 GB | **+0.05 GB total** |

**Total: ~16.6 GB** — comfortably under 24 GB, vs cheddar's 89.5 GB.
Two separate mechanisms drive this, both visible directly in
`orion/backend/desilo/bindings.py`:

1. **One rotation key covers the whole network.**
   `GenerateEvaluationKeys()` calls `engine.create_rotation_key(self._sk)`
   *once*, during initial key generation, before `compile()` even starts.
   `AddRotationKey()` — the hook cheddar uses to add a new per-distance
   key for every layer — is a no-op for desilo:
   `pass  # General rotation key already covers all deltas`. Every
   `Rotate`/`RotateNew` call for any layer, any distance, reuses that one
   key. Cheddar instead generates a distinct key per required rotation
   distance *per layer* (`AddRotationKey` → `PrepareRotationKey`), so its
   cost scales with network depth and shape diversity; desilo's doesn't
   scale with either. This is the entire explanation for the ~75 GB gap
   in the per-layer body of the network (conv1 through the final linear
   layer): cheddar added ~74.9 GB there, desilo added ~0.17 GB.

2. **Desilo already evicts its bootstrap key** — the thing cheddar's
   `RemoveRotationKeys` docstring says has "no release path." Every
   `NewBootstrapper(slots)` call explicitly deletes the previous
   `_boot_key` and runs `gc.collect()` before creating the new one
   (`bindings.py:774-781`, comment: *"each key is ~12GB on GPU"*), so only
   *one* boot key is ever resident, not one per slot count. That's why
   "boot circuits" only added 12.4 GB here despite 3 distinct slot counts
   being requested (16384/8192/4096), vs cheddar which keeps all 3
   resident simultaneously.

**The caveat this measurement doesn't cover:** desilo's boot-key eviction
means that if inference ever needs to bootstrap at a slot count *other
than* the currently-resident one, it must regenerate a ~12 GB key on the
spot. For a plain feedforward network like ResNet20, spatial resolution
only shrinks monotonically (32→16→8), so this should cost at most 2 extra
regenerations across the whole forward pass — but that's an assumption,
not something this compile()-only measurement verifies. More broadly,
this doc only measures memory: desilo's rotation model has its own
compute-side tradeoffs (lazy-relin, rotate_batch) that a memory-only
comparison doesn't capture — a single universal rotation key plausibly
costs more compute per rotation than a dedicated per-distance key would.
A fair backend comparison needs wall-clock inference numbers alongside
these memory numbers, not memory alone.

## Open decisions

1. ~~**Boot-slot crash:** apply the lazy-prepare fix~~ — **done**.
2. ~~**Shared clone:** persist managed memory as an env-gated opt-in~~ —
   **done**, see above.
3. **How to actually get ResNet20 running:** per headline finding 3,
   `io_mode`-style rotation-key streaming *cannot* fit this on its own (the
   preamble + conv1 already exceed 24 GB), so the real choices are (a)
   managed memory alone — already proven to work for `compile()`, expect a
   real slowdown from host-RAM page traffic during inference (unmeasured
   so far), or (b) `io_mode` implemented *in combination with* managed
   memory, to shrink the ~65 GB of required paging down to as little as
   ~4 GB — but cheddar has no key-eviction path at all today (`
   RemoveRotationKeys`/`RemovePlaintextDiagonals` are no-ops — "no release
   path"), rotation keys are shared between the LT and Boot subsystems
   (same `EvkMap`, so naive eviction risks breaking a bootstrap that
   reuses a key an evicted layer needed), and it must be stage/dedup-aware
   rather than naively per-layer (see finding 3). Not started.
4. **Inference-time behavior is unmeasured.** Everything above is from
   `compile()` only. Actual bootstrap execution (38 ops) under managed
   memory — correctness, peak memory, and especially wall-clock time from
   page-fault traffic — still needs a real run.

## Reproduce the measurement

`examples/measure_resnet_memory.py <config>` replicates `Scheme.compile()`
phase-by-phase (rather than modifying `orion/core/orion.py`) with memory
checkpoints after diagonal generation, after boot-circuit generation, and
after every layer's `compile()`. Works for any backend — falls back to
`nvidia-smi` alone when `GetKeyMemoryMB`/`GetPeakDeviceMemoryMB` aren't
available (i.e. everywhere except cheddar). For the cheddar numbers above:

```bash
CUDA_VISIBLE_DEVICES=<idle gpu> ORION_CHEDDAR_MANAGED_MEMORY=1 \
    ORION_CHEDDAR_BOOT_MIN_KS=1 python examples/measure_resnet_memory.py \
    configs/resnet_cheddar.yml
```
