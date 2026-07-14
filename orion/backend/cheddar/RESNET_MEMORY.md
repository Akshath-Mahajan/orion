# Cheddar backend: ResNet memory experiment (`cheddar-resnet-memory`)

Status of the `cheddar-resnet-memory` branch. This branch investigates
whether ResNet20 can run on the Cheddar backend given a single 24 GB card,
by measuring the true memory footprint and testing host-RAM spill via
CUDA managed memory. Bootstrap itself is already implemented and merged on
`cheddar-backend`; this is a follow-on memory study, not new crypto.

## Headline findings

1. **ResNet20 needs ~80 GB of GPU-pool memory** (peak 79.65 GB with
   `min_ks=1`, the memory-optimized rotation-key setting). That is **~3.3x
   over** the 24 GB RTX 3090. Even an 80 GB A100/H100 is too tight (no
   headroom for bootstrap working set); realistically you want ~96 GB+ or
   multiple cards. The cost is dominated by ResNet20's per-layer
   linear-transform rotation keys plus the resident boot circuits; the
   network performs 38 bootstrap operations.

2. **Managed memory solves the OOM but not the run.** Swapping Cheddar's
   RMM upstream to `managed_memory_resource` lets the full 80 GB footprint
   allocate by keeping ~24 GB resident in VRAM and paging the ~56 GB
   overflow to host RAM (244 GB free). Compile + all keys + boot circuits
   fit. This is the "load what fits, stream only the overflow" behavior we
   wanted. **However**, inference then aborts at conv1 on a separate,
   non-memory bug (see below), so ResNet does not yet complete end-to-end.

## The boot-slot-count crash (blocks completion)

Inference aborts at conv1 with:

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
  `Bootstrap` throws for any count not in `boot_prepared_slots`. It cannot
  paper over the missing 15.

### Proposed fix (not yet applied)

Make Cheddar's `Bootstrap` lazily prepare an unprepared slot count instead
of throwing, mirroring desilo's "one key covers all" behavior:

```cpp
int Bootstrap(int ct_id, int slots) {
    if (!g_state.boot_prepared_slots.count(slots))
        NewBootstrapper(0, 0, 0, slots);   // prepare on demand
    ...
}
```

Cost: generates the logslots-15 FFT tables + rotation keys on first use,
adding to the already-80 GB footprint. The cleaner alternative — fix
Orion's placement to generate exactly the slot counts inference requests —
touches backend-agnostic core logic that desilo currently hides the bug in,
so it is deferred.

## Uncommitted / out-of-tree state on this branch

- **`orion/backend/cheddar/ext/cheddar_pybind.cu`** (uncommitted, +11):
  adds `GetPeakDeviceMemoryMB()` (reads the RMM statistics adaptor via
  `cheddar::GetPeakDeviceBytes()`) and the `<core/MemoryPool.h>` include.
  Measurement-only; harmless but depends on the shared-clone changes below.
- **`orion/backend/cheddar/STATUS.md`** (untracked, by design): the older
  `cheddar-backend` porting writeup, kept out of git.
- **Shared clone `matmul-encoding-material/cheddar` (its own git repo, not
  orion's):**
  - `include/core/MemoryPool.h`, `src/core/MemoryPool.cpp` — RMM upstream
    switched to `managed_memory_resource`, added a
    `statistics_resource_adaptor` (`PoolStats`) and a free function
    `GetPeakDeviceBytes()` reading its peak byte counter.
  - **`libcheddar.so` was relinked manually against `libfmt`** (CPM
    spdlog 1.10.0 references `fmt::v12::vformat` but the relink did not
    pull `libfmt`; fixed with an explicit
    `-Wl,--no-as-needed .../libfmt.so`). **This is NOT persisted in
    CMake** — a clean rebuild will fail with `undefined symbol:
    fmt::v12::vformat` until the link fix is added to the build files.

## Open decisions

1. **Boot-slot crash:** apply the lazy-prepare fix to let ResNet complete
   under managed memory, or leave it.
2. **Shared clone:** make managed memory an env-gated opt-in (default the
   original `cuda_async` pool) and persist the libfmt link in CMake, so
   other branches that rebuild `libcheddar` are not affected — or revert
   the clone to pristine and keep only this measurement.

## Reproduce the measurement

`scratchpad/run_resnet_measure.py` runs ResNet20 under managed memory and
prints `GetPeakDeviceMemoryMB()` at each stage (cleartext / fit / compile /
inference). Requires the managed-memory clone + libfmt relink above.
