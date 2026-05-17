# Session 3 handoff — matmul-encoding paper

> Written 2026-05-17 by Akshath + Claude. Follow-up to
> [`HANDOFF2.md`](HANDOFF2.md). The prior doc covers the THOR memory
> rewrite + d=256 GPU/CPU comparison. This one covers the methodology
> hardening that landed in this session: Negar's exact CKKS params,
> NVML-backed GPU energy + memory measurement, and the
> sync/cold-start/clock-pin learnings we picked up smoke-testing on
> the 3090.
>
> **This is the doc to read first if you're picking up the work on a
> new (bigger) GPU.** Section 5 lists exactly what to verify on
> arrival and what flags to use for the paper sweep.

## TL;DR

- **Branch `feat/matmul-encoding`** now contains, in addition to the prior session's THOR memory rewrite:
  - **`--ckks-preset negar`**: a one-line CLI flag to run the harness at LogN=13 Standard / LogQ=55+4·45 / LogP=61 (4096 slots, matching `matmul-encoding-material/MatMult/matmult/init_lattigo.go::DefaultParams` 1:1). Requires the desilo binding fix below.
  - **`Engine(slot_count=…, max_level=…)` desilo binding fix**: previously the binding ignored `slot_count` and silently ran at 8192 slots even when Orion asked for 4096. Now respects what Orion requests.
  - **NVML-backed GPU energy + true memory peak**: new `GpuMonitor` class in [`runners/gpu_sampler.py`](runners/gpu_sampler.py). Uses `nvmlDeviceGetTotalEnergyConsumption` (exact counter delta, not integration; works on this driver even though `nvidia-smi --query-gpu` rejects the field) and a daemon thread polling per-PID memory at ~5 ms.
  - **Per-kernel HBM delta + true idle baseline**: two new columns in the CSV so the cumulative-memory artefact (later kernels inheriting earlier kernels' residue when the harness reuses one `ctx`) is comparable kernel-to-kernel.
  - **Methodology knobs**: `torch.cuda.synchronize()` after every timed call (~1 ms each, halves CV); optional `--cold-start` (pace between rows until GPU temp returns to baseline); optional `--lock-gpu-clocks=MHZ` (calls `nvidia-smi --lock-gpu-clocks` with reset in `finally`).
  - **Two methodology-verification columns**: `start_temp_c` (GPU temp at the moment timing begins) and `mean_clock_mhz` (average graphics clock during the timed window). Lets post-hoc analysis confirm cold-start and clock-pin both worked from the CSV itself.
- **CSV schema is now 21 columns**, fully populated on `--device gpu` and empty for the GPU-only columns on CPU rows; the D8 Go shim [`cpu_baseline/csv_emit.go`](cpu_baseline/csv_emit.go) was widened to match so `pd.concat` of the CPU + GPU sweeps still lines up.
- **`nvidia-ml-py>=12.0`** tracked as an optional extra (`bench-gpu`) in [`pyproject.toml`](../../pyproject.toml). Install with `pip install orion-fhe[bench-gpu]`.
- **Methodology learning (important): cold-start without clock-pin makes variance worse, not better.** The 3090 we developed on does not allow clock-pinning without root; cold-start alone put the GPU in a low P-state and CV went from 1–7% up to 10–43%. **For the paper, drop `--cold-start` and go steady-state instead.** See §3 for the data and reasoning.

## 1. What landed (commits, in branch order)

Final graph on `feat/matmul-encoding`:

```
*   <m4>  Merge feat/matmul-encoding-gpu-methodology: sync + cold-start + clock-pin
|\
| * <h3>  docs(matmul_encodings): HANDOFF3.md (this file)
| * <f4>  feat(matmul_encodings): torch.cuda.synchronize + cold-start + clock-pin + temp/clock columns
|/
*   5c31692  Merge feat/matmul-encoding-gpu-energy: GPU energy + true peak memory
|\
| * fca5cc6  feat(matmul_encodings): per-kernel HBM delta + true idle baseline
| * 3d32443  feat(matmul_encodings): GPU energy + true memory peak via NVML
|/
*   ce12805  Merge feat/matmul-encoding-thor-memory: THOR memory rewrite + --ckks-preset
|\
| * 21b0074  Merge feat/matmul-encoding-negar-params: --ckks-preset for 4096-slot CPU parity
| * 57a7532  feat(matmul_encodings): --ckks-preset CLI flag for Negar's CPU params
| * 27b0013  feat(desilo): pass slot_count to non-bootstrap Engine constructor
| * 8288960  feat(matmul_encodings): THOR memory rewrite -- eager free + fused phases
|/
* 9470782  Merge docs/matmul-encoding-session2-docs: HANDOFF2 + 80GB rationale
...
```

(The `<f4>`, `<h3>`, `<m4>` hashes are filled in when this branch is merged.)

### 1.1 By feature

| Topic | Files touched | Commits |
|---|---|---|
| CKKS param parity with Negar's Go | [`orion/backend/desilo/bindings.py`](../../orion/backend/desilo/bindings.py), [`runners/_common.py`](runners/_common.py), [`runners/__main__.py`](runners/__main__.py), [`tests/oracle/matmul_encodings/conftest.py`](../../tests/oracle/matmul_encodings/conftest.py) | `27b0013` + `57a7532` |
| Energy + true memory peak | [`runners/gpu_sampler.py`](runners/gpu_sampler.py) (new), [`runners/_common.py`](runners/_common.py), [`runners/kernels.py`](runners/kernels.py), [`runners/__main__.py`](runners/__main__.py), [`cpu_baseline/csv_emit.go`](cpu_baseline/csv_emit.go) | `3d32443` |
| Per-kernel HBM delta + true idle | same files | `fca5cc6` |
| Sync + cold-start + clock-pin + temp/clock columns | [`runners/gpu_sampler.py`](runners/gpu_sampler.py), [`runners/_common.py`](runners/_common.py), [`runners/__main__.py`](runners/__main__.py), [`cpu_baseline/csv_emit.go`](cpu_baseline/csv_emit.go), [`pyproject.toml`](../../pyproject.toml) (already added the extras earlier this session) | `<f4>` |

## 2. CSV schema

21 columns. All `BenchResult` field names map 1:1 to CSV headers.

| # | column | populated when | meaning |
|---|---|---|---|
| 1 | `backend` | always | `"lattigo"` / `"desilo"` |
| 2 | `device` | always | `"cpu"` / `"gpu"` |
| 3 | `kernel` | always | `bmm1` / `bmm3` / `thor` / `moai_alg3` / `moai_alg4` / `rowenc` |
| 4 | `shape` | always | human-readable shape label |
| 5 | `n_he` | always | algorithm `n_he` (= `ctx.slots` when not chunked) |
| 6 | `n_trials` | always | timed iterations |
| 7 | `mean_seconds` | always | wall-clock per kernel call, mean over `n_trials` |
| 8 | `std_seconds` | always | 1σ over `n_trials`. 0 when `n_trials==1` |
| 9 | `rotations` | always | per-trial CKKS rotation count |
| 10 | `ct_ct_muls` | always | per-trial ct×ct mult count |
| 11 | `ct_pt_muls` | always | per-trial ct×pt mult count |
| 12 | `peak_hbm_mb` | desilo gpu | cumulative process high-water mark during timed window |
| 13 | `peak_hbm_delta_mb` | desilo gpu | peak − baseline-at-window-enter (per-kernel attributable) |
| 14 | `gross_energy_j` | desilo gpu | raw counter delta / n_trials (joules per trial) |
| 15 | `kernel_energy_j` | desilo gpu | gross − (resident_idle_w × window) / n_trials, clamped ≥ 0 |
| 16 | `mean_power_w` | desilo gpu | gross_energy_total / window_seconds |
| 17 | `true_idle_w` | desilo gpu | true idle (sampled pre-`ctx`, no FHE state) |
| 18 | `start_temp_c` | desilo gpu | GPU temp at the moment timing begins (verifies cold-start) |
| 19 | `mean_clock_mhz` | desilo gpu | average graphics clock during window (verifies clock-pin) |
| 20 | `single_tenant` | desilo gpu | False if any compute PID other than ours was on the GPU |
| 21 | `max_abs_err` | when `--verify` | decrypt error of the last trial vs numpy reference |

### 2.1 Interpreting `kernel_energy_j == 0`

`kernel_energy_j` clamps to 0 when `gross_energy_j < idle_power_w × window`. This happens for **very short kernels** that don't pull the GPU power above the resident-idle line — e.g. THOR at d=2,c=2 (~21 ms) on the 3090 ran at 102 W mean while resident idle was 120 W. The kernel literally did not draw more than idle. **Not a bug**; it's an honest "this kernel is too small to attribute energy to". For paper figures, exclude these rows OR report `gross_energy_j` only.

### 2.2 Interpreting `peak_hbm_mb` vs `peak_hbm_delta_mb`

`peak_hbm_mb` is the cumulative process high-water mark during this row's timed window. Because the harness shares one `ctx` across all kernels in a sweep, later kernels' `peak_hbm_mb` includes residual state from earlier kernels — the column is monotonically non-decreasing through a run, which is **not** comparable kernel-to-kernel.

`peak_hbm_delta_mb = peak_hbm_mb − baseline_at_window_enter`. This is the memory the kernel itself added on top of its starting state. Use this for kernel-to-kernel comparisons in paper figures. The cumulative `peak_hbm_mb` is still useful to report — it's the actual memory pressure your deployment sees in a continuous workload.

## 3. Methodology learnings (read before paper sweep)

### 3.1 `torch.cuda.synchronize()` is wired in — keep it

Smoke test on the 3090, BMM-I at one shape, 10 trials each:

| method | mean | std | CV |
|---|---|---|---|
| no sync | 164.4 ms | 1.8 ms | 1.1% |
| `torch.cuda.synchronize` | 165.7 ms | 0.8 ms | **0.5%** |
| throwaway `ctx.decrypt` | 198.8 ms | 0.7 ms | 0.4% |

Sync barely shifts the mean (~0.8%) but halves the CV. Throwaway decrypt is wrong (it measures decrypt+kernel, not just kernel — 21% overhead). The harness now syncs unconditionally on `--device gpu` if `torch.cuda.is_available()`.

### 3.2 Cold-start without clock-pin makes things WORSE (do not use on 3090)

Same 6 smoke kernels, `--n-trials 5 --warmup 2`, with vs without `--cold-start --cold-start-tolerance-c 3`:

| kernel | no cold-start CV | with cold-start CV | mean_clock w/ cs |
|---|---|---|---|
| bmm1 | 7.5% | **19%** | **1013 MHz** |
| bmm3 | 1.1% | 6.7% | 1641 MHz |
| thor | 4.8% | **43%** | **210 MHz** (idle!) |
| moai_alg3 | 1.1% | 17% | 1653 MHz |
| moai_alg4 | 0.7% | 10.5% | 1573 MHz |
| rowenc | 1.7% | 14% | 1008 MHz |

**Why**: during the 30–50 s cold-start wait, the GPU not only cools but also drops to a low P-state (~210 MHz idle clock). When timing starts the first trial runs on an idle-clock GPU; the clock ramps during the trial; later trials boost. P-state ramp variance dwarfs thermal-drift variance.

**Conclusion**: cold-start syncs the **thermal** state machine. The **P-state** state machine remains unsynchronized. Cold-start + clock-pin together fixes both. Cold-start alone is harmful on hardware with aggressive P-state management.

### 3.3 Steady-state is the right methodology for this paper

We initially wired cold-start as the "right" methodology, but that was wrong. MLPerf and most GPU characterization papers report **steady-state** numbers — GPU warmed up to thermal + clock + power equilibrium before measurement begins. The argument: nobody runs an FHE kernel once and stops; they run thousands back-to-back inside a transformer pass. The steady-state numbers are what production deployments actually pay.

**Paper sweep methodology** (recommended):

| flag | value | reasoning |
|---|---|---|
| `--backend desilo` | — | GPU target |
| `--device gpu` | — | obviously |
| `--ckks-preset negar` | — | matches D8 CPU baseline params exactly |
| `--preset paper` | — | mirrors Negar's `*_runner.go` shape tables |
| `--n-trials 10` | minimum | tight σ |
| `--warmup 5` | minimum | settle CUDA caches + clock + thermal |
| `--lock-gpu-clocks 1395` | if sudoers allows | removes boost variance; pick the GPU's base clock |
| **NOT** `--cold-start` | — | actively harmful without clock-pin; unnecessary with it |

The smoke runs that produced the table in §3.2 had `--warmup 2` which is **not enough for steady-state**. Bump to `--warmup 5` minimum for the paper sweep.

### 3.4 Cold-start fires once per shape, not per trial

When `--cold-start` is on, the wait happens once per shape (post-warmup, pre-timing). Then all `n_trials` run back-to-back; GPU heats up across them. Per-trial cold-start would cost ~30 s × n_trials × n_shapes ≈ multiple hours per sweep — not viable. The current pattern gives **inter-shape** thermal consistency, with intra-shape thermal drift absorbed into `std_seconds`. (Per §3.2/3.3, drop cold-start entirely for the paper.)

## 4. Why clock-pinning needs sudo (and what to ask sysadmin)

`nvidia-smi --lock-gpu-clocks` triggers a driver ioctl that requires `CAP_SYS_ADMIN`. Reasons: hardware risk if locked-high without cooling/power headroom; shared-device fairness; datacenter power policy. The check is in the driver, so no userspace workaround exists.

The minimum-effort sudoers entry to ask for (on the bigger GPU):

```
your_username ALL=(root) NOPASSWD: /usr/bin/nvidia-smi --lock-gpu-clocks=*, /usr/bin/nvidia-smi --reset-gpu-clocks
```

This grants password-less sudo for **only** these two commands. The harness's `--lock-gpu-clocks` flag calls `nvidia-smi` directly (not via `sudo`), so it currently won't use the sudoers entry. If sysadmin grants it, prefix the invocation with `sudo` and the lock will succeed:

```bash
sudo $(which nvidia-smi) --lock-gpu-clocks=1395,1395 -i 0
# run the harness
sudo $(which nvidia-smi) --reset-gpu-clocks -i 0
```

OR I can add `--sudo-lock-clocks` to the harness — 3-line change in [`runners/__main__.py::_set_gpu_clocks`](runners/__main__.py). Easier path is the manual `sudo nvidia-smi` calls before/after the sweep.

## 5. What to do on the new (bigger) HPC

In order:

1. **Pull the branch + checkout** (and the session export branch if you want to feed the JSONL back to Claude):
   ```bash
   git fetch origin
   git checkout feat/matmul-encoding
   ```
2. **Verify environment**:
   ```bash
   python -m pip install -e '.[bench-gpu]'      # installs nvidia-ml-py
   python -m pip install desilofhe-cu129          # or the cuda-version matching driver
   python -c "import desilofhe; print(desilofhe.__version__)"
   python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)))"
   ```
3. **Run the oracle suite** to confirm desilo wheel + bindings work end-to-end on the new card:
   ```bash
   python -m pytest tests/oracle/matmul_encodings/ -v --tb=short
   ```
   Expect 86/86 passing. If anything fails, it's binding or wheel mismatch, not kernel logic.
4. **Smoke-test the harness** on the new GPU to confirm energy + memory columns populate:
   ```bash
   python -m benchmarks.matmul_encodings.runners \
       --backend desilo --device gpu --ckks-preset negar \
       --preset smoke --n-trials 3 --warmup 2 \
       --output /tmp/smoke_newgpu.csv
   ```
   - Sanity check: `start_temp_c` and `mean_clock_mhz` should be populated.
   - Sanity check: `kernel_energy_j` should be positive for the larger kernels.
5. **Check clock-pin permissions**:
   ```bash
   nvidia-smi --lock-gpu-clocks=1395,1395 -i 0
   nvidia-smi --reset-gpu-clocks -i 0
   ```
   If it works, great — use the flag. If "no permission", ask sysadmin for the sudoers entry in §4.
6. **Find the new GPU's base clock**:
   ```bash
   nvidia-smi --query-gpu=clocks.max.gr,clocks.default_applications.gr --format=csv -i 0
   ```
   Pin at the default applications clock — that's the manufacturer-blessed sustained operating clock.
7. **Run the paper sweep**:
   ```bash
   python -m benchmarks.matmul_encodings.runners \
       --backend desilo --device gpu --ckks-preset negar \
       --preset paper \
       --n-trials 10 --warmup 5 \
       --lock-gpu-clocks <BASE_CLOCK_FROM_STEP_6> \
       --output benchmarks/matmul_encodings/results/desilo_gpu_paper_<host>.csv
   ```
   (Skip `--lock-gpu-clocks` if step 5 said no permission; numbers will have ~3% wider σ.)
8. **Re-run D8 CPU baseline on the same host** (Negar's Go via the patched shim) so CPU and GPU rows are from the same machine. See [`cpu_baseline/README.md`](cpu_baseline/README.md). Tooling is already in the repo.
9. **D9 (paper figure)** still TODO. Inputs are now two CSVs with matching 21-column schemas — `pd.concat` them and plot.

## 6. Paper methodology paragraph (draft)

Copy-pasteable for the paper. Fill in the bracketed values from the actual run.

> **GPU measurement methodology.** All GPU numbers were collected on an NVIDIA `[GPU_MODEL]` (`[ARCH]`, `[VRAM]` GB `[VRAM_TYPE]`, `[TDP]` W TDP) using NVML via the `nvidia-ml-py` 13.x wrapper. Per-kernel energy is the delta of `nvmlDeviceGetTotalEnergyConsumption` (cumulative millijoule counter at driver-internal sampling) across the timed window, divided by trial count, minus an idle-power baseline (engine + key state resident, no kernel) multiplied by the window duration. Peak HBM is a daemon thread polling `nvmlDeviceGetComputeRunningProcesses` for the benchmark PID every 5 ms; we report both the cumulative process high-water mark (`peak_hbm_mb`) and the kernel-attributable delta above the kernel's starting memory state (`peak_hbm_delta_mb`). Each shape ran with 5 warmup + 10 timed trials at GPU clocks locked to `[CLOCK]` MHz via `nvidia-smi --lock-gpu-clocks` to remove boost-clock variance; each timed call ends with `torch.cuda.synchronize()` so reported runtime, energy, and memory all refer to the same fully-flushed kernel. The benchmark process was the sole CUDA consumer on the GPU throughout the sweep (verified per-row via NVML's compute-process list). We mirror the CPU baseline's CKKS parameter set exactly (`--ckks-preset negar`: LogN=13 Standard, LogQ=55+4·45, LogP=61, 4096 slots) so CPU vs GPU rows differ only in hardware.

## 7. Open follow-ups (paper-irrelevant unless flagged)

| | item | impact |
|---|---|---|
| 1 | THOR-style memory rewrite for MOAI / BMM-III / RowEnc | unblocks larger shapes on smaller GPUs. THOR is the only kernel that's been rewritten; the others still build O(d²) live ciphertexts. Don't block on this if the new GPU has 80 GB. |
| 2 | `FixedRotationKey` cache (~10 GB at n=2048) | blocks d≥512 even with the THOR memory rewrite. Cache eviction policy or lazy key generation would unblock. Filed but not fixed. |
| 3 | `--cold-start-per-trial` | trivial flag if methodology debate goes that way later. Currently not recommended (see §3). |
| 4 | `--sudo-lock-clocks` | 3-line addition to [`_set_gpu_clocks`](runners/__main__.py) if sysadmin grants the sudoers entry. |
| 5 | `single_tenant == False` handling | currently we warn and emit; might want to skip the row or rerun. |
| 6 | D9 paper figure | matplotlib 2×N bar chart, rotation count annotations, log-scale Y. Inputs ready, just needs the script. |
| 7 | Tighten `atol` in oracle tests from 5e-1 to 5e-2 | documents the actual noise floor; observed errors are ~1e-3. |
| 8 | File `nvidia-smi --query-gpu=total_energy_consumption` upstream | CLI rejects the field even though NVML exposes the counter. Cosmetic. |

## 8. Quick reference

**Active branch:** `feat/matmul-encoding` (this session's merges land on top).
**Python env:** `/home/avm6288/miniconda3/envs/myenv2/bin/python` (has `desilofhe`, `orion`, `lattigo`, `nvidia-ml-py`, `torch`).

**Smoke on this 3090 (baseline, no cold-start, sync on):**
```bash
python -m benchmarks.matmul_encodings.runners \
    --backend desilo --device gpu --ckks-preset negar \
    --preset smoke --n-trials 5 --warmup 5 \
    --output /tmp/smoke.csv
```

**Re-run all matmul-encoding oracle tests:**
```bash
python -m pytest tests/oracle/matmul_encodings/ benchmarks/matmul_encodings/plaintext/ -v
```

**Memory entries that should be consulted:**
- `user_role.md`
- `project_matmul_encoding_paper.md`
- `project_matmul_open_questions.md`
- `project_matmul_testing_strategy.md`
- `reference_matmul_encoding_context_repo.md`
