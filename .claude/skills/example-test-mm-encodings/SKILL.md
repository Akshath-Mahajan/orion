---
name: example-test-mm-encodings
description: End-to-end regression check for the matmul-encoding work. Re-runs examples/run_lola.py, run_mlp.py, run_resnet.py on lattigo + desilo backends and diffs MAE/Precision against the saved reference transcripts in examples/results/. Use after binding changes to catch breakage of existing inference paths.
---

# End-to-end regression: examples vs saved transcripts

The matmul-encoding work modifies `orion/backend/desilo/bindings.py` (adding `MulNoRelin`, `Relinearize`, hoisted `RotateBatch`). These bindings are also used by Orion's existing inference path. This skill confirms LoLA / MLP / ResNet still infer correctly after binding changes.

## What's already in the repo

```
examples/
  run_lola.py       run_mlp.py       run_resnet.py
  results/
    lola.txt        # desilo LoLA reference: MAE 0.0000, Precision 22.7654, ~51s
    mlp.txt
    resnet_f.txt    resnet4.txt
    bsgs-fixed/                  # historical: ResNet variants pre-bsgs-fix
    desilo-bootstrap-cleanup/    # historical: lola/mlp/resnet post desilo bootstrap work
    desilo-native/               # historical: resnet on native desilo multiply_matrix
    naive/                       # historical: desilo vs lattigo resnet baseline

configs/
  lola.yml   lola_desilo.yml
  mlp.yml    mlp_desilo.yml
  resnet.yml resnet_desilo.yml
```

The `<model>.yml` configs are lattigo by default; `<model>_desilo.yml` swaps in the desilo backend.

## Run

From repo root:

```bash
cd examples

# Per model, per backend — capture output for diffing
python run_lola.py   ../configs/lola.yml         > /tmp/lola_lattigo.txt   2>&1
python run_lola.py   ../configs/lola_desilo.yml  > /tmp/lola_desilo.txt    2>&1
python run_mlp.py    ../configs/mlp.yml          > /tmp/mlp_lattigo.txt    2>&1
python run_mlp.py    ../configs/mlp_desilo.yml   > /tmp/mlp_desilo.txt     2>&1
python run_resnet.py ../configs/resnet.yml       > /tmp/resnet_lattigo.txt 2>&1
python run_resnet.py ../configs/resnet_desilo.yml > /tmp/resnet_desilo.txt 2>&1
```

ResNet is the slow one (multi-minute on lattigo CPU; faster on desilo GPU). LoLA + MLP each run in under a minute on either backend.

## What to compare

The reference transcripts end with three canary lines:

```
MAE: 0.0000
Precision: 22.7654
Runtime: 51.3872 secs.
```

**Comparison rules**:
- **MAE / Precision must be stable**: Precision within ~1 bit of the reference (e.g. 21.5–23.5 if reference is 22.76). MAE within ~2× of reference.
- **Runtime is informational, not a regression signal** — varies with machine load, GPU contention. Worth noting if it changes by >2× (could indicate hoisting regression).
- **Per-layer packing diagnostics** (`├── # output rotations`, `├── # diagonals`) should be **identical** to the reference — these are deterministic given the config. A change here means the matmul packing strategy shifted; investigate before accepting.

Suggested diff workflow:

```bash
# Strip volatile lines (runtime, memory deltas) before diffing
grep -v -E "Runtime|secs|MB|^\s*$" examples/results/lola.txt | sort > /tmp/lola_ref.cmp
grep -v -E "Runtime|secs|MB|^\s*$" /tmp/lola_desilo.txt       | sort > /tmp/lola_new.cmp
diff /tmp/lola_ref.cmp /tmp/lola_new.cmp
```

A clean diff for LoLA + MLP + ResNet on both backends = the binding work is non-regressive.

## When to use this skill

- **After any change to `orion/backend/desilo/bindings.py`** — the binding work for [[project-desilo-optimization-gaps]] could break existing inference.
- **After any change to `orion/backend/lattigo/`** for parity (if we ever touch the lattigo wrapper to mirror the desilo binding additions).
- **Before tagging a commit as "ready for paper benchmark"** — pair with `/oracle-test-mm-encodings`.
- **Not after pure kernel work in `benchmarks/matmul_encodings/`** — that's isolated from the inference path; oracle tests cover it.

## Failure triage

| Symptom | Likely cause |
|---|---|
| Precision dropped >2 bits on one backend | binding bug in that backend (lazy relin not equivalent to eager? rotate semantics flipped?) |
| MAE became NaN/inf | rescale or level-management bug; check `# output rotations` and per-layer levels in the transcript |
| Per-layer rotation/diagonal counts changed | matmul packing path took a different branch — check if BSGS path selection changed |
| Only desilo failing | binding work issue; rerun `/oracle-test-mm-encodings` to isolate which op |
| Both failing identically | upstream issue in shared scheme/network code, not the binding |

When triaging, the historical transcripts under `examples/results/<subdir>/` are useful as "this is what it looked like at known-good commit X" anchors.
