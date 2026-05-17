# BMM-3 hoisted-mode OOM check — running

## Why

Session 2 paper-preset GPU sweep (cached mode, RTX 3090, 24 GB):

| shape | CPU | GPU (cached) |
|---|---|---|
| (128, 131, 129) | 8.35 s | 12.43 s (fit) |
| **(256, 259, 257)** | 47.24 s | **OOM** ← first OOM |
| (512, 515, 513) | 303.65 s | **OOM** |
| (1024, 1027, 1025) | 1896.83 s | **OOM** |

Session 3 added a hoisted mode (matches Negar's paper preset
`Bmm3ModeHoisted`) with per-block `ctx.free()` so peak GPU memory grows
like `O(hoist_block_size)` not `O(m)`. Smoke at (64, 67, 65) already
showed hoisted uses **less** memory than cached (4396 MB vs 6194 MB
peak HBM delta) and is **3.4× faster wall-clock**.

This run answers the **direct** question: *does the new hoisted mode
unblock the shapes that previously OOM'd?*

## What's running

Script: [`scripts/bmm3_oom_check.py`](scripts/bmm3_oom_check.py)
Tmux session name: `bmm3-oom-check`
Log file: `/tmp/bmm3_oom_check.log` (live tail)
JSON results: [`results/bmm3_oom_check.json`](results/bmm3_oom_check.json) (flushed after every iteration)

**Test plan**: for each shape, sweep `hoist_block_size` smallest-first;
stop the shape's sweep at first OOM (larger blocks would only use more
memory).

| shape | block sizes tested |
|---|---|
| (256, 259, 257) | 4, 8, 16, 32 |
| (512, 515, 513) | 4, 8, 16 |
| (1024, 1027, 1025) | 4, 8 |

If the smallest block size for one shape OOMs, the script abandons all
larger shapes too (cliff guaranteed).

Fresh Context built per iteration so prior runs' state can't push later
ones into OOM. CKKS preset is `negar` (LogN=13 Standard, 4096 slots) so
results compare apples-to-apples with the session-2 CPU baseline.

## How to check status

```bash
# attach to the running session (Ctrl+B then D to detach)
tmux attach -t bmm3-oom-check

# OR just read the live log without attaching
tail -f /tmp/bmm3_oom_check.log

# OR see the JSON results file (incrementally flushed)
cat /home/avm6288/orion/benchmarks/matmul_encodings/results/bmm3_oom_check.json | python -m json.tool
```

`completed: true` in the JSON means the sweep finished (success or
hit-OOM-cliff). Per-row `success` is the per-(shape, block) pass/fail.

## Expected runtime

Rough estimates (single trial, no warmup; scales linearly with kernel
work):

| shape | per-block runtime | total time for that shape |
|---|---|---|
| (256, 259, 257) | ~30–60 s | ~3 min (4 block sizes) |
| (512, 515, 513) | ~250–300 s | ~15 min (3 block sizes) |
| (1024, 1027, 1025) | ~30–40 min | ~1 hour (2 block sizes) |
| **total worst case** | | **~1.5 hours** |

Hour budget includes `ctx` build (~30 s) per iteration. Could be much
shorter if early shapes OOM and the script bails early.

## Interpretation

- **Best case**: every row in JSON has `success: true`. Hoisted mode
  fixed the OOM cliff for every previously-failing shape on this
  hardware. Paper figure can include all of them.
- **Partial fix**: some shapes pass, larger ones still OOM. Paper
  reports which shapes fit on a 3090 with hoisted mode; bigger ones
  defer to the bigger GPU's data.
- **No improvement**: hoisted at smallest block size (4) still OOMs
  at (256, 259, 257). Means the OOM bottleneck wasn't the per-block
  rotation memory — it's elsewhere (likely the chunk count × `ctx`
  base state). The next move would be to profile peak memory and look
  for other improvable layers (mask cache size? plaintext encode
  count?).

## Headed back to this?

If the JSON shows `completed: true` and you want to delete the tmux
session:

```bash
tmux kill-session -t bmm3-oom-check
```

If you want to re-run with different shapes / block sizes, edit
`TEST_PLAN` in the script and relaunch:

```bash
tmux new-session -d -s bmm3-oom-check 'cd /home/avm6288/orion && \
  /home/avm6288/miniconda3/envs/myenv2/bin/python -u \
  -m benchmarks.matmul_encodings.scripts.bmm3_oom_check \
  2>&1 | tee /tmp/bmm3_oom_check.log'
```
