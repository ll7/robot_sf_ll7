# Issue #193 — SLURM Handoff: Feature Extractor Sweep

**Date:** 2026-04-16  
**Branch:** `codex/193-feature-extractor-evaluation`  
**Status:** Historical submission handoff.  The `feat_sweep_4m` run has completed; see
`docs/context/issue_193_feature_extractor_optuna_study.md` for the April 17 result analysis.

The original low-priority rerun on `l40s` was superseded before it produced useful results.  The
active sweep is now the split `a30` probe + delayed `pro6000` weekend batch documented in
`docs/context/issue_193_feature_extractor_optuna_study.md`.

---

## What happened locally

A 10-trial local smoke test ran on the dev machine (GPU: 7.65 GB VRAM).

- Trials 0–6 **completed successfully** at ~389 fps.
- Trial 7 (`lstm + arch_size=large + policy_arch_size=large`) **OOM'd** — this
  configuration needs ~4.2 GB of free GPU memory for the backward pass, which was
  unavailable after 6 prior trials had fragmented the allocator.
  Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.03 GiB.`

**The OOM is a dev-GPU constraint, not a code bug.**  
A100/V100 cluster nodes (40–80 GB) have ample headroom for all combinations.

**Fix applied since the debug run:**  
`study.optimize(..., catch=(Exception,))` now records a failed trial and continues
instead of crashing the whole process.  An OOM on one trial will not abort the sweep.

---

## Pre-flight checklist before submitting

```
[ ] git status is clean on branch codex/193-feature-extractor-evaluation
[ ] BASE_REF=origin/main scripts/dev/pr_ready_check.sh passes
[ ] Confirm cluster has SLURM, uv, and a GPU partition available
[ ] Read local.machine.md (if it exists at repo root) for cluster-specific limits
[ ] Optionally: run one dry-run trial on the cluster login node to confirm env loads
    uv run python scripts/training/optuna_feature_extractor.py \
        --config configs/training/ppo/feature_extractor_sweep_base.yaml \
        --trials 1 --trial-timesteps 32000 --disable-wandb --log-level INFO
```

---

## Submitting the full sweep

### Option A — shell launcher (recommended)

```bash
# From the repo root on the cluster login node:
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 20 \
    --timesteps 4000000 \
    --partition <YOUR_GPU_PARTITION> \
    --study-name feat_sweep_4m \
    --time 08:00:00 \
    --gpus 1 \
    --cpus 8 \
    --mem 32G
```

Replace `<YOUR_GPU_PARTITION>` with your cluster's GPU partition name (e.g., `gpu`, `a100`, `v100`).

This submits 20 independent jobs that each pick up one Optuna trial from a shared
SQLite database at `output/optuna/feat_extractor/feat_sweep_4m.db`.

**Dry-run first (prints sbatch commands without submitting):**
```bash
bash SLURM/submit_feature_extractor_sweep.sh --dry-run \
    --trials 5 --timesteps 4000000 --partition <YOUR_GPU_PARTITION>
```

### Option B — Python launcher with SLURM flag

```bash
uv run python scripts/training/optuna_feature_extractor.py \
    --config configs/training/ppo/feature_extractor_sweep_base.yaml \
    --trials 20 \
    --trial-timesteps 4000000 \
    --metric eval_episode_return \
    --study-name feat_sweep_4m \
    --storage sqlite:///output/optuna/feat_extractor/feat_sweep_4m.db \
    --slurm \
    --slurm-partition <YOUR_GPU_PARTITION> \
    --slurm-time 08:00:00 \
    --slurm-gpus 1 \
    --slurm-cpus 8 \
    --slurm-mem 32G \
    --fps-warn-threshold 100 \
    --disable-wandb
```

---

## Monitoring progress

```bash
# Check job queue
squeue --me

# Tail logs for a specific job
tail -f output/slurm/feat_sweep_<jobid>.out

# Inspect Optuna study (shows completed/failed trials and best so far)
uv run python scripts/tools/inspect_optuna_db.py \
    --db output/optuna/feat_extractor/feat_sweep_4m.db
```

---

## After the sweep completes

Run the FPS summary and best-trial report:

```bash
uv run python scripts/training/optuna_feature_extractor.py \
    --config configs/training/ppo/feature_extractor_sweep_base.yaml \
    --trials 0 \
    --storage sqlite:///output/optuna/feat_extractor/feat_sweep_4m.db \
    --study-name feat_sweep_4m \
    --disable-wandb
```

This prints:
- Per-extractor mean/min FPS with `[SLOW]` flags for candidates below 100 fps.
- Best trial: extractor type, arch size, policy arch, and metric value.

---

## Decision criteria for promotion to default extractor

A result from this 4 M-step sweep is a **candidate shortlist**, not final evidence.
Before adopting any extractor as the new default, it must:

1. Complete a **≥ 10 M step** training run on the canonical scenario config.
2. Achieve `success_rate ≥ 0.85` and `collision_rate ≤ 0.08` in a 100-episode hold-out eval.
3. Reach **≥ 80 % of DynamicsExtractor FPS** on the target hardware (so long runs stay tractable).
4. Pass the canonical benchmark suite without `snqi` or `path_efficiency` regression.

See [issue_193_feature_extractor_optuna_study.md](issue_193_feature_extractor_optuna_study.md)
for the full decision criteria and known risks (including the LSTM temporal-memory caveat).

---

## Known risk: LSTM large on small GPUs

If the cluster nodes have < 16 GB VRAM, `lstm + large + large` may still OOM.
Since `catch=(Exception,)` is now active, this will just be recorded as a failed
trial — the sweep will not abort.  To pre-empt it, exclude the large LSTM arch:

```bash
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 20 --timesteps 4000000 \
    --partition <YOUR_GPU_PARTITION> \
    --exclude lstm \
    --study-name feat_sweep_4m_nolstmlarge
```

Then run a separate smaller LSTM sweep if needed:
```bash
bash SLURM/submit_feature_extractor_sweep.sh \
    --trials 6 --timesteps 4000000 \
    --partition <YOUR_GPU_PARTITION> \
    --exclude dynamics,mlp,attention,lightweight_cnn \
    --study-name feat_sweep_lstm_only
```
