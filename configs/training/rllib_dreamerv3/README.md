# RLlib DreamerV3 Training Configurations

This directory contains config-first launcher files for RLlib DreamerV3 training workflows.

## Default workflow

- `drive_state_rays.yaml`: Minimal DreamerV3 run on the default Robot SF observation contract
  (`drive_state` + `rays`) without image observations. The current defaults are tuned as a
  fast starter profile for Apple Silicon laptops (for example MacBook Pro M4 Pro):
  `model_size=XS`, `training_ratio=64`, `num_env_runners=8`, `batch_length_T=32`.
- W&B tracking defaults to enabled in offline mode (`tracking.wandb.mode: offline`) so each
  run records metrics locally under `output/wandb/` without requiring immediate login.
- Ray runtime env is hardened for reliability:
  - `disable_uv_run_runtime_env: true` avoids uv-driven worker env reconstruction.
  - `runtime_env.py_executable` defaults to the current interpreter in the launcher.
  - `runtime_env.excludes` trims heavy paths (e.g. `.git`, `.venv`, `output`) from uploads.

## Auxme A30 full-run profile

- `drive_state_rays_auxme_a30_full.yaml`: High-throughput profile tuned for Auxme `a30` runs
  with one A30 GPU and 24 CPUs. Compared to the starter profile it enables GPU learning
  (`resources.num_gpus=auto`), scales env runners (`num_env_runners=auto`), uses a larger model
  (`model_size=S`), and increases Dreamer sequence/training throughput (`training_ratio=128`,
  `batch_length_T=64`).
- This profile assumes W&B online tracking is available on compute nodes.
- The `auto` placeholders are resolved at runtime from Slurm/CUDA visibility
  (`SLURM_CPUS_PER_TASK`, `SLURM_GPUS*`, `CUDA_VISIBLE_DEVICES`) so the same config
  scales across different allocations.

## BR-08 balanced full-run profile

- `benchmark_socnav_grid_br08_full_balanced_gpu_r8.yaml`: Benchmark-aligned SocNav+grid profile
  for longer BR-08 runs when the default full profile is rollout-bound. It keeps a local learner
  on GPU (`num_learners=0`, `num_gpus_per_learner=1`), fixes env runners to `8`, lowers
  `training_ratio` to `64`, shortens `rollout_fragment_length` to `32`, and increases evaluation
  cadence to every `20` iterations to surface reward regressions earlier.

## Canonical command

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml
```

## Canonical command (Auxme A30 profile)

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_auxme_a30_full.yaml
```

## Recommended command (starter profile)

```bash
PYTHONWARNINGS=ignore LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --log-level WARNING
```

To push offline runs to W&B later:

```bash
wandb sync output/wandb/wandb/offline-run-*
```
