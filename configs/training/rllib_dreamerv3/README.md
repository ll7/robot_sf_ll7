# RLlib DreamerV3 Training Configurations

This directory contains config-first launcher files for RLlib DreamerV3 training workflows.

## Default workflow

- `drive_state_rays.yaml`: Minimal DreamerV3 run on the default Robot SF observation contract
  (`drive_state` + `rays`) without image observations. The current defaults are tuned as a
  fast starter profile for Apple Silicon laptops (for example MacBook Pro M4 Pro):
  `model_size=XS`, `training_ratio=64`, `num_env_runners=8`, `batch_length_T=32`.
- W&B tracking defaults to enabled in offline mode (`tracking.wandb.mode: offline`) so each
  run records metrics locally under `output/wandb/` without requiring immediate login.

## Canonical command

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml
```

## Recommended command (starter profile)

```bash
PYTHONWARNINGS=ignore LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
.venv/bin/python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --log-level WARNING
```

To push offline runs to W&B later:

```bash
wandb sync output/wandb/wandb/offline-run-*
```
