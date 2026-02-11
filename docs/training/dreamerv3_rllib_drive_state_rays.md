# DreamerV3 RLlib Runbook (`drive_state` + `rays`)

This runbook documents a reproducible RLlib DreamerV3 workflow for Robot SF using
the default non-image observation contract (`drive_state`, `rays`).

## 1) Prerequisites

Install dependencies and include RLlib:

```bash
uv sync --extra rllib
source .venv/bin/activate
```

Note: the imitation pipeline now lives in a separate dependency group (`--group imitation`).

Headless execution is recommended for training servers:

```bash
export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
```

## 2) Authoritative Config

Primary config:

`configs/training/rllib_dreamerv3/drive_state_rays.yaml`

Important defaults:

- `use_image_obs: false`
- `include_grid_in_observation: false`
- deterministic flattening order: `drive_state` then `rays`
- action-space normalization to `[-1, 1]` before DreamerV3
- starter tuning for Apple Silicon laptops: `num_env_runners=8`, `training_ratio=64`,
  `batch_size_B=16`, `batch_length_T=32`, `horizon_H=10`
- W&B tracking enabled in offline mode by default (`tracking.wandb.mode: offline`)

## 3) Dry-run Validation

Use dry-run to validate YAML parsing and resolved settings:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --dry-run
```

## 4) Training Command

Run DreamerV3 with the config-first entrypoint:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml
```

Recommended command for the starter profile (reduced console noise):

```bash
PYTHONWARNINGS=ignore LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 \
.venv/bin/python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --log-level WARNING
```

This run writes W&B logs locally under `output/wandb/`. To sync later:

```bash
wandb sync output/wandb/wandb/offline-run-*
```

Optional overrides:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --run-id dreamerv3_smoke \
  --train-iterations 5 \
  --checkpoint-every 1 \
  --log-level WARNING
```

## 5) Outputs

Artifacts are written under:

`output/dreamerv3/<run_id>_<timestamp>/`

Key files:

- `checkpoints/` (periodic + final RLlib checkpoints)
- `run_summary.json` (iteration history and run metadata)
