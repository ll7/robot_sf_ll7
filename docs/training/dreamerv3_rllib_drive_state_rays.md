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

## 2) Canonical Command

Use this as the default launch command:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_auxme_a30_full.yaml
```

Why this command is canonical:

- pins Ray workers to the same Python executable as the driver (`runtime_env.py_executable`)
- disables `uv run` runtime-env propagation in Ray (`ray.disable_uv_run_runtime_env: true`)
- uses curated runtime package excludes to avoid large uploads (`ray.runtime_env.excludes`)

Use `configs/training/rllib_dreamerv3/drive_state_rays.yaml` for local smoke runs.

## 3) Config Notes

Both Dreamer configs include these reliability settings:

- `ray.disable_uv_run_runtime_env: true`
- `ray.runtime_env.working_dir: .`
- `ray.runtime_env.excludes: [...]` (e.g. `.git`, `.venv`, `output`, caches, large media)
- action/observation contract fixes for float32-compatible env outputs

These prevent worker-side uv rebuild loops and reduce startup package overhead.

## 4) Dry-run Validation

Use dry-run to validate YAML parsing and resolved settings:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays.yaml \
  --dry-run
```

## 5) Launch Patterns (Auxme)

### 5.1 Interactive allocation + tmux

```bash
srun -p <partition> --gres=gpu:a30:1 --cpus-per-task=24 --mem=64G --time=24:00:00 --pty bash
tmux new -s dreamer
cd /path/to/robot_sf_ll7
source .venv/bin/activate
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_auxme_a30_full.yaml
```

Detach/reattach:

```bash
tmux detach
tmux ls
tmux attach -t dreamer
```

### 5.2 Batch job

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH -p <partition>
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -J dreamer-rllib
set -euo pipefail
cd /path/to/robot_sf_ll7
source .venv/bin/activate
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_auxme_a30_full.yaml
EOF
```

## 6) Monitoring Checklist

### Scheduler/process state

```bash
squeue -u "$USER"
sstat -j <jobid>.batch --format=JobID,MaxRSS,AveCPU
ps -fu "$USER" | rg train_dreamerv3_rllib.py
```

### In-run progress (inside run dir)

```bash
RUN_DIR=output/dreamerv3/<run_id>_<timestamp>
tail -f "$RUN_DIR/result.json"
cat "$RUN_DIR/run_summary.json"
ls -lah "$RUN_DIR/checkpoints"
```

### W&B

- online mode (`*_auxme_a30_full.yaml`): check project dashboard live
- offline mode (`drive_state_rays.yaml`): sync later with:

```bash
wandb sync output/wandb/wandb/offline-run-*
```

## 7) Recovery Playbook

### Case A: allocation alive, training process gone

1. Reattach to tmux (`tmux attach -t dreamer`) or inspect node logs.
2. Confirm no active trainer process.
3. Relaunch canonical command with same config.
4. Verify a new `output/dreamerv3/<run_id>_<timestamp>/` appears.

### Case B: no direct SSH to compute node

Use scheduler attach methods:

```bash
srun --jobid <jobid> --pty bash
# or
sattach <jobid>.0
```

### Case C: startup warnings/failures

- Worker env mismatch warnings: ensure `ray.disable_uv_run_runtime_env: true` is present.
- Large package upload warnings: verify `ray.runtime_env.excludes` in YAML.
- If `result.json` stops updating for prolonged time, treat run as stalled and restart.

## 8) Outputs

Artifacts are written under:

`output/dreamerv3/<run_id>_<timestamp>/`

Key files:

- `checkpoints/` (periodic + final RLlib checkpoints)
- `result.json` (JSONL per-iteration progress stream for live monitoring)
- `run_summary.json` (iteration history and run metadata)
