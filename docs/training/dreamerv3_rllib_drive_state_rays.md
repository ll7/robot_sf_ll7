# DreamerV3 RLlib Runbook ( `drive_state` + `rays` )

This runbook documents a reproducible RLlib DreamerV3 workflow for Robot SF using
the default non-image observation contract ( `drive_state` , `rays` ).

Current scope note:

- BR-08 prep is reproducible and benchmark-reset-v2 aligned.
- The current launcher now supports scenario-matrix-driven training through
  `env.scenario_matrix`, using the same YAML scenario manifests as the PPO benchmark-reset runs.
- Dreamer still remains a challenger track until benchmark evidence shows it beats the current PPO
  champion.

## 1) Prerequisites

Install dependencies and include RLlib:

```bash
uv sync --extra rllib
source .venv/bin/activate
```

Note: the imitation pipeline now lives in a separate dependency group ( `--group imitation` ).

Headless execution is recommended for training servers:

```bash
export DISPLAY=
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
```

## 2) Canonical Command

Use this wrapper as the default full launch command:

```bash
scripts/training/run_dreamerv3_br08.sh full
```

Direct launcher equivalent:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_br08_full.yaml
```

Why this command is canonical:

* pins Ray workers to the same Python executable as the driver (`runtime_env.py_executable`)
* disables `uv run` runtime-env propagation in Ray (`ray.disable_uv_run_runtime_env: true`)
* uses curated runtime package excludes to avoid large uploads (`ray.runtime_env.excludes`)

Use `gate` for the short validation run:

```bash
scripts/training/run_dreamerv3_br08.sh gate
```

Direct launcher equivalent:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/drive_state_rays_br08_gate.yaml
```

## 3) Config Notes

Both BR-08 Dreamer configs include these reliability settings:

- `ray.disable_uv_run_runtime_env: true`
- `ray.runtime_env.working_dir: .`
- `ray.runtime_env.excludes: [...]` (e.g. `.git`, `.venv`, `output`, caches, large media)
- action/observation contract fixes for float32-compatible env outputs
- `env.factory_kwargs.reward_name: route_completion_v3`
- success-priority reward weights aligned with the current benchmark-reset PPO runs
- `env.scenario_matrix.path: configs/scenarios/classic_interactions_francis2023.yaml`
- randomized seed semantics from `env.config.randomize_seeds: true`

These prevent worker-side uv rebuild loops and reduce startup package overhead.

## 4) Dry-run Validation

Use dry-run to validate YAML parsing and resolved settings:

```bash
scripts/training/run_dreamerv3_br08.sh gate --dry-run
```

## 5) Launch Patterns (Auxme)

### 5.1 Interactive allocation + tmux

```bash
srun -p <partition> --gres=gpu:a30:1 --cpus-per-task=24 --mem=64G --time=24:00:00 --pty bash
tmux new -s dreamer
cd /path/to/robot_sf_ll7
source .venv/bin/activate
scripts/training/run_dreamerv3_br08.sh full
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
scripts/training/run_dreamerv3_br08.sh full
EOF
```

## 5.3 BR-08 evaluation contract

DreamerV3 promotion should follow the same success-first philosophy as PPO:

1. train the gate/full run,
2. benchmark the best exported checkpoint,
3. compare against the current promoted PPO champion on success, collision split,
   timeout rate, and then SNQI, 
4. only promote if the full benchmark result is a genuine improvement.

Current limitation:

- Scenario-matrix parity now covers the training env source, but Dreamer still uses its own RLlib
  env-runner topology rather than the PPO `SubprocVecEnv` stack. Compare results at the benchmark
  level, not by assuming identical rollout dynamics.

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
tail -f "$RUN_DIR/result.jsonl"
cat "$RUN_DIR/run_summary.json"
ls -lah "$RUN_DIR/checkpoints"
```

### W&B

* online mode (`*_auxme_a30_full.yaml`): check project dashboard live
* offline mode (`drive_state_rays.yaml`): sync later with:

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

* Worker env mismatch warnings: ensure `ray.disable_uv_run_runtime_env: true` is present.
* Large package upload warnings: verify `ray.runtime_env.excludes` in YAML.
* If `result.jsonl` stops updating for prolonged time, treat run as stalled and restart.

## 8) Outputs

Artifacts are written under:

 `output/dreamerv3/<run_id>_<timestamp>/`

Key files:

- `checkpoints/` (periodic + final RLlib checkpoints)
- `result.jsonl` (JSONL per-iteration progress stream for live monitoring)
- `run_summary.json` (iteration history and run metadata)
- `run_summary.json` also records:
  - `config_path`
  - `scenario_matrix_path`
  - `scenario_matrix_strategy`
  - `scenario_switch_per_reset`
  - `randomize_seeds`
- `run_meta.json` (repo/branch/commit, CLI, seed report, resource resolution, status)
- `resolved_config.json` (fully resolved Dreamer config captured with the run)
