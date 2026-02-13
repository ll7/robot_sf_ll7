# Issue 403 Grid PPO Training Runbook

This runbook is the **reproducible, publication-ready** training procedure for the
grid+SocNav PPO expert (Issue 403). It is designed to be usable by another lab
without additional context.

## 1) Prerequisites

- Python 3.11+ (project uses UV-managed env).
- Dependencies installed:
  ```bash
  uv sync --all-extras
  source .venv/bin/activate
  ```
- Stable-Baselines3 installed (included in core dependencies).
- Artifact root (recommended):
  ```bash
  export ROBOT_SF_ARTIFACT_ROOT=$PWD/output
  ```
- Headless runtime (recommended for training servers):
  ```bash
  export SDL_VIDEODRIVER=dummy
  ```
- CUDA determinism (required on GPU with deterministic algorithms):
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```

## 2) Configuration (authoritative)

**Primary config file:**
`configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml`

Key settings:
- **Scenario matrix**: `configs/scenarios/classic_interactions_francis2023.yaml`
- **Observation contract**: SocNav structured + occupancy grid
- **Grid**: 0.5m resolution, 32×32m, ego frame, 64×64 cells
- **Tracking**: TensorBoard + W&B (disable W&B offline)
- **Eval cadence**: 0.5M steps until 3M, then 1M steps

Important: the config uses `env_overrides` to enforce:
```yaml
observation_mode: socnav_struct
use_occupancy_grid: true
include_grid_in_observation: true
grid_config:
  resolution: 0.5
  width: 32.0
  height: 32.0
  channels: [obstacles, pedestrians, combined]
  use_ego_frame: true
  center_on_robot: true
peds_have_static_obstacle_forces: true
peds_have_robot_repulsion: true
sim_config:
  prf_config:
    is_active: true
```
Note: `peds_have_obstacle_forces` is deprecated and remains as a legacy alias for
static obstacle forces.
This ensures the policy is actually trained on the grid+SocNav observation space.

## 3) Smoke run (fast validation)

This validates the full pipeline without heavy training.
```bash
uv run python scripts/training/train_expert_ppo.py \
  --config configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml \
  --dry-run
```
Expected artifacts (under `output/`):
- `benchmarks/expert_policies/` (config + placeholder checkpoint)
- `benchmarks/ppo_imitation/episodes/`
- `benchmarks/ppo_imitation/runs/`

## 4) Full training run (10M steps)

```bash
uv run python scripts/training/train_expert_ppo.py \
  --config configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml
```

### CPU scaling (automatic)
By default, training uses **one environment per CPU core minus one**. This is controlled by:
```yaml
num_envs: auto
worker_mode: auto
```
You can override it:
```yaml
num_envs: 8
worker_mode: subproc   # or dummy
```

By default this:
- Trains across multiple scenarios via ScenarioSwitchingEnv.
- Saves checkpoints at each evaluation step to:
  `output/benchmarks/expert_policies/checkpoints/<policy_id>/`
- Writes manifests to:
  `output/benchmarks/expert_policies/` and `output/benchmarks/ppo_imitation/runs/`

## 5) Monitoring

**TensorBoard:**
```bash
tensorboard --logdir output/benchmarks/ppo_imitation/<run_id>/tensorboard
```

**W&B:**
Set offline if needed:
```bash
export WANDB_MODE=offline
```

## 6) Expected outputs (must archive)

- Training run manifest: `output/benchmarks/ppo_imitation/runs/*.json`
- Expert policy manifest: `output/benchmarks/expert_policies/*.json`
- Episode logs: `output/benchmarks/ppo_imitation/episodes/*.jsonl`
- Checkpoints: `output/benchmarks/expert_policies/checkpoints/<policy_id>/`

## 7) Troubleshooting

- **Wrong observation shape**: confirm `env_overrides` in the YAML and that
  `observation_mode` is `socnav_struct` + `include_grid_in_observation: true`.
- **No grid channels**: ensure `grid_config.channels` is non-empty and valid.
- **W&B login errors**: `export WANDB_MODE=offline` or set `tracking.wandb.enabled: false`.

## 8) Slurm usage

Use the provided job script:
`SLURM/issue_403_train_expert_ppo_grid.sl`

Submit:
```bash
sbatch SLURM/issue_403_train_expert_ppo_grid.sl
```
