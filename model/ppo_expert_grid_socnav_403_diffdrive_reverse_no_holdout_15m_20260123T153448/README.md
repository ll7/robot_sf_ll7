# PPO Expert (DiffDrive + Reverse, No Holdout, 15M)

## Artifact
- `model.zip` (trained policy)
- Source run: `output/wandb/ppo_expert_grid_socnav_403_diffdrive_reverse_no_holdout_15m_20260123T153448/model.zip`
- Training run ID (W&B): `run-20260123_163502-jx7e1sjx`
- Git commit: `9cce22c89411c0a8ae52051dfa28a0a4055f3f07`

## Training config
- Config file: `configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m.yaml`
- Scenario suite: `configs/scenarios/classic_interactions_francis2023.yaml`
- Total timesteps: **15,000,000**
- Seeds: 123, 231, 777, 992, 1337

### PPO hyperparameters (from W&B config)
- `learning_rate`: 1e-4
- `batch_size`: 256
- `n_steps`: 2048
- `n_epochs`: 4
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_range`: 0.1
- `ent_coef`: 0.01
- `max_grad_norm`: 0.5
- `target_kl`: 0.02
- `n_envs`: 31

## Observation setup
- `observation_mode`: `socnav_struct`
- Occupancy grid enabled and included in observation
- Grid config:
  - `resolution`: 0.2 m
  - `width`/`height`: 32 m x 32 m
  - `channels`: obstacles, pedestrians, combined
  - `use_ego_frame`: true
  - `center_on_robot`: true

## Robot setup
- `type`: differential_drive
- `max_linear_speed`: 3.0
- `max_angular_speed`: 1.0
- `allow_backwards`: true

## Evaluation setup
- Holdout scenarios: **disabled** (`hold_out_scenarios: []`)
- Evaluation episodes: 10
- Note: evaluation is **in-distribution** (training scenarios), so it is a
  **solvability check** rather than a generalization test.

## Final metrics (from W&B summary)
- `eval/success_rate`: **1.0**
- `eval/collision_rate`: **0.0**
- `eval/path_efficiency`: **0.601**
- `eval/snqi`: **1.0**
- `rollout/success_rate`: **0.91**

## Model architecture (from W&B config)
- **Feature extractor:** `GridSocNavExtractor`
  - Grid CNN:
    - Conv2d(3 -> 32, kernel=5, stride=2, padding=2) + ReLU + Dropout(0.1)
    - Conv2d(32 -> 64, kernel=3, stride=2, padding=1) + ReLU + Dropout(0.1)
    - Conv2d(64 -> 64, kernel=3, stride=2, padding=1) + ReLU + Dropout(0.1)
    - Flatten (grid feature dimension ~= 25,600 for 3x160x160)
  - SocNav MLP:
    - Linear(273 -> 128) + ReLU + Dropout(0.1)
    - Linear(128 -> 128) + ReLU + Dropout(0.1)
  - Combined features: 25,728
- **Policy/value heads (SB3 MlpExtractor):**
  - Policy net: Linear(25,728 -> 256) -> Tanh -> Linear(256 -> 256) -> Tanh
  - Value net: Linear(25,728 -> 256) -> Tanh -> Linear(256 -> 256) -> Tanh
  - Action head: Linear(256 -> 2)
  - Value head: Linear(256 -> 1)

## What worked well
- Achieved perfect success on the training distribution with no collisions.
- Stable PPO training dynamics (approx_kl ~0.016, clip_fraction ~0.34).

## Limitations / risks
- **No holdout**: generalization is unknown and likely lower than in-distribution performance.
- **Eval episodes = 10**: metrics may be noisy; increase to 30–50 for more stability.
- **Path efficiency dropped** relative to the 10M holdout run (0.601 vs ~0.65),
  suggesting less efficient trajectories despite perfect success.

## Suggested improvements / next steps
- Re-enable holdout scenarios to measure generalization.
- Increase evaluation episodes for more stable metrics.
- Add scenario diversity (maps, densities, geometry variants) to reduce overfitting.
- Consider modest hyperparameter tuning only after scenario diversity is addressed.

## Registry usage

This model can be referenced from `model/registry.yaml`. Entries are intended to be
auto-populated.

Example auto-population snippet:

```python
from robot_sf.models import upsert_registry_entry

upsert_registry_entry(
    {
        "model_id": "ppo_expert_grid_socnav_403_diffdrive_reverse_no_holdout_15m_20260123T153448",
        "display_name": "PPO diffdrive reverse no-holdout 15M (2026-01-23)",
        "local_path": (
            "model/ppo_expert_grid_socnav_403_diffdrive_reverse_no_holdout_15m_20260123T153448/model.zip"
        ),
        "config_path": (
            "configs/training/ppo_imitation/"
            "expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m.yaml"
        ),
        "commit": "9cce22c89411c0a8ae52051dfa28a0a4055f3f07",
        "wandb_run_id": "jx7e1sjx",
        "wandb_run_path": None,
        "wandb_entity": None,
        "wandb_project": None,
        "wandb_file": "model.zip",
        "tags": ["ppo", "socnav", "diffdrive", "reverse", "no-holdout", "grid-obs"],
        "notes": ["Fill wandb_run_path or wandb_entity/project for auto-download."],
    }
)
```

Programmatic load (with auto-download if configured):

```python
from robot_sf.models import resolve_model_path

path = resolve_model_path(
    "ppo_expert_grid_socnav_403_diffdrive_reverse_no_holdout_15m_20260123T153448",
    allow_download=True,
)
```
