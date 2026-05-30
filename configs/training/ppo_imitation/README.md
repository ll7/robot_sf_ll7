# PPO Imitation Pipeline Configurations

This directory hosts configuration files used by the expert-policy, trajectory collection, and imitation pre-training workflows. Each config captures the exact seed lists, scenario setups, and hyperparameters required for reproducible runs.

## Layout

* `expert_ppo.yaml` – Expert training defaults, including convergence thresholds and evaluation schedule.
* `expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml` – BR-06 sanity pass (0.5M steps) before full retraining.
* `expert_ppo_issue_576_br06_v2_15m_all_maps.yaml` – BR-06 retraining v2 run plan (15M steps, all classic + Francis maps, route-completion v2 reward, full tracking).
* `optuna_expert_ppo.yaml` – Config-first launcher settings for Optuna sweeps on top of `expert_ppo.yaml`.
* `bc_pretrain.yaml` – Behavioural cloning phase settings (optimizer, batch sizes, dataset references).
* `ppo_finetune.yaml` – PPO fine-tuning schedule that resumes from a pre-trained checkpoint.
* `bc_pretrain_issue_749_v10_warm_start.yaml` and
  `ppo_finetune_issue_749_v10_warm_start.yaml` – Issue #749 BC warm-start PPO challenger launch
  packet; see `docs/context/issue_749_bc_preinit_ppo_launch_packet.md` for the dataset collection,
  artifact-persistence, and evaluation boundary.
* `oracle_dataset_issue_1397_launch_packet.yaml` – pre-Slurm oracle-imitation dataset launch
  packet. Validate with
  `uv run python scripts/validation/validate_oracle_imitation_launch_packet.py --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json`.
* Additional files describe scenario coverage manifests referenced by trajectory collection commands.

## Usage Notes

1. Always version-control changes alongside the corresponding manifests under `output/`.
2. Keep seed lists consistent across configs when running comparative studies.
3. Document any temporary overrides in the run manifest to preserve reproducibility.
4. For Optuna launcher configs, prefer `log_level: WARNING` by default and override only when debugging.
5. W&B is enabled by default for Optuna launcher runs; set `disable_wandb: true` only for offline/reliability scenarios.
6. Safety-gated Optuna sweeps can set `constraint_collision_rate_max` (and optional
`constraint_comfort_exposure_max` ) plus `constraint_handling: penalize|prune` .
7. Optuna launcher configs use `schema_version: robot_sf.optuna_expert_ppo_launcher.v1`.
Unsupported versions fail at `scripts/training/launch_optuna_expert_ppo.py --config <path>`
before any trials are launched.
