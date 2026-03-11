# PPO Imitation Pipeline Configurations

This directory hosts configuration files used by the expert-policy, trajectory collection, and imitation pre-training workflows. Each config captures the exact seed lists, scenario setups, and hyperparameters required for reproducible runs.

## Layout

* `expert_ppo.yaml` – Expert training defaults, including convergence thresholds and evaluation schedule.
* `expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml` – BR-06 sanity pass (0.5M steps) before full retraining.
* `expert_ppo_issue_576_br06_v2_15m_all_maps.yaml` – BR-06 retraining v2 run plan (15M steps, all classic + Francis maps, route-completion v2 reward, full tracking).
* `optuna_expert_ppo.yaml` – Config-first launcher settings for Optuna sweeps on top of `expert_ppo.yaml`.
* `bc_pretrain.yaml` – Behavioural cloning phase settings (optimizer, batch sizes, dataset references).
* `ppo_finetune.yaml` – PPO fine-tuning schedule that resumes from a pre-trained checkpoint.
* Additional files describe scenario coverage manifests referenced by trajectory collection commands.

## Usage Notes

1. Always version-control changes alongside the corresponding manifests under `output/`.
2. Keep seed lists consistent across configs when running comparative studies.
3. Document any temporary overrides in the run manifest to preserve reproducibility.
4. For Optuna launcher configs, prefer `log_level: WARNING` by default and override only when debugging.
5. W&B is enabled by default for Optuna launcher runs; set `disable_wandb: true` only for offline/reliability scenarios.
6. Safety-gated Optuna sweeps can set `constraint_collision_rate_max` (and optional
`constraint_comfort_exposure_max` ) plus `constraint_handling: penalize|prune` .
