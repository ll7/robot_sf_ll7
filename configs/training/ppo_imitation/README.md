# PPO Imitation Pipeline Configurations

This directory hosts configuration files used by the expert-policy, trajectory collection, and imitation pre-training workflows. Each config captures the exact seed lists, scenario setups, and hyperparameters required for reproducible runs.

## Layout

- `expert_ppo.yaml` – Expert training defaults, including convergence thresholds and evaluation schedule.
- `bc_pretrain.yaml` – Behavioural cloning phase settings (optimizer, batch sizes, dataset references).
- `ppo_finetune.yaml` – PPO fine-tuning schedule that resumes from a pre-trained checkpoint.
- Additional files describe scenario coverage manifests referenced by trajectory collection commands.

## Usage Notes

1. Always version-control changes alongside the corresponding manifests under `output/`.
2. Keep seed lists consistent across configs when running comparative studies.
3. Document any temporary overrides in the run manifest to preserve reproducibility.
