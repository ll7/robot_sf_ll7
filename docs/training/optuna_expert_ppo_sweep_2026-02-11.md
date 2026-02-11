# Optuna Expert PPO Sweep Report (2026-02-11)

[← Back to Documentation Index](../README.md)

## Overview

This report summarizes the Optuna sweep executed with the launcher config in
`configs/training/ppo_imitation/optuna_expert_ppo.yaml` . The study optimizes
`eval_episode_return` using the `last_n_mean` objective mode (window = 3).

## Study Configuration

* Study name: `weekend_optuna_expert_ppo_last_n_mean`
* Storage: `output/optuna/weekend_optuna_expert_ppo.db`
* Trials requested: 8 per launcher run (28 total trials recorded)
* Trial timesteps: 15, 000, 000
* Eval cadence: every 500, 000 steps
* Eval episodes per checkpoint: 8
* Objective: `eval_episode_return` (maximize)
* Objective mode: `last_n_mean` (window = 3)

Command used for inspection:

```bash
uv run python scripts/tools/inspect_optuna_db.py \
  --db output/optuna/weekend_optuna_expert_ppo.db \
  --show-params --top-n 8 \
  --export-csv output/optuna/weekend_optuna_expert_ppo_trials.csv
```

## Summary

* Trials recorded: 28
* Completed: 25
* Failed: 3
* Best objective value: 0.65
* Median objective value (completed): 0.15

Results are exported to:
* `output/optuna/weekend_optuna_expert_ppo.db`
* `output/optuna/weekend_optuna_expert_ppo_trials.csv`
* `output/optuna/weekend_optuna_expert_ppo_summary.md`

## Best Candidates (Top 5)

| Trial | Value | Policy ID | Hyperparameters |
| --- | --- | --- | --- |
| 12 | 0.65 | ppo_expert_reference_optuna_012 | batch_size=128, n_steps=4096, n_epochs=8, lr=1.04e-05, ent_coef=0.0183, clip_range=0.274, target_kl=0.0296, gamma=0.99, gae_lambda=0.9 |
| 6 | 0.584283 | ppo_expert_reference_optuna_006 | batch_size=128, n_steps=4096, n_epochs=4, lr=2.29e-05, ent_coef=0.0109, clip_range=0.225, target_kl=0.0245, gamma=0.99, gae_lambda=0.98 |
| 20 | 0.529486 | ppo_expert_reference_optuna_020 | batch_size=256, n_steps=4096, n_epochs=8, lr=2.24e-05, ent_coef=0.000489, clip_range=0.242, target_kl=0.0258, gamma=0.98, gae_lambda=0.95 |
| 24 | 0.400775 | ppo_expert_reference_optuna_024 | batch_size=256, n_steps=2048, n_epochs=8, lr=1.38e-05, ent_coef=0.000361, clip_range=0.258, target_kl=0.0249, gamma=0.98, gae_lambda=0.95 |
| 7 | 0.316667 | ppo_expert_reference_optuna_007 | batch_size=128, n_steps=1024, n_epochs=8, lr=1.16e-05, ent_coef=0.0195, clip_range=0.171, target_kl=0.0253, gamma=0.99, gae_lambda=0.9 |

## Observations

* High-performing trials consistently favored larger `n_steps` (2048-4096).
* Lower learning rates (around 1e-05 to 2.3e-05) dominated the top candidates.
* `clip_range` between 0.22 and 0.28 appeared in top trials.
* `target_kl` clustered around 0.024 to 0.030.
* `ent_coef` showed two regimes: very small (~3e-4 to 5e-4) and moderate (~0.01 to 0.02).

## PPO Hyperparameter Guidelines (Expert PPO)

Use these ranges as a starting point for follow-up sweeps or fixed training runs:

* `n_steps`: 2048 to 4096 (prefer 4096 for stability)
* `batch_size`: 128 or 256 (cap by num_envs * 512)
* `n_epochs`: 4 to 8 (higher values paired with smaller learning rates)
* `learning_rate`: 1e-05 to 2.5e-05
* `ent_coef`: 3e-4 to 2e-2 (tune in two bands; avoid mid-range 1e-3 to 5e-3 if unstable)
* `clip_range`: 0.22 to 0.28
* `target_kl`: 0.024 to 0.030
* `gamma`: 0.98 to 0.99
* `gae_lambda`: 0.90 to 0.98

## Recommended Next Runs

1. Validate trial 12 as a full-length expert run using the recorded hyperparameters.
2. Re-run trials 6 and 20 with identical seeds to confirm stability.
3. Narrow the Optuna search space to the ranges above to reduce variance and speed convergence.
