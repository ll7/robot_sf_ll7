# PPO DiffDrive Reverse (No Holdout, 15M) — Training Analysis

Run: `output/wandb/wandb/run-20260123_163502-jx7e1sjx`
Baseline: `output/wandb/wandb/run-20260123_112209-q2wt3fou`

## Summary
- The new run reached **15M steps** and reports **eval success 1.0 / collision 0.0**.
- Hyperparameters are **identical** to the previous run; the key differences are:
  - **Total timesteps:** 10M → 15M
  - **Holdout removed:** evaluation is now on training distribution

This explains the large jump in eval performance. It is not directly comparable to the
previous holdout-based evaluation.

## Training status
- `global_step`: 15,000,000
- `eval/step`: 15,000,000
- Training **completed** (hit the configured total timesteps).

## Key metrics (final snapshot)

| Metric | No-holdout 15M | Holdout 10M |
| --- | --- | --- |
| eval/success_rate | **1.0** | 0.7 |
| eval/collision_rate | **0.0** | 0.2 |
| eval/path_efficiency | 0.601 | 0.652 |
| eval/snqi | 1.0 | 0.6 |
| rollout/success_rate | 0.91 | 0.94 |
| train/approx_kl | 0.0163 | 0.0195 |
| train/clip_fraction | 0.344 | 0.348 |
| train/explained_variance | 0.943 | 0.909 |
| train/value_loss | 0.0086 | 0.0129 |
| train/entropy_loss | -4.27 | -3.71 |
| train/std | 2.07 | 1.55 |

## Hyperparameters (unchanged)
From `config.yaml` for both runs:
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

## Interpretation
- **Eval success is perfect** because the policy is now evaluated on the same
  distribution it trains on (holdout disabled). This is a useful *sanity check*
  but it does **not prove generalization**.
- **Path efficiency is slightly lower** (0.601 vs 0.652), so the policy may be
  reaching goals but with less efficient trajectories.
- **Entropy and std increased**, indicating a more stochastic policy at the end.
  This may reflect the longer training horizon or differences in evaluation setup.

## Oddities / flags
- The leap in evaluation performance is expected due to **no holdout**.
- Because eval episodes are only 10, **metrics may be noisy**. If you want a more
  stable diagnostic, increase eval episodes (e.g., 30–50) for the next run.

## Recommendation
- Treat this run as a **solvability check**: the RL policy *can* solve the training
  distribution when evaluated in-distribution.
- To measure generalization, re-enable the holdout scenarios and compare again.
- If the goal is better generalization, consider increasing scenario diversity
  before changing hyperparameters.

