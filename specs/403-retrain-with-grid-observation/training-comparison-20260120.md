# Issue 403 PPO Expert (Grid + SocNav) Run Comparison

## Runs

- Baseline (pre-tuned): `output/wandb/wandb/run-20260119_171000-n4mvzj5x`
- Tuned PPO params: `output/wandb/wandb/run-20260120_084048-y5inxosu`

## Hyperparameters (from W&B config.yaml)

| Setting | 2026-01-19 | 2026-01-20 |
| --- | --- | --- |
| learning_rate | 3e-4 | 1e-4 |
| batch_size | 64 | 256 |
| n_epochs | 10 | 4 |
| clip_range | 0.2 | 0.1 |
| ent_coef | 0.0 | 0.01 |
| target_kl | None | 0.02 |
| n_envs | 31 | 31 |
| n_steps | 2048 | 2048 |

## Final Metrics (W&B summary @ 10M steps)

| Metric | 2026-01-19 | 2026-01-20 | Delta |
| --- | --- | --- | --- |
| eval/success_rate | 0.10 | 0.30 | +0.20 |
| eval/collision_rate | 0.80 | 0.70 | -0.10 |
| eval/snqi | -0.30 | -0.05 | +0.25 |
| eval/path_efficiency | 0.656 | 0.630 | -0.026 |
| rollout/success_rate | 0.42 | 0.70 | +0.28 |
| train/approx_kl | 0.165 | 0.0078 | -0.157 |
| train/clip_fraction | 0.566 | 0.270 | -0.296 |
| train/std | 0.182 | 3.319 | +3.137 |
| train/entropy_loss | 0.529 | -5.230 | -5.759 |
| train/explained_variance | 0.900 | 0.918 | +0.018 |
| time/fps | 570 | 1386 | +816 |

## Interpretation

- The tuned run is clearly more stable: KL and clip_fraction are much lower, which matches the safer PPO update regime.
- Generalization improved: eval success increased (+0.20) and eval collision decreased (-0.10).
- Exploration is much higher: std and entropy are large, likely helping discovery but also keeping collision rate high.
- The train/eval gap remains big (0.70 rollout success vs 0.30 eval), indicating the evaluation set is still harder or the policy is overfitting to training scenarios.

## Next Change Ideas (if we iterate)

- Reduce exploration slightly: try `ent_coef=0.005` or `0.002` to lower collisions.
- Add or strengthen collision penalties (if available in reward shaping) to target eval collision rate.
- Increase evaluation_episodes to 50–100 to reduce noise in eval metrics.
