# Training Analysis: `ppo_expert_br06_v2_15m_all_maps_20260302T152332`

## Scope
- Data source: W&B run `ll7/robot_sf/asym0xo2`
- Retrieved on: 2026-03-03
- Comparator (legacy, semantically invalid success contract): `ll7/robot_sf/jx7e1sjx`

## Key run facts
- Run state: `killed`
- Max observed training step: ~`7.1M` (`global_step`), not full `15M`.
- Final summary metrics:
  - `eval/success_rate`: `0.70`
  - `eval/collision_rate`: `0.30`
  - `eval/path_efficiency`: `0.833`
  - `eval/comfort_exposure`: `0.0278`
  - `eval/snqi`: `-0.647`
  - `rollout/success_rate`: `0.78`
  - `train/approx_kl`: `0.0158`
  - `train/clip_fraction`: `0.334`
  - `train/explained_variance`: `0.956`

## Training-quality interpretation
- Optimization stability looks healthy (`approx_kl`, `clip_fraction`, `explained_variance` are in plausible PPO ranges).
- Policy quality is not yet benchmark-ready:
  - `30%` eval collision rate is still too high for a promoted default policy.
  - Success (`70%`) is materially below desired production-grade behavior.
- Because the run stopped around `7.1M`, this should be treated as a partial run.

## End-of-training assessment
- Recommendation: **do not treat this as final training**.
- Next step: continue/redo to full budget (>= `15M`) and re-check.
- Practical gate suggestion before promotion:
  - `eval/success_rate >= 0.85`
  - `eval/collision_rate <= 0.10`
  - stable metrics over final 3-5 evaluation windows.

## Hyperparameter optimization need
- HPO is likely useful, but **secondary** to finishing a full-length run first.
- Suggested order:
  1. complete one clean full-budget run with current settings,
  2. if collision remains >`0.10`, run a focused HPO sweep (learning rate, entropy coef, clip range, n_steps, reward weights).

## Legacy model replacement decision
- Legacy model (`jx7e1sjx`) metrics were generated before the success-semantics correction and should not be used for paper claims.
- Registry has been updated to point to this BR-06 v2 run as current PPO entry.
- Keep this model as **candidate** until full-budget retrain + benchmark validation are complete.
