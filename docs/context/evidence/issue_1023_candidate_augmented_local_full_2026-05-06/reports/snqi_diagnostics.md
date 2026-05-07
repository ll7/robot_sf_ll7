# SNQI Diagnostics

- Contract status: `fail`
- Rank alignment (Spearman): `0.1833`
- Outcome separation: `0.2035`
- Objective score: `0.2342`
- Dominant component: `success_reward`
- Dominant component mean |contribution|: `0.0936`

## SNQI Assets

- Weights path: `configs/benchmarks/snqi_weights_camera_ready_v3.json`
- Weights version: `snqi_weights_camera_ready_v3`
- Weights SHA-256: `71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2`
- Baseline path: `configs/benchmarks/snqi_baseline_camera_ready_v3.json`
- Baseline version: `snqi_baseline_camera_ready_v3`
- Baseline SHA-256: `329ca5766491e1587979d0a435c7ba676e148ccdff97040a36546bbb9414035a`

## Baseline Normalization

- Source: `config_file`
- Degeneracy adjustments: `0`

## Positioning

- Recommendation: `strengthen_as_operational_multi_objective_aggregation`
- Claim scope: `benchmark aggregate, not a universal ground-truth utility`
- Aligned variable metrics: `6` / `6`

## Planner Ordering

| Rank | Planner | Kinematics | Mean SNQI | Episodes |
|---:|---|---|---:|---:|
| 1 | ppo | differential_drive | -0.017405 | 144 |
| 2 | socnav_sampling | differential_drive | -0.042773 | 144 |
| 3 | scenario_adaptive_hybrid_orca_v1 | differential_drive | -0.045821 | 144 |
| 4 | hybrid_rule_v3_fast_progress_static_escape | differential_drive | -0.049082 | 144 |
| 5 | orca | differential_drive | -0.062724 | 144 |
| 6 | prediction_planner | differential_drive | -0.130404 | 144 |
| 7 | sacadrl | differential_drive | -0.158229 | 144 |
| 8 | goal | differential_drive | -0.177580 | 144 |
| 9 | social_force | differential_drive | -0.204544 | 144 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | n/a | no | n/a |
| comfort_exposure | negative | -0.325752 | yes | yes |
| force_exceed_events | negative | -0.397857 | yes | yes |
| jerk_mean | negative | -0.077849 | yes | yes |
| near_misses | negative | -0.736912 | yes | yes |
| success | positive | 0.556244 | yes | yes |
| time_to_goal_norm | negative | -0.535121 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.000000 |
| comfort_penalty | 0.005098 |
| force_exceed_penalty | 0.012721 |
| jerk_penalty | 0.012727 |
| near_penalty | 0.084540 |
| success_reward | 0.093613 |
| time_penalty | 0.077256 |

## Caveats

- Degenerate metrics on this slice cannot be used as independent validation signals: collisions
- Planner ordering changed under 4 one-at-a-time weight ablation(s) on this slice.
