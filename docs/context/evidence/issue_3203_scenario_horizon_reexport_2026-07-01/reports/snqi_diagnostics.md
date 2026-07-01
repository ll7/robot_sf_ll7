# SNQI Diagnostics

- Contract status: `fail`
- Rank alignment (Spearman): `-0.2000`
- Outcome separation: `0.2955`
- Objective score: `-0.1261`
- Dominant component: `time_penalty`
- Dominant component mean |contribution|: `0.0823`

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
- Aligned variable metrics: `6` / `7`

## Planner Ordering

| Rank | Planner | Kinematics | Mean SNQI | Episodes |
|---:|---|---|---:|---:|
| 1 | ppo | differential_drive | -0.053098 | 144 |
| 2 | hybrid_rule_v3_fast_progress_static_escape | differential_drive | -0.061743 | 144 |
| 3 | scenario_adaptive_hybrid_orca_v1 | differential_drive | -0.067694 | 144 |
| 4 | orca | differential_drive | -0.084962 | 144 |
| 5 | socnav_sampling | differential_drive | -0.148573 | 144 |
| 6 | prediction_planner | differential_drive | -0.172603 | 144 |
| 7 | social_force | differential_drive | -0.219251 | 144 |
| 8 | sacadrl | differential_drive | -0.235076 | 144 |
| 9 | goal | differential_drive | -0.257856 | 144 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.735042 | yes | yes |
| comfort_exposure | negative | -0.406086 | yes | yes |
| force_exceed_events | negative | -0.409494 | yes | yes |
| jerk_mean | negative | 0.128483 | yes | no |
| near_misses | negative | -0.586642 | yes | yes |
| success | positive | 0.742956 | yes | yes |
| time_to_goal_norm | negative | -0.703049 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.045057 |
| comfort_penalty | 0.006115 |
| force_exceed_penalty | 0.010758 |
| jerk_penalty | 0.001139 |
| near_penalty | 0.075864 |
| success_reward | 0.076712 |
| time_penalty | 0.082320 |

## Caveats

- Planner ordering changed under 3 one-at-a-time weight ablation(s) on this slice.
