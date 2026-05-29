# SNQI Diagnostics

- Contract status: `warn`
- Rank alignment (Spearman): `0.9596`
- Outcome separation: `0.0000`
- Objective score: `0.9596`
- Dominant component: `near_penalty`
- Dominant component mean |contribution|: `0.1153`

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
- Aligned variable metrics: `5` / `5`

## Planner Ordering

| Rank | Planner | Kinematics | Mean SNQI | Episodes |
|---:|---|---|---:|---:|
| 1 | goal | holonomic | -0.099984 | 1 |
| 2 | orca | bicycle_drive | -0.099984 | 1 |
| 3 | prediction_planner | differential_drive | -0.099984 | 1 |
| 4 | socnav_sampling | bicycle_drive | -0.099984 | 1 |
| 5 | socnav_sampling | holonomic | -0.099984 | 1 |
| 6 | prediction_planner | holonomic | -0.100390 | 1 |
| 7 | social_force | bicycle_drive | -0.100771 | 1 |
| 8 | prediction_planner | bicycle_drive | -0.102322 | 1 |
| 9 | ppo | bicycle_drive | -0.103961 | 1 |
| 10 | orca | differential_drive | -0.175372 | 1 |
| 11 | ppo | differential_drive | -0.213906 | 1 |
| 12 | ppo | holonomic | -0.218753 | 1 |
| 13 | social_force | differential_drive | -0.250598 | 1 |
| 14 | socnav_sampling | differential_drive | -0.278332 | 1 |
| 15 | orca | holonomic | -0.311574 | 1 |
| 16 | social_force | holonomic | -0.315167 | 1 |
| 17 | goal | bicycle_drive | -0.408242 | 1 |
| 18 | goal | differential_drive | -0.408242 | 1 |
| 19 | sacadrl | holonomic | -0.408242 | 1 |
| 20 | sacadrl | bicycle_drive | -0.451478 | 1 |
| 21 | sacadrl | differential_drive | -0.453639 | 1 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.111656 | yes | yes |
| comfort_exposure | negative | -0.511610 | yes | yes |
| force_exceed_events | negative | -0.513665 | yes | yes |
| jerk_mean | negative | -0.023561 | yes | yes |
| near_misses | negative | -0.953161 | yes | yes |
| success | positive | n/a | no | n/a |
| time_to_goal_norm | negative | n/a | no | n/a |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.004992 |
| comfort_penalty | 0.003716 |
| force_exceed_penalty | 0.005583 |
| jerk_penalty | 0.004078 |
| near_penalty | 0.115335 |
| success_reward | 0.000000 |
| time_penalty | 0.094911 |

## Caveats

- Degenerate metrics on this slice cannot be used as independent validation signals: success, time_to_goal_norm
- Planner ordering changed under 3 one-at-a-time weight ablation(s) on this slice.
