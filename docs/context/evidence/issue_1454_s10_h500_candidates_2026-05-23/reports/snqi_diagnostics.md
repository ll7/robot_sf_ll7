# SNQI Diagnostics

- Contract status: `fail`
- Rank alignment (Spearman): `-0.2067`
- Outcome separation: `0.2663`
- Objective score: `-0.1401`
- Dominant component: `success_reward`
- Dominant component mean |contribution|: `0.1095`

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
| 1 | hybrid_rule_v3_fast_progress_static_escape_continuous | differential_drive | -0.041680 | 480 |
| 2 | ppo | differential_drive | -0.049324 | 480 |
| 3 | scenario_adaptive_hybrid_orca_v2_collision_guard | differential_drive | -0.060294 | 480 |
| 4 | scenario_adaptive_hybrid_orca_v1 | differential_drive | -0.060298 | 480 |
| 5 | hybrid_rule_v3_fast_progress_static_escape | differential_drive | -0.063535 | 480 |
| 6 | hybrid_rule_v3_fast_progress | differential_drive | -0.077363 | 480 |
| 7 | orca | differential_drive | -0.083298 | 480 |
| 8 | socnav_sampling | differential_drive | -0.109844 | 480 |
| 9 | prediction_planner | differential_drive | -0.168020 | 480 |
| 10 | sacadrl | differential_drive | -0.232426 | 480 |
| 11 | social_force | differential_drive | -0.235593 | 480 |
| 12 | goal | differential_drive | -0.239402 | 480 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.608483 | yes | yes |
| comfort_exposure | negative | -0.354289 | yes | yes |
| force_exceed_events | negative | -0.368253 | yes | yes |
| jerk_mean | negative | 0.079574 | yes | no |
| near_misses | negative | -0.644384 | yes | yes |
| success | positive | 0.661865 | yes | yes |
| time_to_goal_norm | negative | -0.619632 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.027683 |
| comfort_penalty | 0.005099 |
| force_exceed_penalty | 0.011644 |
| jerk_penalty | 0.015087 |
| near_penalty | 0.092579 |
| success_reward | 0.109514 |
| time_penalty | 0.075844 |

## Caveats

- Planner ordering changed under 4 one-at-a-time weight ablation(s) on this slice.
