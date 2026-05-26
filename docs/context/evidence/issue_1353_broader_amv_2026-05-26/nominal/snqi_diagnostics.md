# SNQI Diagnostics

- Contract status: `warn`
- Rank alignment (Spearman): `0.4643`
- Outcome separation: `0.2681`
- Objective score: `0.5313`
- Dominant component: `time_penalty`
- Dominant component mean |contribution|: `0.0834`

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
- Aligned variable metrics: `7` / `7`

## Planner Ordering

| Rank | Planner | Kinematics | Mean SNQI | Episodes |
|---:|---|---|---:|---:|
| 1 | socnav_sampling | differential_drive | -0.081405 | 12 |
| 2 | sacadrl | differential_drive | -0.089064 | 12 |
| 3 | goal | differential_drive | -0.095565 | 12 |
| 4 | ppo | differential_drive | -0.100672 | 12 |
| 5 | orca | differential_drive | -0.112948 | 12 |
| 6 | prediction_planner | differential_drive | -0.113446 | 12 |
| 7 | social_force | differential_drive | -0.129084 | 12 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.625666 | yes | yes |
| comfort_exposure | negative | -0.614471 | yes | yes |
| force_exceed_events | negative | -0.612498 | yes | yes |
| jerk_mean | negative | -0.450381 | yes | yes |
| near_misses | negative | -0.733294 | yes | yes |
| success | positive | 0.638974 | yes | yes |
| time_to_goal_norm | negative | -0.658316 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.019969 |
| comfort_penalty | 0.002473 |
| force_exceed_penalty | 0.002035 |
| jerk_penalty | 0.010388 |
| near_penalty | 0.030232 |
| success_reward | 0.045347 |
| time_penalty | 0.083420 |

## Caveats

- Planner ordering changed under 6 one-at-a-time weight ablation(s) on this slice.
