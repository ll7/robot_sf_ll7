# SNQI Diagnostics

- Contract status: `fail`
- Rank alignment (Spearman): `0.2857`
- Outcome separation: `0.2001`
- Objective score: `0.3358`
- Dominant component: `time_penalty`
- Dominant component mean |contribution|: `0.0929`

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
| 1 | orca | differential_drive | -0.150859 | 144 |
| 2 | socnav_sampling | differential_drive | -0.153456 | 144 |
| 3 | ppo | differential_drive | -0.153760 | 144 |
| 4 | goal | differential_drive | -0.165596 | 144 |
| 5 | sacadrl | differential_drive | -0.173701 | 144 |
| 6 | prediction_planner | differential_drive | -0.194679 | 144 |
| 7 | social_force | differential_drive | -0.206152 | 144 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.641808 | yes | yes |
| comfort_exposure | negative | -0.505240 | yes | yes |
| force_exceed_events | negative | -0.449796 | yes | yes |
| jerk_mean | negative | -0.248675 | yes | yes |
| near_misses | negative | -0.640245 | yes | yes |
| success | positive | 0.355892 | yes | yes |
| time_to_goal_norm | negative | -0.354699 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.027665 |
| comfort_penalty | 0.007087 |
| force_exceed_penalty | 0.009065 |
| jerk_penalty | 0.009217 |
| near_penalty | 0.042420 |
| success_reward | 0.017194 |
| time_penalty | 0.092912 |

## Caveats

- Planner ordering changed under 5 one-at-a-time weight ablation(s) on this slice.
