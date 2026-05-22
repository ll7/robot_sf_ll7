# SNQI Diagnostics

- Contract status: `warn`
- Rank alignment (Spearman): `0.3214`
- Outcome separation: `0.2090`
- Objective score: `0.3737`
- Dominant component: `time_penalty`
- Dominant component mean |contribution|: `0.0927`

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
| 1 | socnav_sampling | differential_drive | -0.146879 | 480 |
| 2 | orca | differential_drive | -0.149461 | 480 |
| 3 | ppo | differential_drive | -0.151885 | 480 |
| 4 | goal | differential_drive | -0.162755 | 480 |
| 5 | sacadrl | differential_drive | -0.179150 | 480 |
| 6 | prediction_planner | differential_drive | -0.195259 | 480 |
| 7 | social_force | differential_drive | -0.196945 | 480 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | -0.592266 | yes | yes |
| comfort_exposure | negative | -0.469880 | yes | yes |
| force_exceed_events | negative | -0.437990 | yes | yes |
| jerk_mean | negative | -0.252355 | yes | yes |
| near_misses | negative | -0.654350 | yes | yes |
| success | positive | 0.368595 | yes | yes |
| time_to_goal_norm | negative | -0.369431 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.025897 |
| comfort_penalty | 0.005780 |
| force_exceed_penalty | 0.008736 |
| jerk_penalty | 0.009743 |
| near_penalty | 0.043622 |
| success_reward | 0.017572 |
| time_penalty | 0.092699 |

## Caveats

- Planner ordering changed under 4 one-at-a-time weight ablation(s) on this slice.
