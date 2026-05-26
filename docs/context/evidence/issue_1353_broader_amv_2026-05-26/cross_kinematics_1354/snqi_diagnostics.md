# SNQI Diagnostics

- Contract status: `warn`
- Rank alignment (Spearman): `0.9829`
- Outcome separation: `0.0000`
- Objective score: `0.9829`
- Dominant component: `near_penalty`
- Dominant component mean |contribution|: `0.1354`

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

- Recommendation: `downgrade_to_appendix_or_implementation_aid`
- Claim scope: `benchmark aggregate, not a universal ground-truth utility`
- Aligned variable metrics: `2` / `2`

## Planner Ordering

| Rank | Planner | Kinematics | Mean SNQI | Episodes |
|---:|---|---|---:|---:|
| 1 | goal | holonomic | -0.099984 | 1 |
| 2 | orca | bicycle_drive | -0.099984 | 1 |
| 3 | social_force | bicycle_drive | -0.100771 | 1 |
| 4 | orca | differential_drive | -0.175372 | 1 |
| 5 | social_force | differential_drive | -0.250598 | 1 |
| 6 | orca | holonomic | -0.311574 | 1 |
| 7 | social_force | holonomic | -0.315167 | 1 |
| 8 | goal | bicycle_drive | -0.408242 | 1 |
| 9 | goal | differential_drive | -0.408242 | 1 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | n/a | no | n/a |
| comfort_exposure | negative | n/a | no | n/a |
| force_exceed_events | negative | n/a | no | n/a |
| jerk_mean | negative | -0.151266 | yes | yes |
| near_misses | negative | -0.982942 | yes | yes |
| success | positive | n/a | no | n/a |
| time_to_goal_norm | negative | n/a | no | n/a |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.000000 |
| comfort_penalty | 0.002248 |
| force_exceed_penalty | 0.002825 |
| jerk_penalty | 0.005747 |
| near_penalty | 0.135373 |
| success_reward | 0.000000 |
| time_penalty | 0.094911 |

## Caveats

- Degenerate metrics on this slice cannot be used as independent validation signals: collisions, comfort_exposure, force_exceed_events, success, time_to_goal_norm
- Planner ordering changed under 1 one-at-a-time weight ablation(s) on this slice.
