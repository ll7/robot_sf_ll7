# SNQI Diagnostics

- Contract status: `warn`
- Rank alignment (Spearman): `0.3214`
- Outcome separation: `0.2200`
- Objective score: `0.3764`
- Dominant component: `time_penalty`
- Dominant component mean |contribution|: `0.0808`

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
| 3 | orca | differential_drive | -0.062724 | 144 |
| 4 | prediction_planner | differential_drive | -0.130404 | 144 |
| 5 | sacadrl | differential_drive | -0.158229 | 144 |
| 6 | goal | differential_drive | -0.177580 | 144 |
| 7 | social_force | differential_drive | -0.204544 | 144 |

## Component Correlations

| Metric | Direction | Spearman | Variable | Aligned |
|---|---|---:|---|---|
| collisions | negative | n/a | no | n/a |
| comfort_exposure | negative | -0.342232 | yes | yes |
| force_exceed_events | negative | -0.403257 | yes | yes |
| jerk_mean | negative | -0.147104 | yes | yes |
| near_misses | negative | -0.702377 | yes | yes |
| success | positive | 0.611807 | yes | yes |
| time_to_goal_norm | negative | -0.581567 | yes | yes |

## Component Dominance (mean absolute contribution)

| Component | Mean |
|---|---:|
| collisions_penalty | 0.000000 |
| comfort_penalty | 0.005250 |
| force_exceed_penalty | 0.012208 |
| jerk_penalty | 0.009991 |
| near_penalty | 0.076220 |
| success_reward | 0.071044 |
| time_penalty | 0.080755 |

## Caveats

- Degenerate metrics on this slice cannot be used as independent validation signals: collisions
- Planner ordering changed under 2 one-at-a-time weight ablation(s) on this slice.
