# Metrics Specification

This document provides the formal specification for all metrics computed in the Social Navigation Benchmark.

## Overview

The benchmark computes a comprehensive set of metrics covering safety, efficiency, comfort, and overall navigation quality. All metrics are computed from episode trajectory data and aggregated across multiple runs.

## Distance-Based Thresholds

The following constants define the distance thresholds used throughout the metrics:

- **Collision distance**: 0.25m (strict boundary for collision events)
- **Near-miss distance**: 0.50m (upper bound for near-miss region)
- **Force comfort threshold**: 2.0 (unitless, force magnitude above which interaction is considered uncomfortable)

Threshold provenance and reproducibility contract:

- Canonical values live in `robot_sf/benchmark/constants.py`.
- Episode records now embed `metric_parameters.threshold_profile` and a deterministic
  `metric_parameters.threshold_signature`.
- Aggregation validates that all episodes in a report share one threshold profile;
  mixed profiles are rejected to prevent invalid comparisons.
- The sensitivity workflow in `scripts/benchmark_threshold_sensitivity.py` quantifies
  how metric summaries change across threshold grids per scenario family.

## Core Metrics

### Safety Metrics

#### Collisions
- **Definition**: Number of timesteps where robot-pedestrian footprint clearance < 0m.
  Legacy records without `metric_parameters.threshold_profile` used center distance < 0.25m.
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Direct safety measure

#### Near Misses
- **Definition**: Number of timesteps where robot-pedestrian footprint clearance ∈ [0m, 0.50m).
  Legacy records without `metric_parameters.threshold_profile` used center distance ∈ [0.25m, 0.50m).
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Proximity safety measure
- **Speed dependence (current default)**: Disabled. Near-miss is distance-only in the
  canonical benchmark metric.
- **Evaluated alternatives**: Relative-speed weighted near-miss score and TTC-gated
  near-miss count (reported by threshold sensitivity tooling, not replacing the
  canonical metric).

#### Minimum Interpersonal Distance
- **Definition**: Global minimum distance to any pedestrian across the episode
- **Type**: Float (meters)
- **Range**: [0, ∞)
- **Purpose**: Worst-case proximity analysis

#### Mean Interpersonal Distance
- **Definition**: Average minimum distance per timestep
- **Type**: Float (meters)
- **Range**: [0, ∞)
- **Purpose**: Overall proximity behavior

### Task Performance Metrics

#### Success
- **Definition**: 1 if goal reached before horizon with zero collisions, 0 otherwise
- **Type**: Binary (0 or 1)
- **Purpose**: Task completion with safety constraint

#### Time to Goal (Normalized)
- **Definition**: Actual time to goal / horizon if successful, 1.0 if unsuccessful
- **Type**: Float
- **Range**: [0, 1]
- **Purpose**: Efficiency measure

#### Path Efficiency
- **Definition**: Shortest path length / actual path length
- **Type**: Float
- **Range**: (0, 1]
- **Purpose**: Path optimality measure

### Comfort Metrics

#### Force Mean
- **Definition**: Mean norm of interaction forces on robot
- **Type**: Float (force units)
- **Range**: [0, ∞)
- **Purpose**: Average interaction intensity

#### Force 95th Percentile
- **Definition**: 95th percentile of force norm distribution
- **Type**: Float (force units)
- **Range**: [0, ∞)
- **Purpose**: Extreme interaction events

#### Force Exceedance Events
- **Definition**: Number of timesteps with force > comfort threshold (2.0)
- **Type**: Integer count
- **Range**: [0, episode_length]
- **Purpose**: Discomfort event frequency

#### Comfort Exposure
- **Definition**: Proportion of timesteps with force > comfort threshold
- **Type**: Float
- **Range**: [0, 1]
- **Purpose**: Relative discomfort exposure

### Pedestrian-Impact Metrics

Pedestrian-impact metrics are opt-in because they are exploratory social-impact signals, not a
headline benchmark score. Enable them in benchmark JSONL output with:

```bash
uv run robot_sf_bench run \
  --matrix <scenario-matrix.yaml> \
  --out output/benchmarks/ped_impact/episodes.jsonl \
  --experimental-ped-impact
```

The flat `ped_impact_*` metrics are preserved for backward compatibility. Episode records also
include `metrics.pedestrian_impact` with `schema_version: pedestrian-impact.v1`.

Canonical reductions:

- `ped_impact_accel_delta_mean`: mean per-pedestrian near-minus-far acceleration delta (`m/s^2`).
- `ped_impact_accel_delta_median`: median per-pedestrian near-minus-far acceleration delta
  (`m/s^2`).
- `ped_impact_turn_rate_delta_mean`: mean per-pedestrian near-minus-far heading turn-rate delta
  (`rad/s`).
- `ped_impact_turn_rate_delta_median`: median per-pedestrian near-minus-far heading turn-rate
  delta (`rad/s`).
- `ped_impact_*_valid`: number of pedestrians with both near and far support for that reduction.

Near samples use robot-pedestrian distance `<= --ped-impact-radius-m` (default `2.0` meters), and
far samples use distances above that threshold. Acceleration and turn-rate signals are smoothed
with a trailing window of `--ped-impact-window-steps` (default `5`). Aggregation flattens
`metrics.pedestrian_impact.canonical_reductions` and `metrics.pedestrian_impact.sample_counts`
into aggregate-ready `ped_impact_*` fields such as `ped_impact_accel_delta_mean`,
`ped_impact_turn_rate_delta_mean`, `ped_impact_ped_count`, `ped_impact_near_samples`,
`ped_impact_far_samples`, and `ped_impact_near_sample_frac`, then reports the normal
mean/median/p95 summaries.

### Social-Acceptability Pilot Metrics

The opt-in pedestrian-impact switch also emits a bounded pilot set of
`social_proxemic_*` metrics. These metrics are exploratory trajectory proxies for proxemic
intrusion analysis. They are not a replacement for SNQI, collision/TTC/clearance reporting,
paper headline metrics, or human-subject validation.

Episode records preserve flat `social_proxemic_*` fields and include a structured
`metrics.social_acceptability` block with `schema_version: social-acceptability-pilot.v1`.
Aggregation flattens the structured block into scalar columns when only the block is present.

Default parameter:

- `social_proxemic_radius_m`: `1.2` meters of robot-pedestrian surface clearance.

Canonical pilot reductions:

- `social_proxemic_available`: true when a trajectory contains at least one timestep and at
  least one pedestrian; false for K=0 or empty trajectories.
- `social_proxemic_ped_count`: number of pedestrians in the episode.
- `social_proxemic_intrusion_steps`: number of timesteps where at least one pedestrian has
  surface clearance below `social_proxemic_radius_m`.
- `social_proxemic_intrusion_frac`: `intrusion_steps / trajectory_timesteps`.
- `social_proxemic_intrusion_area_m_s`: sum of
  `max(social_proxemic_radius_m - clearance_m, 0) * dt` across pedestrians and timesteps.
- `social_proxemic_min_clearance_m`: minimum finite robot-pedestrian surface clearance observed.

Missing pedestrian trajectories fail gracefully: availability is false, counts and intrusion
reductions are zero, and minimum clearance is omitted from sanitized JSON output when unavailable.

### Human-Interaction Proxy Metrics

`compute_all_metrics(..., experimental_human_interaction_proxy=True)` emits a bounded
`human_proxy_*` metric family plus a structured `metrics.human_interaction_proxy` block with
`schema_version: human-interaction-proxy.v1`. These metrics are simulation proxies for
mechanism-report discussion only. They are not validated human comfort, human-subject, safety, or
paper-grade social-compliance metrics.

Default parameters:

- `human_proxy_proxemic_radius_m`: `1.2` meters of robot-pedestrian surface clearance.
- `human_proxy_yield_speed_mps`: `0.15` meters per second robot-speed threshold for proxy yielding.

Canonical reductions:

- `human_discomfort_exposure_m_s`: sum of
  `max(human_proxy_proxemic_radius_m - clearance_m, 0) * dt` across pedestrians and timesteps.
  Unit: meter-seconds (`m*s`).
- `intrusion_duration_s`: number of timesteps where at least one pedestrian has surface clearance
  below `human_proxy_proxemic_radius_m`, multiplied by `dt`. Unit: seconds.
- `time_to_yield_s`: elapsed time from the first proxemic intrusion timestep to the first later
  timestep where robot speed is at or below `human_proxy_yield_speed_mps`. Omitted from sanitized
  JSON output when no intrusion or no proxy-yield timestep exists.
- `robot_yield_distance_m`: nearest center-to-center robot-pedestrian distance at the proxy-yield
  timestep. Omitted when `time_to_yield_s` is unavailable.
- `pedestrian_path_deviation_proxy_m`: mean per-pedestrian extra path length over straight-line
  displacement. This is a trajectory irregularity proxy, not a measured intent or discomfort
  signal.
- `group_split_intrusion_available`: false for the current `EpisodeData` path because the episode
  container does not include social-group membership labels. Group-split intrusion remains excluded
  until a grouped-pedestrian contract exists.

Episode records preserve the flat fields and include a structured block with parameters, units,
sample counts, canonical reductions, exclusions, and an interpretation string. Aggregation flattens
the structured block into scalar columns such as `human_discomfort_exposure_m_s`,
`intrusion_duration_s`, `time_to_yield_s`, `robot_yield_distance_m`, and
`pedestrian_path_deviation_proxy_m`.

### Diagnostic Social Preference Labels

`configs/diagnostics/social_preference_labels.yaml` defines `social-preference-labels.v1`, a
diagnostic annotation contract layered over existing safety, Time-to-Collision (TTC), proxemic,
pedestrian-impact, path-efficiency, and smoothness metrics. These labels are not a new headline
benchmark score, not learned rewards, and not calibrated human preference evidence. Missing trace
fields and missing metric keys remain `not_available`; see
`docs/context/issue_4228_social_preference_labels.md` for the issue #4228 boundary and follow-up
path.

### Motion Quality Metrics

#### Smoothness (Jerk)
- **Definition**: Mean jerk magnitude (third derivative of position)
- **Type**: Float (m/s³)
- **Range**: [0, ∞)
- **Purpose**: Motion smoothness assessment

#### Path Curvature
- **Definition**: Mean path curvature
- **Type**: Float (1/m)
- **Range**: [0, ∞)
- **Purpose**: Path geometry analysis

#### Energy Consumption
- **Definition**: Sum of acceleration magnitudes over episode
- **Type**: Float (m/s²)
- **Range**: [0, ∞)
- **Purpose**: Energy efficiency proxy

### Composite Metrics

#### SNQI (Social Navigation Quality Index)
- **Definition**: Weighted benchmark aggregate over task completion, efficiency, safety, and comfort signals
- **Type**: Float
- **Range**: Depends on normalization and weights
- **Purpose**: Operational multi-objective aggregation for AMV evaluation
- **Formula**:

  `SNQI = w_success * success`

  `      - w_time * time_to_goal_norm`

  `      - w_collisions * norm(collisions)`

  `      - w_near * norm(near_misses)`

  `      - w_comfort * comfort_exposure`

  `      - w_force_exceed * norm(force_exceed_events)`

  `      - w_jerk * norm(jerk_mean)`

- **Interpretation**:
  Higher is better.
  SNQI is a benchmark aggregate for comparing planners under a fixed metric contract.
  It is not a universal ground-truth social utility scalar.

## Normalization Strategy

For SNQI computation, metrics are normalized using baseline statistics:
- **Central tendency**: Median of baseline distribution
- **Scale**: 95th percentile - median range
- **Formula**: normalized = (raw_value - baseline_median) / (baseline_p95 - baseline_median)

## Implementation Notes

- Missing pedestrians (K=0): Distance-based metrics return appropriate defaults (NaN for min/mean distance, 0 for counts)
- Zero-length trajectories: Motion metrics return 0 or NaN as appropriate
- Force availability: Force-based metrics require `record_forces=True` during episode generation
- Paper-facing benchmark runs should use the pinned `camera_ready_v3` SNQI weights and baseline assets.
- Contract health, component correlations, planner ordering, and ablation-based weight sensitivity are emitted in benchmark `reports/snqi_diagnostics.{json,md}` and `reports/snqi_sensitivity.csv`.

## References

- Collision and near-miss thresholds aligned with existing test suite for backward compatibility
- Force threshold calibrated empirically for current fast-pysf physics model
- SNQI methodology follows bounded composite-index practice with versioned weights, baseline normalization, and explicit contract diagnostics

*Last updated: May 2026*
