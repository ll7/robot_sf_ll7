# Issue 4228 Social Preference Labels

This note defines the first diagnostic configuration surface for social preference labels in
Robot SF. It records what the labels mean, where they can connect to existing metrics, and why this
slice does not claim SPLC implementation, reward learning, or calibrated human preference evidence.

## Motivation

The SPLC paper, "Social Preference Learning for Crowd Robot Navigation" (arXiv:2607.01925),
motivates a bridge from hand-authored safety and comfort predicates toward preference-informed
social-compliance analysis. For this repository, issue #4228 is only that bridge's diagnostic
configuration layer.

## Repository Boundary

The config at `configs/diagnostics/social_preference_labels.yaml` is diagnostic-only. It is not a
reward function, not an offline reinforcement learning input, not a replacement for Social
Navigation Quality Index (SNQI), and not calibrated human social preference evidence. The SPLC
paper is cited as literature motivation only.

Missing required trace fields and missing metric keys fail closed as `not_available`. Fallback or
degraded rows remain diagnostic-only and cannot be used as benchmark-strengthening evidence.

## Relationship To Existing Metrics

The v1 label layer points at existing metric families where possible:

| Label | Interpretation | Candidate metric keys | Missing or limited trace contract |
| --- | --- | --- | --- |
| `clearance` | Prefer larger robot-pedestrian surface clearance. | `min_clearance`, `mean_clearance`, social proxemic clearance keys. | Uses existing collision and near-miss threshold provenance, not preference calibration. |
| `ttc_margin` | Prefer larger time-to-collision margin and fewer near-critical intervals. | `near_misses_ttc`, `near_miss_ttc__min_ttc_s`, `time_to_collision_min`. | Requires positions, velocities, and `dt` in traces where TTC is emitted. |
| `pedestrian_displacement` | Prefer lower pedestrian deviation or acceleration change near the robot. | `ped_impact_accel_delta_mean`, `ped_impact_turn_rate_delta_mean`, `pedestrian_path_deviation_proxy_m`. | Current values are exploratory proxies, not measured discomfort or intent. |
| `path_blocking` | Prefer avoiding robot occupancy of likely pedestrian path corridors. | None yet. | Needs pedestrian route or corridor estimates and robot occupancy intervals. |
| `oscillation` | Prefer lower heading, velocity, or path instability. | `jerk_mean`, `curvature_mean`, path irregularity keys. | Motion quality proxy only; no learned social utility. |
| `detour_burden` | Prefer lower extra robot or pedestrian path length and time burden. | `path_efficiency`, path ratio, pedestrian deviation, time-to-goal ratio keys. | Robot and pedestrian burden should remain separate when both are available. |
| `recovery_smoothness` | Prefer smoother robot recovery after near-conflict or yielding. | `jerk_mean`, acceleration and jerk extrema. | Needs conflict-window or yielding-event trace fields for attribution. |

## Follow-Up Path

Useful next slices are:

- compute labels from existing episode traces when the required fields are present;
- emit explicit `not_available` label columns for missing trace contracts;
- compare diagnostic labels with current social proxemic, Time-to-Collision (TTC), comfort, and
  pedestrian-impact metrics;
- explore preference-pair generation offline, still without reward replacement;
- consider reward-model or offline reinforcement learning work only after separate validation.
