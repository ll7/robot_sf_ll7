# H500 Solvability Mechanism Analysis

This analysis uses the aggregate fixed-vs-h500 comparison. It can identify which planner-scenario cells convert fixed-horizon unfinished runs into h500 successes, but it cannot prove a wait-then-go causal mechanism without per-step traces or video.

## Summary

- Base campaign: `paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Candidate campaign: `issue1023_scenario_horizons_candidates_local_2026-05-06`
- Timeout-to-success candidate cells: `123`
- Evidence level: `aggregate_comparison`; waiting claims remain `trace_required`.

## Mechanism Counts

| Mechanism | Count |
|---|---:|
| `budget_limited_clean_completion` | 38 |
| `exposure_enabled_completion` | 40 |
| `late_clean_completion` | 5 |
| `partial_timeout_relief` | 18 |
| `safety_regressed_completion` | 22 |

## Representative Cases

| Mechanism | Planner | Scenario | Success Delta | Collision Delta | Near-Miss Delta | Candidate Time Norm |
|---|---|---|---:|---:|---:|---:|
| `budget_limited_clean_completion` | `orca` | `classic_bottleneck_low` | 1.0000 | 0.0000 | 0.0000 | 0.6193 |
| `budget_limited_clean_completion` | `orca` | `classic_head_on_corridor_low` | 1.0000 | 0.0000 | 0.0000 | 0.5573 |
| `budget_limited_clean_completion` | `orca` | `classic_urban_crossing_medium` | 1.0000 | 0.0000 | 0.0000 | 0.6814 |
| `exposure_enabled_completion` | `orca` | `francis2023_parallel_traffic` | 1.0000 | 0.0000 | 0.3333 | 0.6427 |
| `exposure_enabled_completion` | `ppo` | `francis2023_crowd_navigation` | 1.0000 | 0.0000 | 0.3333 | 0.3671 |
| `exposure_enabled_completion` | `ppo` | `francis2023_parallel_traffic` | 1.0000 | 0.0000 | 0.3333 | 0.6280 |
| `late_clean_completion` | `prediction_planner` | `classic_group_crossing_low` | 1.0000 | 0.0000 | 0.0000 | 0.7987 |
| `late_clean_completion` | `prediction_planner` | `francis2023_intersection_no_gesture` | 1.0000 | 0.0000 | 0.0000 | 0.8075 |
| `late_clean_completion` | `prediction_planner` | `francis2023_intersection_proceed` | 1.0000 | 0.0000 | 0.0000 | 0.8075 |
| `partial_timeout_relief` | `prediction_planner` | `classic_bottleneck_low` | 0.6667 | 0.0000 | 0.0000 | 0.8258 |
| `partial_timeout_relief` | `prediction_planner` | `classic_urban_crossing_medium` | 0.6667 | 0.0000 | 0.0000 | 0.9027 |
| `partial_timeout_relief` | `prediction_planner` | `francis2023_exiting_room` | 0.6667 | 0.0000 | 0.0000 | 0.8588 |
| `safety_regressed_completion` | `prediction_planner` | `francis2023_accompanying_peer` | 0.6667 | 0.3333 | 0.0000 | 0.9628 |
| `safety_regressed_completion` | `prediction_planner` | `francis2023_down_path` | 0.6667 | 0.3333 | 0.0000 | 0.9597 |
| `safety_regressed_completion` | `prediction_planner` | `francis2023_following_human` | 0.6667 | 0.3333 | 0.0000 | 0.9845 |

## Scenario-Family Rollup

| Family | Cases | Mean Success Delta | Mean Collision Delta | Mean Near-Miss Delta | Mechanisms |
|---|---:|---:|---:|---:|---|
| `accompanying_peer` | 4 | 0.8334 | 0.0833 | 0.0000 | `{'budget_limited_clean_completion': 2, 'partial_timeout_relief': 1, 'safety_regressed_completion': 1}` |
| `blind_corner` | 2 | 0.8334 | 0.0000 | 7.3334 | `{'exposure_enabled_completion': 2}` |
| `bottleneck` | 12 | 0.7500 | 0.1389 | 26.2500 | `{'budget_limited_clean_completion': 3, 'exposure_enabled_completion': 4, 'partial_timeout_relief': 1, 'safety_regressed_completion': 4}` |
| `circular_crossing` | 1 | 0.3333 | 0.0000 | 0.0000 | `{'partial_timeout_relief': 1}` |
| `cross_trap` | 8 | 0.6667 | 0.1250 | 8.7500 | `{'exposure_enabled_completion': 4, 'partial_timeout_relief': 2, 'safety_regressed_completion': 2}` |
| `crossing` | 4 | 0.7500 | 0.0000 | 0.0000 | `{'budget_limited_clean_completion': 2, 'partial_timeout_relief': 2}` |
| `crowd_navigation` | 5 | 0.8667 | 0.0000 | 12.4000 | `{'exposure_enabled_completion': 4, 'partial_timeout_relief': 1}` |
| `doorway` | 5 | 0.4667 | 0.1333 | 4.7334 | `{'exposure_enabled_completion': 4, 'safety_regressed_completion': 1}` |
| `down_path` | 3 | 0.8889 | 0.1111 | 0.0000 | `{'budget_limited_clean_completion': 2, 'safety_regressed_completion': 1}` |
| `exiting_elevator` | 3 | 0.7778 | 0.0000 | 5.1111 | `{'budget_limited_clean_completion': 2, 'exposure_enabled_completion': 1}` |
| `exiting_room` | 1 | 0.6667 | 0.0000 | 0.0000 | `{'partial_timeout_relief': 1}` |
| `following_human` | 3 | 0.8889 | 0.1111 | 0.0000 | `{'budget_limited_clean_completion': 2, 'safety_regressed_completion': 1}` |
| `frontal_approach` | 3 | 0.8889 | 0.0000 | 26.3333 | `{'exposure_enabled_completion': 3}` |
| `group_crossing` | 7 | 0.5714 | 0.0000 | 1.6190 | `{'budget_limited_clean_completion': 1, 'exposure_enabled_completion': 1, 'late_clean_completion': 1, 'partial_timeout_relief': 4}` |
| `head_on_corridor` | 8 | 0.6667 | 0.1667 | 3.9167 | `{'budget_limited_clean_completion': 3, 'exposure_enabled_completion': 2, 'partial_timeout_relief': 1, 'safety_regressed_completion': 2}` |
| `intersection_no_gesture` | 4 | 0.7500 | 0.0000 | 0.0000 | `{'budget_limited_clean_completion': 3, 'late_clean_completion': 1}` |
| `intersection_proceed` | 4 | 0.7500 | 0.0000 | 0.0000 | `{'budget_limited_clean_completion': 3, 'late_clean_completion': 1}` |
| `intersection_wait` | 4 | 0.7500 | 0.0000 | 0.0000 | `{'budget_limited_clean_completion': 3, 'late_clean_completion': 1}` |
| `leading_human` | 4 | 0.8334 | 0.0833 | 0.0000 | `{'budget_limited_clean_completion': 2, 'partial_timeout_relief': 1, 'safety_regressed_completion': 1}` |
| `merging` | 2 | 0.5000 | 0.5000 | 29.1667 | `{'safety_regressed_completion': 2}` |
| `narrow_hallway` | 1 | 0.6667 | 0.3333 | 69.3333 | `{'safety_regressed_completion': 1}` |
| `overtaking` | 6 | 0.8889 | 0.0556 | 1.7778 | `{'budget_limited_clean_completion': 1, 'exposure_enabled_completion': 4, 'safety_regressed_completion': 1}` |
| `parallel_traffic` | 4 | 0.6666 | 0.0833 | 5.9166 | `{'exposure_enabled_completion': 2, 'partial_timeout_relief': 1, 'safety_regressed_completion': 1}` |
| `pedestrian_obstruction` | 2 | 1.0000 | 0.0000 | 8.8333 | `{'exposure_enabled_completion': 2}` |
| `pedestrian_overtaking` | 5 | 0.8667 | 0.0000 | 0.0000 | `{'budget_limited_clean_completion': 3, 'late_clean_completion': 1, 'partial_timeout_relief': 1}` |
| `perpendicular_traffic` | 4 | 0.8333 | 0.0000 | 6.5834 | `{'budget_limited_clean_completion': 1, 'exposure_enabled_completion': 3}` |
| `robot_crowding` | 2 | 0.3333 | 0.1667 | 45.0000 | `{'exposure_enabled_completion': 1, 'safety_regressed_completion': 1}` |
| `robot_overtaking` | 4 | 0.8334 | 0.0833 | 1.9167 | `{'budget_limited_clean_completion': 1, 'exposure_enabled_completion': 1, 'partial_timeout_relief': 1, 'safety_regressed_completion': 1}` |
| `t_intersection` | 8 | 0.7083 | 0.1250 | 10.0833 | `{'budget_limited_clean_completion': 4, 'exposure_enabled_completion': 2, 'safety_regressed_completion': 2}` |

## Interpretation Boundary

- `budget_limited_clean_completion` and `late_clean_completion` support the idea that some fixed-horizon failures were budget artifacts.
- `exposure_enabled_completion` means the longer horizon opened a path to success but also increased near-miss exposure; this can be waiting, delayed progress, or simply more time spent in dense pedestrian flow.
- `safety_regressed_completion` should be treated as a caveat, not benchmark-strengthening evidence.
- The aggregate comparison does not contain enough state history to determine whether success came from waiting until dynamic obstacles passed. Use step diagnostics or video for that causal claim.
