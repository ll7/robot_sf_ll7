# Scenario Seed Sensitivity Analysis

## Contract

- Campaign root: `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23`
- Claim boundary: derived analysis over durable compact artifacts; diagnostic scenario prioritization, not causal mechanism proof or paper-facing significance.
- Planner selection: benchmark_success rows ranked by success_mean descending, then collisions_mean, time_to_goal_norm_mean, near_misses_mean, and planner_key ascending.

## Selected Planners

| Rank | Planner | Success | Collision | Time to goal norm | Near misses | Mode |
|---:|---|---:|---:|---:|---:|---|
| 1 | `hybrid_rule_v3_fast_progress_static_escape_continuous` | 0.8771 | 0.0250 | 0.7221 | 18.9146 | `adapter` |
| 2 | `scenario_adaptive_hybrid_orca_v1` | 0.8729 | 0.0333 | 0.7083 | 20.7771 | `adapter` |
| 3 | `scenario_adaptive_hybrid_orca_v2_collision_guard` | 0.8729 | 0.0333 | 0.7084 | 20.7771 | `adapter` |
| 4 | `hybrid_rule_v3_fast_progress_static_escape` | 0.8646 | 0.0354 | 0.7118 | 21.8875 | `adapter` |

## Summary

- Scenarios classified: `48`.
- Seed-sensitive: `25`.
- Not seed-sensitive: `23`.
- Inconclusive: `0`.

## Seed-Sensitive Scenarios

| Scenario | Range | Mean success | Hard seeds | Easy seeds | Most brittle planner |
|---|---:|---:|---|---|---|
| `classic_cross_trap_high` | 1.0000 | 0.7000 | `111 112 115` | `113 114 116 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_cross_trap_medium` | 1.0000 | 0.8000 | `111 115` | `112 113 114 116 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_doorway_high` | 1.0000 | 0.6750 | `112 113 117 118` | `111 114 115 116 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_doorway_low` | 1.0000 | 0.8500 | `111` | `112 113 114 115 116 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_doorway_medium` | 1.0000 | 0.6250 | `114 115 117 118` | `111 112 113 116 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_group_crossing_high` | 1.0000 | 0.8250 | `117 119` | `111 112 113 114 115 116 118 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_group_crossing_low` | 1.0000 | 0.9000 | `117` | `111 112 113 114 115 116 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_head_on_corridor_low` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_head_on_corridor_medium` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_merging_low` | 1.0000 | 0.7000 | `111 117 119` | `112 113 114 115 116 118 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_urban_crossing_medium` | 1.0000 | 0.8250 | `112 116` | `111 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_blind_corner` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_circular_crossing` | 1.0000 | 0.9000 | `111` | `112 113 114 115 116 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_intersection_no_gesture` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_intersection_proceed` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_intersection_wait` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_join_group` | 1.0000 | 0.6000 | `112 113 115 116` | `111 114 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_narrow_hallway` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `francis2023_perpendicular_traffic` | 1.0000 | 0.9000 | `116` | `111 112 113 114 115 117 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| `classic_group_crossing_medium` | 0.7500 | 0.9000 | `117` | `111 112 113 114 115 116 118 119 120` | `scenario_adaptive_hybrid_orca_v2_collision_guard` |

## Hardest Seed IDs Across Scenarios

| Seed | Mean success | Hard scenario count |
|---:|---:|---:|
| 116 | 0.7396 | 12 |
| 117 | 0.8281 | 8 |
| 111 | 0.8490 | 7 |
| 115 | 0.8594 | 7 |
| 112 | 0.8646 | 7 |
| 113 | 0.8958 | 6 |
| 118 | 0.9115 | 4 |
| 119 | 0.9115 | 4 |
| 120 | 0.9271 | 4 |
| 114 | 0.9323 | 3 |

## Interpretation Limit

This derived analysis can prioritize follow-up scenario inspection. It does not prove a causal mechanism for any scenario, and it should not be reused as paper-facing significance evidence without a pre-specified larger seed budget.
