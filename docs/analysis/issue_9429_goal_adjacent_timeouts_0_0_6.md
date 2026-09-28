# Frozen 0.0.6 S30/H600 goal-adjacent timeout counts

Claim boundary: descriptive analysis of the frozen release bundle only. Timeout counts come from retained episode outcomes. Final-waypoint wall distances come from deterministic environment initialization at the bundle's exact source commit. No episode was stepped or rerun, and no runtime or frozen artifact was changed.

Evidence status: timeout and waypoint-geometry counts are available. The goal-adjacent predicate is unavailable for all 0/2,108 classified/timeout rows because the bundle did not retain simulation step traces. `NA` is not zero.

## Input identity

- Bundle: `benchmark_0_0_6_s30_h600_20260911_publication_bundle`
- Campaign: `benchmark_0_0_6_s30_h600_20260911`
- Source commit: `31cdfe0361abe2c520117a17f99c1b7a0aba4359`
- Campaign config hash: `60b554cd35c66aa0`
- Scenario matrix hash: `152eba3969a9`
- Scenario manifest: `configs/scenarios/classic_interactions_francis2023.yaml`
- Checksum verification: 111 files, 740933980 bytes
- Episode rows read: 20160

## Exact predicate

Predicate `goal_adjacent_timeout.v1` was configured with `N=100`, `goal_adjacent_radius_m=4.0`, and each episode's runtime completion radius (`robot radius + goal radius`, 2.0 m in this bundle):

```text
outcome.timeout_event is true
AND outcome.collision_event is false
AND min(distance(robot_position[t], final_waypoint) for t in final 100 recorded steps) < 4.0 m
AND min(distance(robot_position[t], final_waypoint) for t in every recorded episode step) > completion_radius_m
```

The last comparison is strict because runtime completion is true at distance `<= completion_radius_m`. A timeout with a missing, malformed, non-finite, or shorter-than-N simulation step trace is unavailable rather than false.

Applied result: the frozen episode rows set `scenario_params.record_simulation_step_trace=false` and contain no `algorithm_metadata.simulation_step_trace.steps`; therefore the trace-dependent clauses were not evaluable.

Unavailable classification reasons:

- `missing_simulation_step_trace`: 2108

## Counts by arm

| arm | episodes | timeouts | trace-classified timeouts | goal-adjacent timeouts | goal-adjacent / timeouts |
| --- | --- | --- | --- | --- | --- |
| goal__differential_drive | 1440 | 36 | 0 | NA | NA |
| guarded_ppo__differential_drive | 1440 | 605 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | 1440 | 86 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | 1440 | 111 | 0 | NA | NA |
| orca__differential_drive | 1440 | 37 | 0 | NA | NA |
| ppo__differential_drive | 1440 | 35 | 0 | NA | NA |
| prediction_planner__differential_drive | 1440 | 24 | 0 | NA | NA |
| predictive_mppi__differential_drive | 1440 | 306 | 0 | NA | NA |
| risk_dwa__differential_drive | 1440 | 86 | 0 | NA | NA |
| sacadrl__differential_drive | 1440 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | 1440 | 89 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | 1440 | 87 | 0 | NA | NA |
| social_force__differential_drive | 1440 | 604 | 0 | NA | NA |
| socnav_sampling__differential_drive | 1440 | 0 | 0 | 0 | 0.000000 |

## Counts by arm and scenario family

| arm | scenario family | episodes | timeouts | trace-classified timeouts | goal-adjacent timeouts | goal-adjacent / timeouts |
| --- | --- | --- | --- | --- | --- | --- |
| goal__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | merging | 60 | 36 | 0 | NA | NA |
| goal__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | station_platform | 30 | 0 | 0 | 0 | 0.000000 |
| goal__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | accompanying_peer | 30 | 20 | 0 | NA | NA |
| guarded_ppo__differential_drive | blind_corner | 30 | 3 | 0 | NA | NA |
| guarded_ppo__differential_drive | bottleneck | 120 | 61 | 0 | NA | NA |
| guarded_ppo__differential_drive | circular_crossing | 30 | 13 | 0 | NA | NA |
| guarded_ppo__differential_drive | cross_trap | 90 | 13 | 0 | NA | NA |
| guarded_ppo__differential_drive | crossing | 30 | 27 | 0 | NA | NA |
| guarded_ppo__differential_drive | crowd_navigation | 30 | 29 | 0 | NA | NA |
| guarded_ppo__differential_drive | doorway | 90 | 37 | 0 | NA | NA |
| guarded_ppo__differential_drive | down_path | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | exiting_elevator | 30 | 29 | 0 | NA | NA |
| guarded_ppo__differential_drive | exiting_room | 30 | 28 | 0 | NA | NA |
| guarded_ppo__differential_drive | following_human | 30 | 2 | 0 | NA | NA |
| guarded_ppo__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | group_crossing | 90 | 58 | 0 | NA | NA |
| guarded_ppo__differential_drive | head_on_corridor | 60 | 27 | 0 | NA | NA |
| guarded_ppo__differential_drive | intersection_no_gesture | 30 | 1 | 0 | NA | NA |
| guarded_ppo__differential_drive | intersection_proceed | 30 | 1 | 0 | NA | NA |
| guarded_ppo__differential_drive | intersection_wait | 30 | 1 | 0 | NA | NA |
| guarded_ppo__differential_drive | join_group | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | leave_group | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | merging | 60 | 2 | 0 | NA | NA |
| guarded_ppo__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | narrow_hallway | 30 | 5 | 0 | NA | NA |
| guarded_ppo__differential_drive | overtaking | 60 | 6 | 0 | NA | NA |
| guarded_ppo__differential_drive | parallel_traffic | 30 | 24 | 0 | NA | NA |
| guarded_ppo__differential_drive | pedestrian_obstruction | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| guarded_ppo__differential_drive | perpendicular_traffic | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | robot_crowding | 30 | 28 | 0 | NA | NA |
| guarded_ppo__differential_drive | robot_overtaking | 30 | 30 | 0 | NA | NA |
| guarded_ppo__differential_drive | station_platform | 30 | 10 | 0 | NA | NA |
| guarded_ppo__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | cross_trap | 90 | 13 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | crossing | 30 | 2 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | doorway | 90 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | intersection_no_gesture | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | intersection_proceed | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | intersection_wait | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | merging | 60 | 11 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | narrow_hallway | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | parallel_traffic | 30 | 2 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | perpendicular_traffic | 30 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | station_platform | 30 | 14 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | cross_trap | 90 | 8 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | crossing | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | doorway | 90 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | exiting_elevator | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | intersection_no_gesture | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | intersection_proceed | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | intersection_wait | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | join_group | 30 | 2 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | leave_group | 30 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | merging | 60 | 18 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | narrow_hallway | 30 | 6 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | overtaking | 60 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | parallel_traffic | 30 | 1 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | perpendicular_traffic | 30 | 3 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | robot_crowding | 30 | 5 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | station_platform | 30 | 16 | 0 | NA | NA |
| hybrid_rule_v3_fast_progress_static_escape_continuous__differential_drive | t_intersection | 60 | 4 | 0 | NA | NA |
| orca__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | doorway | 90 | 3 | 0 | NA | NA |
| orca__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | merging | 60 | 3 | 0 | NA | NA |
| orca__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| orca__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| orca__differential_drive | station_platform | 30 | 1 | 0 | NA | NA |
| orca__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | bottleneck | 120 | 2 | 0 | NA | NA |
| ppo__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | cross_trap | 90 | 6 | 0 | NA | NA |
| ppo__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | crowd_navigation | 30 | 3 | 0 | NA | NA |
| ppo__differential_drive | doorway | 90 | 1 | 0 | NA | NA |
| ppo__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | exiting_elevator | 30 | 2 | 0 | NA | NA |
| ppo__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | group_crossing | 90 | 7 | 0 | NA | NA |
| ppo__differential_drive | head_on_corridor | 60 | 1 | 0 | NA | NA |
| ppo__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | join_group | 30 | 3 | 0 | NA | NA |
| ppo__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | leave_group | 30 | 1 | 0 | NA | NA |
| ppo__differential_drive | merging | 60 | 1 | 0 | NA | NA |
| ppo__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | overtaking | 60 | 3 | 0 | NA | NA |
| ppo__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| ppo__differential_drive | robot_overtaking | 30 | 1 | 0 | NA | NA |
| ppo__differential_drive | station_platform | 30 | 4 | 0 | NA | NA |
| ppo__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | join_group | 30 | 11 | 0 | NA | NA |
| prediction_planner__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | leave_group | 30 | 5 | 0 | NA | NA |
| prediction_planner__differential_drive | merging | 60 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| prediction_planner__differential_drive | station_platform | 30 | 8 | 0 | NA | NA |
| prediction_planner__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | accompanying_peer | 30 | 21 | 0 | NA | NA |
| predictive_mppi__differential_drive | blind_corner | 30 | 3 | 0 | NA | NA |
| predictive_mppi__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | cross_trap | 90 | 21 | 0 | NA | NA |
| predictive_mppi__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | crowd_navigation | 30 | 14 | 0 | NA | NA |
| predictive_mppi__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | down_path | 30 | 24 | 0 | NA | NA |
| predictive_mppi__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | following_human | 30 | 20 | 0 | NA | NA |
| predictive_mppi__differential_drive | frontal_approach | 30 | 19 | 0 | NA | NA |
| predictive_mppi__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | head_on_corridor | 60 | 45 | 0 | NA | NA |
| predictive_mppi__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | leading_human | 30 | 21 | 0 | NA | NA |
| predictive_mppi__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | merging | 60 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | narrow_hallway | 30 | 1 | 0 | NA | NA |
| predictive_mppi__differential_drive | overtaking | 60 | 27 | 0 | NA | NA |
| predictive_mppi__differential_drive | parallel_traffic | 30 | 21 | 0 | NA | NA |
| predictive_mppi__differential_drive | pedestrian_obstruction | 30 | 15 | 0 | NA | NA |
| predictive_mppi__differential_drive | pedestrian_overtaking | 30 | 30 | 0 | NA | NA |
| predictive_mppi__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| predictive_mppi__differential_drive | robot_overtaking | 30 | 7 | 0 | NA | NA |
| predictive_mppi__differential_drive | station_platform | 30 | 17 | 0 | NA | NA |
| predictive_mppi__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | crowd_navigation | 30 | 2 | 0 | NA | NA |
| risk_dwa__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | frontal_approach | 30 | 20 | 0 | NA | NA |
| risk_dwa__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | head_on_corridor | 60 | 13 | 0 | NA | NA |
| risk_dwa__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | merging | 60 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | overtaking | 60 | 24 | 0 | NA | NA |
| risk_dwa__differential_drive | parallel_traffic | 30 | 9 | 0 | NA | NA |
| risk_dwa__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| risk_dwa__differential_drive | robot_overtaking | 30 | 1 | 0 | NA | NA |
| risk_dwa__differential_drive | station_platform | 30 | 15 | 0 | NA | NA |
| risk_dwa__differential_drive | t_intersection | 60 | 2 | 0 | NA | NA |
| sacadrl__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | merging | 60 | 1 | 0 | NA | NA |
| sacadrl__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| sacadrl__differential_drive | station_platform | 30 | 1 | 0 | NA | NA |
| sacadrl__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | bottleneck | 120 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | cross_trap | 90 | 13 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | crossing | 30 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | doorway | 90 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | intersection_no_gesture | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | intersection_proceed | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | intersection_wait | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | merging | 60 | 11 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | narrow_hallway | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | overtaking | 60 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | parallel_traffic | 30 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | perpendicular_traffic | 30 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | station_platform | 30 | 14 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | cross_trap | 90 | 13 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | crossing | 30 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | doorway | 90 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | intersection_no_gesture | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | intersection_proceed | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | intersection_wait | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | merging | 60 | 11 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | narrow_hallway | 30 | 3 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | overtaking | 60 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | parallel_traffic | 30 | 2 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | perpendicular_traffic | 30 | 1 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | station_platform | 30 | 14 | 0 | NA | NA |
| scenario_adaptive_hybrid_orca_v2_collision_guard__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | bottleneck | 120 | 60 | 0 | NA | NA |
| social_force__differential_drive | circular_crossing | 30 | 23 | 0 | NA | NA |
| social_force__differential_drive | cross_trap | 90 | 77 | 0 | NA | NA |
| social_force__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | crowd_navigation | 30 | 2 | 0 | NA | NA |
| social_force__differential_drive | doorway | 90 | 90 | 0 | NA | NA |
| social_force__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | entering_elevator | 30 | 30 | 0 | NA | NA |
| social_force__differential_drive | entering_room | 30 | 30 | 0 | NA | NA |
| social_force__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | exiting_room | 30 | 30 | 0 | NA | NA |
| social_force__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | frontal_approach | 30 | 1 | 0 | NA | NA |
| social_force__differential_drive | group_crossing | 90 | 90 | 0 | NA | NA |
| social_force__differential_drive | head_on_corridor | 60 | 4 | 0 | NA | NA |
| social_force__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | leading_human | 30 | 1 | 0 | NA | NA |
| social_force__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | merging | 60 | 60 | 0 | NA | NA |
| social_force__differential_drive | narrow_doorway | 30 | 30 | 0 | NA | NA |
| social_force__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | overtaking | 60 | 55 | 0 | NA | NA |
| social_force__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | robot_crowding | 30 | 1 | 0 | NA | NA |
| social_force__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| social_force__differential_drive | station_platform | 30 | 20 | 0 | NA | NA |
| social_force__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | accompanying_peer | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | blind_corner | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | bottleneck | 120 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | circular_crossing | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | cross_trap | 90 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | crossing | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | crowd_navigation | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | doorway | 90 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | down_path | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | entering_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | entering_room | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | exiting_elevator | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | exiting_room | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | following_human | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | frontal_approach | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | group_crossing | 90 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | head_on_corridor | 60 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | intersection_no_gesture | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | intersection_proceed | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | intersection_wait | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | join_group | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | leading_human | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | leave_group | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | merging | 60 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | narrow_doorway | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | narrow_hallway | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | overtaking | 60 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | parallel_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | pedestrian_obstruction | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | pedestrian_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | perpendicular_traffic | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | robot_crowding | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | robot_overtaking | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | station_platform | 30 | 0 | 0 | 0 | 0.000000 |
| socnav_sampling__differential_drive | t_intersection | 60 | 0 | 0 | 0 | 0.000000 |

## Final-waypoint distance to the nearest static wall

One final waypoint was reconstructed for every scenario/seed pair. All arms share the same scenario-seed initialization, so the geometry denominator is 48 scenarios × 30 seeds = 1,440 waypoints, not 20,160 arm-episode rows. The distance is the Shapely point distance to the nearest parsed static obstacle polygon or map-boundary segment under the legacy SVG geometry contract at the frozen source commit.

`within threshold` means wall distance `<= completion_radius_m + 0.5 m`.

| scenario family | waypoints | completion radius m | threshold m | min m | p25 m | median m | p75 m | p95 m | max m | within threshold | fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accompanying_peer | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| blind_corner | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| bottleneck | 120 | 2.000000 | 2.500000 | 2.966015 | 4.041215 | 4.515901 | 4.945734 | 5.338006 | 5.547676 | 0 | 0.000000 |
| circular_crossing | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| cross_trap | 90 | 2.000000 | 2.500000 | 3.004997 | 3.458594 | 3.974364 | 4.635639 | 5.232523 | 5.682213 | 0 | 0.000000 |
| crossing | 30 | 2.000000 | 2.500000 | 1.004997 | 1.352513 | 1.654470 | 2.123657 | 2.431309 | 2.754289 | 29 | 0.966667 |
| crowd_navigation | 30 | 2.000000 | 2.500000 | 1.033870 | 1.301968 | 1.573745 | 1.836186 | 2.332172 | 2.509363 | 29 | 0.966667 |
| doorway | 90 | 2.000000 | 2.500000 | 2.564905 | 3.018502 | 3.534272 | 4.195547 | 4.792431 | 5.242121 | 0 | 0.000000 |
| down_path | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| entering_elevator | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| entering_room | 30 | 2.000000 | 2.500000 | 4.036595 | 4.318474 | 4.591159 | 4.852645 | 5.218015 | 5.450146 | 0 | 0.000000 |
| exiting_elevator | 30 | 2.000000 | 2.500000 | 1.658894 | 2.205774 | 2.512818 | 2.765865 | 2.938467 | 2.997502 | 14 | 0.466667 |
| exiting_room | 30 | 2.000000 | 2.500000 | 1.658894 | 2.205774 | 2.512818 | 2.765865 | 2.938467 | 2.997502 | 14 | 0.466667 |
| following_human | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| frontal_approach | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| group_crossing | 90 | 2.000000 | 2.500000 | 3.512273 | 4.778597 | 5.214686 | 5.498754 | 5.958811 | 6.061397 | 0 | 0.000000 |
| head_on_corridor | 60 | 2.000000 | 2.500000 | 1.684604 | 3.202574 | 3.774545 | 4.323535 | 4.632092 | 4.883674 | 6 | 0.100000 |
| intersection_no_gesture | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| intersection_proceed | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| intersection_wait | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| join_group | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| leading_human | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| leave_group | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| merging | 60 | 2.000000 | 2.500000 | 3.727741 | 5.171855 | 5.452569 | 5.755092 | 6.565770 | 6.673652 | 0 | 0.000000 |
| narrow_doorway | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| narrow_hallway | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| overtaking | 60 | 2.000000 | 2.500000 | 1.472283 | 3.200674 | 3.671920 | 4.375507 | 4.876632 | 5.069745 | 8 | 0.133333 |
| parallel_traffic | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| pedestrian_obstruction | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| pedestrian_overtaking | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| perpendicular_traffic | 30 | 2.000000 | 2.500000 | 1.002498 | 1.176256 | 1.327235 | 1.561829 | 1.715655 | 1.877144 | 30 | 1.000000 |
| robot_crowding | 30 | 2.000000 | 2.500000 | 1.022650 | 1.274590 | 1.417047 | 1.867635 | 2.396794 | 2.547856 | 29 | 0.966667 |
| robot_overtaking | 30 | 2.000000 | 2.500000 | 1.002498 | 1.234135 | 1.487182 | 1.794226 | 2.103533 | 2.341106 | 30 | 1.000000 |
| station_platform | 30 | 2.000000 | 2.500000 | 2.004997 | 2.468271 | 2.974364 | 3.588453 | 4.207066 | 4.682213 | 9 | 0.300000 |
| t_intersection | 60 | 2.000000 | 2.500000 | 0.536139 | 1.096278 | 1.536893 | 1.967413 | 2.291095 | 2.391682 | 60 | 1.000000 |
| ALL | 1440 | 2.000000 | 2.500000 | 0.536139 | 1.383893 | 2.087976 | 4.115754 | 5.450449 | 6.673652 | 798 | 0.554167 |

## Reproduction

Run the committed analyzer from any checkout while pointing `--source-root` at an unchanged checkout of the source commit above:

```bash
uv run python scripts/analysis/analyze_goal_adjacent_timeouts_issue_9429.py \
  --bundle-root /path/to/benchmark_0_0_6_s30_h600_20260911_publication_bundle \
  --source-root /path/to/robot_sf_ll7-at-31cdfe03 \
  --output docs/analysis/issue_9429_goal_adjacent_timeouts_0_0_6.md
```

Report schema: `issue_9429_goal_adjacent_timeout_report.v1`.
