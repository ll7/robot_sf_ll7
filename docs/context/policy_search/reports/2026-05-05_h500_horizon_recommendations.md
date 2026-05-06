# Policy Search Horizon Recommendations

Generated from safe incumbent policy-search summaries.

## Selected Summaries

| Candidate | Stage | Success | Collision | Near Miss |
|---|---|---:|---:|---:|
| `hybrid_rule_v3_dynamic_relaxed` | `full_matrix_h500` | 0.7778 | 0.0139 | 0.4167 |
| `hybrid_rule_v3_fast_progress` | `full_matrix_h500` | 0.8264 | 0.0139 | 0.4236 |
| `hybrid_rule_v3_fast_progress_static_escape` | `full_matrix_h500` | 0.9028 | 0.0208 | 0.4236 |
| `hybrid_rule_v3_progress_2p4` | `full_matrix_h500` | 0.8056 | 0.0139 | 0.4097 |
| `hybrid_rule_v3_teb_like_rollout` | `full_matrix_h500` | 0.7708 | 0.0139 | 0.4097 |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static02` | `full_matrix_h500` | 0.7778 | 0.0139 | 0.4097 |
| `hybrid_rule_v3_waypoint2_route_lookahead8_static05` | `full_matrix_h500` | 0.7778 | 0.0139 | 0.4097 |
| `hybrid_rule_v4_recovery_aware` | `full_matrix_h500` | 0.8056 | 0.0139 | 0.4097 |
| `scenario_adaptive_hybrid_orca_v1` | `full_matrix_h500` | 0.9097 | 0.0208 | 0.4236 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `full_matrix_h500` | 0.9028 | 0.0139 | 0.4236 |

## Scenario Recommendations

| Scenario | Horizon | Status | Bucket | Success Episodes | P95 Steps | Timeouts/Failures |
|---|---:|---|---|---:|---:|---:|
| `classic_bottleneck_high` | 236 | `recommended` | `medium` | 30 | 216.0 | 0/0 |
| `classic_bottleneck_low` | 150 | `recommended` | `short` | 30 | 130.0 | 0/0 |
| `classic_bottleneck_medium` | 279 | `recommended` | `medium` | 30 | 259.0 | 0/0 |
| `classic_cross_trap_high` | 418 | `recommended` | `long` | 6 | 398.0 | 14/24 |
| `classic_cross_trap_low` | 431 | `recommended` | `long` | 10 | 411.0 | 20/20 |
| `classic_cross_trap_medium` | 445 | `recommended` | `long` | 10 | 425.0 | 20/20 |
| `classic_doorway_high` | 220 | `recommended` | `medium` | 25 | 200.0 | 5/5 |
| `classic_doorway_low` | 347 | `recommended` | `long` | 17 | 327.0 | 13/13 |
| `classic_doorway_medium` | 153 | `recommended` | `medium` | 21 | 133.0 | 9/9 |
| `classic_group_crossing_high` | 141 | `recommended` | `short` | 30 | 121.0 | 0/0 |
| `classic_group_crossing_low` | 127 | `recommended` | `short` | 30 | 107.0 | 0/0 |
| `classic_group_crossing_medium` | 144 | `recommended` | `short` | 30 | 124.0 | 0/0 |
| `classic_head_on_corridor_low` | 221 | `recommended` | `medium` | 30 | 201.0 | 0/0 |
| `classic_head_on_corridor_medium` | 232 | `recommended` | `medium` | 30 | 211.2 | 0/0 |
| `classic_merging_low` | 376 | `recommended` | `long` | 2 | 356.0 | 26/28 |
| `classic_merging_medium` | 500 | `planner_blocked` | `planner_blocked` | 0 | n/a | 30/30 |
| `classic_overtaking_low` | 302 | `recommended` | `long` | 28 | 282.0 | 2/2 |
| `classic_overtaking_medium` | 320 | `recommended` | `long` | 29 | 300.0 | 1/1 |
| `classic_realworld_double_bottleneck_high` | 471 | `recommended` | `long` | 30 | 451.0 | 0/0 |
| `classic_station_platform_medium` | 500 | `planner_blocked` | `planner_blocked` | 0 | n/a | 30/30 |
| `classic_t_intersection_low` | 281 | `recommended` | `medium` | 30 | 261.0 | 0/0 |
| `classic_t_intersection_medium` | 264 | `recommended` | `medium` | 30 | 244.0 | 0/0 |
| `classic_urban_crossing_medium` | 191 | `recommended` | `medium` | 30 | 171.0 | 0/0 |
| `francis2023_accompanying_peer` | 182 | `recommended` | `medium` | 30 | 162.0 | 0/0 |
| `francis2023_blind_corner` | 266 | `recommended` | `medium` | 30 | 246.0 | 0/0 |
| `francis2023_circular_crossing` | 90 | `recommended` | `short` | 20 | 70.0 | 0/10 |
| `francis2023_crowd_navigation` | 258 | `recommended` | `medium` | 30 | 237.9 | 0/0 |
| `francis2023_down_path` | 182 | `recommended` | `medium` | 30 | 162.0 | 0/0 |
| `francis2023_entering_elevator` | 88 | `recommended` | `short` | 30 | 68.0 | 0/0 |
| `francis2023_entering_room` | 112 | `recommended` | `short` | 30 | 92.0 | 0/0 |
| `francis2023_exiting_elevator` | 228 | `recommended` | `medium` | 24 | 208.0 | 6/6 |
| `francis2023_exiting_room` | 123 | `recommended` | `short` | 30 | 103.0 | 0/0 |
| `francis2023_following_human` | 182 | `recommended` | `medium` | 30 | 162.0 | 0/0 |
| `francis2023_frontal_approach` | 210 | `recommended` | `medium` | 30 | 190.0 | 0/0 |
| `francis2023_intersection_no_gesture` | 137 | `recommended` | `short` | 30 | 117.0 | 0/0 |
| `francis2023_intersection_proceed` | 137 | `recommended` | `short` | 30 | 117.0 | 0/0 |
| `francis2023_intersection_wait` | 137 | `recommended` | `short` | 30 | 117.0 | 0/0 |
| `francis2023_join_group` | 132 | `recommended` | `short` | 24 | 112.0 | 6/6 |
| `francis2023_leading_human` | 182 | `recommended` | `medium` | 30 | 162.0 | 0/0 |
| `francis2023_leave_group` | 145 | `recommended` | `short` | 23 | 125.0 | 7/7 |
| `francis2023_narrow_doorway` | 500 | `planner_blocked` | `planner_blocked` | 0 | n/a | 30/30 |
| `francis2023_narrow_hallway` | 272 | `recommended` | `medium` | 26 | 252.0 | 4/4 |
| `francis2023_parallel_traffic` | 211 | `recommended` | `medium` | 30 | 191.0 | 0/0 |
| `francis2023_pedestrian_obstruction` | 228 | `recommended` | `medium` | 30 | 208.0 | 0/0 |
| `francis2023_pedestrian_overtaking` | 188 | `recommended` | `medium` | 30 | 168.0 | 0/0 |
| `francis2023_perpendicular_traffic` | 176 | `recommended` | `medium` | 24 | 156.0 | 6/6 |
| `francis2023_robot_crowding` | 293 | `recommended` | `medium` | 30 | 273.0 | 0/0 |
| `francis2023_robot_overtaking` | 204 | `recommended` | `medium` | 30 | 184.0 | 0/0 |
