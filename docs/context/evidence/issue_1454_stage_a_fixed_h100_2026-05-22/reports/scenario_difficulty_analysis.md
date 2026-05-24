# Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

## Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | francis2023_robot_overtaking | robot_overtaking | 0.7521 | 0.0000 | 0.6333 | 10.5333 | 1.0000 | 0.0000 |
| 2 | classic_doorway_medium | doorway | 0.7085 | 0.0000 | 0.3333 | 5.8000 | 1.0000 | 0.0000 |
| 3 | francis2023_perpendicular_traffic | perpendicular_traffic | 0.7021 | 0.0000 | 0.3333 | 4.4333 | 1.0000 | 0.0000 |
| 4 | classic_doorway_high | doorway | 0.7011 | 0.0000 | 0.2667 | 9.6667 | 1.0000 | 0.0000 |
| 5 | classic_station_platform_medium | station_platform | 0.6957 | 0.0000 | 0.4000 | 2.5667 | 1.0000 | 0.0000 |
| 6 | francis2023_following_human | following_human | 0.6856 | 0.0000 | 0.3000 | 4.3333 | 1.0000 | 0.0000 |
| 7 | francis2023_crowd_navigation | crowd_navigation | 0.6846 | 0.0000 | 0.1667 | 8.9667 | 1.0000 | 0.0000 |
| 8 | classic_cross_trap_high | cross_trap | 0.6809 | 0.0000 | 0.3667 | 2.3000 | 1.0000 | 0.0000 |

## Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| accompanying_peer | 1 | 0.6473 | 0.0000 | 0.3000 | 1.6000 | 0.0000 | francis2023_accompanying_peer |
| blind_corner | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_blind_corner |
| bottleneck | 4 | 0.4483 | 0.0083 | 0.0250 | 1.7000 | 0.0167 | classic_realworld_double_bottleneck_high, classic_bottleneck_high, classic_bottleneck_medium |
| circular_crossing | 1 | 0.2500 | 0.3000 | 0.1000 | 0.3333 | 0.0667 | francis2023_circular_crossing |
| cross_trap | 3 | 0.6105 | 0.0000 | 0.2111 | 2.4444 | 0.0000 | classic_cross_trap_high, classic_cross_trap_medium, classic_cross_trap_low |
| crossing | 1 | 0.5287 | 0.0000 | 0.0333 | 0.2000 | 0.0000 | classic_urban_crossing_medium |
| crowd_navigation | 1 | 0.6846 | 0.0000 | 0.1667 | 8.9667 | 0.0000 | francis2023_crowd_navigation |
| doorway | 3 | 0.6138 | 0.0333 | 0.2889 | 7.8111 | 0.0333 | classic_doorway_medium, classic_doorway_high, classic_doorway_low |
| down_path | 1 | 0.5324 | 0.0000 | 0.0000 | 4.5333 | 0.0000 | francis2023_down_path |
| entering_elevator | 1 | 0.1872 | 0.3333 | 0.1000 | 0.0000 | 0.2000 | francis2023_entering_elevator |
| entering_room | 1 | 0.1745 | 0.3333 | 0.1000 | 0.0000 | 0.0000 | francis2023_entering_room |
| exiting_elevator | 1 | 0.4814 | 0.1333 | 0.6667 | 6.6000 | 0.1000 | francis2023_exiting_elevator |
| exiting_room | 1 | 0.1830 | 0.3333 | 0.1000 | 0.0000 | 0.0000 | francis2023_exiting_room |
| following_human | 1 | 0.6856 | 0.0000 | 0.3000 | 4.3333 | 0.0000 | francis2023_following_human |
| frontal_approach | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_frontal_approach |
| group_crossing | 3 | 0.3076 | 0.2667 | 0.1000 | 3.3667 | 0.0889 | classic_group_crossing_high, classic_group_crossing_medium, classic_group_crossing_low |
| head_on_corridor | 2 | 0.6356 | 0.0000 | 0.1500 | 2.9000 | 0.0000 | classic_head_on_corridor_medium, classic_head_on_corridor_low |
| intersection_no_gesture | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_no_gesture |
| intersection_proceed | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_proceed |
| intersection_wait | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_wait |
| join_group | 1 | 0.3931 | 0.3333 | 0.6667 | 10.8333 | 0.0000 | francis2023_join_group |
| leading_human | 1 | 0.6574 | 0.0000 | 0.3333 | 1.3667 | 0.0000 | francis2023_leading_human |
| leave_group | 1 | 0.4681 | 0.2333 | 0.7667 | 13.4667 | 0.1000 | francis2023_leave_group |
| merging | 2 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | classic_merging_low, classic_merging_medium |
| narrow_doorway | 1 | 0.5628 | 0.0000 | 0.1333 | 0.0000 | 0.0000 | francis2023_narrow_doorway |
| narrow_hallway | 1 | 0.5362 | 0.0000 | 0.1000 | 0.0000 | 0.0000 | francis2023_narrow_hallway |
| overtaking | 2 | 0.5633 | 0.0000 | 0.0500 | 2.5167 | 0.0000 | classic_overtaking_low, classic_overtaking_medium |
| parallel_traffic | 1 | 0.6202 | 0.0000 | 0.0667 | 8.7000 | 0.0000 | francis2023_parallel_traffic |
| pedestrian_obstruction | 1 | 0.4734 | 0.0000 | 0.0000 | 0.1333 | 0.0000 | francis2023_pedestrian_obstruction |
| pedestrian_overtaking | 1 | 0.4431 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_pedestrian_overtaking |
| perpendicular_traffic | 1 | 0.7021 | 0.0000 | 0.3333 | 4.4333 | 0.0000 | francis2023_perpendicular_traffic |
| robot_crowding | 1 | 0.5362 | 0.0667 | 0.8000 | 13.9000 | 0.1000 | francis2023_robot_crowding |
| robot_overtaking | 1 | 0.7521 | 0.0000 | 0.6333 | 10.5333 | 0.0000 | francis2023_robot_overtaking |
| station_platform | 1 | 0.6957 | 0.0000 | 0.4000 | 2.5667 | 0.0000 | classic_station_platform_medium |
| t_intersection | 2 | 0.4582 | 0.0000 | 0.0000 | 0.0667 | 0.0000 | classic_t_intersection_medium, classic_t_intersection_low |

## Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.0457 | 0.4417 | 9 | classic_group_crossing_medium, classic_group_crossing_low, classic_group_crossing_high |
| orca | core | -0.0872 | 0.3333 | 0 | classic_urban_crossing_medium, classic_overtaking_medium, classic_head_on_corridor_medium |
| ppo | experimental | 56609.1549 | 1474249.9167 | 7 | classic_bottleneck_medium, francis2023_narrow_hallway, classic_merging_low |
| prediction_planner | experimental | 21781.4128 | 674999.9167 | 10 | francis2023_narrow_hallway, classic_cross_trap_low, francis2023_blind_corner |
| sacadrl | experimental | 17187.6403 | 250000.0000 | 11 | francis2023_frontal_approach, francis2023_down_path, francis2023_pedestrian_obstruction |
| social_force | core | 0.0415 | 0.3333 | 0 | francis2023_accompanying_peer, francis2023_leading_human, francis2023_following_human |
| socnav_sampling | experimental | 65359.5622 | 900001.6667 | 8 | classic_bottleneck_high, francis2023_narrow_hallway, classic_bottleneck_medium |

## Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

## Difficulty Findings

- No additional scenario-difficulty warnings.
