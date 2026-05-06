# Camera-Ready Campaign Analysis

- Campaign ID: `issue1023_scenario_horizons_candidates_local_2026-05-06`
- Campaign root: `/home/luttkule/git/robot_sf_ll7.worktrees/origin-1023-validate-scenario-h500-bench/output/benchmarks/issue_1023/issue1023_scenario_horizons_candidates_local_2026-05-06`
- Runtime sec: `1966.5879543470073`
- Episodes/sec: `0.6590094265223588`

## Planner Diagnostics

| planner | algo | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | ok | 144 | 0.0556 | 0.6181 | -0.1904378873091616 | 0 | 44.7352 | 3.2189 |
| hybrid_rule_v3_fast_progress_static_escape | hybrid_rule_local_planner | ok | 144 | 0.9028 | 0.0278 | -0.08744030345552531 | 0 | 576.6053 | 0.2497 |
| orca | orca | ok | 144 | 0.7569 | 0.1667 | -0.25134526846413413 | 0 | 71.2254 | 2.0217 |
| ppo | ppo | ok | 144 | 0.8056 | 0.1667 | -0.20744673065513033 | 0 | 140.1452 | 1.0275 |
| prediction_planner | prediction_planner | ok | 144 | 0.4931 | 0.4514 | -0.14075642225341856 | 0 | 396.6133 | 0.3631 |
| sacadrl | sacadrl | ok | 144 | 0.0833 | 0.6667 | -0.2725885642962835 | 0 | 57.2194 | 2.5166 |
| scenario_adaptive_hybrid_orca_v1 | hybrid_rule_local_planner | ok | 144 | 0.9097 | 0.0278 | -0.0834875718412503 | 0 | 570.8285 | 0.2523 |
| social_force | social_force | ok | 144 | 0.0139 | 0.3819 | -0.9537106074382992 | 0 | 75.1397 | 1.9164 |
| socnav_sampling | socnav_sampling | ok | 144 | 0.4028 | 0.5972 | -0.08483026461988694 | 0 | 29.0170 | 4.9626 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| hybrid_rule_v3_fast_progress_static_escape | 576.6053 | 7.9583 | 19.7112 |
| scenario_adaptive_hybrid_orca_v1 | 570.8285 | 7.8809 | 19.6967 |
| prediction_planner | 396.6133 | 5.4713 | 9.2060 |

- `hybrid_rule_v3_fast_progress_static_escape` top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=22.3525s p95=25.8219s (episodes=3)
  - `classic_station_platform_medium` mean=21.4814s p95=22.6859s (episodes=3)
  - `francis2023_narrow_doorway` mean=18.4120s p95=18.8406s (episodes=3)
- `scenario_adaptive_hybrid_orca_v1` top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=22.4131s p95=25.9149s (episodes=3)
  - `classic_station_platform_medium` mean=21.4876s p95=22.7110s (episodes=3)
  - `francis2023_narrow_doorway` mean=18.4079s p95=18.8192s (episodes=3)
- `prediction_planner` top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=13.0431s p95=15.5502s (episodes=3)
  - `classic_station_platform_medium` mean=9.8937s p95=21.4729s (episodes=3)
  - `francis2023_parallel_traffic` mean=9.2619s p95=9.3927s (episodes=3)

## Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

### Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | classic_cross_trap_high | cross_trap | 0.9356 | 0.0000 | 0.7778 | 16.4444 | 1.0000 | 0.0000 |
| 2 | francis2023_narrow_hallway | narrow_hallway | 0.9106 | 0.0000 | 1.0000 | 8.1111 | 1.0000 | 0.0000 |
| 3 | francis2023_robot_crowding | robot_crowding | 0.8968 | 0.1111 | 0.8889 | 44.5556 | 0.9639 | 0.2222 |
| 4 | classic_station_platform_medium | station_platform | 0.8894 | 0.0000 | 0.5556 | 24.2222 | 1.0000 | 0.0000 |
| 5 | classic_merging_medium | merging | 0.8213 | 0.1111 | 0.5556 | 20.6667 | 0.9663 | 0.2222 |
| 6 | classic_doorway_high | doorway | 0.8160 | 0.1111 | 0.5556 | 12.6667 | 0.9893 | 0.2222 |
| 7 | classic_bottleneck_high | bottleneck | 0.8154 | 0.0000 | 0.3333 | 26.7778 | 1.0000 | 0.0000 |
| 8 | classic_cross_trap_medium | cross_trap | 0.7862 | 0.1111 | 0.5556 | 10.7778 | 0.9350 | 0.2222 |

### Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| accompanying_peer | 1 | 0.4207 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | francis2023_accompanying_peer |
| blind_corner | 1 | 0.4793 | 0.3333 | 0.4444 | 5.3333 | 0.0000 | francis2023_blind_corner |
| bottleneck | 4 | 0.5733 | 0.2222 | 0.2778 | 18.5000 | 0.1389 | classic_bottleneck_high, classic_bottleneck_medium, classic_realworld_double_bottleneck_high |
| circular_crossing | 1 | 0.4106 | 0.3333 | 0.3333 | 1.0000 | 0.3333 | francis2023_circular_crossing |
| cross_trap | 3 | 0.7046 | 0.1481 | 0.5556 | 12.9259 | 0.0741 | classic_cross_trap_high, classic_cross_trap_medium, classic_cross_trap_low |
| crossing | 1 | 0.4378 | 0.3333 | 0.3333 | 4.4444 | 0.0000 | classic_urban_crossing_medium |
| crowd_navigation | 1 | 0.5133 | 0.3333 | 0.6667 | 10.8889 | 0.0000 | francis2023_crowd_navigation |
| doorway | 3 | 0.7241 | 0.1111 | 0.4815 | 8.0741 | 0.1111 | classic_doorway_high, classic_doorway_medium, classic_doorway_low |
| down_path | 1 | 0.3585 | 0.3333 | 0.0000 | 4.2222 | 0.0000 | francis2023_down_path |
| entering_elevator | 1 | 0.3809 | 0.3333 | 0.1111 | 0.0000 | 0.3333 | francis2023_entering_elevator |
| entering_room | 1 | 0.2745 | 0.3333 | 0.1111 | 0.0000 | 0.0000 | francis2023_entering_room |
| exiting_elevator | 1 | 0.4537 | 0.3333 | 0.6667 | 6.6667 | 0.0000 | francis2023_exiting_elevator |
| exiting_room | 1 | 0.3043 | 0.3333 | 0.1111 | 0.0000 | 0.0000 | francis2023_exiting_room |
| following_human | 1 | 0.4495 | 0.3333 | 0.3333 | 4.3333 | 0.0000 | francis2023_following_human |
| frontal_approach | 1 | 0.5335 | 0.3333 | 0.6667 | 6.0000 | 0.0000 | francis2023_frontal_approach |
| group_crossing | 3 | 0.2062 | 0.4074 | 0.0741 | 5.3333 | 0.1111 | classic_group_crossing_low, classic_group_crossing_high, classic_group_crossing_medium |
| head_on_corridor | 2 | 0.4412 | 0.3333 | 0.3889 | 10.0556 | 0.0000 | classic_head_on_corridor_medium, classic_head_on_corridor_low |
| intersection_no_gesture | 1 | 0.3676 | 0.3333 | 0.2222 | 1.5556 | 0.0000 | francis2023_intersection_no_gesture |
| intersection_proceed | 1 | 0.3106 | 0.3333 | 0.0000 | 1.4444 | 0.0000 | francis2023_intersection_proceed |
| intersection_wait | 1 | 0.3106 | 0.3333 | 0.0000 | 1.4444 | 0.0000 | francis2023_intersection_wait |
| join_group | 1 | 0.4899 | 0.3333 | 0.6667 | 10.4444 | 0.0000 | francis2023_join_group |
| leading_human | 1 | 0.4207 | 0.3333 | 0.3333 | 1.0000 | 0.0000 | francis2023_leading_human |
| leave_group | 1 | 0.7239 | 0.2222 | 0.7778 | 11.6666 | 0.2222 | francis2023_leave_group |
| merging | 2 | 0.7338 | 0.1667 | 0.5000 | 12.0000 | 0.2222 | classic_merging_medium, classic_merging_low |
| narrow_doorway | 1 | 0.6814 | 0.0000 | 0.3333 | 0.0000 | 0.0000 | francis2023_narrow_doorway |
| narrow_hallway | 1 | 0.9106 | 0.0000 | 1.0000 | 8.1111 | 0.0000 | francis2023_narrow_hallway |
| overtaking | 2 | 0.3649 | 0.3333 | 0.3333 | 3.7778 | 0.0000 | classic_overtaking_medium, classic_overtaking_low |
| parallel_traffic | 1 | 0.2745 | 0.4444 | 0.1111 | 21.5556 | 0.1111 | francis2023_parallel_traffic |
| pedestrian_obstruction | 1 | 0.5064 | 0.3333 | 0.5556 | 8.3333 | 0.0000 | francis2023_pedestrian_obstruction |
| pedestrian_overtaking | 1 | 0.1617 | 0.4444 | 0.0000 | 0.0000 | 0.1111 | francis2023_pedestrian_overtaking |
| perpendicular_traffic | 1 | 0.2824 | 0.4444 | 0.4444 | 7.7778 | 0.2222 | francis2023_perpendicular_traffic |
| robot_crowding | 1 | 0.8968 | 0.1111 | 0.8889 | 44.5556 | 0.2222 | francis2023_robot_crowding |
| robot_overtaking | 1 | 0.5856 | 0.3333 | 0.6667 | 9.3333 | 0.0000 | francis2023_robot_overtaking |
| station_platform | 1 | 0.8894 | 0.0000 | 0.5556 | 24.2222 | 0.0000 | classic_station_platform_medium |
| t_intersection | 2 | 0.4468 | 0.3333 | 0.5556 | 8.2222 | 0.0000 | classic_t_intersection_medium, classic_t_intersection_low |

### Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.1802 | 0.3833 | 13 | classic_head_on_corridor_medium, classic_group_crossing_high, francis2023_crowd_navigation |
| hybrid_rule_v3_fast_progress_static_escape | experimental | -30682.4825 | 3.0131 | 3 | classic_cross_trap_low, francis2023_crowd_navigation, francis2023_leave_group |
| orca | core | -0.3022 | 0.2407 | 0 | classic_cross_trap_high, francis2023_narrow_hallway, classic_station_platform_medium |
| ppo | experimental | -8000.5442 | 0.2972 | 0 | classic_merging_low, classic_merging_medium, francis2023_narrow_doorway |
| prediction_planner | experimental | -12504.4929 | 83324.8856 | 9 | classic_bottleneck_low, francis2023_down_path, classic_group_crossing_medium |
| sacadrl | experimental | 34436.6171 | 1416675.1667 | 15 | classic_group_crossing_low, francis2023_down_path, classic_group_crossing_medium |
| scenario_adaptive_hybrid_orca_v1 | experimental | -30682.5150 | 3.0131 | 3 | classic_cross_trap_low, francis2023_crowd_navigation, classic_overtaking_low |
| social_force | core | 0.1220 | 0.5000 | 12 | francis2023_accompanying_peer, francis2023_leading_human, francis2023_intersection_no_gesture |
| socnav_sampling | experimental | 8680.0337 | 250000.0833 | 5 | francis2023_down_path, classic_group_crossing_low, classic_group_crossing_medium |

### Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- scenario_adaptive_hybrid_orca_v1: snqi_mean mismatch: row=-0.079500, episodes=-0.083488
