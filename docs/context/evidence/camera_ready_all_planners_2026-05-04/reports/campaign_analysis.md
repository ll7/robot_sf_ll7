# Camera-Ready Campaign Analysis

- Campaign ID: `paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Campaign root: `/home/luttkule/git/robot_sf_ll7.worktrees/benchmark_2026-05-04/output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_main_latest_all_20260504_171217`
- Runtime sec: `1894.2798869700637`
- Episodes/sec: `0.5321283337977657`

## Planner Diagnostics

| planner | algo | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | ok | 144 | 0.0139 | 0.2361 | -0.16556360558976968 | 0 | 92.5234 | 1.5564 |
| orca | orca | ok | 144 | 0.1806 | 0.0347 | -0.25887773381978324 | 0 | 153.8507 | 0.9360 |
| ppo | ppo | ok | 144 | 0.2500 | 0.0903 | -0.3059706716624499 | 0 | 233.9481 | 0.6155 |
| prediction_planner | prediction_planner | ok | 144 | 0.0694 | 0.2083 | -0.1945193436079888 | 0 | 1006.0669 | 0.1431 |
| sacadrl | sacadrl | ok | 144 | 0.0000 | 0.3889 | -0.2833891285061912 | 0 | 186.4920 | 0.7722 |
| social_force | social_force | ok | 144 | 0.0000 | 0.2083 | -0.8534776695868902 | 0 | 108.2956 | 1.3297 |
| socnav_bench | socnav_bench | unknown | 0 | 0.0000 | 0.0000 | nan | 0 | 0.1830 | 0.0000 |
| socnav_sampling | socnav_sampling | ok | 144 | 0.1736 | 0.5278 | -0.13902763418765163 | 0 | 103.4304 | 1.3922 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| prediction_planner | 1006.0669 | 6.9455 | 7.9159 |
| ppo | 233.9481 | 1.5093 | 1.9424 |
| sacadrl | 186.4920 | 1.2493 | 1.7730 |

- `prediction_planner` top slow scenarios:
  - `classic_bottleneck_low` mean=10.4252s p95=20.1699s (episodes=3)
  - `classic_realworld_double_bottleneck_high` mean=8.1599s p95=8.1852s (episodes=3)
  - `classic_bottleneck_high` mean=7.9822s p95=10.2280s (episodes=3)
- `ppo` top slow scenarios:
  - `classic_bottleneck_low` mean=2.1698s p95=2.9497s (episodes=3)
  - `classic_realworld_double_bottleneck_high` mean=1.9588s p95=2.0217s (episodes=3)
  - `francis2023_intersection_no_gesture` mean=1.9445s p95=2.0311s (episodes=3)
- `sacadrl` top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=1.8329s p95=1.9779s (episodes=3)
  - `francis2023_perpendicular_traffic` mean=1.7770s p95=1.8159s (episodes=3)
  - `francis2023_intersection_no_gesture` mean=1.7589s p95=2.2447s (episodes=3)

## Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

### Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | francis2023_robot_crowding | robot_crowding | 0.7803 | 0.0000 | 0.7778 | 15.8889 | 1.0000 | 0.0000 |
| 2 | francis2023_robot_overtaking | robot_overtaking | 0.7511 | 0.0000 | 0.6667 | 9.3333 | 1.0000 | 0.0000 |
| 3 | francis2023_exiting_elevator | exiting_elevator | 0.7351 | 0.0000 | 0.6667 | 6.6667 | 1.0000 | 0.0000 |
| 4 | classic_doorway_high | doorway | 0.7314 | 0.0000 | 0.3333 | 11.3333 | 1.0000 | 0.0000 |
| 5 | francis2023_crowd_navigation | crowd_navigation | 0.7122 | 0.0000 | 0.3333 | 7.4445 | 1.0000 | 0.0000 |
| 6 | francis2023_following_human | following_human | 0.6867 | 0.0000 | 0.3333 | 4.3333 | 1.0000 | 0.0000 |
| 7 | classic_cross_trap_high | cross_trap | 0.6782 | 0.0000 | 0.3333 | 6.3333 | 1.0000 | 0.0000 |
| 8 | classic_doorway_medium | doorway | 0.6771 | 0.0000 | 0.3333 | 3.4445 | 1.0000 | 0.0000 |

### Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| accompanying_peer | 1 | 0.6644 | 0.0000 | 0.3333 | 1.0000 | 0.0000 | francis2023_accompanying_peer |
| blind_corner | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_blind_corner |
| bottleneck | 4 | 0.5032 | 0.0000 | 0.0000 | 1.5556 | 0.0000 | classic_realworld_double_bottleneck_high, classic_bottleneck_high, classic_bottleneck_low |
| circular_crossing | 1 | 0.3324 | 0.2222 | 0.3333 | 1.0000 | 0.2222 | francis2023_circular_crossing |
| cross_trap | 3 | 0.6335 | 0.0000 | 0.1481 | 6.5556 | 0.0000 | classic_cross_trap_high, classic_cross_trap_medium, classic_cross_trap_low |
| crossing | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | classic_urban_crossing_medium |
| crowd_navigation | 1 | 0.7122 | 0.0000 | 0.3333 | 7.4445 | 0.0000 | francis2023_crowd_navigation |
| doorway | 3 | 0.6248 | 0.0370 | 0.3704 | 7.1111 | 0.0741 | classic_doorway_high, classic_doorway_medium, classic_doorway_low |
| down_path | 1 | 0.5479 | 0.0000 | 0.0000 | 4.2222 | 0.0000 | francis2023_down_path |
| entering_elevator | 1 | 0.2245 | 0.3333 | 0.1111 | 0.0000 | 0.3333 | francis2023_entering_elevator |
| entering_room | 1 | 0.2117 | 0.3333 | 0.1111 | 0.0000 | 0.0000 | francis2023_entering_room |
| exiting_elevator | 1 | 0.7351 | 0.0000 | 0.6667 | 6.6667 | 0.0000 | francis2023_exiting_elevator |
| exiting_room | 1 | 0.2394 | 0.3333 | 0.1111 | 0.0000 | 0.0000 | francis2023_exiting_room |
| following_human | 1 | 0.6867 | 0.0000 | 0.3333 | 4.3333 | 0.0000 | francis2023_following_human |
| frontal_approach | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_frontal_approach |
| group_crossing | 3 | 0.2479 | 0.2963 | 0.0741 | 4.0741 | 0.0370 | classic_group_crossing_high, classic_group_crossing_medium, classic_group_crossing_low |
| head_on_corridor | 2 | 0.5761 | 0.0000 | 0.0555 | 2.3889 | 0.0000 | classic_head_on_corridor_medium, classic_head_on_corridor_low |
| intersection_no_gesture | 1 | 0.2372 | 0.1111 | 0.0000 | 0.0000 | 0.1111 | francis2023_intersection_no_gesture |
| intersection_proceed | 1 | 0.2372 | 0.1111 | 0.0000 | 0.0000 | 0.1111 | francis2023_intersection_proceed |
| intersection_wait | 1 | 0.2372 | 0.1111 | 0.0000 | 0.0000 | 0.1111 | francis2023_intersection_wait |
| join_group | 1 | 0.3941 | 0.3333 | 0.6667 | 10.4444 | 0.0000 | francis2023_join_group |
| leading_human | 1 | 0.6644 | 0.0000 | 0.3333 | 1.0000 | 0.0000 | francis2023_leading_human |
| leave_group | 1 | 0.4622 | 0.2222 | 0.7778 | 11.6666 | 0.2222 | francis2023_leave_group |
| merging | 2 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | classic_merging_low, classic_merging_medium |
| narrow_doorway | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_narrow_doorway |
| narrow_hallway | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_narrow_hallway |
| overtaking | 2 | 0.5096 | 0.0000 | 0.0000 | 1.1111 | 0.0000 | classic_overtaking_medium, classic_overtaking_low |
| parallel_traffic | 1 | 0.5734 | 0.0000 | 0.0000 | 6.8889 | 0.0000 | francis2023_parallel_traffic |
| pedestrian_obstruction | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_pedestrian_obstruction |
| pedestrian_overtaking | 1 | 0.4840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_pedestrian_overtaking |
| perpendicular_traffic | 1 | 0.6553 | 0.0000 | 0.2222 | 4.8889 | 0.0000 | francis2023_perpendicular_traffic |
| robot_crowding | 1 | 0.7803 | 0.0000 | 0.7778 | 15.8889 | 0.0000 | francis2023_robot_crowding |
| robot_overtaking | 1 | 0.7511 | 0.0000 | 0.6667 | 9.3333 | 0.0000 | francis2023_robot_overtaking |
| station_platform | 1 | 0.5910 | 0.0000 | 0.0000 | 10.4444 | 0.0000 | classic_station_platform_medium |
| t_intersection | 2 | 0.5016 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | classic_t_intersection_medium, classic_t_intersection_low |

### Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.0586 | 0.4348 | 6 | classic_group_crossing_high, francis2023_exiting_room, francis2023_entering_room |
| orca | core | -0.1051 | 0.1667 | 0 | classic_overtaking_medium, classic_head_on_corridor_low, classic_realworld_double_bottleneck_high |
| ppo | experimental | 36197.7293 | 1250000.0000 | 7 | classic_bottleneck_medium, classic_merging_medium, francis2023_narrow_doorway |
| prediction_planner | experimental | 44946.9866 | 916675.0000 | 15 | francis2023_narrow_hallway, classic_merging_medium, francis2023_narrow_doorway |
| sacadrl | experimental | 79861.0256 | 1416675.1667 | 14 | classic_group_crossing_low, classic_overtaking_low, classic_urban_crossing_medium |
| social_force | core | 0.0465 | 0.3333 | 0 | francis2023_accompanying_peer, francis2023_leading_human, francis2023_following_human |
| socnav_sampling | experimental | 154513.4164 | 1500000.0000 | 12 | classic_bottleneck_high, classic_bottleneck_medium, classic_merging_medium |

### Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- No inconsistencies detected by automated checks.
