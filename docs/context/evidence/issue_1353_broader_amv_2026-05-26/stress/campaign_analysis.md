# Camera-Ready Campaign Analysis

- Campaign ID: `issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603`
- Campaign root: `output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603`
- Runtime sec: `1228.392698256299`
- Episodes/sec: `0.8205844934041484`

## Planner Diagnostics

| planner | algo | kinematics | preflight | episodes | success(ep) | collision(ep) | snqi(ep) | abs map paths | runtime(s) | eps/s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| goal | goal | differential_drive | ok | 144 | 0.0000 | 0.2500 | -0.17522669481718603 | 0 | 77.6868 | 1.8536 |
| orca | orca | differential_drive | ok | 144 | 0.1667 | 0.0764 | -0.2465878625397102 | 0 | 110.8397 | 1.2992 |
| ppo | ppo | differential_drive | ok | 144 | 0.2222 | 0.1667 | -0.3196007071612242 | 0 | 156.7022 | 0.9189 |
| prediction_planner | prediction_planner | differential_drive | ok | 144 | 0.0694 | 0.2153 | -0.20385415254292483 | 0 | 613.4111 | 0.2348 |
| sacadrl | sacadrl | differential_drive | ok | 144 | 0.0208 | 0.3472 | -0.2918732243255027 | 0 | 113.4495 | 1.2693 |
| social_force | social_force | differential_drive | ok | 144 | 0.0000 | 0.2500 | -0.8603302781853034 | 0 | 81.2125 | 1.7731 |
| socnav_bench | socnav_bench | differential_drive | skipped | 0 | 0.0000 | 0.0000 | nan | 0 | 0.0895 | 0.0000 |
| socnav_sampling | socnav_sampling | differential_drive | ok | 144 | 0.1528 | 0.5417 | -0.19458225645838884 | 0 | 69.3675 | 2.0759 |

## Runtime Hotspots

| planner | runtime(s) | wall_time_mean(s) | wall_time_p95(s) |
|---|---:|---:|---:|
| prediction_planner (differential_drive) | 613.4111 | 4.2192 | 5.0777 |
| ppo (differential_drive) | 156.7022 | 1.0209 | 1.3866 |
| sacadrl (differential_drive) | 113.4495 | 0.7443 | 1.1031 |

- `prediction_planner` (differential_drive) top slow scenarios:
  - `classic_t_intersection_low` mean=5.1834s p95=5.2843s (episodes=3)
  - `classic_t_intersection_medium` mean=5.0726s p95=5.3467s (episodes=3)
  - `classic_realworld_double_bottleneck_high` mean=5.0289s p95=5.2001s (episodes=3)
- `ppo` (differential_drive) top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=1.4261s p95=1.4811s (episodes=3)
  - `classic_t_intersection_medium` mean=1.3764s p95=1.4219s (episodes=3)
  - `francis2023_intersection_wait` mean=1.3635s p95=1.5196s (episodes=3)
- `sacadrl` (differential_drive) top slow scenarios:
  - `classic_realworld_double_bottleneck_high` mean=1.1676s p95=1.1816s (episodes=3)
  - `classic_t_intersection_medium` mean=1.1015s p95=1.1542s (episodes=3)
  - `classic_t_intersection_low` mean=1.0799s p95=1.1031s (episodes=3)

## Scenario Difficulty

- Primary proxy: `consensus_outcome_rank_v1`
- Description: Weighted consensus score across core benchmark-success planners using success, collisions, near-misses, and normalized time-to-goal.
- Consensus planners: `3` (core benchmark-success planners)
- Supporting metric: `snqi_mean`

### Hardest Scenarios

| rank | scenario | family | difficulty | success | collisions | near misses | time_to_goal_norm | seed success CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | francis2023_robot_crowding | robot_crowding | 0.7532 | 0.0000 | 0.7778 | 15.8889 | 1.0000 | 0.0000 |
| 2 | francis2023_robot_overtaking | robot_overtaking | 0.7298 | 0.0000 | 0.6667 | 10.3333 | 1.0000 | 0.0000 |
| 3 | classic_cross_trap_high | cross_trap | 0.7245 | 0.0000 | 0.8889 | 4.7778 | 1.0000 | 0.0000 |
| 4 | classic_doorway_low | doorway | 0.7074 | 0.0000 | 0.4444 | 9.3333 | 1.0000 | 0.0000 |
| 5 | classic_cross_trap_medium | cross_trap | 0.6952 | 0.0000 | 0.5556 | 4.7778 | 1.0000 | 0.0000 |
| 6 | classic_doorway_high | doorway | 0.6920 | 0.0000 | 0.3333 | 9.8889 | 1.0000 | 0.0000 |
| 7 | classic_doorway_medium | doorway | 0.6856 | 0.0000 | 0.3333 | 8.8889 | 1.0000 | 0.0000 |
| 8 | francis2023_crowd_navigation | crowd_navigation | 0.6665 | 0.0000 | 0.3333 | 4.7778 | 1.0000 | 0.0000 |

### Family Rollup

| family | scenarios | difficulty | success | collisions | near misses | seed success CI | hardest scenarios |
|---|---:|---:|---:|---:|---:|---:|---|
| accompanying_peer | 1 | 0.6410 | 0.0000 | 0.3333 | 1.7778 | 0.0000 | francis2023_accompanying_peer |
| blind_corner | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_blind_corner |
| bottleneck | 4 | 0.5411 | 0.0000 | 0.0833 | 1.6944 | 0.0000 | classic_realworld_double_bottleneck_high, classic_bottleneck_high, classic_bottleneck_low |
| circular_crossing | 1 | 0.3080 | 0.2222 | 0.3333 | 0.3333 | 0.2222 | francis2023_circular_crossing |
| cross_trap | 3 | 0.6495 | 0.0000 | 0.4815 | 4.6667 | 0.0000 | classic_cross_trap_high, classic_cross_trap_medium, classic_cross_trap_low |
| crossing | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | classic_urban_crossing_medium |
| crowd_navigation | 1 | 0.6665 | 0.0000 | 0.3333 | 4.7778 | 0.0000 | francis2023_crowd_navigation |
| doorway | 3 | 0.6950 | 0.0000 | 0.3704 | 9.3704 | 0.0000 | classic_doorway_low, classic_doorway_high, classic_doorway_medium |
| down_path | 1 | 0.5287 | 0.0000 | 0.0000 | 4.4444 | 0.0000 | francis2023_down_path |
| entering_elevator | 1 | 0.1782 | 0.2222 | 0.0000 | 0.0000 | 0.2222 | francis2023_entering_elevator |
| entering_room | 1 | 0.1144 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | francis2023_entering_room |
| exiting_elevator | 1 | 0.4426 | 0.2222 | 0.6667 | 8.2222 | 0.2222 | francis2023_exiting_elevator |
| exiting_room | 1 | 0.1186 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | francis2023_exiting_room |
| following_human | 1 | 0.6569 | 0.0000 | 0.3333 | 4.2222 | 0.0000 | francis2023_following_human |
| frontal_approach | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_frontal_approach |
| group_crossing | 3 | 0.2551 | 0.2963 | 0.0741 | 2.6296 | 0.0370 | classic_group_crossing_high, classic_group_crossing_medium, classic_group_crossing_low |
| head_on_corridor | 2 | 0.5000 | 0.0000 | 0.0000 | 1.1111 | 0.0000 | classic_head_on_corridor_medium, classic_head_on_corridor_low |
| intersection_no_gesture | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_no_gesture |
| intersection_proceed | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_proceed |
| intersection_wait | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_intersection_wait |
| join_group | 1 | 0.3862 | 0.3333 | 0.6667 | 11.0000 | 0.0000 | francis2023_join_group |
| leading_human | 1 | 0.6441 | 0.0000 | 0.3333 | 1.8889 | 0.0000 | francis2023_leading_human |
| leave_group | 1 | 0.4846 | 0.1111 | 0.8889 | 12.0000 | 0.1111 | francis2023_leave_group |
| merging | 2 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | classic_merging_low, classic_merging_medium |
| narrow_doorway | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_narrow_doorway |
| narrow_hallway | 1 | 0.5394 | 0.0000 | 0.1111 | 0.0000 | 0.0000 | francis2023_narrow_hallway |
| overtaking | 2 | 0.4809 | 0.0000 | 0.0000 | 0.7222 | 0.0000 | classic_overtaking_medium, classic_overtaking_low |
| parallel_traffic | 1 | 0.6543 | 0.0000 | 0.1111 | 13.4444 | 0.0000 | francis2023_parallel_traffic |
| pedestrian_obstruction | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_pedestrian_obstruction |
| pedestrian_overtaking | 1 | 0.4569 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | francis2023_pedestrian_overtaking |
| perpendicular_traffic | 1 | 0.6181 | 0.0000 | 0.2222 | 2.3333 | 0.0000 | francis2023_perpendicular_traffic |
| robot_crowding | 1 | 0.7532 | 0.0000 | 0.7778 | 15.8889 | 0.0000 | francis2023_robot_crowding |
| robot_overtaking | 1 | 0.7298 | 0.0000 | 0.6667 | 10.3333 | 0.0000 | francis2023_robot_overtaking |
| station_platform | 1 | 0.6324 | 0.0000 | 0.3333 | 3.2222 | 0.0000 | classic_station_platform_medium |
| t_intersection | 2 | 0.4761 | 0.0000 | 0.0000 | 0.2222 | 0.0000 | classic_t_intersection_medium, classic_t_intersection_low |

### Planner Residual Summary

| planner | group | mean residual | max residual | easy-scenario mismatches | worst scenarios |
|---|---|---:|---:|---:|---|
| goal | core | 0.0372 | 0.4737 | 6 | classic_group_crossing_medium, francis2023_exiting_elevator, classic_group_crossing_high |
| orca | core | -0.0856 | 0.1667 | 0 | francis2023_narrow_hallway, classic_overtaking_medium, classic_realworld_double_bottleneck_high |
| ppo | experimental | 51145.6030 | 2166674.9167 | 2 | classic_bottleneck_medium, francis2023_narrow_hallway, classic_realworld_double_bottleneck_high |
| prediction_planner | experimental | 14948.4831 | 250000.0000 | 8 | francis2023_blind_corner, francis2023_narrow_doorway, francis2023_narrow_hallway |
| sacadrl | experimental | 29514.0136 | 250000.1667 | 11 | francis2023_entering_elevator, classic_urban_crossing_medium, francis2023_frontal_approach |
| social_force | core | 0.0484 | 0.3333 | 6 | francis2023_accompanying_peer, francis2023_following_human, francis2023_leading_human |
| socnav_sampling | experimental | 84184.2556 | 750000.6667 | 8 | classic_bottleneck_high, francis2023_narrow_hallway, classic_urban_crossing_medium |

### Verified-Simple Assessment

- Status: `rerun_required`
- Matched scenarios: `0`
- Rank correlation: `nan`
- Recommendation: The current camera-ready campaign does not include the verified-simple candidate scenarios. Keep the subset as a debugging or promotion gate for now and run one bounded pilot before treating it as a calibration set.

### Difficulty Findings

- No additional scenario-difficulty warnings.

## Findings

- No inconsistencies detected by automated checks.
