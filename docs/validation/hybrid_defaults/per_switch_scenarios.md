# Per-scenario hybrid switch outcomes

Diagnostic-only. Each cell is successes / collisions / timeouts out of 30.
Flag order: physical exclusion / goal validity / validity sensor.
Literal goal-only (010) was checked on every cell: 30 configuration errors per scenario, before any environment step. These are excluded from success/collision/timeout denominators.
The executable validity arm includes its required sensor; compare it with sensor-only.

| Scenario | Off (000) | Static (100) | Sensor (001) | Validity + sensor (011) | All (111) |
| --- | ---: | ---: | ---: | ---: | ---: |
| classic_bottleneck_high | 30 / 0 / 0 | 29 / 0 / 1 | 30 / 0 / 0 | 30 / 0 / 0 | 29 / 0 / 1 |
| classic_bottleneck_low | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_bottleneck_medium | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_cross_trap_high | 29 / 0 / 1 | 30 / 0 / 0 | 29 / 0 / 1 | 29 / 0 / 1 | 30 / 0 / 0 |
| classic_cross_trap_low | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_cross_trap_medium | 28 / 0 / 2 | 30 / 0 / 0 | 28 / 0 / 2 | 28 / 0 / 2 | 30 / 0 / 0 |
| classic_doorway_high | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_doorway_low | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_doorway_medium | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_group_crossing_high | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_group_crossing_low | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_group_crossing_medium | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_head_on_corridor_low | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_head_on_corridor_medium | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_merging_low | 29 / 0 / 1 | 28 / 0 / 2 | 29 / 0 / 1 | 29 / 0 / 1 | 28 / 0 / 2 |
| classic_merging_medium | 23 / 0 / 7 | 22 / 0 / 8 | 23 / 0 / 7 | 23 / 0 / 7 | 22 / 0 / 8 |
| classic_overtaking_low | 29 / 0 / 1 | 30 / 0 / 0 | 29 / 0 / 1 | 29 / 0 / 1 | 30 / 0 / 0 |
| classic_overtaking_medium | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_realworld_double_bottleneck_high | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| classic_station_platform_medium | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 |
| classic_t_intersection_low | 29 / 0 / 1 | 30 / 0 / 0 | 29 / 0 / 1 | 29 / 0 / 1 | 30 / 0 / 0 |
| classic_t_intersection_medium | 29 / 0 / 1 | 30 / 0 / 0 | 29 / 0 / 1 | 29 / 0 / 1 | 30 / 0 / 0 |
| classic_urban_crossing_medium | 27 / 0 / 3 | 30 / 0 / 0 | 27 / 0 / 3 | 27 / 0 / 3 | 30 / 0 / 0 |
| francis2023_accompanying_peer | 28 / 0 / 2 | 27 / 0 / 3 | 28 / 0 / 2 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_blind_corner | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 29 / 0 / 1 |
| francis2023_circular_crossing | 29 / 0 / 1 | 30 / 0 / 0 | 29 / 0 / 1 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_crowd_navigation | 25 / 0 / 5 | 26 / 0 / 4 | 25 / 0 / 5 | 28 / 0 / 2 | 28 / 0 / 2 |
| francis2023_down_path | 26 / 0 / 4 | 27 / 0 / 3 | 26 / 0 / 4 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_entering_elevator | 26 / 0 / 4 | 27 / 0 / 3 | 26 / 0 / 4 | 29 / 0 / 1 | 29 / 0 / 1 |
| francis2023_entering_room | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_exiting_elevator | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 29 / 0 / 1 | 28 / 0 / 2 |
| francis2023_exiting_room | 30 / 0 / 0 | 30 / 0 / 0 | 30 / 0 / 0 | 29 / 0 / 1 | 29 / 0 / 1 |
| francis2023_following_human | 25 / 0 / 5 | 23 / 0 / 7 | 25 / 0 / 5 | 27 / 0 / 3 | 27 / 0 / 3 |
| francis2023_frontal_approach | 26 / 0 / 4 | 29 / 0 / 1 | 26 / 0 / 4 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_intersection_no_gesture | 29 / 0 / 1 | 28 / 0 / 2 | 29 / 0 / 1 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_intersection_proceed | 29 / 0 / 1 | 28 / 0 / 2 | 29 / 0 / 1 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_intersection_wait | 29 / 0 / 1 | 28 / 0 / 2 | 29 / 0 / 1 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_join_group | 25 / 0 / 5 | 25 / 0 / 5 | 25 / 0 / 5 | 29 / 0 / 1 | 29 / 0 / 1 |
| francis2023_leading_human | 25 / 0 / 5 | 26 / 0 / 4 | 25 / 0 / 5 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_leave_group | 24 / 0 / 6 | 25 / 0 / 5 | 24 / 0 / 6 | 29 / 0 / 1 | 28 / 0 / 2 |
| francis2023_narrow_doorway | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 | 0 / 0 / 30 |
| francis2023_narrow_hallway | 26 / 0 / 4 | 5 / 0 / 25 | 26 / 0 / 4 | 30 / 0 / 0 | 5 / 0 / 25 |
| francis2023_parallel_traffic | 26 / 0 / 4 | 24 / 0 / 6 | 26 / 0 / 4 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_pedestrian_obstruction | 26 / 0 / 4 | 24 / 0 / 6 | 26 / 0 / 4 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_pedestrian_overtaking | 23 / 0 / 7 | 23 / 0 / 7 | 23 / 0 / 7 | 30 / 0 / 0 | 29 / 0 / 1 |
| francis2023_perpendicular_traffic | 28 / 0 / 2 | 27 / 0 / 3 | 28 / 0 / 2 | 30 / 0 / 0 | 30 / 0 / 0 |
| francis2023_robot_crowding | 19 / 0 / 11 | 15 / 1 / 14 | 19 / 0 / 11 | 19 / 0 / 11 | 18 / 1 / 11 |
| francis2023_robot_overtaking | 28 / 0 / 2 | 26 / 0 / 4 | 28 / 0 / 2 | 30 / 0 / 0 | 30 / 0 / 0 |

All times and rows: [CSV](per_switch_scenarios.csv). Paired times, every failure classification,
control checks and complete arm totals: [summary](per_switch_summary.json).
Source/dependency identity: [manifest](per_switch_manifest.json). Raw per-step traces remain
in the owned lane; their full SHA-256 identities and sizes are tracked separately.

## Executed contacts

Stationary pedestrian contacts remain collisions. Classification describes the native trace,
without excluding passive contacts or assigning intent.

| Scenario / seed / arm | Contact | Time (s) | Stationary ending at contact (s) | Displacement on contact (m) | Last mode |
| --- | --- | ---: | ---: | ---: | --- |
| francis2023_robot_crowding / 1013 / static_only | pedestrian_contact_while_robot_stationary | 5.8 | 0.8 | 0.000000 | PROTECTIVE_STOP |
| francis2023_robot_crowding / 1013 / current_defaults | pedestrian_contact_while_robot_stationary | 5.8 | 0.8 | 0.000000 | PROTECTIVE_STOP |

Literal goal-only records: [all 1,440 cells](literal_goal_only.json.gz); [probe identity](literal_goal_only_manifest.json).
