# ORCA Plausibility Sweep (2026-01-29)

## Run Summary
- Policy: `socnav_orca`
- Scenario config: `/Users/lennart/git/robot_sf_ll7/configs/scenarios/classic_interactions_francis2023.yaml`
- Episodes: 43
- Success rate: 0.953
- Collision rate: 0.000
- Benchmark output: `output/benchmarks/policy_analysis_socnav_orca_20260129_140828`
- Recordings: `output/recordings/policy_analysis_socnav_orca_20260129_140828`

## Interaction Highlights (by metric)
Lowest robot_ped_within_5m_frac:
- classic_doorway_medium: 0.000
- classic_group_crossing_low: 0.000
- classic_group_crossing_medium: 0.000
- classic_merging_low: 0.000
- classic_merging_medium: 0.000

Highest robot_ped_within_5m_frac:
- francis2023_leave_group: 1.000
- francis2023_robot_crowding: 0.899
- francis2023_crowd_navigation: 0.878
- francis2023_join_group: 0.861
- francis2023_exiting_elevator: 0.679

Lowest ped_force_mean:
- francis2023_exiting_room: 0.035
- classic_doorway_low: 0.047
- francis2023_down_path: 0.065
- classic_overtaking_medium: 0.072
- classic_group_crossing_low: 0.073

Highest ped_force_mean:
- francis2023_leave_group: 0.704
- classic_t_intersection_low: 0.499
- francis2023_robot_crowding: 0.490
- francis2023_narrow_hallway: 0.466
- francis2023_join_group: 0.384

## Per-Scenario Metrics

| scenario | success_rate | collisions_mean | min_distance | mean_distance | within_5m_frac | ped_force_mean | force_q95 | video |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| classic_bottleneck_high | 1.000 | 0.000 | na | na | na | na | na | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_bottleneck_high_seed123_socnav_orca.mp4` |
| classic_bottleneck_low | 1.000 | 0.000 | na | na | na | na | na | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_bottleneck_low_seed123_socnav_orca.mp4` |
| classic_bottleneck_medium | 1.000 | 0.000 | na | na | na | na | na | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_bottleneck_medium_seed123_socnav_orca.mp4` |
| classic_crossing_high | 1.000 | 0.000 | 4.641 | 8.467 | 0.191 | 0.148 | 0.581 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_crossing_high_seed123_socnav_orca.mp4` |
| classic_crossing_low | 1.000 | 0.000 | 4.366 | 8.047 | 0.255 | 0.129 | 0.440 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_crossing_low_seed123_socnav_orca.mp4` |
| classic_crossing_medium | 1.000 | 0.000 | 3.803 | 7.369 | 0.357 | 0.090 | 0.260 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_crossing_medium_seed123_socnav_orca.mp4` |
| classic_doorway_low | 1.000 | 0.000 | 4.881 | 6.894 | 0.038 | 0.047 | 0.149 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_doorway_low_seed123_socnav_orca.mp4` |
| classic_doorway_medium | 1.000 | 0.000 | 5.502 | 6.633 | 0.000 | 0.247 | 1.547 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_doorway_medium_seed123_socnav_orca.mp4` |
| classic_group_crossing_low | 1.000 | 0.000 | 7.721 | 8.969 | 0.000 | 0.073 | 0.249 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_group_crossing_low_seed123_socnav_orca.mp4` |
| classic_group_crossing_medium | 1.000 | 0.000 | 6.350 | 7.863 | 0.000 | 0.148 | 0.529 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_group_crossing_medium_seed123_socnav_orca.mp4` |
| classic_head_on_corridor_low | 1.000 | 0.000 | 4.617 | 16.676 | 0.026 | 0.131 | 0.439 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_head_on_corridor_low_seed123_socnav_orca.mp4` |
| classic_head_on_corridor_medium | 1.000 | 0.000 | 4.377 | 6.682 | 0.261 | 0.138 | 0.461 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_head_on_corridor_medium_seed123_socnav_orca.mp4` |
| classic_merging_low | 1.000 | 0.000 | 11.685 | 15.031 | 0.000 | 0.110 | 0.304 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_merging_low_seed123_socnav_orca.mp4` |
| classic_merging_medium | 1.000 | 0.000 | 11.553 | 15.347 | 0.000 | 0.352 | 1.634 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_merging_medium_seed123_socnav_orca.mp4` |
| classic_overtaking_low | 0.000 | 0.000 | 12.425 | 13.715 | 0.000 | 0.086 | 0.257 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_overtaking_low_seed123_socnav_orca.mp4` |
| classic_overtaking_medium | 0.000 | 0.000 | 11.840 | 13.297 | 0.000 | 0.072 | 0.280 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_overtaking_medium_seed123_socnav_orca.mp4` |
| classic_t_intersection_low | 1.000 | 0.000 | 9.342 | 12.555 | 0.000 | 0.499 | 2.030 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_t_intersection_low_seed123_socnav_orca.mp4` |
| classic_t_intersection_medium | 1.000 | 0.000 | 7.077 | 11.457 | 0.000 | 0.211 | 0.686 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/classic_t_intersection_medium_seed123_socnav_orca.mp4` |
| francis2023_accompanying_peer | 1.000 | 0.000 | 1.318 | 7.831 | 0.318 | 0.113 | 0.423 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_accompanying_peer_seed123_socnav_orca.mp4` |
| francis2023_blind_corner | 1.000 | 0.000 | 0.393 | 9.484 | 0.254 | 0.169 | 0.810 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_blind_corner_seed123_socnav_orca.mp4` |
| francis2023_circular_crossing | 1.000 | 0.000 | 1.843 | 4.639 | 0.463 | 0.319 | 1.106 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_circular_crossing_seed123_socnav_orca.mp4` |
| francis2023_crowd_navigation | 1.000 | 0.000 | 1.890 | 3.424 | 0.878 | 0.357 | 1.371 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_crowd_navigation_seed123_socnav_orca.mp4` |
| francis2023_down_path | 1.000 | 0.000 | 2.169 | 8.020 | 0.299 | 0.065 | 0.274 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_down_path_seed123_socnav_orca.mp4` |
| francis2023_entering_elevator | 1.000 | 0.000 | 2.281 | 6.497 | 0.349 | 0.171 | 0.424 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_entering_elevator_seed123_socnav_orca.mp4` |
| francis2023_entering_room | 1.000 | 0.000 | 7.867 | 12.332 | 0.000 | 0.203 | 0.469 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_entering_room_seed123_socnav_orca.mp4` |
| francis2023_exiting_elevator | 1.000 | 0.000 | 2.016 | 4.024 | 0.679 | 0.274 | 0.588 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_exiting_elevator_seed123_socnav_orca.mp4` |
| francis2023_exiting_room | 1.000 | 0.000 | 5.214 | 8.278 | 0.000 | 0.035 | 0.090 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_exiting_room_seed123_socnav_orca.mp4` |
| francis2023_following_human | 1.000 | 0.000 | 1.069 | 7.712 | 0.323 | 0.108 | 0.441 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_following_human_seed123_socnav_orca.mp4` |
| francis2023_frontal_approach | 1.000 | 0.000 | 0.889 | 7.970 | 0.319 | 0.178 | 0.883 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_frontal_approach_seed123_socnav_orca.mp4` |
| francis2023_intersection_no_gesture | 1.000 | 0.000 | 3.256 | 6.450 | 0.364 | 0.083 | 0.344 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_intersection_no_gesture_seed123_socnav_orca.mp4` |
| francis2023_intersection_proceed | 1.000 | 0.000 | 3.498 | 6.629 | 0.343 | 0.094 | 0.342 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_intersection_proceed_seed123_socnav_orca.mp4` |
| francis2023_intersection_wait | 1.000 | 0.000 | 3.498 | 6.629 | 0.343 | 0.094 | 0.342 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_intersection_wait_seed123_socnav_orca.mp4` |
| francis2023_join_group | 1.000 | 0.000 | 1.407 | 3.091 | 0.861 | 0.384 | 2.331 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_join_group_seed123_socnav_orca.mp4` |
| francis2023_leading_human | 1.000 | 0.000 | 0.951 | 7.866 | 0.323 | 0.125 | 0.461 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_leading_human_seed123_socnav_orca.mp4` |
| francis2023_leave_group | 1.000 | 0.000 | 0.687 | 1.962 | 1.000 | 0.704 | 2.810 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_leave_group_seed123_socnav_orca.mp4` |
| francis2023_narrow_doorway | 1.000 | 0.000 | 8.076 | 13.009 | 0.000 | 0.254 | 0.729 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_narrow_doorway_seed123_socnav_orca.mp4` |
| francis2023_narrow_hallway | 1.000 | 0.000 | 1.761 | 8.673 | 0.257 | 0.466 | 1.534 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_narrow_hallway_seed123_socnav_orca.mp4` |
| francis2023_parallel_traffic | 1.000 | 0.000 | 1.884 | 11.271 | 0.247 | 0.271 | 0.875 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_parallel_traffic_seed123_socnav_orca.mp4` |
| francis2023_pedestrian_obstruction | 1.000 | 0.000 | 1.639 | 10.599 | 0.311 | 0.193 | 1.304 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_pedestrian_obstruction_seed123_socnav_orca.mp4` |
| francis2023_pedestrian_overtaking | 1.000 | 0.000 | 0.293 | 8.038 | 0.324 | 0.260 | 1.404 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_pedestrian_overtaking_seed123_socnav_orca.mp4` |
| francis2023_perpendicular_traffic | 1.000 | 0.000 | 5.882 | 8.069 | 0.000 | 0.349 | 1.361 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_perpendicular_traffic_seed123_socnav_orca.mp4` |
| francis2023_robot_crowding | 1.000 | 0.000 | 1.060 | 3.003 | 0.899 | 0.490 | 1.476 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_robot_crowding_seed123_socnav_orca.mp4` |
| francis2023_robot_overtaking | 1.000 | 0.000 | 0.838 | 8.122 | 0.316 | 0.199 | 1.111 | `output/recordings/policy_analysis_socnav_orca_20260129_140828/francis2023_robot_overtaking_seed123_socnav_orca.mp4` |

## Notes
- `ped_force_mean` and `force_q95` come from total pedestrian force vectors (sum of all force components).
- Plausibility status remains **unverified**; metrics were recorded via `--write-plausibility-metrics`.

