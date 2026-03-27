# Issue 596 Atomic Scenario Matrix

This table summarizes the issue-596 atomic suite and its stricter verified-simple subset.

| Scenario | Map ID | Primary Capability | Target Failure Mode | Verified-Simple | Invalid Fixture |
| --- | --- | --- | --- | --- | --- |
| `empty_map_8_directions_east` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | yes | no |
| `empty_map_8_directions_northeast` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | no | no |
| `empty_map_8_directions_north` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | yes | no |
| `empty_map_8_directions_northwest` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | no | no |
| `empty_map_8_directions_west` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | yes | no |
| `empty_map_8_directions_southwest` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | no | no |
| `empty_map_8_directions_south` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | no | no |
| `empty_map_8_directions_southeast` | `atomic_empty_frame_test` | `frame_consistency` | `coordinate_transform` | no | no |
| `goal_behind_robot` | `atomic_empty_frame_test` | `frame_consistency` | `angle_wrap` | yes | no |
| `small_angle_precision` | `atomic_empty_frame_test` | `frame_consistency` | `oscillation` | no | no |
| `single_obstacle_circle` | `atomic_single_circle_obstacle` | `static_avoidance` | `clearance_regression` | yes | no |
| `single_obstacle_rectangle` | `atomic_single_rectangle_obstacle` | `static_avoidance` | `clearance_regression` | no | no |
| `line_wall_detour` | `atomic_line_wall_obstacle` | `static_avoidance` | `oscillation` | yes | no |
| `narrow_passage` | `atomic_narrow_passage_test` | `static_avoidance` | `clearance_regression` | yes | no |
| `corner_90_turn` | `atomic_corner_90_test` | `topology` | `oscillation` | no | no |
| `u_trap_local_minimum` | `atomic_u_trap_test` | `topology` | `local_minima` | no | no |
| `corridor_following` | `atomic_corridor_test` | `topology` | `oscillation` | no | no |
| `single_ped_crossing_orthogonal` | `francis2023_intersection_no_gesture` | `dynamic_interaction` | `social_collision` | yes | no |
| `head_on_interaction` | `francis2023_frontal_approach` | `dynamic_interaction` | `deadlock` | yes | no |
| `overtaking_interaction` | `francis2023_robot_overtaking` | `dynamic_interaction` | `social_collision` | yes | no |
| `start_near_obstacle` | `atomic_start_near_obstacle_test` | `robustness` | `clearance_regression` | no | no |
| `goal_very_close` | `atomic_goal_close_test` | `robustness` | `goal_termination` | no | no |
| `symmetry_ambiguous_choice` | `atomic_symmetry_split_test` | `robustness` | `deadlock` | no | no |
| `goal_inside_obstacle_invalid` | `atomic_goal_inside_obstacle_invalid` | `robustness` | `invalid_geometry` | no | yes |
