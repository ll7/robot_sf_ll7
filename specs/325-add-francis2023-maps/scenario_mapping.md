# Francis 2023 scenario mapping (Phase 1)

Legend
- Geometry status: done = SVG map exists; pending = map not created yet
- Behavior status: pending = needs behavior scripting or per-ped speed control

| Fig | Scenario | Geometry (map) | Geometry status | Behavior needs | Notes |
| --- | --- | --- | --- | --- | --- |
| a | Frontal Approach | `maps/svg_maps/francis2023/francis2023_frontal_approach.svg` | done | none (single ped) | Head-on corridor encounter. |
| b | Pedestrian Obstruction | `maps/svg_maps/francis2023/francis2023_ped_obstruction.svg` | done | per-ped speed (slow ped) | Obstruction intent needs slower pedestrian. |
| c | Pedestrian Overtaking | `maps/svg_maps/francis2023/francis2023_ped_overtaking.svg` | done | per-ped speed (fast ped) | Overtaking intent needs faster pedestrian. |
| d | Robot Overtaking | `maps/svg_maps/francis2023/francis2023_robot_overtaking.svg` | done | per-ped speed (slow ped) | Robot overtakes slower pedestrian. |
| e | Down Path | `maps/svg_maps/francis2023/francis2023_down_path.svg` | done | none (single ped) | Parallel same-direction paths. |
| f | Intersection No Gesture | `maps/svg_maps/francis2023/francis2023_intersection_no_gesture.svg` | done | none (single ped) | Perpendicular crossing, no gesture. |
| g | Intersection Wait | candidate: `francis2023_intersection_no_gesture.svg` | done | wait/gesture behavior | Likely reuse intersection geometry; needs stop/wait behavior. |
| h | Intersection Proceed | candidate: `francis2023_intersection_no_gesture.svg` | done | proceed/gesture behavior | Likely reuse intersection geometry; needs proceed behavior. |
| i | Blind Corner | `maps/svg_maps/francis2023/francis2023_blind_corner.svg` | done | none (single ped) | L-corner corridor with large obstacle. |
| j | Narrow Hallway | `maps/svg_maps/francis2023/francis2023_narrow_hallway.svg` | done | none (single ped) | Tight corridor encounter. |
| k | Narrow Doorway | `maps/svg_maps/francis2023/francis2023_narrow_doorway.svg` | done | none (single ped) | Doorway constriction. |
| l | Entering Room | `maps/svg_maps/francis2023/francis2023_entering_room.svg` | done | none (single ped) | Robot enters room with ped inside. |
| m | Exiting Room | `maps/svg_maps/francis2023/francis2023_exiting_room.svg` | done | none (single ped) | Robot exits room with ped outside. |
| n | Entering Elevator | `maps/svg_maps/francis2023/francis2023_entering_elevator.svg` | done | none (single ped) | Robot enters elevator; ped inside. |
| o | Exiting Elevator | `maps/svg_maps/francis2023/francis2023_exiting_elevator.svg` | done | none (single ped) | Robot exits elevator; ped outside. |
| p | Join Group | `maps/svg_maps/francis2023/francis2023_join_group.svg` | done | group join behavior | Requires group formation or join logic. |
| q | Leave Group | `maps/svg_maps/francis2023/francis2023_leave_group.svg` | done | group leave behavior | Requires group split logic. |
| r | Following Human | `maps/svg_maps/francis2023/francis2023_down_path.svg` | done | follow behavior (role) | Ped acts as leader; robot follows. |
| s | Leading Human | `maps/svg_maps/francis2023/francis2023_down_path.svg` | done | lead behavior (role) | Ped follows robot; requires scripted follower. |
| t | Accompanying Peer | `maps/svg_maps/francis2023/francis2023_down_path.svg` | done | side-by-side behavior (role) | Requires paired formation/maintained offset. |
| u | Crowd Navigation | `maps/svg_maps/francis2023/francis2023_crowd_navigation.svg` | done | crowd density + crowded zone | Needs density tuning for desired crowding. |
| v | Parallel Traffic | `maps/svg_maps/francis2023/francis2023_parallel_traffic.svg` | done | multi-ped parallel flow | Multiple pedestrian routes in same direction. |
| w | Perpendicular Traffic | `maps/svg_maps/francis2023/francis2023_perpendicular_traffic.svg` | done | multi-ped perpendicular flow | Crossing flow perpendicular to robot. |
| x | Circular Crossing | `maps/svg_maps/francis2023/francis2023_circular_crossing.svg` | done | circular flow route | Pedestrians circulate on a looped route. |
| y | Robot Crowding | `maps/svg_maps/francis2023/francis2023_robot_crowding.svg` | done | high crowd density | Dense crowd around the robot path. |

Notes
- Geometry-only SVGs include outer obstacle boundaries to keep agents inside the map.
- Scenario YAML definitions live in `configs/scenarios/francis2023.yaml`; crowd/traffic
  entries still need density tuning for best fidelity.
