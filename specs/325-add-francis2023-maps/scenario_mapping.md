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
| p | Join Group | TBD (open area or corridor map) | pending | group join behavior | Requires group formation or join logic. |
| q | Leave Group | TBD (open area or corridor map) | pending | group leave behavior | Requires group split logic. |
| r | Following Human | TBD (corridor map) | pending | follow behavior | Ped acts as leader; robot follows. |
| s | Leading Human | TBD (corridor map) | pending | lead behavior | Ped follows robot; requires scripted follower. |
| t | Accompanying Peer | TBD (corridor map) | pending | side-by-side behavior | Requires paired formation/maintained offset. |
| u | Crowd Navigation | TBD (open area + crowded zones) | pending | crowd density + routes | Needs crowd spawning and density tuning. |
| v | Parallel Traffic | TBD (corridor + routes) | pending | multi-ped parallel flow | Multiple pedestrian routes in same direction. |
| w | Perpendicular Traffic | TBD (crossing routes) | pending | multi-ped perpendicular flow | Crossing flows with routes. |
| x | Circular Crossing | TBD (circular layout) | pending | multi-ped circulate flow | Requires circular route topology. |
| y | Robot Crowding | TBD (open area + crowded zones) | pending | crowd density + routes | High-density crowd around robot. |

Notes
- Geometry-only SVGs include outer obstacle boundaries to keep agents inside the map.
- Scenario YAML definitions are still pending for all Francis 2023 entries.
