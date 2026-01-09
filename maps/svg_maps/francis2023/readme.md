# Francis 2023 SVG maps (geometry-only)

These SVGs implement the geometry-first subset of the Francis et al. (2023)
scenario figure. Each map uses the standard SVG labeling conventions from
`docs/SVG_MAP_EDITOR.md`. Most maps place a single pedestrian via
`single_ped_h1_start` / `single_ped_h1_goal` markers; group scenarios include
multiple `single_ped_h*` markers for the cluster.

Conventions
- Scale: 1 SVG unit = 1 meter.
- Robot uses `robot_spawn_zone`, `robot_goal_zone`, and `robot_route_0_0`.
- Single pedestrian markers define the human start/goal only; speed and
  waiting/gesture behaviors are not encoded yet.
- POI markers are duplicated at the single-ped start/goal positions using
  `poi_h1_start` and `poi_h1_goal` (class `poi`) to support scenario overlays.
- Each map includes an outer obstacle boundary to keep agents inside the map.

Maps
- `francis2023_frontal_approach.svg` (Fig 7a): head-on corridor encounter.
- `francis2023_ped_obstruction.svg` (Fig 7b): pedestrian ahead in same corridor
  direction (obstruction intent; speed tuning pending).
- `francis2023_ped_overtaking.svg` (Fig 7c): pedestrian starts behind robot
  in the same corridor (overtaking intent; speed tuning pending).
- `francis2023_robot_overtaking.svg` (Fig 7d): pedestrian starts ahead of the
  robot (robot overtaking intent; speed tuning pending).
- `francis2023_down_path.svg` (Fig 7e): robot and pedestrian move in the same
  direction on offset lines within a corridor.
- `francis2023_intersection_no_gesture.svg` (Fig 7f): cross intersection with
  perpendicular robot and pedestrian routes.
- `francis2023_blind_corner.svg` (Fig 7i): L-shaped corridor around a large
  obstacle to create a blind corner.
- `francis2023_narrow_hallway.svg` (Fig 7j): narrow corridor with opposing
  robot and pedestrian routes.
- `francis2023_narrow_doorway.svg` (Fig 7k): doorway gap in a corridor wall.
- `francis2023_entering_room.svg` (Fig 7l): robot enters a room through a
  doorway; pedestrian stays inside the room.
- `francis2023_exiting_room.svg` (Fig 7m): robot exits the room; pedestrian is
  outside near the doorway.
- `francis2023_entering_elevator.svg` (Fig 7n): robot enters a small elevator
  room; pedestrian inside the elevator.
- `francis2023_exiting_elevator.svg` (Fig 7o): robot exits the elevator; pedestrian
  outside the doorway.
- `francis2023_join_group.svg` (Fig 7p): open area with a small pedestrian cluster and
  a joiner start position.
- `francis2023_leave_group.svg` (Fig 7q): open area with a small pedestrian cluster and
  a designated leaver start position.
- `francis2023_crowd_navigation.svg` (Fig 7u): open area with a crowded zone for
  dense pedestrian sampling as the robot crosses.
- `francis2023_parallel_traffic.svg` (Fig 7v): corridor with two parallel pedestrian
  routes moving in the robot's direction.
- `francis2023_perpendicular_traffic.svg` (Fig 7w): intersection geometry with a
  perpendicular pedestrian route crossing the robot path.
- `francis2023_circular_crossing.svg` (Fig 7x): open area with a circular pedestrian
  route intersecting the robot's straight path.
- `francis2023_robot_crowding.svg` (Fig 7y): dense crowded zone covering the robot
  route to induce crowding.

Notes
- Overtaking and obstruction scenarios will need per-ped speed control to match
  the intended behavior in the paper.
- Gesture-based scenarios (wait/proceed, join/leave, follow/lead/accompany) now
  rely on scenario YAML overrides and runtime behavior logic; geometry in these maps
  provides only the base layout.
- Crowd/traffic scenarios depend on `ped_density` tuning plus the ped routes or
  crowded zones embedded in each SVG.
