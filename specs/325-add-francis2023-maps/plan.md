# Francis 2023 scenarios: implementation plan

## Decisions captured so far

- Use SVG maps for geometry and labeled spawn/goal/routes, consistent with existing
  `maps/svg_maps/*.svg` conventions.
- Use a dedicated scenario matrix YAML (for example
  `configs/scenarios/francis2023.yaml`) with `name`, `map_file`, `simulation_config`,
  and `metadata` to define the scenarios.
- For single-pedestrian scenarios, set `ped_density=0` and rely on
  `single_ped_<id>_start/goal` markers in the map.
- For crowd/flow scenarios, use `ped_route_*` paths and `ped_crowded_zone`
  rectangles rather than ad-hoc spawning.
- Behavior-only differences (wait, follow, lead, join/leave, etc.) will require
  code support beyond maps and scenario configs.

## Phase 1: Map inventory and scenario mapping

- Build a table mapping each Figure 7 scenario (a) to (y) to:
  - map geometry needed
  - number of humans/robots
  - required behaviors (static, slow, wait, follow, lead, group)
  - whether current capabilities suffice or code changes are required
- Identify which existing maps can be reused and which must be created anew.

## Phase 2: Base SVG maps (geometry-first)

- Create new SVG maps under `maps/svg_maps/` for the scenarios that are
  geometry-only or can be represented with single pedestrians:
  - frontal approach, obstruction, overtaking, down path
  - blind corner, narrow hallway, narrow doorway
  - entering/exiting room, entering/exiting elevator
- Encode single pedestrians with `single_ped_<id>_start/goal` markers.
- Use obstacles and boundaries to constrain routes and visibility as in the figure.

## Phase 3: Scenario matrix configuration

- Add `configs/scenarios/francis2023.yaml` with:
  - `map_file` entries pointing to the new SVGs
  - `simulation_config` settings for episode length, ped density, and speed scaling
  - `metadata` fields to record archetype, flow, and behavioral tags
  - seed lists for reproducibility

## Phase 4: Behavior support for non-geometry scenarios

- Extend `SinglePedestrianDefinition` to optionally include:
  - `speed_m_s` for per-ped speed control
  - `behavior` or `role` tag (wait, follow, lead, accompany, join, leave)
  - optional waypoint timing or a simple "wait at point" flag
- Add a `SinglePedestrianBehavior` controller that:
  - advances trajectory waypoints when reached
  - applies waits where specified
  - updates the pedestrian goal so PySocialForce moves toward the next waypoint
- Update the SVG/JSON map loaders to accept these optional fields:
  - SVG: add a lightweight encoding strategy (for example, use POI labels or a
    small JSON companion file per map if SVG labeling becomes too complex).
  - JSON: parse the new fields in `robot_sf/nav/map_config.py`.
- Wire the new behavior controller into `robot_sf/sim/simulator.py` alongside
  the existing `CrowdedZoneBehavior` and `FollowRouteBehavior`.

## Phase 5: Crowd and flow scenarios

- Implement maps and configs for:
  - crowd navigation, parallel traffic, perpendicular traffic
  - circular crossing, robot crowding
- Calibrate `ped_density` and routes to approximate the intended flows in the figure.

## Phase 6: Validation and documentation

- Run map verification for the new SVGs.
- Generate scenario thumbnails for visual sanity checks.
- Update `docs/README.md` and `CHANGELOG.md` if these scenarios are user-facing.
- Add or extend tests if new behavior logic is introduced (especially for
  trajectory waypoint advancement and per-ped speed).
