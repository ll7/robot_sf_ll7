# Francis 2023 scenarios: current state

## Existing capabilities in the repo

- SVG-based maps are the primary way to describe geometry, spawn/goal zones, and routes.
  - Parser: `robot_sf/nav/svg_map_parser.py`
  - Labels: `robot_spawn_zone`, `robot_goal_zone`, `ped_spawn_zone`, `ped_goal_zone`,
    `robot_route_<spawn>_<goal>`, `ped_route_<spawn>_<goal>`, `ped_crowded_zone`.
  - Single pedestrians can be placed via circles labeled
    `single_ped_<id>_start` and `single_ped_<id>_goal`.
  - POIs are supported via circles with class `poi` and a descriptive label.
- MapDefinition supports:
  - obstacles, robot routes, ped routes, ped crowded zones
  - `single_pedestrians` (start + goal or trajectory)
  - code: `robot_sf/nav/map_config.py`
- Pedestrian spawning:
  - crowd and route populations via `PedSpawnConfig` in `robot_sf/ped_npc/ped_population.py`
  - single pedestrians are injected from `MapDefinition.single_pedestrians`
  - simulator uses `peds_speed_mult` as a global max speed multiplier and
    `PedSpawnConfig.initial_speed` for initial velocities
- Scenario configs (YAML) already exist for other archetypes, for example
  `configs/scenarios/classic_interactions.yaml`.
  - The planner expects scenarios with `name`, `map_file`, `simulation_config`,
    and `metadata` (see `robot_sf/benchmark/full_classic/planning.py`).
- Baseline map examples already exist under `maps/svg_maps/` (classic interactions,
  overtaking, crossing, doorway, static humans).
- Scenario overlays can override single pedestrians by id, including `goal`/`trajectory`,
  POI-based overrides, `speed_m_s`, `wait_at`, and `note` fields.
- Preview helper: `scripts/tools/preview_scenario_trajectories.py` renders overlays
  on top of map geometry.

## What is missing for the Francis 2023 scenarios

- Scenario YAML exists (`configs/scenarios/francis2023.yaml`), but only covers
  geometry-first entries.
- Several scenarios imply scripted pedestrian behavior that is not supported:
  - "wait" / "proceed" behaviors at intersections
  - "join group", "leave group"
  - "follow human", "lead human", "accompany peer"
  - pedestrians that stop or gesture
- Single-pedestrian trajectories do not currently advance across waypoints.
  - `SinglePedestrianDefinition` can include a trajectory, but there is no
    behavior class that updates the goal to the next waypoint during simulation.
- Per-pedestrian speed control is not supported.
  - Only global `peds_speed_mult` and `PedSpawnConfig.initial_speed` exist.
  - "slow walking pedestrian" would currently require a global slow-down.
- Crowd/flow scenarios (parallel traffic, perpendicular traffic, circular crossing,
  robot crowding) are not yet mapped or configured.

## Spec context

- This spec folder includes:
  - `Readme.md` with the citation
  - `Fig7_Francis2023_scenarios.png` (reference diagram)
  - `plan.md` and `scenario_mapping.md` for the build plan
  - `start.md` (this document)

## Current Francis 2023 assets

- SVG maps under `maps/svg_maps/francis2023/` for Fig. 7a, 7b, 7c, 7d, 7e, 7f, 7i–7o.
  - Each map includes boundary obstacles and POI markers (`poi_h1_start`, `poi_h1_goal`).
- Scenario matrix: `configs/scenarios/francis2023.yaml` (geometry-first entries).
  - Uses `goal_poi: poi_h1_goal` overlays for readability.
- Map verification: `output/validation/francis2023_map_verification.json` (latest run).
