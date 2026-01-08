# Francis 2023 scenarios: current state

## Existing capabilities in the repo

- SVG-based maps are the primary way to describe geometry, spawn/goal zones, and routes.
  - Parser: `robot_sf/nav/svg_map_parser.py`
  - Labels: `robot_spawn_zone`, `robot_goal_zone`, `ped_spawn_zone`, `ped_goal_zone`,
    `robot_route_<spawn>_<goal>`, `ped_route_<spawn>_<goal>`, `ped_crowded_zone`.
  - Single pedestrians can be placed via circles labeled
    `single_ped_<id>_start` and `single_ped_<id>_goal`.
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

## What is missing for the Francis 2023 scenarios

- No maps for the Francis 2023 Figure 7 scenarios exist yet in `maps/svg_maps/`.
- No scenario matrix (YAML) exists to define these scenarios for training or
  benchmarking workflows.
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

## Spec context

- This spec folder currently contains only:
  - `Readme.md` with the citation
  - `Fig7_Francis2023_scenarios.png` (reference diagram)
