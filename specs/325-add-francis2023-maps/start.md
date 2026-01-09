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
- Single pedestrians now advance trajectory waypoints at runtime, honor wait rules,
  and support role tags (wait/follow/lead/accompany/join/leave).
- Preview helper: `scripts/tools/preview_scenario_trajectories.py` renders overlays
  on top of map geometry.

## Spec context

- This spec folder includes:
  - `Readme.md` with the citation
  - `Fig7_Francis2023_scenarios.png` (reference diagram)
  - `plan.md` and `scenario_mapping.md` for the build plan
  - `start.md` (this document)

## Current Francis 2023 assets

- SVG maps under `maps/svg_maps/francis2023/` cover Fig. 7a through 7y, including the
  join/leave and crowd/traffic layouts.
  - Each map includes boundary obstacles and POI markers (`poi_h1_start`, `poi_h1_goal`).
- Scenario matrix: `configs/scenarios/francis2023.yaml`.
  - Geometry-first entries plus behavior-driven scenarios (wait/proceed, follow/lead,
    accompany, join/leave) and crowd/traffic scenarios with initial `ped_density` tuning.
  - Speed overrides and waits are encoded directly in the per-ped overlays.
- Map verification: `output/validation/francis2023_map_verification.json` (latest run).
- Scenario previews: `output/preview/scenario_trajectories/` (full-style renders for
  the Francis scenarios, including crowd/traffic flows).

## Remaining work

- Review the crowd/traffic preview PNGs and adjust routes/zones or `ped_density` to
  better match the intended Fig. 7 flows.
- If these scenarios are user-facing, link them in `docs/README.md` and update
  `CHANGELOG.md`.
