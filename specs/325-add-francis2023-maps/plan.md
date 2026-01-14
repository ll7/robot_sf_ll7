# Francis 2023 scenarios: implementation plan

## Decisions captured so far

- Use SVG maps for geometry and labeled spawn/goal/routes, consistent with existing
  `maps/svg_maps/*.svg` conventions.
- Use a dedicated scenario matrix YAML (for example
  `configs/scenarios/francis2023.yaml`) with `name`, `map_file`, `simulation_config`,
  and `metadata` to define the scenarios.
- For single-pedestrian scenarios, set `ped_density=0` and rely on
  `single_ped_<id>_start/goal` markers in the map as the base geometry. Use scenario
  YAML overlays for per-scenario goals/trajectories, `speed_m_s`, `wait_at`, and `note`.
- For crowd/flow scenarios, use `ped_route_*` paths and `ped_crowded_zone`
  rectangles rather than ad-hoc spawning.
- Behavior-only differences (wait, follow, lead, join/leave, etc.) will require
  code support beyond maps and scenario configs.
- Prefer POI labels in SVGs for readability, with a preview helper to visualize
  trajectories and waits (`scripts/tools/preview_scenario_trajectories.py`).

## Phase 1: Map inventory and scenario mapping

- Build a table mapping each Figure 7 scenario (a) to (y) to:
  - map geometry needed
  - number of humans/robots
  - required behaviors (static, slow, wait, follow, lead, group)
  - whether current capabilities suffice or code changes are required
- Identify which existing maps can be reused and which must be created anew.
- Mapping table: `specs/325-add-francis2023-maps/scenario_mapping.md`.
- Status: done.

## Phase 2: Base SVG maps (geometry-first)

- Create new SVG maps under `maps/svg_maps/` for the scenarios that are
  geometry-only or can be represented with single pedestrians:
  - frontal approach, obstruction, overtaking, down path
  - blind corner, narrow hallway, narrow doorway
  - entering/exiting room, entering/exiting elevator
- Encode single pedestrians with `single_ped_<id>_start/goal` markers.
- Use obstacles and boundaries to constrain routes and visibility as in the figure.
- Status: done (initial geometry + boundary obstacles in `maps/svg_maps/francis2023/`).

## Phase 3: Scenario matrix configuration

- Add `configs/scenarios/francis2023.yaml` with:
  - `map_file` entries pointing to the new SVGs
  - `simulation_config` settings for episode length, ped density, and speed scaling
  - `single_pedestrians` overlays using:
    - `goal` or `goal_poi`
    - `trajectory` or `trajectory_poi`
    - optional `speed_m_s`, `wait_at`, `note`
  - `metadata` fields to record archetype, flow, and behavioral tags
  - seed lists for reproducibility
- Status: draft complete for geometry-first scenarios; behavior-driven entries pending.

## Phase 3b: POI annotations and verification

- Add POI circles (with labels) to the Francis SVGs to make overlays readable.
- Re-run map verification for the updated SVGs.
- Status: POI circles added; verification complete.

## Phase 4: Behavior support for non-geometry scenarios

### Phase 4a: Overlay parsing and preview helper

- Extend `SinglePedestrianDefinition` to optionally include `speed_m_s`, `wait_at`, and `note`.
- Parse scenario overlays in `robot_sf/training/scenario_loader.py` (goal/trajectory, POI support).
- Add a preview helper (`scripts/tools/preview_scenario_trajectories.py`) to visualize overlays.
- Status: done.

### Phase 4b: Runtime behavior controller

- Extend single-ped runtime behavior to honor waits and role tags
  (wait, follow, lead, accompany, join, leave).
- Add a `SinglePedestrianBehavior` controller that:
  - advances trajectory waypoints when reached
  - applies waits where specified
  - updates the pedestrian goal so PySocialForce moves toward the next waypoint
- Update the SVG/JSON map loaders to accept these optional fields:
  - SVG: add a lightweight encoding strategy (POI labels or companion overlay file).
  - JSON: parse the new fields in `robot_sf/nav/map_config.py`.
- Wire the new behavior controller into `robot_sf/sim/simulator.py` alongside
  the existing `CrowdedZoneBehavior` and `FollowRouteBehavior`.
 - Status: done.

## Phase 5: Crowd and flow scenarios

- Implement maps and configs for:
  - crowd navigation, parallel traffic, perpendicular traffic
  - circular crossing, robot crowding
- Calibrate `ped_density` and routes to approximate the intended flows in the figure.
- Status: done (initial density tuning applied; refine as needed).

## Phase 6: Validation and documentation

- Run map verification for the new SVGs.
- Generate scenario thumbnails for visual sanity checks.
- Update `docs/README.md` and `CHANGELOG.md` if these scenarios are user-facing.
- Add or extend tests if new behavior logic is introduced (especially for
  trajectory waypoint advancement and per-ped speed).

## Current state (session snapshot)

- New crowd/traffic SVGs added under `maps/svg_maps/francis2023/`:
  `francis2023_crowd_navigation.svg`, `francis2023_parallel_traffic.svg`,
  `francis2023_perpendicular_traffic.svg`, `francis2023_circular_crossing.svg`,
  `francis2023_robot_crowding.svg`.
- Join/leave maps updated with per-ped goal POIs; scenario overrides now include
  static anchors (`role: wait`) and leave uses `goal_poi: poi_h1_goal`.
- `configs/scenarios/francis2023.yaml` includes behavior-driven entries plus
  crowd/traffic scenarios with initial `ped_density` tuning:
  crowd=0.06, parallel=0.04, perpendicular=0.10, circular=0.06, crowding=0.10.
- Map verification run (local mode) for all new crowd/traffic SVGs.
- Full-style previews rendered for the new crowd/traffic scenarios under
  `output/preview/scenario_trajectories/`.
- Sanity sweep (10 steps each) confirms nonzero ped counts:
  crowd=7, parallel=8, perpendicular=7, circular=6, crowding=20.
- New SVGs are still untracked (need `git add` when ready).

## Next steps

- Review the crowd/traffic preview PNGs and adjust routes/zones if flows look off.
- Refine `ped_density` values to better match the Francis 2023 intent.
- Regenerate previews after tuning (consider `--style full` for route visibility).
- Update `docs/README.md` to link the Francis 2023 scenarios if this is a user-facing addition.

## Validation / testing commands

- Verify new maps (local):
  - `source .venv/bin/activate`
  - `MPLCONFIGDIR=$PWD/output/tmp/mplconfig XDG_CACHE_HOME=$PWD/output/tmp/cache MPLBACKEND=Agg \\`
    `python scripts/validation/verify_maps.py --scope francis2023_crowd_navigation.svg --mode local`
  - Repeat for: `francis2023_parallel_traffic.svg`, `francis2023_perpendicular_traffic.svg`,
    `francis2023_circular_crossing.svg`, `francis2023_robot_crowding.svg`.
- Generate previews (full style for routes/zones):
  - `MPLCONFIGDIR=$PWD/output/tmp/mplconfig XDG_CACHE_HOME=$PWD/output/tmp/cache MPLBACKEND=Agg \\`
    `python scripts/tools/preview_scenario_trajectories.py --scenario configs/scenarios/francis2023.yaml --scenario-id francis2023_parallel_traffic --style full`
  - Repeat for the other crowd/traffic scenario IDs (or use `--all` for full batch output).
- Quick sanity sweep (ped counts + 10 steps):
  - `MPLCONFIGDIR=$PWD/output/tmp/mplconfig XDG_CACHE_HOME=$PWD/output/tmp/cache MPLBACKEND=Agg \\`
    `python - <<'PY'`
    `from pathlib import Path`
    `from robot_sf.gym_env.environment_factory import make_robot_env`
    `from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios, select_scenario`
    `scenario_path = Path("configs/scenarios/francis2023.yaml").resolve()`
    `scenarios = load_scenarios(scenario_path)`
    `scenario_ids = ["francis2023_crowd_navigation","francis2023_parallel_traffic","francis2023_perpendicular_traffic","francis2023_circular_crossing","francis2023_robot_crowding"]`
    `for scenario_id in scenario_ids:`
    `    scenario = select_scenario(scenarios, scenario_id)`
    `    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)`
    `    env = make_robot_env(config=config)`
    `    env.reset()`
    `    ped_pos = getattr(env.simulator, "ped_pos", None)`
    `    ped_count = int(ped_pos.shape[0]) if ped_pos is not None else 0`
    `    print(f"{scenario_id}: ped_count={ped_count}")`
    `    for _ in range(10):`
    `        _, _, terminated, truncated, _ = env.step(env.action_space.sample())`
    `        if terminated or truncated:`
    `            break`
    `    env.close()`
    `PY`

## Handoff checklist

- Confirm untracked SVGs are staged:
  `maps/svg_maps/francis2023/francis2023_crowd_navigation.svg`,
  `maps/svg_maps/francis2023/francis2023_parallel_traffic.svg`,
  `maps/svg_maps/francis2023/francis2023_perpendicular_traffic.svg`,
  `maps/svg_maps/francis2023/francis2023_circular_crossing.svg`,
  `maps/svg_maps/francis2023/francis2023_robot_crowding.svg`,
  `maps/svg_maps/francis2023/francis2023_join_group.svg`,
  `maps/svg_maps/francis2023/francis2023_leave_group.svg`.
- Review previews in `output/preview/scenario_trajectories/` for crowd/traffic flows
  and update routes/zones if the geometry does not match Fig. 7 intent.
- Re-run map verification for the new maps after any edits.
- Re-run the sanity sweep (ped counts + 10 steps) and confirm ped counts are within
  expected ranges after density tweaks.
- Decide whether to link the Francis 2023 scenarios in `docs/README.md` and update
  if this should be user-facing.
- Keep `output/tmp` as the location for matplotlib caches; ensure `.cache` and
  `.mplconfig` do not reappear at the repo root.
