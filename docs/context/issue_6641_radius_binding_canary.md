# Issue #6641 — Gate 1 runtime radius-binding canary (#6600 collision-envelope campaign)

Status: implemented (Gate 1 binding canary). Parent: #6600 (quantify planner-ranking
sensitivity to collision-envelope radius). Oracle foundation: #5574 (planner-free
feasibility oracle, see `evidence/issue_5574_feasibility_oracle_2026-07-14/README.md`).

## What it proves

Before any production SLURM sweep (Gate 2), Gate 1 proves on a geometry-sensitive scenario
that a declared robot collision-envelope radius propagates consistently to the five binding
surfaces, and emits a machine-readable go/no-go verdict per surface. The canonical radius
source is the scenario-level `robot_config.radius`, written by
`robot_sf.scenario_certification.feasibility_oracle.make_envelope_scenario` (the same
surface the campaign's envelope-sensitivity sweep uses).

The five binding surfaces and the real code path each probe reads:

1. `simulator_collision_geometry` — `build_robot_config_from_scenario(...).robot_config.radius`
   (the simulator's robot collision circle; also sizes the pedestrian reserved zone).
2. `obstacle_pedestrian_contact_logic` — the radius-aware contact boundary
   `clearance = center_distance - (robot_radius + ped_radius)`
   (`robot_sf/benchmark/collision_definition_inventory.classify_clearance_regime`).
3. `feasibility_oracle` — `make_envelope_scenario` injection plus the planner-free geometric
   inflation (`envelope_radius_m` / `envelope_diameter_m = 2 * radius`).
4. `metric_metadata_and_output_rows` — the runner row extraction
   (`robot_sf.benchmark.runner._scenario_robot_radius_m`) and the orchestrator metric-data
   binding (`getattr(robot_cfg, "radius", 1.0)`).
5. `planner_inputs` — `replace(force_config, robot_radius=robot.config.radius)` for the
   ped-robot and adversarial-ped force configs (mirrors `robot_sf/sim/simulator.py`).

Semantics are fail-closed: any surface that binds a radius differing from the declared
target by more than the tolerance, or that cannot be observed, is a no-go that stops the
campaign. The canary does not change the frozen `0.0.3.post1` metric semantics; it only
observes which radius each surface binds.

## Run it

```bash
uv run python scripts/benchmark/run_radius_binding_canary_issue_6641.py \
  --out-json output/radius_binding_canary_6641.json
```

Defaults to the geometry-sensitive `configs/scenarios/single/francis2023_narrow_doorway.yaml`
at the #6600 fixed treatment (0.5 m, 0.8 m, 1.0 m). Exit code 0 = go (all surfaces bind at
every radius), 1 = no-go (fail-closed), 2 = usage error.

Machine-readable schemas: per-radius `radius_binding_canary.v1`, wrapped by the runner in
`radius_binding_canary_report.v1`.

## Current result

On `francis2023_narrow_doorway`, all five surfaces bind the declared radius at 0.5 m, 0.8 m,
and 1.0 m: overall `go: true`. The oracle probe runs the real certifier on the real map
(planner-free geometric margin). Negative controls in the tests confirm the canary fails
closed on a silently divergent metric binding and on an unobservable surface.

## Claim boundary

Within-simulator radius-binding canary only. It is not a physical-footprint validation, a
realism result, or a safety guarantee, and it is not Gate 2 benchmark evidence.
