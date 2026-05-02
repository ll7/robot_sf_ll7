# Hybrid Rule Local Planner V0 Inspection Note

Date: 2026-04-30

## Existing Planner Interface

Map-based benchmark planners are built in `robot_sf/benchmark/map_runner.py::_build_policy`.
Native local planner adapters expose `plan(observation) -> (linear_velocity, angular_velocity)`.
The map runner projects those commands through the benchmark kinematics model and records
feasibility metadata. Optional `reset()` and `diagnostics()` hooks are supported; `diagnostics()`
is copied into episode `algorithm_metadata.planner_runtime`.

The new family should therefore integrate as an experimental-testing algorithm, not as a
baseline-ready planner. It must require `allow_testing_algorithms: true` in config.

## Available Observations

The planner-facing structured observation includes:

- `robot.position`, `robot.heading`, `robot.speed`, and `robot.radius`.
- `goal.current` and `goal.next`.
- `pedestrians.positions`, `pedestrians.velocities`, `pedestrians.count`, and shared
  `pedestrians.radius`.
- optional `occupancy_grid` plus flattened `occupancy_grid_meta_*` fields for static-obstacle
  checks.
- optional `sim.timestep`/`dt` fields; otherwise local configs must declare rollout timing.

The map runner handles differential-drive, bicycle, and holonomic environment action conversion
after the planner emits the unicycle command.

## Existing Metrics

Episode metrics are computed by `robot_sf/benchmark/metrics.py` through `compute_all_metrics()` and
post-processed for optional SNQI. Relevant existing outputs include success, collisions,
near-misses, min distance, path efficiency, force/comfort exposure when forces are recorded,
smoothness/energy metrics, and SNQI when calibration assets are supplied. This change must not
alter metric definitions.

## Benchmark Entry Points

Small local proof should use:

- smoke: `uv run python scripts/validation/run_policy_search_candidate.py --candidate <name> --stage smoke`
- direct map-runner proof: `robot_sf.benchmark.map_runner.run_map_batch(...)`
- wider policy-search stages only after smoke evidence is clean.

Paper-facing release runs remain out of scope for this first implementation step.

## Constraints and Open Questions

- No training, learned weights, or benchmark-result fitting.
- No changes to scenarios, seeds, metric definitions, or benchmark schema.
- Constants must stay manually specified and documented in config/docs.
- V0 scope is a minimal DWA-style control variant with diagnostics. Later ORCA, TEB-like,
  recovery, and ensemble variants require separate benchmark-driven iterations.
- Existing experimental planners (`risk_dwa`, `mppi_social`, `teb`, `hybrid_portfolio`) provide
  useful implementation patterns but are not renamed or promoted by this work.

