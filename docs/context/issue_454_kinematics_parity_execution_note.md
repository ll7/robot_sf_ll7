# Issue #454 Execution Note (Kinematics Contract + Parity)

## Tracker Close-out Status

Issue #454 is a coordination tracker. The runtime and benchmark work landed through child issues
and should stay there rather than being reimplemented in this parent issue.

- #529 closed via PR #534: first-class `KinematicsModel` contract and planner/runtime wiring.
- #530 closed via PR #555: kinematics feasibility diagnostics in benchmark artifacts and reports.
- #523 closed via PR #555: cross-kinematics parity campaign contract and reporting artifacts.

## Final Support Contract

- Runtime scenario `robot_config` controls benchmark robot kinematics for `differential_drive`,
  `bicycle_drive`, and `holonomic`.
- Holonomic robot support includes two explicit command modes:
  - `vx_vy`
  - `unicycle_vw`
- Planner/runtime command projection is centralized through
  `robot_sf/planner/kinematics_model.py`, including differential-drive, bicycle-drive, and
  holonomic-passthrough models.
- Benchmark episode and summary metadata distinguish planner intent from robot feasibility through
  `algorithm_metadata.planner_kinematics` and `algorithm_metadata.kinematics_feasibility`.
- Camera-ready campaign execution supports a `kinematics_matrix` and writes parity and skip
  artifacts under the campaign `reports/` directory:
  - `kinematics_parity_table.csv`
  - `kinematics_parity_table.md`
  - `kinematics_skipped_combinations.csv`
  - `kinematics_skipped_combinations.md`

## Interpretation Caveat

Kinematics parity should be interpreted jointly with:

- `planner_kinematics.execution_mode` (native vs adapter),
- `planner_kinematics.robot_command_space`,
- `planner_kinematics.projection_policy`,
- `kinematics_feasibility.projection_rate`,
- `kinematics_feasibility.infeasible_rate`.

High projection or infeasible rates mean metric deltas may be driven by robot constraints or adapter
routing, not planner intent alone. This supports issue #454's original hypothesis that planner
capability, adapter effects, and robot feasibility must not be collapsed into one metric signal.

## Evidence Anchors

- Runtime contract: `robot_sf/planner/kinematics_model.py`
- Classic planner wiring: `robot_sf/planner/classic_planner_adapter.py`
- Benchmark feasibility aggregation: `robot_sf/benchmark/map_runner.py`
- Campaign parity reporting: `robot_sf/benchmark/camera_ready_campaign.py`
- Schema field: `robot_sf/benchmark/schemas/episode.schema.v1.json`
- Contract tests:
  - `tests/test_classic_planner_adapter.py`
  - `tests/benchmark/test_map_runner_utils.py`
  - `tests/benchmark/test_camera_ready_campaign.py`

## Boundary

This note does not claim that a fresh full paper campaign was run for issue #454. The tracker close
condition is the merged runtime contract plus the benchmark artifact/reporting surface. Any new
paper-facing claim still needs the campaign output path and validation command from the specific
benchmark run that produced the cited values.
