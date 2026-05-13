# Open issues implementation status pass

Date: 2026-05-12

## Scope

This note summarizes the broad open-issue implementation pass that touched multiple low-risk slices across environment cleanup, manual-control foundations, planner features, visibility filtering, multi-AMV records, CARLA parity tooling, and documentation cleanup.

No validation commands were run during this pass.

## Implemented / partially implemented slices

### #1150 / #1141 / #1146 legacy env cleanup

Implemented:

- Added `robot_sf/gym_env/crowd_sim_env.py` with `CrowdSimulationConfig` and `CrowdSimEnv`.
- Added `make_crowd_sim_env(...)` to `robot_sf/gym_env/environment_factory.py`.
- Removed `robot_sf/gym_env/simple_robot_env.py`.
- Removed `robot_sf/gym_env/empty_robot_env.py` after replacing the useful crowd-only capability.
- Added `tests/test_crowd_sim_env_contract.py`.
- Updated refactoring docs and factory signature tests for the new crowd env surface.

Remaining:

- Run targeted env/factory tests.
- Decide whether to close or split any remaining broad #1150 follow-ups after validation.

### #1142 video FPS

Implemented:

- `SimulationView` now uses effective target FPS for video encoding when explicit `video_fps` is not set.
- Added targeted coverage in `tests/visuals/test_sim_view_coverage_paths.py`.

Remaining:

- Run visual targeted tests.

### #1143 / #1149 sensor history

Implemented:

- Documented SensorFusion temporal stack order as oldest-to-newest, current sample at `[-1]`.
- Added shared `append_history_row(...)` helper and used it in both sensor fusion classes.
- Added focused SensorFusion stack test.

Remaining:

- Run targeted sensor tests.

### #1144 WheelSpeedState units

Implemented:

- Documented `WheelSpeedState` as left/right wheel angular velocities in radians per second.
- Added targeted differential-drive test.

Remaining:

- Run targeted differential-drive tests.

### #1145 TODO docstring cleanup

Implemented:

- Replaced clear TODO docstrings in several public modules touched or inspected during this pass, including goal sensor, simulator registry/config helpers, dummy backend, TensorBoard logging callback, fast-pysf wrapper, bicycle state, and render helper docstrings.

Remaining:

- Broad repository-wide placeholder cleanup remains out of scope for this pass.

### #1147 classic_benchmark_full fail-closed

Implemented:

- Added `FullBenchmarkUnavailableError` fallback and CLI exit-code handling.
- Added targeted CLI test for actionable stderr/no traceback behavior.

Remaining:

- Run benchmark-full CLI targeted tests.

### #1148 env constructor defaults

Implemented:

- Changed env constructors touched in this pass to avoid shared config instance defaults.
- Added constructor signature regression test.

Remaining:

- Run factory/signature tests.

### #1151 / #1152 / #1153 / #1154 manual-control foundations

Implemented:

- Added `robot_sf/manual_control/` package foundations:
  - `input_mapping.py`
  - `modes.py`
  - `session.py`
  - `recording.py`
  - `baseline.py`
  - `export.py`
  - `replay.py`
  - `manifest.py`
- Added CLI `scripts/manual_control/export_bc_samples.py`.
- Added focused tests for input mapping, modes, session state, recording, baseline comparison, BC export, replay grouping, manifests, and export CLI.
- Added context note `docs/context/issue_1151_manual_control_mvp_foundation.md`.
- Added web-game context note `docs/context/issue_1154_web_game_data_collection_path.md` earlier in the pass.

Remaining:

- Implement the actual Pygame manual-control runner.
- Wire runner events to session controller, mapper, recorder, manifest, and renderer overlays.
- Validate the schema end to end before web-game work.
- Implement alternate steering/view modes, rewind, and visual replay in follow-up passes.

### #1138 predictive obstacle features

Implemented:

- Added `robot_sf/planner/obstacle_features.py` with deterministic `predictive_obstacle_features_v1` extractor.
- Added schema metadata validation and `append_obstacle_features(...)` composition helper.
- Added tests in `tests/planner/test_obstacle_features.py`.
- Added context note `docs/context/issue_1138_predictive_obstacle_features_schema.md`.

Remaining:

- Wire collection, training/evaluation, checkpoint metadata, and runtime inference to the shared schema.
- Add config-first runnable path only after lifecycle wiring exists.

### #1124 dynamic pedestrian occlusion

Implemented:

- Added `dynamic_pedestrian_occlusion_mask(...)` helper.
- Added opt-in `ObservationVisibilitySettings.dynamic_occlusion`, scenario-loader parsing, metadata propagation, and SocNav observation filtering.
- Added tests for helper semantics, observation filtering, and scenario parsing.
- Added context note `docs/context/issue_1124_dynamic_pedestrian_occlusion_contract.md`.

Remaining:

- Run targeted visibility tests and visibility smoke.

### #1128 multi-AMV episode extension

Implemented:

- Added `multi_amv_episode_extension(...)` for additive namespaced multi-AMV episode blocks.
- Updated `scripts/validation/run_multi_amv_smoke.py` to emit the same block.
- Added tests in `tests/benchmark/test_multi_amv.py`.
- Added context note `docs/context/issue_1128_multi_amv_episode_extension.md`.

Remaining:

- Integrate multi-AMV into primary benchmark command path or a documented equivalent.
- Add aggregate report support.

### #1110 / #872 CARLA parity tooling

Implemented:

- Added import-safe `robot_sf/carla_bridge/parity.py` and package export.
- Added CLI `scripts/carla_bridge/compare_oracle_replay_metrics.py`.
- Added tests under `tests/carla_bridge/`.
- Added context note `docs/context/issue_1110_carla_oracle_replay_parity_adapter.md`.

Remaining:

- Wire to actual CARLA/Robot-SF replay artifact shapes after #1111 live smoke output exists.

## Blocked or execution-only issues

### #1119 Docker reproduction smoke

Blocked on Docker-capable host/runner and digest/checksum evidence.

### #1111 live CARLA T1 replay smoke

Blocked on CARLA-capable host/runtime.

### #1134 SocNavBench ETH map conversion

Blocked on official SocNavBench/S3DIS ETH source assets staged locally with checksums.

### #1126 SDD-derived scenario curation

Blocked on official SDD source data staging, scene/video selection, scale assumptions, and importer run.

### #1108 BC warm-start PPO experiment

Execution/artifact task. Requires real BC pretraining, PPO fine-tuning, durable artifact promotion, and policy-analysis comparison.

## Required validation before PR handoff

At minimum, run targeted tests covering touched surfaces:

```bash
uv run pytest \
  tests/test_crowd_sim_env_contract.py \
  tests/test_environment_factory_signatures.py \
  tests/differential_drive_test.py \
  tests/test_sensor_fusion_stack.py \
  tests/test_manual_control_*.py \
  tests/planner/test_obstacle_features.py \
  tests/test_socnav_dynamic_occlusion.py \
  tests/test_socnav_observation.py \
  tests/training/test_scenario_loader.py \
  tests/benchmark/test_multi_amv.py \
  tests/carla_bridge/test_parity.py \
  tests/carla_bridge/test_parity_cli.py \
  tests/visuals/test_sim_view_coverage_paths.py \
  tests/benchmark_full/test_classic_benchmark_full_cli.py \
  -q
```

Before PR handoff, run the repository PR readiness gate after syncing with latest `origin/main`.

## Validation status

No validation commands have been run during this pass.
