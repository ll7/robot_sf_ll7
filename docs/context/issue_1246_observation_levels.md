# Issue #1246 Graded Observation Levels 2026-05-18

## Goal

Issue #1246 adds a benchmark-facing observation-level vocabulary so results can
state whether a planner ran with privileged state, tracked agents, noisy tracks,
lidar-style inputs, or partial/occluded observations.

## Implementation Boundary

This is a contract and provenance layer first. The new
`robot_sf/benchmark/observation_levels.py` module defines:

- `oracle_full_state`
- `tracked_agents_no_noise`
- `tracked_agents_with_noise`
- `lidar_2d`
- `occluded_partial_state`

`algorithm_metadata`, `planner_command_contract`, and `map_runner` now resolve
the active observation level alongside the existing observation mode. Unsupported
planner/level combinations fail before episodes are written. Episode records
include top-level `observation_level`, `scenario_params.observation_level`, and
`algorithm_metadata.observation_level`.

The first slice deliberately does not add real camera perception, detector
training, calibrated tracking, or a new environment observation implementation.
The levels are benchmark evidence labels and compatibility gates, not
sim-to-real validity claims.

## Validation

- Red state: `tests/benchmark/test_observation_levels.py` initially failed at
  collection because `robot_sf.benchmark.observation_levels` did not exist.
- `uv run pytest tests/benchmark/test_observation_levels.py tests/benchmark/test_map_runner_utils.py::test_run_map_episode_smoke tests/benchmark/test_map_runner_utils.py::test_run_map_batch_serial_and_resume -q`
  - `6 passed`
- `uv run pytest tests/benchmark/test_observation_levels.py tests/benchmark/test_algorithm_metadata_contract.py tests/benchmark/test_planner_command_contract.py tests/benchmark/test_map_runner_utils.py tests/benchmark/test_map_runner_resume_identity.py tests/benchmark/test_observation_noise.py tests/test_observation_stack_config.py tests/test_socnav_observation_mode.py tests/test_cli.py -q`
  - `140 passed`
- `uv run ruff check robot_sf/benchmark/observation_levels.py robot_sf/benchmark/algorithm_metadata.py robot_sf/benchmark/planner_command_contract.py robot_sf/benchmark/map_runner.py robot_sf/benchmark/runner.py robot_sf/benchmark/cli.py robot_sf/baselines/interface.py tests/benchmark/test_observation_levels.py tests/benchmark/test_map_runner_utils.py`
  - passed

## Related Surfaces

- Issue: https://github.com/ll7/robot_sf_ll7/issues/1246
- Vocabulary: `robot_sf/benchmark/observation_levels.py`
- Planner metadata: `robot_sf/benchmark/algorithm_metadata.py`
- Compatibility validation: `robot_sf/benchmark/planner_command_contract.py`
- Episode propagation: `robot_sf/benchmark/map_runner.py`
- Observation docs: `docs/dev/observation_contract.md`
