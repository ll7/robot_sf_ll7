# Issue #1659 LiDAR Occupancy Adapter

Date: 2026-05-29

Related issue: https://github.com/ll7/robot_sf_ll7/issues/1659

## Goal

Add the first concrete adapter path for LiDAR-observation planner compatibility: convert range rays
into an ego-frame local occupancy grid, then run one occupancy-backed classical planner without
reading simulator-backed occupancy or direct pedestrian state.

## Implemented Slice

- `robot_sf/planner/lidar_occupancy.py` converts `rays` plus allowed ego goal state from
  `drive_state` into a synthetic planner observation:
  - robot at ego origin,
  - goal reconstructed from target distance and angle,
  - no pedestrian tracks,
  - 3-channel occupancy grid with obstacle and combined channels populated from ray endpoints,
  - ego-frame occupancy metadata compatible with existing occupancy-aware planner helpers.
- `LidarOccupancyPlannerAdapter` wraps an occupancy-aware planner and fails closed with `(0.0, 0.0)`
  if rays are missing or invalid.
- `safety_barrier` can opt into this path through `algo_config["lidar_occupancy_adapter"]`; a
  `lidar_2d` safety-barrier run without that explicit adapter config fails before an episode is
  written.
- Map-runner episodes that resolve to `sensor_fusion_state` now materialize DEFAULT_GYM
  observations and disable simulator-backed occupancy-grid observations before environment
  construction.
- Benchmark metadata marks the path as adapter execution and exposes
  `LidarOccupancySafetyBarrierAdapter`.

## Boundary

This PR does not claim a full LiDAR benchmark campaign is ready. The adapter-level smoke path is
implemented and tested, but the broader map-runner environment setup still needs Issue #1613 to
materialize complete LiDAR-observation benchmark scenarios. The adapter is intentionally
testing-only and must not be reported as fallback or degraded benchmark success.

## Validation

```bash
PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  uv run pytest tests/test_planner/test_lidar_occupancy.py \
    tests/benchmark/test_lidar_occupancy_adapter.py \
    tests/benchmark/test_algorithm_metadata_contract.py::test_safety_barrier_accepts_lidar_level_through_sensor_fusion_contract \
    tests/benchmark/test_observation_levels.py::test_observation_level_rejects_unsupported_planner_combination -q
# 8 passed

PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  uv run ruff check robot_sf/planner/lidar_occupancy.py robot_sf/benchmark/map_runner.py \
    robot_sf/benchmark/algorithm_metadata.py tests/test_planner/test_lidar_occupancy.py \
    tests/benchmark/test_lidar_occupancy_adapter.py \
    tests/benchmark/test_algorithm_metadata_contract.py
# All checks passed

PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  uv run pytest tests/test_range_sensor.py tests/planner/test_safety_barrier.py -q
# 37 passed

PYTHONPATH=$PWD UV_PROJECT_ENVIRONMENT=/home/luttkule/git/robot_sf_ll7/.venv UV_NO_SYNC=1 \
  scripts/dev/check_docs_proof_consistency_diff.sh
# passed for 8 changed files
```

## Follow-Up

Issue #1613 should own the full benchmark setup and validation contract for running scenario
episodes under `lidar_2d`. Issue #1660 owns the separate ray-to-tracked-agent adapter path for
social-state planners.
