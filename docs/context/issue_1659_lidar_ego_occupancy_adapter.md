# Issue #1659 LiDAR Ego Occupancy Adapter

## Scope

Issue #1659 asked for a narrow `lidar_rays -> ego occupancy_grid` adapter for one representative
classical local planner, with explicit opt-in and fail-closed handling for unsupported
planner/observation-level combinations.

This slice implements the representative planner as `lidar_grid_route`, a testing-only wrapper
around the existing `grid_route` planner. It does not change `grid_route` itself into a LiDAR
planner; `grid_route` continues to reject `observation_level=lidar_2d` unless the explicit
LiDAR adapter algorithm is selected.

## Runtime Contract

- Algorithm key: `lidar_grid_route`
- Alias: `lidar_occupancy_grid_route`
- Readiness tier: `experimental`
- Opt-in: `allow_testing_algorithms: true`
- Observation level: `lidar_2d`
- Active observation mode: `sensor_fusion_state`
- Runtime inputs consumed: `drive_state`, `rays`
- Derived payload: ego-frame occupancy grid with obstacle and combined channels
- Forbidden runtime dependencies: privileged `map`, simulator obstacle lists, SocNav `robot`,
  `pedestrians`, or precomputed `occupancy_grid`

The adapter uses the same ray-angle convention as `robot_sf.sensor.range_sensor`: evenly spaced
angles over the configured symmetric opening, with the duplicated upper endpoint excluded.
Max-range rays are treated as no obstacle endpoint. Finite shorter returns mark obstacle cells in
the ego grid.

Sensor-fusion observations are normalized by the environment, so
`configs/algos/lidar_grid_route_issue_1659.yaml` records the unnormalization scales used before
creating the grid-route observation.

## Files

- `robot_sf/planner/lidar_occupancy_grid.py`
- `robot_sf/benchmark/algorithm_readiness.py`
- `robot_sf/benchmark/algorithm_metadata.py`
- `robot_sf/benchmark/map_runner.py`
- `configs/algos/lidar_grid_route_issue_1659.yaml`
- `tests/planner/test_lidar_occupancy_grid.py`
- `tests/benchmark/test_lidar_grid_route_contract.py`

## Validation

Commands run:

```bash
uv run pytest tests/planner/test_lidar_occupancy_grid.py tests/benchmark/test_lidar_grid_route_contract.py tests/benchmark/test_algorithm_metadata_contract.py
PYTEST_NUM_WORKERS=8 BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Observed results:

- Targeted adapter/metadata suite: `37 passed`.
- Full PR readiness suite on the working tree before commit:
  `4370 passed, 11 skipped, 9 warnings`.

Final branch-head readiness is rerun after commit before PR creation.

## Limitations

This is not benchmark-strengthening evidence for classical planners under LiDAR. It is a
testing-only adapter that proves one bounded compatibility path and preserves fail-closed behavior
for the original `grid_route` planner. It only rasterizes ray endpoints; it does not infer hidden
obstacles, dynamic agents, occlusions, or map topology beyond the current LiDAR scan.
