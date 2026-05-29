# Issue #1660 LiDAR Tracked-Agent Adapter

## Scope

Issue #1660 asked for one narrow `lidar_rays -> tracked_agents` adapter path so a
non-learning social-state planner can run under the LiDAR observation track without consuming
direct simulator pedestrian state.

This slice implements `lidar_social_force`, a testing-only wrapper around the existing
`SocialForcePlannerAdapter`. It does not change `social_force` itself into a LiDAR planner;
`social_force` continues to reject `observation_level=lidar_2d` unless the explicit LiDAR adapter
algorithm is selected.

## Runtime Contract

- Algorithm key: `lidar_social_force`
- Alias: `lidar_tracked_social_force`
- Readiness tier: `experimental`
- Opt-in: `allow_testing_algorithms: true`
- Observation level: `lidar_2d`
- Active observation mode: `sensor_fusion_state`
- Runtime inputs consumed: `drive_state`, `rays`
- Derived payload: `tracked_agents`
- Tracking assumption: current-frame LiDAR endpoint clusters
- Velocity assumption: zero velocity, no identity persistence
- Occlusion policy: visible ray endpoints only; hidden agents are unavailable
- Noise policy: raw range noise only; no separate tracker noise model
- Forbidden runtime dependencies: privileged `robot`, `pedestrians`, `tracked_agents`, `map`,
  or `static_obstacles` inputs from the caller

The adapter uses the same ray-angle convention as `robot_sf.sensor.range_sensor`: evenly spaced
angles over the configured symmetric opening, with the duplicated upper endpoint excluded.
Finite non-max range returns are grouped into adjacent ray clusters and converted into synthetic
track positions. Track velocities are zero by construction.

## Files

- `robot_sf/planner/lidar_tracked_agents.py`
- `robot_sf/benchmark/algorithm_readiness.py`
- `robot_sf/benchmark/algorithm_metadata.py`
- `robot_sf/benchmark/map_runner.py`
- `configs/algos/lidar_social_force_issue_1660.yaml`
- `tests/planner/test_lidar_tracked_agents.py`
- `tests/benchmark/test_lidar_social_force_contract.py`

## Validation

Commands run before full branch-head readiness:

```bash
uv run pytest tests/planner/test_lidar_tracked_agents.py tests/benchmark/test_lidar_social_force_contract.py tests/benchmark/test_algorithm_metadata_contract.py
uv run ruff check robot_sf/planner/lidar_tracked_agents.py robot_sf/benchmark/algorithm_readiness.py robot_sf/benchmark/algorithm_metadata.py robot_sf/benchmark/map_runner.py tests/planner/test_lidar_tracked_agents.py tests/benchmark/test_lidar_social_force_contract.py
codex-agent-worker --provider qwen --model Qwen3.6-27B --timeout 900 --slug issue-1660-qwen-scout --task-file /tmp/qwen_issue1660_scout.md
```

Observed results:

- Targeted adapter/metadata suite after merging `origin/main`: `38 passed`.
- Ruff targeted check: passed.
- Qwen read-only scout returned successfully and suggested `risk_dwa` as the smallest possible
  second target, but this issue requires only one representative planner path; this slice keeps
  the implementation scoped to `lidar_social_force`.

Full branch-head readiness is run before PR creation and recorded in the PR body.

## Limitations

This is not perception-tracking research and not benchmark-strengthening evidence for
LiDAR-only social-state planners. It is a testing-only adapter proving that one social-state
planner path can be wired from LiDAR-derived visible endpoint clusters while preserving the
fail-closed contract for the original planner. It does not infer hidden agents, maintain
identities, estimate velocities, or distinguish pedestrians from static obstacles in the scan.
