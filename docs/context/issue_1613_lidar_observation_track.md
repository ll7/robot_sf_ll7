# Issue #1613 LiDAR Observation Track Setup

Date: 2026-05-29

## Scope

This note records the initial LiDAR-observation benchmark-track setup for parent epic #1611. It
defines the track contract, creates a minimal smoke packet, and validates benchmark-row metadata
with a stubbed compatible policy. It does not train policies, adapt classical planners, launch a
full campaign, or claim comparability with grid/socnav-state benchmarks.

## Launch Packet

- Packet: `configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml`
- Scenario source: `configs/scenarios/sanity_v1.yaml`
- Observation source of truth: `docs/dev/observation_contract.md`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

The packet reuses the existing sanity scenario matrix so the first smoke changes the observation
track, not the scenario-generation machinery.

## Observation Contract

The track is labeled with benchmark observation level `lidar_2d` and active observation mode
`sensor_fusion_state`. Runtime policy inputs are limited to:

- robot state and current goal information encoded in `drive_state`,
- LiDAR/range observations encoded in `rays`.

The track excludes occupancy grids, simulator-backed global map occupancy, direct SocNav
pedestrian positions/velocities, future pedestrian trajectories, and collision or success labels.

## Smoke Validation

Targeted validation:

```bash
uv run pytest -q tests/benchmark/test_lidar_observation_track.py
```

The smoke uses a stubbed PPO-compatible policy because no trained LiDAR checkpoint is promoted by
this issue. The test still exercises the map-runner episode path, verifies the policy sees only
`drive_state` and `rays`, and checks that the emitted row records `observation_level=lidar_2d` and
`observation_mode=sensor_fusion_state` in both top-level and algorithm metadata.

## Limitations

The current setup is a contract smoke, not benchmark evidence. A real LiDAR-track campaign needs a
compatible trained learned policy, a diagnostic dummy planner that is explicitly marked
non-evidence, or LiDAR-derived adapters for classical planners. Fallback, degraded, or unavailable
rows must remain caveats or exclusions rather than successful LiDAR-track evidence.
