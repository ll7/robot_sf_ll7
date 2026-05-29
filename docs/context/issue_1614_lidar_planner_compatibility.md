# Issue #1614 LiDAR Planner Compatibility Audit

Date: 2026-05-29

## Scope

This note classifies representative planners for the LiDAR-observation benchmark track under
parent epic #1611. It does not implement adapters, launch benchmark campaigns, or claim benchmark
performance. The machine-readable matrix is
`configs/benchmarks/lidar/planner_compatibility_issue_1614.yaml`.

## Contract Boundary

The LiDAR track uses benchmark observation level `lidar_2d`. Runtime inputs must be limited to
robot state, goal information, and LiDAR/range rays. The track must not read simulator-backed
global occupancy, hidden obstacles, direct SocNav pedestrian positions/velocities, future
trajectories, or outcome labels.

The current contract gate already rejects most structured-state planners when
`observation_level=lidar_2d`. That fail-closed behavior is desirable: a planner should be
unavailable until a real LiDAR-derived adapter supplies its required inputs.

## Classification Summary

| Planner family | Current classification | Why |
| --- | --- | --- |
| PPO / guarded PPO | Training required | The contract gate accepts `sensor_fusion_state` under `lidar_2d`, but a LiDAR-trained checkpoint and registry provenance are still required. |
| SAC | Training and metadata work required | SAC is plausible for `drive_state+rays`, but current benchmark metadata does not yet expose it as a LiDAR-compatible benchmark row. |
| CrowdNav HEIGHT | Adapter required | The current `lidar_human_state` contract includes LiDAR rays plus human fields; it must not count as LiDAR-only until those human fields are LiDAR-derived. |
| `grid_route`, `teb`, `safety_barrier` | Ego-occupancy adapter required | These planners need local occupancy/clearance structure. The safe path is `lidar_rays -> ego occupancy_grid`, with no global map or hidden-obstacle access. After #1659, `safety_barrier` may pass the metadata gate only when the explicit LiDAR occupancy adapter preflight is satisfied. |
| `orca`, `social_force`, `risk_dwa` | Tracked-agent adapter required | These planners need agent positions/velocities. The safe path is `lidar_rays -> tracked_agents` with explicit noise, occlusion, and history assumptions. |
| Prediction-aware planners | Excluded from first LiDAR track | Current paths depend on structured pedestrians, history, or prediction features that are not yet derived from LiDAR observations. |
| SocNavBench-style adapters | Excluded until wrapped | Current external-style adapters consume structured state and should remain unavailable for LiDAR evidence unless a LiDAR-derived wrapper is built. |

## Selected Adapter Paths

- #1659: implement a `lidar_rays -> ego occupancy_grid` adapter and smoke one occupancy-backed
  classical planner such as `grid_route`, `teb`, or `safety_barrier`.
- #1660: implement a `lidar_rays -> tracked_agents` adapter and smoke one social-state planner
  such as `orca`, `social_force`, or `risk_dwa`.
- #1662: run the first LiDAR learned-policy smoke from the #1615 launch packet before any longer
  PPO/SAC/DreamerV3 training or registry promotion.

## Validation

Contract and matrix smoke:

```bash
uv run pytest -q tests/benchmark/test_lidar_planner_compatibility.py
```

Expected result: current PPO and guarded-PPO metadata pass the `lidar_2d` gate through
`sensor_fusion_state`; `safety_barrier` passes the sensor-fusion contract only when the explicit
LiDAR occupancy adapter config is present; other structured/grid planners fail closed; CrowdNav
HEIGHT keeps its `humans`-field caveat visible; and selected follow-up issues are recorded.

## Claim Boundary

Fallback, degraded, or unavailable rows are caveats or exclusions, not successful LiDAR-track
evidence. A planner becomes LiDAR-track evidence only after a targeted smoke run proves it consumes
LiDAR-derived inputs and the benchmark metadata records observation level, active observation mode,
execution mode, and adapter assumptions.
