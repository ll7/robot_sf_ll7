# Issue #690 Holonomic Benchmark Feasibility Note

## What Changed

- Added a parallel holonomic camera-ready profile at `configs/benchmarks/camera_ready_all_planners_holonomic.yaml`.
- Added a strict PPO benchmark config at `configs/baselines/ppo_15m_grid_socnav_holonomic.yaml` with `fallback_to_goal: false`.
- Split the ORCA execution path so holonomic `vx_vy` runs can forward a world-frame velocity
  vector directly instead of round-tripping through `unicycle_vw`.
- Applied the same direct holonomic world-velocity path to the
  `social_navigation_pyenvs_orca` wrapper so the upstream `ActionXY` contract is preserved in the
  holonomic benchmark.
- Added explicit action/observation contract docs at
  `docs/dev/holonomic_action_contract.md` and corrected `docs/dev/observation_contract.md`.
- Kept the current differential-drive camera-ready benchmark untouched.

## Feasibility Assessment

The holonomic profile is feasible with the current repository shape.

Why:

- The runtime already supports `HolonomicDriveSettings` and `HolonomicDriveRobot`.
- The benchmark runner already maps holonomic commands through `holonomic_command_mode: vx_vy` or `unicycle_vw`.
- The existing kinematics-parity benchmark preset already includes holonomic execution.
- PPO already has a benchmark-facing command conversion path; the remaining benchmark concern is strictness, not basic kinematics support.

## Recommendation

- Keep `camera_ready_all_planners.yaml` as the current differential-drive baseline.
- Use the new holonomic sibling profile as a co-existing investigation surface.
- Treat incompatible planner/dependency combinations as unavailable rather than silently fallback-successful.
- Keep PPO in prototype-only mode for the holonomic profile unless a later retrained checkpoint proves it can run cleanly without fallback.

## Compatibility Snapshot

| Planner | Holonomic path | Notes |
| --- | --- | --- |
| `goal` | yes | Native command projection already exists. |
| `social_force` | yes | Classical adapter path; no fallback needed in strict mode. |
| `orca` | yes, if prereqs are present | World-frame velocity can now be forwarded directly in `vx_vy`; use fail-fast instead of fallback when dependencies are missing. |
| `ppo` | yes, prototype-only | Existing command conversion works; strict no-fallback config used here. |
| `prediction_planner` | yes | Uses the existing adapter path and the current checkpoint lookup. |
| `socnav_sampling` | yes, if prereqs are present | Strictly fail-closed in the holonomic profile. |
| `sacadrl` | yes, if prereqs are present | Strictly fail-closed in the holonomic profile. |
| `socnav_bench` | yes, if prereqs are present | Strictly fail-closed in the holonomic profile. |

## Suggested Proof Step

Run a preflight-only holonomic campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_all_planners_holonomic.yaml \
  --mode preflight \
  --label issue690_holonomic_preflight
```

This should validate the matrix, seed policy, and strict fail-closed planner policy without
overwriting the current benchmark contract.

## Follow-up Evidence

- Added `configs/benchmarks/holonomic_upstream_wrappers_probe.yaml` to generate real holonomic
  benchmark artifacts for upstream wrapper planners.
- Real probe artifacts now show both `social_navigation_pyenvs_orca` and
  `social_navigation_pyenvs_sfm_helbing` with:
  - `execution_detail = direct_holonomic_world_velocity`
  - `planner_command_space = holonomic_vxy_world`
  - `benchmark_command_space = holonomic_vxy_world`
  - `projection_policy = world_velocity_passthrough`
- Probe campaign root:
  `output/benchmarks/camera_ready/holonomic_upstream_wrappers_probe_issue690_upstream_wrappers_holonomic_probe_20260326_140518`
- Local `social_force` was also upgraded to expose a direct holonomic world-velocity path.
- That change improved contract fidelity, but not benchmark quality. Compared with the previous
  corrected holonomic campaign:
  - success: `0.0071 -> 0.0000`
  - collisions: `0.2411 -> 0.1135`
  - SNQI: `-3.1917 -> -3.6294`
  - jerk: `0.4338 -> 1.7103`
- Interpretation:
  - Direct holonomic passthrough is clearly the right contract for ORCA-family wrappers.
  - It is not automatically beneficial for every planner that internally computes a world-frame
    velocity.
  - For local `social_force`, the direct holonomic path changes closed-loop behavior in a way that
    lowers collisions but worsens overall benchmark quality and smoothness.
