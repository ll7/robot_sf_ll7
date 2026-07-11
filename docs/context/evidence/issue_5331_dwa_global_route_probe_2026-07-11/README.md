<!-- AI-GENERATED (robot_sf#5331, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5331 — DWA Global-Route Integration Probe for Bottleneck Convergence

## Scope

This diagnostic probes global-route waypoint integration for the classical DWA planner to test whether waypoint-following helps navigate through bottleneck corridors where the constant-velocity rollout cannot directly see the goal.

- Config: `configs/algos/dwa_global_route_probe.yaml`
- Matrix: `configs/scenarios/classic_interactions.yaml`
- Commit: `8d84eda28d2b72e9ebebd1bd47ae4aa9198ea6f0`

## Episodes

### Bottleneck timeout (seed 131)

- Termination: max_steps
- Steps: 100
- Net progress: -2.885 m
- Min distance to goal: 0.474 m
- Global-route probe activated: False

### T-intersection collision (seed 161)

- Termination: collision
- Steps: 96
- Net progress: 1.178 m
- Min distance to goal: 1.575 m
- Global-route probe activated: False

## Outcome comparison

The probe did not activate on either recorded step. Under the fail-closed scoring contract, its waypoint term is therefore zero and DWA uses the baseline score. This is a scoring-contract inference, not an independently rerun baseline comparator.

## Acceptance criteria

- [x] Planner/config tests cover the new contract and malformed-input failure mode
- [x] The evidence packet states whether the probe activates and whether either original mechanism changes
- [x] Results remain diagnostic-only unless a separate benchmark decision establishes a broader claim

## Claim boundary

This is a diagnostic-only trace. It makes no benchmark, metric, paper, or dissertation claim. Results indicate whether the global-route probe activates and whether it changes the episode outcome relative to the baseline.

## Limitations

- Two fixed-seed episodes only; not a representative sample.
- CPU-only, no training, no benchmark suite.
- The probe requires `route_waypoints` in the observation; episodes without waypoints fall back to baseline DWA behavior.
- Activation depends on the waypoint being within `global_route_probe_waypoint_distance` of the robot.
