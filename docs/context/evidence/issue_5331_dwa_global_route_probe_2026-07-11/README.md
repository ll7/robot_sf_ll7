<!-- AI-GENERATED (robot_sf#5331, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5331 — DWA Global-Route Integration Probe for Bottleneck Convergence

## Scope

This diagnostic probes global-route waypoint integration for the classical DWA planner to test whether waypoint-following helps navigate through bottleneck corridors where the constant-velocity rollout cannot directly see the goal.

- Config: `configs/algos/dwa_global_route_probe.yaml`
- Matrix: `configs/scenarios/classic_interactions.yaml`
- Commit: `b6a374de6ee7150a9b67f16036e2bda05634a877`

## Episodes

### Bottleneck timeout (seed 131)

- Termination: max_steps
- Steps: 100
- Net progress: -2.876 m
- Min distance to goal: 0.461 m
- Global-route probe activated: True
- Global-route probe first activation step: 0

### T-intersection collision (seed 161)

- Termination: max_steps
- Steps: 100
- Net progress: 1.336 m
- Min distance to goal: 0.982 m
- Global-route probe activated: True
- Global-route probe first activation step: 0

## Claim boundary

This is a diagnostic-only trace. It makes no benchmark, metric, paper, or dissertation claim. Results indicate whether the global-route probe activates and whether it changes the episode outcome relative to the baseline.

## Limitations

- Two fixed-seed episodes only; not a representative sample.
- CPU-only, no training, no benchmark suite.
- The probe requires `route_waypoints` in the observation; episodes without waypoints fall back to baseline DWA behavior.
- Activation depends on the waypoint being within `global_route_probe_waypoint_distance` of the robot.
