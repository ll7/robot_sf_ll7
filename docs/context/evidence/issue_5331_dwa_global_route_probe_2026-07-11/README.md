<!-- AI-GENERATED (robot_sf#5331, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5331 — DWA Global-Route Integration Probe for Bottleneck Convergence

## Scope

This diagnostic probes global-route waypoint integration for the classical DWA
planner to test whether waypoint-following helps navigate through bottleneck
corridors where the constant-velocity rollout cannot directly see the goal.

This is a successor to the #5319 route-rescue probe (PR #5328), which showed
that route-rescue did not activate on the bottleneck case because DWA's
constant-velocity rollout cannot "see" through the bottleneck corridor.

- Config: `configs/algos/dwa_global_route_probe.yaml`
- Matrix: `configs/scenarios/classic_interactions.yaml`
- Probe mechanism: biases DWA toward the next global-route waypoint by computing
  a heading-alignment score at the rollout endpoint

## Mechanism

The global-route integration probe adds a waypoint-following bias to the DWA
scoring function. When `global_route_probe_enabled=true` and `route_waypoints`
are present in the observation, the probe:

1. Finds the nearest usable waypoint within `global_route_probe_waypoint_distance`
   and advances to the next route waypoint when one is available
2. Computes heading alignment from the rollout endpoint to that forward waypoint
3. Adds `global_route_probe_heading_weight * waypoint_score` to the base DWA score

This helps DWA navigate through bottleneck corridors where the constant-velocity
rollout cannot directly see the goal.

## Acceptance criteria

- [x] Planner/config tests cover the new contract and malformed-input failure mode
- [ ] The evidence packet states whether the probe activates and whether either original mechanism changes
- [x] Results remain diagnostic-only unless a separate benchmark decision establishes a broader claim

## Test results

- 54 targeted DWA planner tests pass, including 17 global-route probe tests.
- 2 trace-runner production-call contract tests pass.

## Validation commands

```bash
uv run pytest tests/planner/test_dwa_global_route_probe.py tests/planner/test_dwa.py -v
uv run pytest tests/benchmark/test_trace_dwa_global_route_probe_issue_5331.py -v
uv run ruff check robot_sf/planner/dwa.py scripts/benchmark/trace_dwa_global_route_probe_issue_5331.py tests/planner/test_dwa_global_route_probe.py tests/benchmark/test_trace_dwa_global_route_probe_issue_5331.py
uv run ruff format --check robot_sf/planner/dwa.py scripts/benchmark/trace_dwa_global_route_probe_issue_5331.py tests/planner/test_dwa_global_route_probe.py tests/benchmark/test_trace_dwa_global_route_probe_issue_5331.py
```

## Trace script

To run the diagnostic trace:

```bash
uv run python scripts/benchmark/trace_dwa_global_route_probe_issue_5331.py --evidence-dir docs/context/evidence/issue_5331_dwa_global_route_probe_2026-07-11
```

## Claim boundary

This is a diagnostic-only trace. It makes no benchmark, metric, paper, or
dissertation claim. Results indicate whether the global-route probe activates
and whether it changes the episode outcome relative to the baseline.

## Limitations

- Two fixed-seed episodes only; not a representative sample.
- CPU-only, no training, no benchmark suite.
- The probe requires `route_waypoints` in the observation; episodes without
  waypoints fall back to baseline DWA behavior.
- Activation depends on the waypoint being within
  `global_route_probe_waypoint_distance` of the robot.
