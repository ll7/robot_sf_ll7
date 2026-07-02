# Issue #4015 Hybrid Global/RL Local Planner Design

This first slice adds a route-conditioned learned local planner adapter: an existing global
waypoint provider rewrites the local goal passed to an existing reinforcement learning (RL) local
policy.

## Scope

- Adds `HybridGlobalRLLocalAdapter` as an adapter/interface lane, not a new global planner.
- Uses `GridRoutePlannerAdapter` as the initial waypoint provider.
- Supports `SACPlanner` and `PPOPlanner` as existing learned local policy wrappers.
- Registers `hybrid_global_rl` and aliases through the map-runner policy-builder registry.

## Claim Boundary

Evidence status: `diagnostic-only`. This PR proves route waypoint injection, fail-closed behavior,
action conversion, and map-runner construction. It is not benchmark evidence and makes no robustness
claim.

## Remaining Work

- Run a representative route/occupancy scenario with a real learned local policy checkpoint.
- Produce a paired-seed diagnostic comparison against the same end-to-end RL policy without route
  conditioning.
- Add the analysis report that excludes fallback/degraded rows from benchmark-strength evidence.
