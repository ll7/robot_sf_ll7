# Issue #4183 Hybrid Global/RL Diagnostic

This packet records a diagnostic-only paired route/occupancy comparison for `hybrid_global_rl` against the same learned local policy without route conditioning.

- Evidence status: `diagnostic-only`
- Run status: `blocked_no_valid_episode_rows`
- Claim boundary: diagnostic-only paired route/occupancy comparison; not benchmark-improvement or paper-facing evidence
- Included diagnostic rows: 0
- Fallback/degraded rows excluded: 0
- Invalid pair rows: 0
- Linked work: #4161 and #4015

Rows marked fallback, degraded, unavailable, or missing-pair are not evidence for a route-conditioned effect. They remain diagnostic rows only.

## Fail-Closed Runner Failures

- arm=route_conditioned_hybrid_global_rl, scenario_id=francis2023_intersection_wait, seed=4183, reason=RuntimeError('hybrid_global_rl route waypoint unavailable'), source=bounded CPU run_map_batch smoke on imech036, row_classification=fail_closed_no_episode_row
- arm=learned_local_no_route_conditioning, scenario_id=francis2023_intersection_wait, seed=4183, reason=RuntimeError('PPO model unavailable or prediction failed'), source=bounded CPU run_map_batch smoke on imech036, row_classification=fail_closed_no_episode_row
