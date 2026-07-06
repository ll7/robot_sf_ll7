# Issue #4183 Hybrid Global/RL Diagnostic

This packet records a diagnostic-only paired route/occupancy comparison for `hybrid_global_rl` against the same learned local policy without route conditioning.

- Evidence status: `diagnostic-only`
- Run status: `completed_with_fail_closed_exclusions`
- Claim boundary: diagnostic-only paired route/occupancy comparison; not benchmark-improvement or paper-facing evidence
- Included diagnostic rows: 0
- Fallback/degraded rows excluded: 0
- Invalid pair rows: 3
- Linked work: #4161 and #4015

Rows marked fallback, degraded, unavailable, or missing-pair are not evidence for a route-conditioned effect. They remain diagnostic rows only.

## Integration Report

This section classifies the diagnostic packet state so the next empirical action is clear without promoting diagnostic-only evidence.

- New blockers: 0
- Next empirical action: Resolve the fail-closed runner blockers, then rerun the same paired route/occupancy diagnostic builder so route-conditioned and unconditioned arms emit matched native episode rows for the predeclared seeds.

### Blockers Remaining

- no_included_route_conditioned_effect_rows: status=remaining, evidence=The packet has no native, paired route-conditioned effect rows.
- fail_closed_route_conditioned_hybrid_global_rl: status=remaining, evidence=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row
- fail_closed_route_conditioned_hybrid_global_rl: status=remaining, evidence=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row
- fail_closed_route_conditioned_hybrid_global_rl: status=remaining, evidence=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row

### New Blockers

- none

### Intentional Exclusions

- fallback_or_degraded_rows_excluded: status=intentional, evidence=0 rows excluded from route-conditioned effect evidence
- pairing_errors_excluded: status=intentional, evidence=3 rows excluded because the scenario, seed, or checkpoint pairing contract was not satisfied

## Fail-Closed Runner Failures

- arm=route_conditioned_hybrid_global_rl, scenario_id=francis2023_intersection_wait, seed=4183, reason=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row, source=issue_4183_paired_runner
- arm=route_conditioned_hybrid_global_rl, scenario_id=francis2023_intersection_wait, seed=4184, reason=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row, source=issue_4183_paired_runner
- arm=route_conditioned_hybrid_global_rl, scenario_id=francis2023_intersection_wait, seed=4185, reason=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row, source=issue_4183_paired_runner
