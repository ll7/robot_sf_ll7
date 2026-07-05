# Issue #4183 Hybrid Global/RL Diagnostic

This packet records a diagnostic-only paired route/occupancy comparison for `hybrid_global_rl` against the same learned local policy without route conditioning.

- Evidence status: `diagnostic-only`
- Run status: `blocked_no_valid_episode_rows`
- Claim boundary: diagnostic-only paired route/occupancy comparison; not benchmark-improvement or paper-facing evidence
- Included diagnostic rows: 0
- Fallback/degraded rows excluded: 0
- Invalid pair rows: 0
- Linked work: #4161 and #4015
- Registry preflight: `preflight_registry_checkpoint_20260705.json` records the durable
  `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` checkpoint
  reference and the current `blocked_missing_learned_checkpoint` cache state.

Rows marked fallback, degraded, unavailable, or missing-pair are not evidence for a route-conditioned effect. They remain diagnostic rows only.

## Integration Report

This section classifies the diagnostic packet state so the next empirical action is clear without promoting diagnostic-only evidence.

- New blockers: 0
- Next empirical action: Resolve the fail-closed runner blockers, then rerun the same paired route/occupancy diagnostic builder so route-conditioned and unconditioned arms emit matched native episode rows for the predeclared seeds.

### Blockers Remaining

- no_valid_episode_rows: status=remaining, evidence=Fail-closed runner failures produced no paired episode rows.
- fail_closed_route_conditioned_hybrid_global_rl: status=remaining, evidence=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row
- fail_closed_learned_local_no_route_conditioning: status=remaining, evidence=RuntimeError('PPO model unavailable or prediction failed'), row_classification=fail_closed_no_episode_row

### New Blockers

- none

### Intentional Exclusions

- fallback_or_degraded_rows_excluded: status=intentional, evidence=0 rows excluded from route-conditioned effect evidence
- pairing_errors_excluded: status=intentional, evidence=0 rows excluded because the scenario, seed, or checkpoint pairing contract was not satisfied

## Fail-Closed Runner Failures

- arm=route_conditioned_hybrid_global_rl, reason=RuntimeError('hybrid_global_rl route waypoint unavailable'), row_classification=fail_closed_no_episode_row, scenario_id=francis2023_intersection_wait, seed=4183, source=bounded CPU run_map_batch smoke on imech036
- arm=learned_local_no_route_conditioning, reason=RuntimeError('PPO model unavailable or prediction failed'), row_classification=fail_closed_no_episode_row, scenario_id=francis2023_intersection_wait, seed=4183, source=bounded CPU run_map_batch smoke on imech036
