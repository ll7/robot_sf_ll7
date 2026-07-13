# Issue #4882 S30 interpretation packet

**Claim boundary:** Diagnostic-only S30 interpretation. This packet does not promote a paper, dissertation, record-breaking, universal-planner, or real-world claim.

**Evidence status:** `diagnostic-only`. The packet composes the scheduler-frozen final-attempt
five-arm slice from job 13376 with the clean PPO-only job 13388. It excludes the contaminated
job 13378 aggregate and excludes SNQI ranking because the SNQI contract failed.

## Result

- Branch verdict: `branch_a_separation`.
- Hybrid v3 continuous versus ORCA success delta: `+0.0910` with 95% CI
  `[+0.0646, +0.1167]`.
- Collision-event delta: `-0.1347` with 95% CI
  `[-0.1590, -0.1097]`.
- Success-rank Kendall tau from the preregistered S20 prefix to S30: `0.8667`.

## Evidence boundaries

- Success is `metrics.success`; collision is `outcome.collision_event`.
- Intervals are paired or per-arm percentile bootstraps over seed-level means, not individual rows.
- S20 is seeds 111-130 derived from the identical frozen S30 rows, not an independent rerun.
- Hybrid/ORCA arms are adapter-mode; PPO is native. No fallback/degraded rows are admitted.
- The scenario-normalized quality index (SNQI) contract failed and is descriptive only.
- There is no independent clean-machine or real-world validation.

See `input_audit.json`, `campaign_crosscheck.json`, and `SHA256SUMS` for provenance and integrity.
