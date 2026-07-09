<!-- AI-GENERATED (robot_sf#4831) - NEEDS-REVIEW -->

# Issue #4831: Trace-Verified Failure-Mechanism Labels

Generated: 2026-07-09T04:23:17+00:00

Campaign root: output/issue4206-trace-rerun/13334

- Total episodes: 6480
- Success: 3868
- Failure: 2612
- Labeled: 2612
- Unlabeled residual: 0
- Coverage: 100.00%

## guarded_ppo

guarded_ppo is excluded from derivation (accepted-unavailable status).

## Label Distribution

- proxemic_or_clearance_tradeoff: 2265
- time_budget_artifact: 197
- static_deadlock_or_local_minimum: 95
- dynamic_phase_or_order_sensitivity: 55

## Evidence Mode
All derived labels use `paired_trace` evidence from event_ledger, outcome,
and safety predicates in the episode JSONL records.

## Claim Boundary
This packet derives failure-mechanism labels from trace surfaces available in
the episode JSONL. Confidence levels reflect the strength of derivation evidence.
Labels are diagnostic, not paper-facing claims.
