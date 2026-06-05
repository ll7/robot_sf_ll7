# Issue #2268 AMV Timeout Decomposition Manifest 2026-06-05

This directory preserves compact durable evidence for the #2268 AMV timeout decomposition.

## Tracked Files

- `summary.json`: machine-readable diagnostic classification, source rows, deltas, missing
  evidence, and recommendation.
- `timeout_decomposition.csv`: two-row timeout classification table comparing the baseline and
  actuation-aware candidates.

## Source Evidence

- `docs/context/issue_2224_amv_actuation_ranking.md`
- `docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json`
- `docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.md`
- `docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/manifest.json`
- `docs/context/policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_amv_actuation_smoke.md`
- `docs/context/policy_search/reports/2026-06-04_actuation_aware_hybrid_rule_v0_amv_actuation_smoke.md`

## Claim Boundary

This is analysis-only evidence over one synthetic AMV actuation smoke row. It may guide the next
instrumentation or trace-level rerun, but it must not be cited as planner-ranking,
hardware-calibrated, safety, benchmark-strength, or paper-facing AMV evidence.

The durable source evidence does not include per-step route-progress or command-feasibility time
series. The route-progress blocker classification is based on the preserved `timeout_low_progress`
failure mode plus unchanged speed/yaw/braking aggregates; finer mechanism attribution requires a
trace-level follow-up.
