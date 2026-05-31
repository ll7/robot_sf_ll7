# Candidate Report: tentabot_value_scorer_v3_trace_recovery (smoke)

## Decision

pass

## Hypothesis

A clean-room Tentabot-style primitive value scorer can keep the v1 static safety gate while adding an explicit trace-level route recovery selector that only chooses already accepted corridor-subgoal or route-guide candidates under recent route regression or combined route/goal stall. This tests route-recovery policy rather than another speed, clearance, static-gate, or scalar route-progress retune.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/tentabot_value_scorer_v3_trace_recovery/smoke/issue1908_v3/summary.json`
- Git commit: `33a067ec166c41967a6a1e9cd94d584b180c3e4b`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 1.5545 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
