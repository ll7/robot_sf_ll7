# Candidate Report: tentabot_value_scorer_v3_trace_recovery (nominal_sanity)

## Decision

revise

## Hypothesis

A clean-room Tentabot-style primitive value scorer can keep the v1 static safety gate while adding an explicit trace-level route recovery selector that only chooses already accepted corridor-subgoal or route-guide candidates under recent route regression or combined route/goal stall. This tests route-recovery policy rather than another speed, clearance, static-gate, or scalar route-progress retune.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/tentabot_value_scorer_v3_trace_recovery/nominal_sanity/issue1908_v3/summary.json`
- Git commit: `33a067ec166c41967a6a1e9cd94d584b180c3e4b`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.1111 | 0.1667 | 4.1333 | 1.4183 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.1667 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `static_collision`: `2`
- `timeout_low_progress`: `9`

## Claim Boundary

This report is diagnostic-only wiring or stage evidence. Treat aggregate metrics and baseline deltas as arithmetic context, not benchmark-strength evidence for comfort, near-miss behavior, generalization, or planner superiority.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1300 | n/a |
| orca | +0.0378 | +0.0756 | n/a |
| ppo | -0.0260 | +0.0118 | n/a |
