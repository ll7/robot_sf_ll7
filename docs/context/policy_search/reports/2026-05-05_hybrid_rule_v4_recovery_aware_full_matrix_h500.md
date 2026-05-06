# Candidate Report: hybrid_rule_v4_recovery_aware (full_matrix_h500)

## Decision

tracked

## Hypothesis

Adding a narrow static-deadlock recovery mode to the route-guided hybrid planner should convert some safe low-progress stalls into goal progress without weakening dynamic-agent fail-closed behavior.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v4_recovery_aware/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.8056 | 0.0139 | 0.4097 | 2.8475 | 1.3448 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6957 | 0.0145 | 0.4638 |
| francis2023 | 75 | 0.9067 | 0.0133 | 0.3600 |

## Failure Taxonomy

- `bottleneck_yield_failure`: `1`
- `near_miss_intrusive`: `11`
- `static_collision`: `2`
- `timeout_low_progress`: `14`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7914 | -0.2272 | +0.0000 |
| orca | +0.6212 | -0.0216 | -3.9233 |
| ppo | +0.5574 | -0.0854 | -3.1153 |
