# Candidate Report: hybrid_rule_v4_recovery_aware (full_matrix)

## Decision

tracked

## Hypothesis

Adding a narrow static-deadlock recovery mode to the route-guided hybrid planner should convert some safe low-progress stalls into goal progress without weakening dynamic-agent fail-closed behavior.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v4_recovery_aware/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Durable summary evidence: `not promoted`; the `output/` path is retained only as regeneration context.
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2500 | 0.0139 | 0.3125 | 3.3974 | 1.5178 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3188 |
| francis2023 | 75 | 0.3333 | 0.0133 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `38`
- `static_collision`: `2`
- `timeout_low_progress`: `68`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2358 | -0.2272 | +0.0000 |
| orca | +0.0656 | -0.0216 | -4.0205 |
| ppo | +0.0018 | -0.0854 | -3.2125 |
