# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead6 (full_matrix)

## Decision

tracked

## Hypothesis

A smaller route-guide lookahead increase may retain the nominal success gain suggested by 8 cells while avoiding its static-collision regression.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead6/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2431 | 0.0972 | 0.3333 | 3.4947 | 1.5671 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1014 | 0.3768 |
| francis2023 | 75 | 0.3200 | 0.0933 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `34`
- `static_collision`: `14`
- `timeout_low_progress`: `61`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2289 | -0.1439 | +0.0000 |
| orca | +0.0587 | +0.0617 | -3.9997 |
| ppo | -0.0051 | -0.0021 | -3.1917 |
