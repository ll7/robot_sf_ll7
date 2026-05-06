# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8_inflation4 (full_matrix)

## Decision

tracked

## Hypothesis

The 8-cell route lookahead improved nominal success but caused one static collision; increasing only route-guide obstacle inflation may preserve the progress gain while steering away from the static hazard.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_inflation4/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2222 | 0.1250 | 0.3403 | 3.5411 | 1.5594 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1304 | 0.3768 |
| francis2023 | 75 | 0.2800 | 0.1200 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `34`
- `static_collision`: `18`
- `timeout_low_progress`: `60`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.1161 | +0.0000 |
| orca | +0.0378 | +0.0895 | -3.9927 |
| ppo | -0.0260 | +0.0257 | -3.1847 |
