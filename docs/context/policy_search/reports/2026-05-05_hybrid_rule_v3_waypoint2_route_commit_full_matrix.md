# Candidate Report: hybrid_rule_v3_waypoint2_route_commit (full_matrix)

## Decision

tracked

## Hypothesis

Giving the route-guide candidate a small deterministic bonus only after weak 3 s progress may reduce local route stalls while leaving normal DWA behavior unchanged in progressing scenes.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_commit/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2431 | 0.0903 | 0.3472 | 3.4905 | 1.5468 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1014 | 0.3768 |
| francis2023 | 75 | 0.3200 | 0.0800 | 0.3200 |

## Failure Taxonomy

- `near_miss_intrusive`: `37`
- `static_collision`: `13`
- `timeout_low_progress`: `59`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2289 | -0.1508 | +0.0000 |
| orca | +0.0587 | +0.0548 | -3.9858 |
| ppo | -0.0051 | -0.0090 | -3.1778 |
