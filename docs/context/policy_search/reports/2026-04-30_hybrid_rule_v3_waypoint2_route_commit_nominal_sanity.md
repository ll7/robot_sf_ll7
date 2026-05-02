# Candidate Report: hybrid_rule_v3_waypoint2_route_commit (nominal_sanity)

## Decision

revise

## Hypothesis

Giving the route-guide candidate a small deterministic bonus only after weak 3 s progress may reduce local route stalls while leaving normal DWA behavior unchanged in progressing scenes.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_commit_nominal/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.1111 | 0.1667 | 3.7788 | 1.6813 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.1667 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `static_collision`: `2`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.1300 | +0.0000 |
| orca | -0.0177 | +0.0756 | -4.1663 |
| ppo | -0.0815 | +0.0118 | -3.3583 |
