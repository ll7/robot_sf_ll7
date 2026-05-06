# Candidate Report: hybrid_rule_v3_waypoint2_route_commit (full_matrix_h500)

## Decision

tracked

## Hypothesis

Giving the route-guide candidate a small deterministic bonus only after weak 3 s progress may reduce local route stalls while leaving normal DWA behavior unchanged in progressing scenes.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_commit/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7708 | 0.1111 | 0.4653 | 2.9658 | 1.4021 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6087 | 0.1449 | 0.5362 |
| francis2023 | 75 | 0.9200 | 0.0800 | 0.4000 |

## Failure Taxonomy

- `near_miss_intrusive`: `7`
- `static_collision`: `16`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7566 | -0.1300 | +0.0000 |
| orca | +0.5864 | +0.0756 | -3.8677 |
| ppo | +0.5226 | +0.0118 | -3.0597 |
