# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8 (nominal_sanity)

## Decision

revise

## Hypothesis

Looking farther along the route-guide grid path may reduce long-route stalls in crossing, corridor, and Francis-style scenarios without changing safety margins or social comfort terms.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_nominal/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.3333 | 0.0556 | 0.1667 | 3.9712 | 1.7855 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.2500 | 0.0833 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `1`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.3191 | -0.1855 | +0.0000 |
| orca | +0.1489 | +0.0201 | -4.1663 |
| ppo | +0.0851 | -0.0437 | -3.3583 |
