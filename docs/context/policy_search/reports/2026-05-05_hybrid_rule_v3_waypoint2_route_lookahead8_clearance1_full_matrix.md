# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8_clearance1 (full_matrix)

## Decision

tracked

## Hypothesis

Stronger route-guide clearance penalty may keep the 8-cell lookahead progress gain while avoiding the static collision without globally tightening the hard static margin.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_clearance1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2292 | 0.1250 | 0.3264 | 3.5422 | 1.5628 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1304 | 0.3623 |
| francis2023 | 75 | 0.2933 | 0.1200 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `32`
- `static_collision`: `18`
- `timeout_low_progress`: `61`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2150 | -0.1161 | +0.0000 |
| orca | +0.0448 | +0.0895 | -4.0066 |
| ppo | -0.0190 | +0.0257 | -3.1986 |
