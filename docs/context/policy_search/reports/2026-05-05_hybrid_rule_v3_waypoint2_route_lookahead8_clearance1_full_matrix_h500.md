# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8_clearance1 (full_matrix_h500)

## Decision

tracked

## Hypothesis

Stronger route-guide clearance penalty may keep the 8-cell lookahead progress gain while avoiding the static collision without globally tightening the hard static margin.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_clearance1/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.6944 | 0.1597 | 0.4097 | 3.0318 | 1.3876 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5652 | 0.2029 | 0.4638 |
| francis2023 | 75 | 0.8133 | 0.1200 | 0.3600 |

## Failure Taxonomy

- `bottleneck_yield_failure`: `1`
- `near_miss_intrusive`: `6`
- `static_collision`: `23`
- `timeout_low_progress`: `14`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.6802 | -0.0814 | +0.0000 |
| orca | +0.5100 | +0.1242 | -3.9233 |
| ppo | +0.4462 | +0.0604 | -3.1153 |
