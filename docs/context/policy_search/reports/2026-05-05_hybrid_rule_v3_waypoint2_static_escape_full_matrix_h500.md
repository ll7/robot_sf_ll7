# Candidate Report: hybrid_rule_v3_waypoint2_static_escape (full_matrix_h500)

## Decision

tracked

## Hypothesis

Allowing only slow, non-worsening commands when the robot is already inside the conservative occupancy-grid static clearance band may reduce static deadlock timeouts without weakening occupied-cell collision rejection.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_static_escape/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7222 | 0.1597 | 0.4167 | 3.0364 | 1.4273 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5942 | 0.2029 | 0.4783 |
| francis2023 | 75 | 0.8400 | 0.1200 | 0.3600 |

## Failure Taxonomy

- `near_miss_intrusive`: `8`
- `static_collision`: `23`
- `timeout_low_progress`: `9`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7080 | -0.0814 | +0.0000 |
| orca | +0.5378 | +0.1242 | -3.9163 |
| ppo | +0.4740 | +0.0604 | -3.1083 |
