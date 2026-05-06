# Candidate Report: hybrid_rule_v3_waypoint2_static_escape (full_matrix)

## Decision

tracked

## Hypothesis

Allowing only slow, non-worsening commands when the robot is already inside the conservative occupancy-grid static clearance band may reduce static deadlock timeouts without weakening occupied-cell collision rejection.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_static_escape/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2153 | 0.1319 | 0.3264 | 3.5359 | 1.5676 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1449 | 0.3623 |
| francis2023 | 75 | 0.2667 | 0.1200 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `34`
- `static_collision`: `19`
- `timeout_low_progress`: `60`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2011 | -0.1092 | +0.0000 |
| orca | +0.0309 | +0.0964 | -4.0066 |
| ppo | -0.0329 | +0.0326 | -3.1986 |
