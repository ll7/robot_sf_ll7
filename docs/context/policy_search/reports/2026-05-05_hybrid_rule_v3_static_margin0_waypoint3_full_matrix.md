# Candidate Report: hybrid_rule_v3_static_margin0_waypoint3 (full_matrix)

## Decision

tracked

## Hypothesis

A 3.0 m waypoint handoff distance may further reduce route-local stalls in long crossing/corridor scenes, but may be brittle around doorway waypoints.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0_waypoint3/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.1458 | 0.1111 | 0.3472 | 3.7082 | 1.5590 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1594 | 0.3478 |
| francis2023 | 75 | 0.1333 | 0.0667 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `40`
- `static_collision`: `16`
- `timeout_low_progress`: `67`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1316 | -0.1300 | +0.0000 |
| orca | -0.0386 | +0.0756 | -3.9858 |
| ppo | -0.1024 | +0.0118 | -3.1778 |
