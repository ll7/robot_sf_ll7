# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead6 (full_matrix_h500)

## Decision

tracked

## Hypothesis

A smaller route-guide lookahead increase may retain the nominal success gain suggested by 8 cells while avoiding its static-collision regression.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead6/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7639 | 0.1181 | 0.4097 | 3.0037 | 1.4261 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6232 | 0.1449 | 0.4783 |
| francis2023 | 75 | 0.8933 | 0.0933 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `7`
- `static_collision`: `17`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7497 | -0.1230 | +0.0000 |
| orca | +0.5795 | +0.0826 | -3.9233 |
| ppo | +0.5157 | +0.0188 | -3.1153 |
