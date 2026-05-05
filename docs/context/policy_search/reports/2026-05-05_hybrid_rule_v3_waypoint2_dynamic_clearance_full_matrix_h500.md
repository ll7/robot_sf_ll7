# Candidate Report: hybrid_rule_v3_waypoint2_dynamic_clearance (full_matrix_h500)

## Decision

tracked

## Hypothesis

Slightly stronger dynamic-clearance scoring may reduce intrusive near misses without the static-collision regression caused by speed-cap comfort variants.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_dynamic_clearance/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7222 | 0.1319 | 0.4306 | 3.0266 | 1.3863 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5942 | 0.1449 | 0.4928 |
| francis2023 | 75 | 0.8400 | 0.1200 | 0.3733 |

## Failure Taxonomy

- `near_miss_intrusive`: `10`
- `static_collision`: `19`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7080 | -0.1092 | +0.0000 |
| orca | +0.5378 | +0.0964 | -3.9024 |
| ppo | +0.4740 | +0.0326 | -3.0944 |
