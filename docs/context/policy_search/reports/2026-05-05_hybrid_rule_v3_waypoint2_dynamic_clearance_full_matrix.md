# Candidate Report: hybrid_rule_v3_waypoint2_dynamic_clearance (full_matrix)

## Decision

tracked

## Hypothesis

Slightly stronger dynamic-clearance scoring may reduce intrusive near misses without the static-collision regression caused by speed-cap comfort variants.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_dynamic_clearance/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2153 | 0.1042 | 0.3264 | 3.5385 | 1.5518 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0870 | 0.3478 |
| francis2023 | 75 | 0.2667 | 0.1200 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `35`
- `static_collision`: `15`
- `timeout_low_progress`: `63`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2011 | -0.1369 | +0.0000 |
| orca | +0.0309 | +0.0687 | -4.0066 |
| ppo | -0.0329 | +0.0049 | -3.1986 |
