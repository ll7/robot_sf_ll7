# Candidate Report: hybrid_rule_v3_waypoint2_mild_comfort (full_matrix_h500)

## Decision

tracked

## Hypothesis

A mild dynamic-clearance increase on top of the waypoint2 candidate may reduce intrusive near misses without the static collision regression seen in the stronger comfort variant.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_mild_comfort/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7222 | 0.1458 | 0.3750 | 3.0519 | 1.3441 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5942 | 0.1739 | 0.4058 |
| francis2023 | 75 | 0.8400 | 0.1200 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `static_collision`: `21`
- `timeout_low_progress`: `13`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7080 | -0.0953 | +0.0000 |
| orca | +0.5378 | +0.1103 | -3.9580 |
| ppo | +0.4740 | +0.0465 | -3.1500 |
