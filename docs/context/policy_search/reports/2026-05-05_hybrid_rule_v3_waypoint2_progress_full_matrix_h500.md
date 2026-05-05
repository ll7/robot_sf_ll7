# Candidate Report: hybrid_rule_v3_waypoint2_progress (full_matrix_h500)

## Decision

tracked

## Hypothesis

Stronger manual progress and speed preference on top of waypoint2 may convert low-progress timeouts while preserving the same hard safety filters.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_progress/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7292 | 0.1389 | 0.4306 | 3.0218 | 1.4194 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5942 | 0.1594 | 0.4783 |
| francis2023 | 75 | 0.8533 | 0.1200 | 0.3867 |

## Failure Taxonomy

- `near_miss_intrusive`: `9`
- `static_collision`: `20`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7150 | -0.1022 | +0.0000 |
| orca | +0.5448 | +0.1034 | -3.9024 |
| ppo | +0.4810 | +0.0396 | -3.0944 |
