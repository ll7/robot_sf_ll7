# Candidate Report: hybrid_rule_v3_waypoint2_speed2p2 (full_matrix_h500)

## Decision

tracked

## Hypothesis

A mild 2.2 m/s speed envelope on top of the selected waypoint2 policy may recover long-route timeout cases without the safety regressions seen in the more aggressive 2.4 m/s progress-weight retune.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_speed2p2/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7292 | 0.1389 | 0.4514 | 3.0273 | 1.4089 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5652 | 0.1739 | 0.4928 |
| francis2023 | 75 | 0.8800 | 0.1067 | 0.4133 |

## Failure Taxonomy

- `near_miss_intrusive`: `8`
- `static_collision`: `20`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7150 | -0.1022 | +0.0000 |
| orca | +0.5448 | +0.1034 | -3.8816 |
| ppo | +0.4810 | +0.0396 | -3.0736 |
