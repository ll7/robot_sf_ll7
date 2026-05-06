# Candidate Report: hybrid_rule_v3_waypoint2_speed2p2 (full_matrix)

## Decision

tracked

## Hypothesis

A mild 2.2 m/s speed envelope on top of the selected waypoint2 policy may recover long-route timeout cases without the safety regressions seen in the more aggressive 2.4 m/s progress-weight retune.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_speed2p2/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2361 | 0.1042 | 0.3889 | 3.5322 | 1.5579 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1739 | 0.1014 | 0.4058 |
| francis2023 | 75 | 0.2933 | 0.1067 | 0.3733 |

## Failure Taxonomy

- `near_miss_intrusive`: `42`
- `static_collision`: `15`
- `timeout_low_progress`: `53`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2219 | -0.1369 | +0.0000 |
| orca | +0.0517 | +0.0687 | -3.9441 |
| ppo | -0.0121 | +0.0049 | -3.1361 |
