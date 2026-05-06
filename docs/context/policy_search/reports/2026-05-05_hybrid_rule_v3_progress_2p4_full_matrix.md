# Candidate Report: hybrid_rule_v3_progress_2p4 (full_matrix)

## Decision

tracked

## Hypothesis

A moderate 2.4 m/s speed envelope with stronger progress pressure should recover long-route nominal-sanity timeouts without the near-miss and doorway regressions observed at 3.0 m/s.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_progress_2p4/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2708 | 0.0139 | 0.3403 | 3.4642 | 1.5119 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1739 | 0.0145 | 0.3623 |
| francis2023 | 75 | 0.3600 | 0.0133 | 0.3200 |

## Failure Taxonomy

- `near_miss_intrusive`: `39`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `63`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2566 | -0.2272 | +0.0000 |
| orca | +0.0864 | -0.0216 | -3.9927 |
| ppo | +0.0226 | -0.0854 | -3.1847 |
