# Candidate Report: planner_selector_v1 (full_matrix)

## Decision

tracked

## Hypothesis

Existing non-learning and predictive heads can be combined into a stronger selector without introducing any new training dependency.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_portfolio`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/planner_selector_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.0486 | 0.3056 | 0.2569 | 4.4297 | 1.1146 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.0290 | 0.4058 | 0.2174 |
| francis2023 | 75 | 0.0667 | 0.2133 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `19`
- `static_collision`: `44`
- `timeout_low_progress`: `74`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.0344 | +0.0645 | +0.0000 |
| orca | -0.1358 | +0.2701 | -4.0761 |
| ppo | -0.1996 | +0.2063 | -3.2681 |
