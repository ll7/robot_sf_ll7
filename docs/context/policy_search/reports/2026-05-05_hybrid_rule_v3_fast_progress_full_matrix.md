# Candidate Report: hybrid_rule_v3_fast_progress (full_matrix)

## Decision

tracked

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2569 | 0.0139 | 0.3333 | 3.4707 | 1.4979 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3913 |
| francis2023 | 75 | 0.3467 | 0.0133 | 0.2800 |

## Failure Taxonomy

- `near_miss_intrusive`: `40`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `64`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2427 | -0.2272 | +0.0000 |
| orca | +0.0725 | -0.0216 | -3.9997 |
| ppo | +0.0087 | -0.0854 | -3.1917 |
