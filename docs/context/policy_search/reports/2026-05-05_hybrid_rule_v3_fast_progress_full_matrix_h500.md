# Candidate Report: hybrid_rule_v3_fast_progress (full_matrix_h500)

## Decision

tracked

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.8264 | 0.0139 | 0.4236 | 2.9111 | 1.3279 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.7391 | 0.0145 | 0.5217 |
| francis2023 | 75 | 0.9067 | 0.0133 | 0.3333 |

## Failure Taxonomy

- `near_miss_intrusive`: `10`
- `overconservative_stop`: `2`
- `static_collision`: `2`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.8122 | -0.2272 | +0.0000 |
| orca | +0.6420 | -0.0216 | -3.9094 |
| ppo | +0.5782 | -0.0854 | -3.1014 |
