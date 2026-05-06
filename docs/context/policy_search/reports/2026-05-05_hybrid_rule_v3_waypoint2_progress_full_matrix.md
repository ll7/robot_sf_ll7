# Candidate Report: hybrid_rule_v3_waypoint2_progress (full_matrix)

## Decision

tracked

## Hypothesis

Stronger manual progress and speed preference on top of waypoint2 may convert low-progress timeouts while preserving the same hard safety filters.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_progress/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2292 | 0.1250 | 0.3333 | 3.5287 | 1.5668 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1739 | 0.1304 | 0.3333 |
| francis2023 | 75 | 0.2800 | 0.1200 | 0.3333 |

## Failure Taxonomy

- `near_miss_intrusive`: `32`
- `static_collision`: `18`
- `timeout_low_progress`: `61`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2150 | -0.1161 | +0.0000 |
| orca | +0.0448 | +0.0895 | -3.9997 |
| ppo | -0.0190 | +0.0257 | -3.1917 |
