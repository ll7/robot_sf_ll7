# Candidate Report: scenario_adaptive_hybrid_orca_v1 (full_matrix)

## Decision

tracked

## Hypothesis

The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/scenario_adaptive_hybrid_orca_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2778 | 0.0139 | 0.3403 | 3.3588 | 1.5241 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3913 |
| francis2023 | 75 | 0.3867 | 0.0133 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `38`
- `static_collision`: `2`
- `timeout_low_progress`: `64`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.2272 | +0.0000 |
| orca | +0.0934 | -0.0216 | -3.9927 |
| ppo | +0.0296 | -0.0854 | -3.1847 |

## Family Override Runs

- `classic`: success `0.1594`, collision `0.0145`
- `francis2023`: success `0.3623`, collision `0.0145`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `0.3333`, collision `0.0000`
