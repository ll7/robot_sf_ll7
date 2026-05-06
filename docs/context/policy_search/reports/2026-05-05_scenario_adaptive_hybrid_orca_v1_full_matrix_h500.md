# Candidate Report: scenario_adaptive_hybrid_orca_v1 (full_matrix_h500)

## Decision

tracked

## Hypothesis

The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/scenario_adaptive_hybrid_orca_v1/full_matrix_h500/policy_search_full_matrix_h500_leaders_clean_20260505_204501/summary.json`
- Git commit: `2b796ea92104467d3bc913528801fb8bb11034dd`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.9097 | 0.0208 | 0.4236 | 2.8377 | 1.4256 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.8696 | 0.0290 | 0.5072 |
| francis2023 | 75 | 0.9467 | 0.0133 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `static_collision`: `3`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.8955 | -0.2203 | +0.0000 |
| orca | +0.7253 | -0.0147 | -3.9094 |
| ppo | +0.6615 | -0.0785 | -3.1014 |

## Family Override Runs

- `classic`: success `0.8696`, collision `0.0290`
- `francis2023`: success `0.9420`, collision `0.0145`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
