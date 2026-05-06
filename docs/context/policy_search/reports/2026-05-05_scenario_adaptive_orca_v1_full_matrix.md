# Candidate Report: scenario_adaptive_orca_v1 (full_matrix)

## Decision

tracked

## Hypothesis

Family-specific ORCA parameterization should reduce classic bottleneck risk without slowing Francis-style flowing interactions too aggressively.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `orca`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/scenario_adaptive_orca_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.0486 | 0.0347 | 0.2708 | 4.2730 | 0.9414 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.0000 | 0.0145 | 0.2899 |
| francis2023 | 75 | 0.0933 | 0.0533 | 0.2533 |

## Failure Taxonomy

- `near_miss_intrusive`: `32`
- `static_collision`: `5`
- `timeout_low_progress`: `100`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.0344 | -0.2064 | +0.0000 |
| orca | -0.1358 | -0.0008 | -4.0622 |
| ppo | -0.1996 | -0.0646 | -3.2542 |

## Family Override Runs

- `classic`: success `0.0000`, collision `0.0145`
- `francis2023`: success `0.0933`, collision `0.0533`
