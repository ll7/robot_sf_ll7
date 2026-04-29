# Candidate Report: scenario_adaptive_orca_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

Family-specific ORCA parameterization should reduce classic bottleneck risk without slowing Francis-style flowing interactions too aggressively.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `orca`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/scenario_adaptive_orca_v1/nominal_sanity/latest/summary.json`
- Git commit: `612a0db6c6de70c8078793dfb6037625567d53ed`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.1111 | 3.8592 | 1.0630 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.1667 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `timeout_low_progress`: `13`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | +0.0000 |
| orca | -0.0177 | -0.0355 | -4.2219 |
| ppo | -0.0815 | -0.0993 | -3.4139 |

## Family Override Runs

- `classic`: success `0.0000`, collision `0.0000`
- `francis2023`: success `0.0000`, collision `0.0000`
- `nominal`: success `1.0000`, collision `0.0000`
