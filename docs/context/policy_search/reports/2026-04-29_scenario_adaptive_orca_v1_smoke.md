# Candidate Report: scenario_adaptive_orca_v1 (smoke)

## Decision

pass

## Hypothesis

Family-specific ORCA parameterization should reduce classic bottleneck risk without slowing Francis-style flowing interactions too aggressively.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `orca`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/scenario_adaptive_orca_v1/smoke/latest/summary.json`
- Git commit: `69cfe6461a5cf90c8555d53bfda0a56b050d381d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 3 | 1.0000 | 0.0000 | 0.0000 | n/a | n/a |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |

## Family Override Runs

- `nominal`: success `1.0000`, collision `0.0000`
