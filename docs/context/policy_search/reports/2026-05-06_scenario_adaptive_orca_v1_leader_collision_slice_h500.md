# Candidate Report: scenario_adaptive_orca_v1 (leader_collision_slice_h500)

## Decision

tracked

## Hypothesis

Family-specific ORCA parameterization should reduce classic bottleneck risk without slowing Francis-style flowing interactions too aggressively.


## Evaluation Scope

- Stage: `leader_collision_slice_h500`
- Algorithm: `orca`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `configs/policy_search/leader_collision_slice_h500_seeds.yaml`
- Summary JSON: `output/policy_search/scenario_adaptive_orca_v1/leader_collision_slice_h500/policy_search_h500_collision_repair_micro_repair_20260506_0747/summary.json`
- Git commit: `bdffdb36f12357eaf26808c09d889e75cce8c69a`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.4444 | 0.1667 | 0.6111 | 1.9181 | 0.8050 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 9 | 0.1111 | 0.1111 | 0.5556 |
| francis2023 | 9 | 0.7778 | 0.2222 | 0.6667 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `static_collision`: `3`
- `timeout_low_progress`: `3`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.4302 | -0.0744 | +0.0000 |
| orca | +0.2600 | +0.1312 | -3.7219 |
| ppo | +0.1962 | +0.0674 | -2.9139 |

## Family Override Runs

- `classic`: success `0.1111`, collision `0.1111`
- `francis2023`: success `0.7778`, collision `0.2222`
