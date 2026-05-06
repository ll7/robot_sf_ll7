# Candidate Report: scenario_adaptive_hybrid_orca_v1 (leader_collision_slice_h500)

## Decision

tracked

## Hypothesis

The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios.


## Evaluation Scope

- Stage: `leader_collision_slice_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `configs/policy_search/leader_collision_slice_h500_seeds.yaml`
- Summary JSON: `output/policy_search/scenario_adaptive_hybrid_orca_v1/leader_collision_slice_h500/policy_search_h500_collision_repair_micro_20260506_0740/summary.json`
- Git commit: `661505bdde0bd9743fd45d2b5c34b610216ee639`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.6111 | 0.1667 | 0.3333 | 2.1941 | 1.0780 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 9 | 0.3333 | 0.2222 | 0.2222 |
| francis2023 | 9 | 0.8889 | 0.1111 | 0.4444 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `3`
- `timeout_low_progress`: `3`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.5969 | -0.0744 | +0.0000 |
| orca | +0.4267 | +0.1312 | -3.9997 |
| ppo | +0.3629 | +0.0674 | -3.1917 |

## Family Override Runs

- `classic`: success `0.3333`, collision `0.2222`
- `francis2023`: success `0.6667`, collision `0.3333`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
