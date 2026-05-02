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
- Summary JSON: `output/ai/autoresearch/best_policy_next/scenario_adaptive_hybrid_orca_v1_full_h500_w2/summary.json`
- Git commit: `a381a5b30514c741b1dce9178af9d9fc9a420447`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 141 | 0.9291 | 0.0213 | 0.4113 | 2.8623 | 1.4375 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 66 | 0.9091 | 0.0303 | 0.4848 |
| francis2023 | 75 | 0.9467 | 0.0133 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `3`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9149 | -0.2198 | +0.0000 |
| orca | +0.7447 | -0.0142 | -3.9217 |
| ppo | +0.6809 | -0.0780 | -3.1137 |

## Family Override Runs

- `classic`: success `0.9091`, collision `0.0303`
- `francis2023`: success `0.9420`, collision `0.0145`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
