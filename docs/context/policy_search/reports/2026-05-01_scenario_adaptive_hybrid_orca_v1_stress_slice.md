# Candidate Report: scenario_adaptive_hybrid_orca_v1 (stress_slice)

## Decision

tracked

## Hypothesis

The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/best_policy_next/scenario_adaptive_hybrid_orca_v1_stress_h500_w2/summary.json`
- Git commit: `8e5ceb432d67fe46e58f2349959bfbe8520dad88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 1.0000 | 0.0000 | 0.5000 | 3.2623 | 1.4886 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 1.0000 | 0.0000 | 0.5000 |
| francis2023 | 12 | 1.0000 | 0.0000 | 0.5000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -3.8330 |
| ppo | +0.7518 | -0.0993 | -3.0250 |

## Family Override Runs

- `classic`: success `1.0000`, collision `0.0000`
- `francis2023`: success `1.0000`, collision `0.0000`
