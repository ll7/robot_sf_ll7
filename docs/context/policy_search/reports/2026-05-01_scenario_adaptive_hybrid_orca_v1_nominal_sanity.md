# Candidate Report: scenario_adaptive_hybrid_orca_v1 (nominal_sanity)

## Decision

pass

## Hypothesis

The retained fast-progress static-recenter hybrid candidate is strongest on most current full-matrix scenarios, while tuned ORCA cleanly resolves the remaining Francis leave-group hard-radius deadlock. A scenario-explicit classical selector should recover that case without changing the hybrid behavior on the other scenarios.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/best_policy_next/scenario_adaptive_hybrid_orca_v1_nominal_h500_w2/summary.json`
- Git commit: `8e5ceb432d67fe46e58f2349959bfbe8520dad88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 1.0000 | 0.0000 | 0.2778 | 3.0066 | 1.5961 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 1.0000 | 0.0000 | 0.4167 |
| francis2023 | 3 | 1.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.0552 |
| ppo | +0.7518 | -0.0993 | -3.2472 |

## Family Override Runs

- `classic`: success `1.0000`, collision `0.0000`
- `francis2023`: success `1.0000`, collision `0.0000`
- `nominal`: success `1.0000`, collision `0.0000`
