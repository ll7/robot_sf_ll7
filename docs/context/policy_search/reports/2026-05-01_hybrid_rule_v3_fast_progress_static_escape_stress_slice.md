# Candidate Report: hybrid_rule_v3_fast_progress_static_escape (stress_slice)

## Decision

tracked

## Hypothesis

Remaining fast-progress failures include static-clearance stalls where forward rollout is rejected but a short rotation can reopen a safe slow-forward heading. Reusing the slow static-escape gate, adding a scored static-recenter probe, and using a faster scenario-specific static-corridor creep should recover doorway, crossing, and long corridor stalls while preserving the hard static-collision gate.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_scenario_override_stress_h500_w2/summary.json`
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
