# Candidate Report: hybrid_rule_v3_fast_progress_static_escape (full_matrix)

## Decision

tracked

## Hypothesis

Remaining fast-progress failures include static-clearance stalls where forward rollout is rejected but a short rotation can reopen a safe slow-forward heading. Reusing the slow static-escape gate, adding a scored static-recenter probe, and using a faster scenario-specific static-corridor creep should recover doorway, crossing, and long corridor stalls while preserving the hard static-collision gate.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/ai/autoresearch/best_policy_next/fast_progress_static_recenter07_scenario_override_full_h500_w2/summary.json`
- Git commit: `8e5ceb432d67fe46e58f2349959bfbe8520dad88`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 141 | 0.9220 | 0.0213 | 0.4113 | 2.8617 | 1.4294 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 66 | 0.9091 | 0.0303 | 0.4848 |
| francis2023 | 75 | 0.9333 | 0.0133 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `static_collision`: `3`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9078 | -0.2198 | +0.0000 |
| orca | +0.7376 | -0.0142 | -3.9217 |
| ppo | +0.6738 | -0.0780 | -3.1137 |

## Family Override Runs

- `classic`: success `0.9091`, collision `0.0303`
- `francis2023`: success `0.9306`, collision `0.0139`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
