# Candidate Report: hybrid_rule_v3_fast_progress_static_escape (full_matrix_h500)

## Decision

tracked

## Hypothesis

Remaining fast-progress failures include static-clearance stalls where forward rollout is rejected but a short rotation can reopen a safe slow-forward heading. Reusing the slow static-escape gate, adding a scored static-recenter probe, and using a faster scenario-specific static-corridor creep should recover doorway, crossing, and long corridor stalls while preserving the hard static-collision gate.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress_static_escape/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.9028 | 0.0208 | 0.4236 | 2.8372 | 1.4177 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.8696 | 0.0290 | 0.5072 |
| francis2023 | 75 | 0.9333 | 0.0133 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `5`
- `static_collision`: `3`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.8886 | -0.2203 | +0.0000 |
| orca | +0.7184 | -0.0147 | -3.9094 |
| ppo | +0.6546 | -0.0785 | -3.1014 |

## Family Override Runs

- `classic`: success `0.8696`, collision `0.0290`
- `francis2023`: success `0.9306`, collision `0.0139`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
