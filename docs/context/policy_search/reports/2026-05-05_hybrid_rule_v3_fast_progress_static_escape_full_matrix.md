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
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress_static_escape/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2639 | 0.0139 | 0.3403 | 3.3583 | 1.5200 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3913 |
| francis2023 | 75 | 0.3600 | 0.0133 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `40`
- `static_collision`: `2`
- `timeout_low_progress`: `64`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2497 | -0.2272 | +0.0000 |
| orca | +0.0795 | -0.0216 | -3.9927 |
| ppo | +0.0157 | -0.0854 | -3.1847 |

## Family Override Runs

- `classic`: success `0.1594`, collision `0.0145`
- `francis2023`: success `0.3611`, collision `0.0139`
- `francis2023__francis2023_perpendicular_traffic`: success `0.3333`, collision `0.0000`
