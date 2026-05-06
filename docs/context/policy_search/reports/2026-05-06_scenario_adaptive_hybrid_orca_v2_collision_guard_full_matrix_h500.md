# Candidate Report: scenario_adaptive_hybrid_orca_v2_collision_guard (full_matrix_h500)

## Decision

tracked

## Hypothesis

The h500 leader misses the strict collision gate by one controllable merging collision plus two first-step dynamic-deadlock episodes that tuned ORCA does not repair. Disable the static-escape/recenter extras only on `classic_merging_low`, preserving the v1 selector elsewhere while reducing the controllable h500 collision slice.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/scenario_adaptive_hybrid_orca_v2_collision_guard/full_matrix_h500/policy_search_full_matrix_h500_collision_guard_20260506_0800/summary.json`
- Git commit: `d22c5e6c91b8f690a4ba0a1c6e32bf7aa3cf1b21`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.9028 | 0.0139 | 0.4236 | 2.8551 | 1.4072 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.8551 | 0.0145 | 0.5072 |
| francis2023 | 75 | 0.9467 | 0.0133 | 0.3467 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `static_collision`: `2`
- `timeout_low_progress`: `8`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.8886 | -0.2272 | +0.0000 |
| orca | +0.7184 | -0.0216 | -3.9094 |
| ppo | +0.6546 | -0.0854 | -3.1014 |

## Family Override Runs

- `classic`: success `0.8939`, collision `0.0152`
- `classic__classic_merging_low`: success `0.0000`, collision `0.0000`
- `francis2023`: success `0.9420`, collision `0.0145`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
