# Candidate Report: scenario_adaptive_hybrid_orca_v2_collision_guard (leader_collision_slice_h500)

## Decision

tracked

## Hypothesis

The h500 leader misses the strict collision gate by one controllable merging collision plus two first-step dynamic-deadlock episodes that tuned ORCA does not repair. Disable the static-escape/recenter extras only on `classic_merging_low`, preserving the v1 selector elsewhere while reducing the controllable h500 collision slice.


## Evaluation Scope

- Stage: `leader_collision_slice_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `configs/policy_search/leader_collision_slice_h500_seeds.yaml`
- Summary JSON: `output/policy_search/scenario_adaptive_hybrid_orca_v2_collision_guard/leader_collision_slice_h500/policy_search_h500_collision_repair_v2_only_20260506_0752/summary.json`
- Git commit: `235caf755c0fe7e1e967c91d47fefa22c5dae517`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.5556 | 0.1111 | 0.3333 | 2.3305 | 0.9309 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 9 | 0.2222 | 0.1111 | 0.2222 |
| francis2023 | 9 | 0.8889 | 0.1111 | 0.4444 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `5`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.5414 | -0.1300 | +0.0000 |
| orca | +0.3712 | +0.0756 | -3.9997 |
| ppo | +0.3074 | +0.0118 | -3.1917 |

## Family Override Runs

- `classic`: success `0.3333`, collision `0.1667`
- `classic__classic_merging_low`: success `0.0000`, collision `0.0000`
- `francis2023`: success `0.6667`, collision `0.3333`
- `francis2023__francis2023_leave_group`: success `1.0000`, collision `0.0000`
- `francis2023__francis2023_perpendicular_traffic`: success `1.0000`, collision `0.0000`
