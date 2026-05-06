# Candidate Report: hybrid_rule_v3_fast_progress (leader_collision_slice_h500)

## Decision

tracked

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `leader_collision_slice_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `configs/policy_search/leader_collision_slice_h500_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress/leader_collision_slice_h500/policy_search_h500_collision_repair_micro_20260506_0740/summary.json`
- Git commit: `661505bdde0bd9743fd45d2b5c34b610216ee639`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.3333 | 0.1111 | 0.3333 | 3.1246 | 0.6266 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 9 | 0.0000 | 0.1111 | 0.2222 |
| francis2023 | 9 | 0.6667 | 0.1111 | 0.4444 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `6`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.3191 | -0.1300 | +0.0000 |
| orca | +0.1489 | +0.0756 | -3.9997 |
| ppo | +0.0851 | +0.0118 | -3.1917 |
