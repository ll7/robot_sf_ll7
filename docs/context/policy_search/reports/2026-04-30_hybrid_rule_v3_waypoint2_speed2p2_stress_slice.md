# Candidate Report: hybrid_rule_v3_waypoint2_speed2p2 (stress_slice)

## Decision

tracked

## Hypothesis

A mild 2.2 m/s speed envelope on top of the selected waypoint2 policy may recover long-route timeout cases without the safety regressions seen in the more aggressive 2.4 m/s progress-weight retune.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_speed2p2_stress/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.2500 | 0.0833 | 0.2917 | 4.7483 | 1.5932 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.2500 | 0.1667 | 0.3333 |
| francis2023 | 12 | 0.2500 | 0.0000 | 0.2500 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `static_collision`: `2`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2358 | -0.1578 | +0.0000 |
| orca | +0.0656 | +0.0478 | -4.0413 |
| ppo | +0.0018 | -0.0160 | -3.2333 |
