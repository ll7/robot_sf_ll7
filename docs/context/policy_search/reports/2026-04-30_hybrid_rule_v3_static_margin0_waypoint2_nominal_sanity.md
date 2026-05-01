# Candidate Report: hybrid_rule_v3_static_margin0_waypoint2 (nominal_sanity)

## Decision

revise

## Hypothesis

Switching from the active route waypoint at 2.0 m instead of 0.9 m may reduce route-local stalls in crossing and corridor scenes while keeping the selected v3 static safety settings.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/highest_success_policy/baseline_waypoint2_nominal/summary.json`
- Git commit: `449de7a3d36b723760ba9bd6e4bd7c9c065c6434`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.1111 | 0.1667 | 3.8824 | 1.6684 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.1667 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `static_collision`: `2`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.1300 | +0.0000 |
| orca | -0.0177 | +0.0756 | -4.1663 |
| ppo | -0.0815 | +0.0118 | -3.3583 |
