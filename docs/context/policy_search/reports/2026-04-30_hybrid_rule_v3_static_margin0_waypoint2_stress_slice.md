# Candidate Report: hybrid_rule_v3_static_margin0_waypoint2 (stress_slice)

## Decision

tracked

## Hypothesis

Switching from the active route waypoint at 2.0 m instead of 0.9 m may reduce route-local stalls in crossing and corridor scenes while keeping the selected v3 static safety settings.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0_waypoint2_stress/summary.json`
- Git commit: `1ca3d03cc3228c08ec1cbd61efc1675fd28b5f5a`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.3333 | 0.0000 | 0.2083 | 4.7580 | 1.6936 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.4167 | 0.0000 | 0.1667 |
| francis2023 | 12 | 0.2500 | 0.0000 | 0.2500 |

## Failure Taxonomy

- `near_miss_intrusive`: `5`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.3191 | -0.2411 | +0.0000 |
| orca | +0.1489 | -0.0355 | -4.1247 |
| ppo | +0.0851 | -0.0993 | -3.3167 |
