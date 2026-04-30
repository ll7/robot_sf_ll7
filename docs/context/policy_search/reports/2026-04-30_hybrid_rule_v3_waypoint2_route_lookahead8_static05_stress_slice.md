# Candidate Report: hybrid_rule_v3_waypoint2_route_lookahead8_static05 (stress_slice)

## Decision

tracked

## Hypothesis

The 8-cell route lookahead may become acceptable if paired with a 5 cm static hard margin, trading some doorway progress for collision-free behavior.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_route_lookahead8_static05_stress/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.2917 | 0.0000 | 0.2917 | 4.7495 | 1.6285 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.3333 | 0.0000 | 0.3333 |
| francis2023 | 12 | 0.2500 | 0.0000 | 0.2500 |

## Failure Taxonomy

- `near_miss_intrusive`: `6`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2775 | -0.2411 | +0.0000 |
| orca | +0.1073 | -0.0355 | -4.0413 |
| ppo | +0.0435 | -0.0993 | -3.2333 |
