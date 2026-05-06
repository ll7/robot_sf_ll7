# Candidate Report: hybrid_rule_v3_teb_like_rollout (full_matrix_h500)

## Decision

tracked

## Hypothesis

Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_teb_like_rollout/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7708 | 0.0139 | 0.4097 | 2.9438 | 1.3102 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6667 | 0.0145 | 0.4638 |
| francis2023 | 75 | 0.8667 | 0.0133 | 0.3600 |

## Failure Taxonomy

- `bottleneck_yield_failure`: `1`
- `near_miss_intrusive`: `13`
- `overconservative_stop`: `2`
- `static_collision`: `2`
- `timeout_low_progress`: `15`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7566 | -0.2272 | +0.0000 |
| orca | +0.5864 | -0.0216 | -3.9233 |
| ppo | +0.5226 | -0.0854 | -3.1153 |
