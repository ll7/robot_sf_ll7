# Candidate Report: hybrid_rule_v3_teb_like_rollout (nominal_sanity)

## Decision

revise

## Hypothesis

Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/highest_success_policy/baseline_teb_nominal/summary.json`
- Git commit: `449de7a3d36b723760ba9bd6e4bd7c9c065c6434`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.1111 | 3.8577 | 1.6099 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.1667 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `timeout_low_progress`: `13`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | +0.0000 |
| orca | -0.0177 | -0.0355 | -4.2219 |
| ppo | -0.0815 | -0.0993 | -3.4139 |
