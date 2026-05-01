# Candidate Report: hybrid_rule_v3_teb_like_rollout (smoke)

## Decision

pass

## Hypothesis

Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_teb_like_rollout_smoke_static_full_rollout/summary.json`
- Git commit: `309a143d4052de3f1bd8cc0b11ffa155f786a017`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9082 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |
