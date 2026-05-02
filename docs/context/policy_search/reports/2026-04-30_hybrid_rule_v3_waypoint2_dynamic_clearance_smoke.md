# Candidate Report: hybrid_rule_v3_waypoint2_dynamic_clearance (smoke)

## Decision

pass

## Hypothesis

Slightly stronger dynamic-clearance scoring may reduce intrusive near misses without the static-collision regression caused by speed-cap comfort variants.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_dynamic_clearance_smoke/summary.json`
- Git commit: `a819e4071ba5fdc1177b96462c1c191edd175502`

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
