# Candidate Report: hybrid_rule_v3_waypoint2_mild_comfort (full_matrix)

## Decision

tracked

## Hypothesis

A mild dynamic-clearance increase on top of the waypoint2 candidate may reduce intrusive near misses without the static collision regression seen in the stronger comfort variant.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_waypoint2_mild_comfort/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2014 | 0.0903 | 0.2847 | 3.5731 | 1.5063 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1159 | 0.0580 | 0.3043 |
| francis2023 | 75 | 0.2800 | 0.1200 | 0.2667 |

## Failure Taxonomy

- `near_miss_intrusive`: `34`
- `static_collision`: `13`
- `timeout_low_progress`: `68`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1872 | -0.1508 | +0.0000 |
| orca | +0.0170 | +0.0548 | -4.0483 |
| ppo | -0.0468 | -0.0090 | -3.2403 |
