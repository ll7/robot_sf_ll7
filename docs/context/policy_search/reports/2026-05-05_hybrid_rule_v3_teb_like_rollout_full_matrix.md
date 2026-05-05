# Candidate Report: hybrid_rule_v3_teb_like_rollout (full_matrix)

## Decision

tracked

## Hypothesis

Adding an occupancy-grid route-guide candidate to the safety-filtered DWA scorer should recover progress in static/corridor local minima without giving back the collision improvement from the static footprint filter.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_teb_like_rollout/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2431 | 0.0139 | 0.3194 | 3.4865 | 1.4983 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3333 |
| francis2023 | 75 | 0.3200 | 0.0133 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `39`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `67`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2289 | -0.2272 | +0.0000 |
| orca | +0.0587 | -0.0216 | -4.0136 |
| ppo | -0.0051 | -0.0854 | -3.2056 |
