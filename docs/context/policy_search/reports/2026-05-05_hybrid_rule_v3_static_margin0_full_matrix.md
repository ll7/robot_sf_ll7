# Candidate Report: hybrid_rule_v3_static_margin0 (full_matrix)

## Decision

tracked

## Hypothesis

Enforcing static hard clearance over the full rollout while using no extra static margin beyond the reported robot radius should preserve static safety without the doorway freezing caused by a 5 cm buffer in tight passages.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2153 | 0.1111 | 0.3264 | 3.5365 | 1.5564 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.1014 | 0.3623 |
| francis2023 | 75 | 0.2667 | 0.1200 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `35`
- `static_collision`: `16`
- `timeout_low_progress`: `62`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2011 | -0.1300 | +0.0000 |
| orca | +0.0309 | +0.0756 | -4.0066 |
| ppo | -0.0329 | +0.0118 | -3.1986 |
