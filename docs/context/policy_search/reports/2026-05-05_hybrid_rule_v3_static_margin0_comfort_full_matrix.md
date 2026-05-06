# Candidate Report: hybrid_rule_v3_static_margin0_comfort (full_matrix)

## Decision

tracked

## Hypothesis

Increasing dynamic clearance pressure and applying a lower speed cap out to 2.5 m should reduce doorway and crowd intrusive near misses without altering static safety.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0_comfort/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.1806 | 0.0972 | 0.2361 | 3.6245 | 1.4260 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1159 | 0.0580 | 0.2464 |
| francis2023 | 75 | 0.2400 | 0.1333 | 0.2267 |

## Failure Taxonomy

- `near_miss_intrusive`: `29`
- `static_collision`: `14`
- `timeout_low_progress`: `75`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1664 | -0.1439 | +0.0000 |
| orca | -0.0038 | +0.0617 | -4.0969 |
| ppo | -0.0676 | -0.0021 | -3.2889 |
