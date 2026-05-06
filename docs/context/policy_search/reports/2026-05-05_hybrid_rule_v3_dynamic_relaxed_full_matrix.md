# Candidate Report: hybrid_rule_v3_dynamic_relaxed (full_matrix)

## Decision

tracked

## Hypothesis

Shortening only the hard dynamic collision horizon should reduce freezing in doorway and crossing scenes while retaining hard radius-based dynamic collision rejection, static footprint clearance, and route-guided progress.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_dynamic_relaxed/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `a395f943ce0c1d10c9cf88e7e2a52ca302ebc38e`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2431 | 0.0139 | 0.3125 | 3.4878 | 1.5001 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3478 |
| francis2023 | 75 | 0.3200 | 0.0133 | 0.2800 |

## Failure Taxonomy

- `near_miss_intrusive`: `38`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `68`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2289 | -0.2272 | +0.0000 |
| orca | +0.0587 | -0.0216 | -4.0205 |
| ppo | -0.0051 | -0.0854 | -3.2125 |
