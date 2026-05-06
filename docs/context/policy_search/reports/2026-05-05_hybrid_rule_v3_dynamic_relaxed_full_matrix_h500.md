# Candidate Report: hybrid_rule_v3_dynamic_relaxed (full_matrix_h500)

## Decision

tracked

## Hypothesis

Shortening only the hard dynamic collision horizon should reduce freezing in doorway and crossing scenes while retaining hard radius-based dynamic collision rejection, static footprint clearance, and route-guided progress.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_dynamic_relaxed/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7778 | 0.0139 | 0.4167 | 2.9400 | 1.3168 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6667 | 0.0145 | 0.4783 |
| francis2023 | 75 | 0.8800 | 0.0133 | 0.3600 |

## Failure Taxonomy

- `bottleneck_yield_failure`: `1`
- `near_miss_intrusive`: `12`
- `overconservative_stop`: `2`
- `static_collision`: `2`
- `timeout_low_progress`: `15`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7636 | -0.2272 | +0.0000 |
| orca | +0.5934 | -0.0216 | -3.9163 |
| ppo | +0.5296 | -0.0854 | -3.1083 |
