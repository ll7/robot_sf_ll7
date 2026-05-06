# Candidate Report: hybrid_rule_v3_static_margin0 (full_matrix_h500)

## Decision

tracked

## Hypothesis

Enforcing static hard clearance over the full rollout while using no extra static margin beyond the reported robot radius should preserve static safety without the doorway freezing caused by a 5 cm buffer in tight passages.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_static_margin0/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7222 | 0.1389 | 0.4236 | 3.0258 | 1.3952 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.5942 | 0.1594 | 0.4928 |
| francis2023 | 75 | 0.8400 | 0.1200 | 0.3600 |

## Failure Taxonomy

- `near_miss_intrusive`: `10`
- `static_collision`: `20`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7080 | -0.1022 | +0.0000 |
| orca | +0.5378 | +0.1034 | -3.9094 |
| ppo | +0.4740 | +0.0396 | -3.1014 |
