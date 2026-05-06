# Candidate Report: hybrid_rule_v0_minimal (full_matrix_h500)

## Decision

tracked

## Hypothesis

A clean deterministic DWA-style hybrid-rule control variant with explicit safety filtering and score diagnostics should provide a transparent non-learning baseline for later social, ORCA, recovery, and ensemble mechanisms.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v0_minimal/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.7361 | 0.0208 | 0.4167 | 2.8873 | 1.2580 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.6667 | 0.0290 | 0.4783 |
| francis2023 | 75 | 0.8000 | 0.0133 | 0.3600 |

## Failure Taxonomy

- `bottleneck_yield_failure`: `1`
- `near_miss_intrusive`: `10`
- `overconservative_stop`: `1`
- `static_collision`: `3`
- `timeout_low_progress`: `23`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.7219 | -0.2203 | +0.0000 |
| orca | +0.5517 | -0.0147 | -3.9163 |
| ppo | +0.4879 | -0.0785 | -3.1083 |
