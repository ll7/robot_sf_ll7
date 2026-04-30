# Candidate Report: hybrid_rule_v0_minimal (nominal_sanity)

## Decision

revise

## Hypothesis

A clean deterministic DWA-style hybrid-rule control variant with explicit safety filtering and score diagnostics should provide a transparent non-learning baseline for later social, ORCA, recovery, and ensemble mechanisms.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v0_minimal_nominal_sanity/summary.json`
- Git commit: `74481bb532ecd0a28d7c5e97110d07a788c8cb35`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.4444 | 0.2222 | 3.9469 | 1.7281 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.4167 | 0.3333 |
| francis2023 | 3 | 0.0000 | 1.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `2`
- `static_collision`: `8`
- `timeout_low_progress`: `5`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | +0.2033 | +0.0000 |
| orca | -0.0177 | +0.4089 | -4.1108 |
| ppo | -0.0815 | +0.3451 | -3.3028 |
