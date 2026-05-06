# Candidate Report: hybrid_rule_v0_minimal (full_matrix)

## Decision

tracked

## Hypothesis

A clean deterministic DWA-style hybrid-rule control variant with explicit safety filtering and score diagnostics should provide a transparent non-learning baseline for later social, ORCA, recovery, and ensemble mechanisms.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v0_minimal/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `1d7acbaac53b32dd4d656c5a31466b018dd131f6`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2569 | 0.0139 | 0.3264 | 3.4491 | 1.4842 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1594 | 0.0145 | 0.3478 |
| francis2023 | 75 | 0.3467 | 0.0133 | 0.3067 |

## Failure Taxonomy

- `near_miss_intrusive`: `39`
- `overconservative_stop`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `65`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2427 | -0.2272 | +0.0000 |
| orca | +0.0725 | -0.0216 | -4.0066 |
| ppo | +0.0087 | -0.0854 | -3.1986 |
