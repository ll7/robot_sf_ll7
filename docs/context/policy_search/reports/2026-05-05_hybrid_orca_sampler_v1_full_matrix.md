# Candidate Report: hybrid_orca_sampler_v1 (full_matrix)

## Decision

tracked

## Hypothesis

Keep ORCA-like safety behavior while allowing a short-horizon sampler to recover progress in constrained geometry.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_orca_sampler`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_orca_sampler_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `1d7acbaac53b32dd4d656c5a31466b018dd131f6`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.0139 | 0.1111 | 0.1667 | 4.2580 | 0.8610 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.0000 | 0.1739 | 0.1884 |
| francis2023 | 75 | 0.0267 | 0.0533 | 0.1467 |

## Failure Taxonomy

- `near_miss_intrusive`: `22`
- `static_collision`: `16`
- `timeout_low_progress`: `104`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0003 | -0.1300 | +0.0000 |
| orca | -0.1705 | +0.0756 | -4.1663 |
| ppo | -0.2343 | +0.0118 | -3.3583 |
