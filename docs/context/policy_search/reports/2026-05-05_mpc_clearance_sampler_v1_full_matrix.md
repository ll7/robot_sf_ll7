# Candidate Report: mpc_clearance_sampler_v1 (full_matrix)

## Decision

tracked

## Hypothesis

A deterministic NMPC-style rollout scorer should improve constrained-geometry progress without giving up clearance control.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `nmpc_social`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/mpc_clearance_sampler_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.2361 | 0.2847 | 0.1319 | 3.8400 | 1.5660 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.1449 | 0.4203 | 0.1594 |
| francis2023 | 75 | 0.3200 | 0.1600 | 0.1067 |

## Failure Taxonomy

- `static_collision`: `41`
- `timeout_low_progress`: `69`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2219 | +0.0436 | +0.0000 |
| orca | +0.0517 | +0.2492 | -4.2011 |
| ppo | -0.0121 | +0.1854 | -3.3931 |
