# Candidate Report: mpc_clearance_sampler_v1 (full_matrix_h500)

## Decision

tracked

## Hypothesis

A deterministic NMPC-style rollout scorer should improve constrained-geometry progress without giving up clearance control.


## Evaluation Scope

- Stage: `full_matrix_h500`
- Algorithm: `nmpc_social`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/mpc_clearance_sampler_v1/full_matrix_h500/policy_search_full_matrix_h500_gt20_20260505/summary.json`
- Git commit: `47fecd938482949b7989f1011ec6e34237d8b45d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.6111 | 0.3750 | 0.1597 | 3.5100 | 1.5500 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.3913 | 0.6087 | 0.2174 |
| francis2023 | 75 | 0.8133 | 0.1600 | 0.1067 |

## Failure Taxonomy

- `overconservative_stop`: `1`
- `static_collision`: `54`
- `timeout_low_progress`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.5969 | +0.1339 | +0.0000 |
| orca | +0.4267 | +0.3395 | -4.1733 |
| ppo | +0.3629 | +0.2757 | -3.3653 |
