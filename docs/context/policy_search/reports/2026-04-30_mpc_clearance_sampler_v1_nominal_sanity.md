# Candidate Report: mpc_clearance_sampler_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

A deterministic NMPC-style rollout scorer should improve constrained-geometry progress without giving up clearance control.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `nmpc_social`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/mpc_clearance_sampler_v1_nominal/summary.json`
- Git commit: `93edf63efb9a5d91095387f157bc11ae072dbd74`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.2778 | 0.2222 | 4.0883 | 1.5560 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.4167 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `static_collision`: `5`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | +0.0367 | +0.0000 |
| orca | -0.0177 | +0.2423 | -4.1108 |
| ppo | -0.0815 | +0.1785 | -3.3028 |
