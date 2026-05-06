# Candidate Report: orca_primary_guarded_ppo_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

If residual blending cannot remove static failures, a stricter ORCA-primary guard can keep the learning stack inside the reciprocal-avoidance safety envelope while still allowing the PPO and risk-DWA branches to contribute when ORCA does not provide a safe/progressive command.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/20260505T134203+0200/orca_primary_guarded_ppo_v1/nominal_sanity/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.1667 | 4.1915 | 1.1974 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `timeout_low_progress`: `12`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | +0.0000 |
| orca | -0.0177 | -0.0355 | -4.1663 |
| ppo | -0.0815 | -0.0993 | -3.3583 |
