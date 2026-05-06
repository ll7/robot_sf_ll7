# Candidate Report: risk_guarded_ppo_v1 (full_matrix)

## Decision

tracked

## Hypothesis

The existing PPO success signal can be preserved if unsafe short-horizon actions are vetoed and replaced with safer local controls.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/risk_guarded_ppo_v1/full_matrix/policy_search_full_matrix_all_20260505/summary.json`
- Git commit: `3da4af0a6f424ee819cc3c7904d54745b45ac3c8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 144 | 0.1181 | 0.1736 | 0.2292 | 4.1621 | 1.1686 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 69 | 0.0290 | 0.2029 | 0.1594 |
| francis2023 | 75 | 0.2000 | 0.1467 | 0.2933 |

## Failure Taxonomy

- `near_miss_intrusive`: `31`
- `static_collision`: `25`
- `timeout_low_progress`: `71`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1039 | -0.0675 | +0.0000 |
| orca | -0.0663 | +0.1381 | -4.1038 |
| ppo | -0.1301 | +0.0743 | -3.2958 |
