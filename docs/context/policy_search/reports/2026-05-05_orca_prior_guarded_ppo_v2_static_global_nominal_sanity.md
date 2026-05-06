# Candidate Report: orca_prior_guarded_ppo_v2_static_global (nominal_sanity)

## Decision

revise

## Hypothesis

The first ORCA-prior guarded PPO pass recovered some success but still allowed static collision terminations when the PPO branch looked clear of pedestrians. Applying the ORCA prior globally with stricter obstacle clearance should reduce static-risk exposure while preserving enough learned PPO progress to remain competitive with ORCA.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v2_static_global/nominal_sanity/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.1111 | 0.1667 | 4.5559 | 1.6276 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.1667 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`
- `static_collision`: `2`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.1300 | +0.0000 |
| orca | +0.0934 | +0.0756 | -4.1663 |
| ppo | +0.0296 | +0.0118 | -3.3583 |
