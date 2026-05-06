# Candidate Report: ppo_issue791_best_v1 (nominal_sanity)

## Decision

revise

## Hypothesis

The repository's current best learned checkpoint from issue 791 should be the strongest feasible learning-only candidate: eval-aligned large-capacity PPO with predictive foresight, reward curriculum, and fail-closed model inference, without additional action-level ORCA wrapping that degraded the 2026-05-05 guarded-PPO attempts.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `ppo`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/20260505_best_learning/ppo_issue791_best_v1/nominal_sanity/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2778 | 0.0000 | 0.2222 | 3.7136 | 1.8471 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.1667 | 0.0000 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `timeout_low_progress`: `10`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2636 | -0.2411 | +0.0000 |
| orca | +0.0934 | -0.0355 | -4.1108 |
| ppo | +0.0296 | -0.0993 | -3.3028 |
