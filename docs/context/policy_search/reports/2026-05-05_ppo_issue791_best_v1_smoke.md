# Candidate Report: ppo_issue791_best_v1 (smoke)

## Decision

pass

## Hypothesis

The repository's current best learned checkpoint from issue 791 should be the strongest feasible learning-only candidate: eval-aligned large-capacity PPO with predictive foresight, reward curriculum, and fail-closed model inference, without additional action-level ORCA wrapping that degraded the 2026-05-05 guarded-PPO attempts.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/20260505_best_learning/ppo_issue791_best_v1/smoke/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9998 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |
