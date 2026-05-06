# Candidate Report: orca_prior_guarded_ppo_v2_static_global (smoke)

## Decision

pass

## Hypothesis

The first ORCA-prior guarded PPO pass recovered some success but still allowed static collision terminations when the PPO branch looked clear of pedestrians. Applying the ORCA prior globally with stricter obstacle clearance should reduce static-risk exposure while preserving enough learned PPO progress to remain competitive with ORCA.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/20260505T134203+0200/orca_prior_guarded_ppo_v2_static_global/smoke/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9527 |

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
