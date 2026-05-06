# Candidate Report: orca_primary_guarded_ppo_v1 (smoke)

## Decision

pass

## Hypothesis

If residual blending cannot remove static failures, a stricter ORCA-primary guard can keep the learning stack inside the reciprocal-avoidance safety envelope while still allowing the PPO and risk-DWA branches to contribute when ORCA does not provide a safe/progressive command.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/20260505T134203+0200/orca_primary_guarded_ppo_v1/smoke/summary.json`
- Git commit: `25282b457c8b83ebf128a3fcc311597af3d76763`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 1.3485 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | +0.0000 |
| orca | -0.1844 | -0.0355 | -4.3330 |
| ppo | -0.2482 | -0.0993 | -3.5250 |
