# Candidate Report: orca_residual_guarded_ppo_v0 (smoke)

## Decision

pass

## Claim Boundary

This is runtime-surface smoke proof only. It does not train a learned residual policy and must not
be counted as learned-residual contribution evidence for #1428 or #1358.

## Hypothesis

Unsafe PPO proposals can be reinterpreted as a bounded residual over the nominal ORCA command before falling through to the existing prior/fallback shield. This v0 entry wires the benchmark surface, clipping bounds, and diagnostics needed before launching the #1358 training campaign.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/orca_residual_guarded_ppo_v0/smoke/latest/summary.json`
- Git commit: `e14e2f8bc2058d9f0e071219629915dd5b5dd5a8`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.9962 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
