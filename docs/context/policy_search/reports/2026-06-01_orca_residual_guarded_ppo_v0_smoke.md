# Candidate Report: orca_residual_guarded_ppo_v0 (smoke)

## Decision

pass

## Hypothesis

Unsafe PPO proposals can be reinterpreted as a bounded residual over the nominal ORCA command before falling through to the existing prior/fallback shield. This v0 entry wires the benchmark surface, clipping bounds, and diagnostics needed before launching the #1358 training campaign.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/slurm/issue1475-orca-residual-bc-local-1967-rerun2-20260601T043657Z/policy_search/orca_residual_guarded_ppo_v0/smoke/issue1475_smoke/summary.json`
- Git commit: `2469d1d0f5adf0d9a4afc95c40b483c6f2855c1f`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.8041 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Issue #1475 Gate Note

This candidate-runner smoke decision is `pass` because the adapter fix produced a usable episode row. The stricter #1475 wrapper gate is still `revise`: success_rate is `0.0000`, collision_rate is `0.0000`, and nominal_sanity was not submitted.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
