# Candidate Report: orca_residual_guarded_ppo_v0 (smoke)

## Decision

pass

## Issue #1475 Gate

failed_closed / revise.

The smoke runner produced one valid episode row, so the runner-level decision is `pass`, but the
#1475 wrapper gate exited nonzero because `success_rate=0.0` and success is required before nominal
escalation. Do not submit `nominal_sanity` from this result.

## Hypothesis

Unsafe PPO proposals can be reinterpreted as a bounded residual over the nominal ORCA command before falling through to the existing prior/fallback shield. This v0 entry wires the benchmark surface, clipping bounds, and diagnostics needed before launching the #1358 training campaign.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `guarded_ppo`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/slurm/issue1475-orca-residual-bc-job-12749/policy_search/orca_residual_guarded_ppo_v0/smoke/issue1475_smoke/summary.json`
- Git commit: `5faaa318d609f87730757d7fbda65b799178b5c5`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.8038 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
