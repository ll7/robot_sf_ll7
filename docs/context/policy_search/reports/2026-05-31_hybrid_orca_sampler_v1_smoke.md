# Candidate Report: hybrid_orca_sampler_v1 (smoke)

## Decision

pass

## Hypothesis

Keep ORCA-like safety behavior while allowing a short-horizon sampler to recover progress in constrained geometry.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_orca_sampler`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_orca_sampler_v1/smoke/pr1830_current_h120_probe/summary.json`
- Git commit: `b152a729e25499626010ef54dd61f594f948af23`
- Command override: `--horizon 120`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.6852 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Issue #1829 Comparison - 2026-05-31

This current-head rerun retunes the experimental candidate's speed envelope from the inherited
`1.1 m/s` cap to `2.0 m/s` for both the ORCA head and MPPI sampler, and keeps the route-stall
sampler handoff diagnostic added in the same branch.

The default 80-step smoke horizon still times out with low progress for this candidate. With the
explicit 120-step smoke horizon override above, the same `planner_sanity_simple` seed completes in
83 steps with no collision or near miss. This supports only the narrow claim that the retuned
candidate can recover progress on the smoke surface when given the same 120-step horizon used by
the broader policy-search sanity stages.

This is smoke evidence only. It does not promote `hybrid_orca_sampler_v1` beyond experimental
policy-search candidate status.

## Baseline Deltas

_Diagnostic-only arithmetic context; not a benchmark comparison claim._

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
