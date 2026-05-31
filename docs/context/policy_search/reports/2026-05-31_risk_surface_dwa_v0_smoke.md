# Candidate Report: risk_surface_dwa_v0 (smoke)

## Decision

pass

## Hypothesis

A deterministic local risk-surface producer can expose pedestrian proximity as an occupancy-compatible ego grid and let the existing Risk-DWA planner consume that surface through the normal policy-search smoke path. This is a prototype interface proof only, not learned-risk model or benchmark-strength evidence.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `risk_surface_dwa`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/risk_surface_dwa_v0/smoke/issue1820_progress_h80/summary.json`
- Git commit: `541ce2cb06bf2c39e88859e2a62ddc06d24fa0fe`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8153 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Issue #1820 Progress-Recovery Check

This run follows PR #1810, where the candidate first became executable through the normal
policy-search smoke path but still ended in low-progress timeout. The #1820 change keeps the
deterministic risk-surface fixture prototype-only and raises only the wrapped Risk-DWA smoke
command envelope to the environment's 2.0 m/s local limit.

Before #1820 on the same h80 smoke command:

- `termination_reason`: `max_steps`
- `failure_mode_counts`: `timeout_low_progress=1`
- `success_rate`: `0.0`
- `collision_rate`: `0.0`
- `mean_avg_speed`: `1.1999999999999975`

After #1820:

- `termination_reason`: `success`
- `success_rate`: `1.0`
- `collision_rate`: `0.0`
- `near_miss_rate`: `0.0`
- `mean_avg_speed`: `1.815306729624374`

## Claim Boundary

This is smoke evidence that the executable risk-surface DWA candidate no longer times out on
`planner_sanity_simple` under the h80 smoke path. It is not learned-risk model evidence, not
benchmark-strength promotion evidence, and not evidence of general nominal/stress performance.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | n/a |
| orca | +0.8156 | -0.0355 | n/a |
| ppo | +0.7518 | -0.0993 | n/a |
