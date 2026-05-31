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
- Summary JSON: `output/policy_search/risk_surface_dwa_v0/smoke/latest/summary.json`
- Git commit: `ae01fb0a3b7888bf789753a35be4428ff71c4a98`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 0.0000 | n/a | 0.0000 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `overconservative_stop`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | n/a |
| orca | -0.1844 | -0.0355 | n/a |
| ppo | -0.2482 | -0.0993 | n/a |
