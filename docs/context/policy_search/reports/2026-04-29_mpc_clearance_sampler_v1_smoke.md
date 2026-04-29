# Candidate Report: mpc_clearance_sampler_v1 (smoke)

## Decision

pass

## Hypothesis

A deterministic NMPC-style rollout scorer should improve constrained-geometry progress without giving up clearance control.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `nmpc_social`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/mpc_clearance_sampler_v1/smoke/latest/summary.json`
- Git commit: `0cc24586dd1323705d4bab7fbdda43d71929649b`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 3 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.7896 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |
