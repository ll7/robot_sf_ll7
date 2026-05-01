# Candidate Report: planner_selector_v1 (smoke)

## Decision

pass

## Hypothesis

Existing non-learning and predictive heads can be combined into a stronger selector without introducing any new training dependency.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_portfolio`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/planner_selector_v1/smoke/latest/summary.json`
- Git commit: `69cfe6461a5cf90c8555d53bfda0a56b050d381d`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 3 | 0.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `timeout_low_progress`: `3`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | -0.0142 | -0.2411 | +0.0000 |
| orca | -0.1844 | -0.0355 | -4.3330 |
| ppo | -0.2482 | -0.0993 | -3.5250 |
