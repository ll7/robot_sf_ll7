# Candidate Report: issue_2170_static_recenter_only (full_matrix)

## Decision

tracked

## Hypothesis

One-factor diagnostic row for #2104/#2174: enable static recenter scoring on the fast-progress baseline while keeping static escape, corridor transit, corridor subgoal, and continuous static checks disabled.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/issue2221/issue_2170_static_recenter_only/heldout_smoke/summary.json`
- Git commit: `ed8c6817650105f5ebf8420d3d12c98c67a24ecb`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.5000 | 0.0000 | 0.5000 | 4.8987 | 1.5643 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 1 | 0.0000 | 0.0000 | 1.0000 |
| francis2023 | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `1`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.4858 | -0.2411 | n/a |
| orca | +0.3156 | -0.0355 | n/a |
| ppo | +0.2518 | -0.0993 | n/a |

## Family Override Runs

- `classic`: success `0.0000`, collision `0.0000`
- `francis2023`: success `1.0000`, collision `0.0000`
