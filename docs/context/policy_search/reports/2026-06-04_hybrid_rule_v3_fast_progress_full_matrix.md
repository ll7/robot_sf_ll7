# Candidate Report: hybrid_rule_v3_fast_progress (full_matrix)

## Decision

tracked

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `full_matrix`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/issue2221/hybrid_rule_v3_fast_progress/heldout_smoke/summary.json`
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
