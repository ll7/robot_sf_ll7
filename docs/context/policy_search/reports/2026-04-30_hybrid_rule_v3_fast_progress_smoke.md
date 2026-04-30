# Candidate Report: hybrid_rule_v3_fast_progress (smoke)

## Decision

pass

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `smoke`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/scenarios/single/planner_sanity_simple.yaml`
- Seed manifest: `suite default`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress_smoke/summary.json`
- Git commit: `93edf63efb9a5d91095387f157bc11ae072dbd74`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000 | 0.0000 | 0.0000 | n/a | 1.8604 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| nominal | 1 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- No failures recorded.

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.9858 | -0.2411 | +0.0000 |
| orca | +0.8156 | -0.0355 | -4.3330 |
| ppo | +0.7518 | -0.0993 | -3.5250 |
