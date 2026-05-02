# Candidate Report: hybrid_rule_v3_fast_progress (stress_slice)

## Decision

tracked

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `stress_slice`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/stress_slice_matrix.yaml`
- Seed manifest: `configs/policy_search/stress_slice_seeds.yaml`
- Summary JSON: `output/ai/autoresearch/highest_success_policy/incumbent_fast_progress_stress/summary.json`
- Git commit: `7dd2beeba0bef04f5e642c42034a4645a4040f08`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.2917 | 0.0000 | 0.3333 | 4.7691 | 1.5942 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.3333 | 0.0000 | 0.4167 |
| francis2023 | 12 | 0.2500 | 0.0000 | 0.2500 |

## Failure Taxonomy

- `near_miss_intrusive`: `8`
- `timeout_low_progress`: `9`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2775 | -0.2411 | +0.0000 |
| orca | +0.1073 | -0.0355 | -3.9997 |
| ppo | +0.0435 | -0.0993 | -3.1917 |
