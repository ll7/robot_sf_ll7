# Candidate Report: hybrid_rule_v3_fast_progress (nominal_sanity)

## Decision

revise

## Hypothesis

Raising the manually specified speed envelope to the 3.0 m/s robot limit used elsewhere in the repository should reduce low-progress timeouts on long route slices while preserving the v3 static and dynamic safety filters.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_fast_progress_nominal/summary.json`
- Git commit: `93edf63efb9a5d91095387f157bc11ae072dbd74`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.1667 | 0.0000 | 0.2222 | 3.7599 | 1.5624 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0000 | 0.0000 | 0.3333 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `4`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.1525 | -0.2411 | +0.0000 |
| orca | -0.0177 | -0.0355 | -4.1108 |
| ppo | -0.0815 | -0.0993 | -3.3028 |
