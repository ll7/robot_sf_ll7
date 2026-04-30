# Candidate Report: hybrid_rule_v3_dynamic_relaxed (nominal_sanity)

## Decision

revise

## Hypothesis

Shortening only the hard dynamic collision horizon should reduce freezing in doorway and crossing scenes while retaining hard radius-based dynamic collision rejection, static footprint clearance, and route-guided progress.


## Evaluation Scope

- Stage: `nominal_sanity`
- Algorithm: `hybrid_rule_local_planner`
- Scenario matrix: `configs/policy_search/nominal_sanity_matrix.yaml`
- Seed manifest: `configs/policy_search/nominal_sanity_seeds.yaml`
- Summary JSON: `output/policy_search/hybrid_rule_v3_dynamic_relaxed_nominal/summary.json`
- Git commit: `93edf63efb9a5d91095387f157bc11ae072dbd74`

## Aggregate Results

| Episodes | Success | Collision | Near Miss | Mean MinDist | Mean AvgSpeed |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.2222 | 0.0000 | 0.1667 | 3.7283 | 1.6644 |

## Scenario-Family Split

| Family | Episodes | Success | Collision | Near Miss |
|---|---:|---:|---:|---:|
| classic | 12 | 0.0833 | 0.0000 | 0.2500 |
| francis2023 | 3 | 0.0000 | 0.0000 | 0.0000 |
| nominal | 3 | 1.0000 | 0.0000 | 0.0000 |

## Failure Taxonomy

- `near_miss_intrusive`: `3`
- `timeout_low_progress`: `11`

## Baseline Deltas

| Baseline | Success Delta | Collision Delta | Near-Miss Delta |
|---|---:|---:|---:|
| goal | +0.2080 | -0.2411 | +0.0000 |
| orca | +0.0378 | -0.0355 | -4.1663 |
| ppo | -0.0260 | -0.0993 | -3.3583 |
